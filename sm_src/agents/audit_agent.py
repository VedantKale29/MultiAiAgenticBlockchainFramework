"""
agents/audit_agent.py
=====================
STAGE 2 -- Audit Agent

WHAT IT DOES:
  1. After every batch, writes a structured incident report to
     audit_log.jsonl (append-only, one JSON object per line --
     simulates blockchain immutability locally).

  2. Feeds the incident report BACK into the RAG knowledge base
     (FraudKnowledgeAgent) -- this is the self-improving loop from
     the framework doc Section 4.2.

  3. Optionally uploads the audit log to S3 alongside other run
     artifacts (uses existing S3Manager -- no new AWS setup).

AUDIT RECORD SCHEMA:
  Every record has:
    - incident_id     : deterministic ID (run_name + batch + timestamp hash)
    - batch           : batch number
    - timestamp       : UTC ISO timestamp
    - trigger         : what caused this record (BLOCK/WATCHLIST/batch_summary)
    - detection       : risk_score, p_rf, s_if, decision
    - policy          : policy_action, policy_reason
    - rag_context     : retrieved similar past events (if RAG available)
    - outcome         : y_true label (ground truth for supervised eval)
    - system_state    : tau_alert, tau_block, w at time of decision
    - batch_metrics   : precision, recall, f1 for the batch

IMMUTABILITY SIMULATION:
  audit_log.jsonl is append-only. The AuditAgent NEVER overwrites
  existing records. Each record includes a sha256 hash of the
  previous record's content -- creating a tamper-evident chain that
  simulates on-chain audit log semantics without needing a real
  blockchain. (In Stage 5, this gets replaced with a real Hardhat
  transaction.)

ZERO BREAKING CHANGES:
  - Does not modify any existing agent
  - Does not modify coordinator_agent.py (coordinator calls it
    as an optional post-step after the main pipeline)
  - If FraudKnowledgeAgent is unavailable, audit still runs --
    rag_context field is empty string
  - All failures are caught and logged -- never crashes main pipeline
"""

import os
import json
import hashlib
from datetime import datetime
from typing import Optional

from agents.base_agent import BaseAgent, AgentMessage
import numpy as np


class AuditAgent(BaseAgent):
    """
    Append-only audit logger with RAG self-improvement loop.
    """

    def __init__(
        self,
        run_dir: str,
        run_name: str,
        knowledge_agent=None,   # FraudKnowledgeAgent instance (optional)
        s3=None,                # S3Manager (optional -- same as other agents)
    ):
        """
        Parameters
        ----------
        run_dir : str
            Run directory (same as all other agents).
        run_name : str
            Run identifier, e.g. "run_seed42_v1".
        knowledge_agent : FraudKnowledgeAgent | None
            If provided, audit reports are fed back into the RAG store.
        s3 : S3Manager | None
            If provided, audit_log.jsonl is uploaded to S3 after each batch.
        """
        super().__init__(name="AuditAgent")
        self.run_dir         = run_dir
        self.run_name        = run_name
        self.knowledge_agent = knowledge_agent
        self.s3              = s3

        self.audit_log_path = os.path.join(run_dir, "audit_log.jsonl")
        self._last_hash      = "GENESIS"   # chain anchor for first record

        os.makedirs(run_dir, exist_ok=True)

        # If a partial audit log already exists (resume scenario),
        # read the hash of its last record to continue the chain.
        self._last_hash = self._read_last_hash()

    # ═══════════════════════════════════════════════════════════
    # CHAIN INTEGRITY
    # ═══════════════════════════════════════════════════════════
    def _read_last_hash(self) -> str:
        """Read the hash of the last written record (for chain continuity)."""
        if not os.path.exists(self.audit_log_path):
            return "GENESIS"
        try:
            last_line = None
            with open(self.audit_log_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        last_line = line
            if last_line:
                record = json.loads(last_line)
                return record.get("record_hash", "GENESIS")
        except Exception:
            pass
        return "GENESIS"

    @staticmethod
    def _hash_record(record: dict) -> str:
        """SHA-256 hash of a record's content (for tamper-evident chain)."""
        content = json.dumps(record, sort_keys=True, default=str)
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    # ═══════════════════════════════════════════════════════════
    # WRITE -- append a record (never overwrites)
    # ═══════════════════════════════════════════════════════════
    def _append_record(self, record: dict):
        """Append one JSON record to the append-only audit log."""
        record["prev_hash"]    = self._last_hash
        record["record_hash"]  = self._hash_record(record)
        self._last_hash        = record["record_hash"]

        with open(self.audit_log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, default=str) + "\n")

    # ═══════════════════════════════════════════════════════════
    # BUILD INCIDENT ID
    # ═══════════════════════════════════════════════════════════
    def _incident_id(self, batch_idx: int, suffix: str = "") -> str:
        ts = datetime.utcnow().strftime("%Y%m%dT%H%M%S")
        raw = f"{self.run_name}:batch{batch_idx+1}:{ts}{suffix}"
        return hashlib.sha256(raw.encode()).hexdigest()[:12]

    # ═══════════════════════════════════════════════════════════
    # MAIN _run -- called by coordinator after each batch
    # ═══════════════════════════════════════════════════════════
    def _run(self, msg: AgentMessage) -> AgentMessage:
        """
        Reads the full batch context from the AgentMessage payload
        and writes audit records for:
          - every BLOCK or WATCHLIST decision (individual records)
          - one batch_summary record (aggregate metrics)

        Then feeds the batch's fraud events back into RAG.
        """
        # ── Extract payload ──────────────────────────────────────
        decisions      = np.asarray(msg.payload.get("decisions",     []), dtype=object)
        policy_actions = np.asarray(msg.payload.get("policy_actions",[]), dtype=object)
        policy_reasons = np.asarray(msg.payload.get("policy_reasons",[]), dtype=object)
        risk_scores    = np.asarray(msg.payload.get("risk_scores",   []), dtype=float)
        p_rf           = np.asarray(msg.payload.get("p_rf",          []), dtype=float)
        s_if           = np.asarray(msg.payload.get("s_if",          []), dtype=float)
        y_batch        = msg.payload.get("y_batch",    [])
        batch_idx      = msg.payload.get("batch_idx",  0)
        agent_state    = msg.payload.get("agent_state", {})
        batch_log      = msg.payload.get("batch_log",  {})
        tx_meta        = msg.payload.get("tx_meta",    {})

        tx_hashes  = tx_meta.get("tx_hash",       ["unknown"] * len(decisions)) if tx_meta else ["unknown"] * len(decisions)
        from_addrs = tx_meta.get("from_address",  ["unknown"] * len(decisions)) if tx_meta else ["unknown"] * len(decisions)

        # ── Individual event records for BLOCK / WATCHLIST ────────
        records_written = 0
        for i in range(len(decisions)):
            action = str(policy_actions[i]) if i < len(policy_actions) else "ALLOW"
            if action not in ("BLOCK", "WATCHLIST"):
                continue

            risk  = float(risk_scores[i]) if i < len(risk_scores) else 0.0
            rf    = float(p_rf[i])        if i < len(p_rf)        else 0.0
            sf    = float(s_if[i])        if i < len(s_if)        else 0.0
            dec   = str(decisions[i])     if i < len(decisions)   else "CLEAR"
            reas  = str(policy_reasons[i])if i < len(policy_reasons) else ""
            wallet= str(from_addrs[i])    if hasattr(from_addrs, '__getitem__') else "unknown"
            ytrue = int(y_batch.iloc[i])  if hasattr(y_batch, "iloc") else int(y_batch[i]) if i < len(y_batch) else -1
            txh   = str(tx_hashes[i])     if hasattr(tx_hashes, '__getitem__') else "unknown"

            # RAG context -- retrieve similar past events
            rag_context = ""
            if self.knowledge_agent is not None:
                try:
                    rag_context = self.knowledge_agent.build_rag_context(
                        risk_score=risk, p_rf=rf, s_if=sf,
                        decision=dec, wallet=wallet, n_results=3,
                    )
                except Exception as e:
                    self.logger.warning(f"[{self.name}] RAG query failed: {e}")

            record = {
                "incident_id"   : self._incident_id(batch_idx, suffix=txh[:6]),
                "batch"         : batch_idx + 1,
                "timestamp"     : datetime.utcnow().isoformat(),
                "trigger"       : action,
                "detection"     : {
                    "risk_score": risk,
                    "p_rf"      : rf,
                    "s_if"      : sf,
                    "decision"  : dec,
                },
                "policy"        : {
                    "action": action,
                    "reason": reas,
                    "wallet": wallet,
                    "tx_hash": txh,
                },
                "rag_context"   : rag_context,
                "outcome"       : {
                    "y_true"    : ytrue,
                    "label"     : "fraud" if ytrue == 1 else "normal",
                    "correct"   : (
                        (ytrue == 1 and action in ("BLOCK",))
                        or (ytrue == 0 and action == "WATCHLIST")
                    ),
                },
                "system_state"  : {
                    "tau_alert" : float(agent_state.get("tau_alert", 0)),
                    "tau_block" : float(agent_state.get("tau_block", 0)),
                    "w"         : float(agent_state.get("w", 0)),
                },
            }

            try:
                self._append_record(record)
                records_written += 1
            except Exception as e:
                self.logger.warning(f"[{self.name}] Failed to write audit record: {e}")

        # ── Batch summary record ──────────────────────────────────
        summary_record = {
            "incident_id"   : self._incident_id(batch_idx, suffix="summary"),
            "batch"         : batch_idx + 1,
            "timestamp"     : datetime.utcnow().isoformat(),
            "trigger"       : "batch_summary",
            "batch_metrics" : {
                "precision" : float(batch_log.get("precision",  0)),
                "recall"    : float(batch_log.get("recall",     0)),
                "f1"        : float(batch_log.get("f1",         0)),
                "roc_auc"   : float(batch_log.get("roc_auc",    0) or 0),
                "tp"        : int(batch_log.get("tp",           0)),
                "fp"        : int(batch_log.get("fp",           0)),
                "fn"        : int(batch_log.get("fn",           0)),
                "tn"        : int(batch_log.get("tn",           0)),
            },
            "system_state"  : {
                "tau_alert" : float(agent_state.get("tau_alert", 0)),
                "tau_block" : float(agent_state.get("tau_block", 0)),
                "w"         : float(agent_state.get("w", 0)),
            },
            "rag_store_size": (
                self.knowledge_agent.get_store_size()
                if self.knowledge_agent else 0
            ),
        }

        try:
            self._append_record(summary_record)
        except Exception as e:
            self.logger.warning(f"[{self.name}] Failed to write batch summary: {e}")

        # ── RAG self-improvement: re-index after writing new events ─
        if self.knowledge_agent is not None:
            try:
                fraud_events_path = os.path.join(self.run_dir, "fraud_events.csv")
                self.knowledge_agent.index_fraud_events(fraud_events_path)
            except Exception as e:
                self.logger.warning(f"[{self.name}] RAG re-indexing failed: {e}")

        # ── S3 upload of audit log (optional, non-blocking) ────────
        if self.s3 is not None:
            try:
                s3_key = f"runs/{self.run_name}/audit_log.jsonl"
                self.s3.upload_file(self.audit_log_path, s3_key)
                self.logger.info(f"[{self.name}] Audit log uploaded to s3://{s3_key}")
            except Exception as e:
                self.logger.warning(f"[{self.name}] S3 audit upload failed ({e}) -- continuing.")

        self.logger.info(
            f"[{self.name}] Batch {batch_idx+1} | "
            f"records_written={records_written} | "
            f"audit_log={self.audit_log_path} | "
            f"rag_store_size={self.knowledge_agent.get_store_size() if self.knowledge_agent else 'n/a'}"
        )

        # Pass-through -- audit agent does not modify the message payload
        return AgentMessage(
            sender=self.name,
            payload={
                **msg.payload,
                "audit_records_written": records_written,
                "audit_log_path": self.audit_log_path,
            },
            status="ok",
        )