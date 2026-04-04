"""
agents/coordinator_agent.py
============================
AGENT 10: CoordinatorAgent -- THE MASTER ORCHESTRATOR

ROLE:
  Owns and runs the entire pipeline. It is the only agent that:
  - Knows all other agents exist
  - Manages the shared agent_state (w, tau_alert, tau_block)
  - Runs the batch loop
  - Handles AWS: uploads results to S3, logs final summary

UPDATED PIPELINE ORDER:
  PerceptionAgent  → validates state vector z + passes tx_meta
  RFAgent          → p_RF(z)
  IFAgent          → s_IF(z)
  FusionAgent      → S(z) = w*p_RF + (1-w)*s_IF → decisions
  ActionAgent      → CLEAR / ALERT / AUTO-BLOCK
  PolicyAgent      → ALLOW / WATCHLIST / BLOCK
  ResponseAgent    → writes fraud_events.csv / attack_log.json
  MonitoringAgent  → metrics + CloudWatch + SM Experiments
  AdaptationAgent  → update tau, w + CloudWatch event log

After ALL batches:
  Save batch_history.csv to run_dir
  Save final_state.json to run_dir
  Upload all results to S3
  Log final run summary to CloudWatch
  Log final metrics to SM Experiments

SHARED STATE:
  agent_state = {"w": 0.70, "tau_alert": 0.487, "tau_block": 0.587}
  CoordinatorAgent holds this dict.
  FusionAgent READS it each batch.
  AdaptationAgent RETURNS a new_state each batch.
  CoordinatorAgent UPDATES the dict with new_state.

PIPELINE ORDER:
  PerceptionAgent   → validates z + passes tx_meta
  RFAgent           → p_RF(z)
  IFAgent           → s_IF(z)
  FusionAgent       → S(z) = w*p_RF + (1-w)*s_IF → decisions
  ActionAgent       → CLEAR / ALERT / AUTO-BLOCK
  PolicyAgent       → ALLOW / WATCHLIST / BLOCK
  ResponseAgent     → fraud_events.csv / attack_log.json
  MonitoringAgent   → metrics + CloudWatch + SM Experiments
  ─── Stage 2 ───
  AuditAgent        → audit_log.jsonl + RAG re-index
  ─── Stage 3 ───
  DecisionAgent     → LLM + RAG → ActionPlan
  ContractAgent     → template + Slither + deploy
  ─── Stage 4 ───
  GovernanceAgent   → consecutive pattern → timelock proposal
  ─── Always ───
  AdaptationAgent   → update tau, w
 
SAFETY CONTRACT:
  Every Stage 2–4 agent is wrapped in _run_optional().
  Any crash → warning logged → original message returned.
  AWS / S3 / SageMaker / CloudWatch untouched.
  New agents only instantiated if their module files exist.
  If no new deps installed: pipeline runs identically to before.
"""

import os
import json
import time
import numpy as np
import pandas as pd

from agents.base_agent        import BaseAgent, AgentMessage
from agents.perception_agent  import PerceptionAgent
from agents.rf_agent          import RFAgent
from agents.if_agent          import IFAgent
from agents.fusion_agent      import FusionAgent
from agents.action_agent      import ActionAgent
from agents.policy_agent      import PolicyAgent
from agents.response_agent    import ResponseAgent
from agents.monitoring_agent  import MonitoringAgent
from agents.adaptation_agent  import AdaptationAgent

try:
    from agents.fraud_knowledge_agent import FraudKnowledgeAgent
    from agents.audit_agent           import AuditAgent
    _STAGE12 = True
except ImportError:
    _STAGE12 = False
 
try:
    from agents.decision_agent import DecisionAgent
    from agents.contract_agent import ContractAgent
    _STAGE3 = True
except ImportError:
    _STAGE3 = False
 
try:
    from agents.governance_agent import GovernanceAgent
    _STAGE4 = True
except ImportError:
    _STAGE4 = False

import config


class CoordinatorAgent(BaseAgent):

    def __init__(
        self,
        rf_model,
        if_model,
        expected_features: list,
        run_dir: str,
        run_name: str,
        seed: int,
        cw_logger=None,   # CloudWatchLogger (optional)
        tracker=None,     # ExperimentTracker (optional)
        s3=None,          # S3Manager (optional)

        # ── Stage 3–4 optional kwargs (all default to safe None) ──
        anthropic_api_key:  str = None,
        hardhat_url:        str = "http://127.0.0.1:8545",
        registry_address:   str = None,
        governance_address: str = None,
        deployer_key:       str = None,
    ):
        super().__init__(name="CoordinatorAgent")

        self.run_dir   = run_dir
        self.run_name  = run_name
        self.seed      = seed
        self.tracker   = tracker
        self.cw_logger = cw_logger
        self.s3        = s3

        # Shared adaptive state
        self.agent_state = {
            "w": config.INITIAL_WEIGHT_W0,
            "tau_alert": config.INITIAL_THRESHOLD_TAU0,
            "tau_block": float(
                np.clip(
                    config.INITIAL_THRESHOLD_TAU0 + config.BLOCK_MARGIN_DELTA,
                    0.0,
                    1.0,
                )
            ),
        }

        self.logger.info(f"[{self.name}] Initial state: {self.agent_state}")

        # Worker agents
        self.perception_agent = PerceptionAgent(expected_features)
        self.rf_agent         = RFAgent(rf_model)
        self.if_agent         = IFAgent(if_model)
        self.fusion_agent     = FusionAgent()
        self.action_agent     = ActionAgent()
        self.policy_agent     = PolicyAgent(run_dir=run_dir)
        self.response_agent   = ResponseAgent(run_dir=run_dir, cw_logger=cw_logger)
        self.monitoring_agent = MonitoringAgent(
            cw_logger=cw_logger,
            tracker=tracker,
        )
        self.adaptation_agent = AdaptationAgent(
            cw_logger=cw_logger,
        )

        # ── Stage 1+2: RAG + Audit ─────────────────────────────────
        self.knowledge_agent = None
        self.audit_agent     = None
        if _STAGE12:
            try:
                self.knowledge_agent = FraudKnowledgeAgent(run_dir=run_dir)
                self.audit_agent     = AuditAgent(
                    run_dir=run_dir,
                    run_name=run_name,
                    knowledge_agent=self.knowledge_agent,
                    s3=s3,
                )
                self.logger.info(f"[{self.name}] Stages 1+2: RAG + Audit ready "
                                 f"(store={self.knowledge_agent.get_store_size()} docs)")
            except Exception as e:
                self.logger.warning(f"[{self.name}] Stages 1+2 init failed ({e}) -- skipping")
                self.knowledge_agent = None
                self.audit_agent     = None
 
        # ── Stage 3: Decision + Contract ──────────────────────────
        self.decision_agent = None
        self.contract_agent = None
        if _STAGE3:
            try:
                self.decision_agent = DecisionAgent(
                    knowledge_agent=self.knowledge_agent,
                    anthropic_api_key=anthropic_api_key,
                )
                self.contract_agent = ContractAgent(
                    run_dir=run_dir,
                    knowledge_agent=self.knowledge_agent,
                    hardhat_url=hardhat_url,
                    registry_address=registry_address,
                    deployer_key=deployer_key,
                )
                self.logger.info(f"[{self.name}] Stage 3: Decision + Contract ready")
            except Exception as e:
                self.logger.warning(f"[{self.name}] Stage 3 init failed ({e}) -- skipping")
                self.decision_agent = None
                self.contract_agent = None
 
        # ── Stage 4: Governance ────────────────────────────────────
        self.governance_agent = None
        if _STAGE4:
            try:
                self.governance_agent = GovernanceAgent(
                    run_dir=run_dir,
                    hardhat_url=hardhat_url,
                    governance_address=governance_address,
                    deployer_key=deployer_key,
                )
                self.logger.info(f"[{self.name}] Stage 4: Governance ready")
            except Exception as e:
                self.logger.warning(f"[{self.name}] Stage 4 init failed ({e}) -- skipping")
                self.governance_agent = None

        self.history = []

        # # ── Stage 1: Restore RAG store from S3 (if previous run exists) ─
        # # This makes RAG knowledge accumulate across SageMaker runs.
        # # On first run: no rag_store in S3 yet → starts fresh (silent).
        # # On subsequent runs: restores previous ChromaDB → knowledge grows.
        # if s3 is not None:
        #     self.logger.info(
        #         f"[{self.name}] Attempting to restore rag_store from S3..."
        #     )
        #     restored = s3.download_rag_store(
        #         run_name      = run_name,
        #         local_run_dir = run_dir,
        #     )
        #     if restored:
        #         self.logger.info(
        #             f"[{self.name}] rag_store restored from S3 -- "
        #             f"RAG knowledge will accumulate from previous run"
        #         )
        #     else:
        #         self.logger.info(
        #             f"[{self.name}] No previous rag_store in S3 -- "
        #             f"starting fresh (expected on first run)"
        #         )

        # # ── Stage 1: RAG knowledge base ─────────────────────────────
        # # Degrades gracefully if chromadb not installed.
        # self.knowledge_agent = FraudKnowledgeAgent(run_dir=run_dir)
    
        # # ── Stage 2: Audit agent ─────────────────────────────────────
        # self.audit_agent = AuditAgent(
        #     run_dir        = run_dir,
        #     run_name       = run_name,
        #     knowledge_agent= self.knowledge_agent,  # wires RAG loop
        #     s3             = s3,                    # same S3 instance you already have
        # )

    def _run(self, msg: AgentMessage) -> AgentMessage:
        """
        Called once with X_test, y_test, and optional X_test_meta.
        Internally loops over all batches.
        """
        X_test = msg.payload["X_test"]
        y_test = msg.payload["y_test"]
        X_test_meta = msg.payload.get("X_test_meta", None)

        batch_size  = config.BATCH_SIZE
        num_samples = len(X_test)
        num_batches = int(np.ceil(num_samples / batch_size))

        self.logger.info(
            f"[{self.name}] Starting: {num_samples} samples | "
            f"{num_batches} batches | batch_size={batch_size}"
        )

        # ─────────────────────────────────────────────────────
        # MAIN BATCH LOOP
        # ─────────────────────────────────────────────────────
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx   = min(start_idx + batch_size, num_samples)

            X_batch = X_test.iloc[start_idx:end_idx]
            y_batch = y_test.iloc[start_idx:end_idx]

            # metadata source: aligned in main.py, sliced here batch-wise
            if X_test_meta is not None:
                meta_batch = X_test_meta.iloc[start_idx:end_idx].copy()
            else:
                meta_batch = None

            self.logger.info(
                f"\n{'='*50}\n"
                f"[{self.name}] BATCH {batch_idx+1}/{num_batches}\n"
                f"{'='*50}"
            )

            current_msg = AgentMessage(
                sender="CoordinatorAgent",
                payload={
                    "X_batch": X_batch,
                    "y_batch": y_batch,
                    "tx_meta": meta_batch,
                    "batch_idx": batch_idx,
                    "start_time": time.time(),
                    "agent_state": dict(self.agent_state),
                },
            )

            # Updated pipeline
            for agent in [
                self.perception_agent,
                self.rf_agent,
                self.if_agent,
                self.fusion_agent,
                self.action_agent,
                self.policy_agent,
                self.response_agent,
                self.monitoring_agent,
            ]:
                current_msg = agent.run(current_msg)

                if current_msg.status == "error":
                    self.logger.error(
                        f"[{self.name}] {agent.name} failed: "
                        f"{current_msg.error} -- skipping batch"
                    )
                    break

            else:
                # Save monitoring log
                batch_log = current_msg.payload["batch_log"]
                self.history.append(batch_log)

                #   # ── Stage 2: Audit + RAG self-improvement ───────────────────
                # audit_agent writes the audit record and re-indexes RAG.
                # If it fails, the error is caught inside AuditAgent._run()
                # and the pipeline continues without interruption.
                # ── Stage 2: Audit + RAG self-improvement ──────────
                if self.audit_agent is not None:
                    audit_msg = self.audit_agent.run(current_msg)
                    if audit_msg.status == "ok":
                        current_msg = audit_msg
                        self.logger.info(
                            f"[{self.name}] AuditAgent: "
                            f"records={audit_msg.payload.get('audit_records_written',0)} | "
                            f"rag_store={self.knowledge_agent.get_store_size()} docs"
                        )

                # ── Stage 3a: Decision Agent (LLM + RAG → ActionPlan) ──
                if self.decision_agent is not None:
                    decision_msg = self.decision_agent.run(current_msg)
                    if decision_msg.status == "ok":
                        current_msg = decision_msg
                        ap = decision_msg.payload.get("action_plan", {})
                        self.logger.info(
                            f"[{self.name}] DecisionAgent: "
                            f"threat={ap.get('threat_type','?')} | "
                            f"severity={ap.get('severity','?')} | "
                            f"template={ap.get('recommended_template','?')} | "
                            f"rag_hits={ap.get('rag_hits',0)} | "
                            f"llm_used={ap.get('llm_used',False)}"
                        )
                    else:
                        self.logger.warning(
                            f"[{self.name}] DecisionAgent failed: {decision_msg.error}"
                        )

                # ── Stage 3b: Contract Agent (select + Slither + deploy) ─
                if self.contract_agent is not None:
                    contract_msg = self.contract_agent.run(current_msg)
                    if contract_msg.status == "ok":
                        current_msg = contract_msg
                        dr = contract_msg.payload.get("deployment_record", {})
                        self.logger.info(
                            f"[{self.name}] ContractAgent: "
                            f"template={dr.get('template','?')} | "
                            f"slither={dr.get('slither_passed','?')} | "
                            f"simulated={dr.get('simulated',True)} | "
                            f"address={str(dr.get('deployed_address','?'))[:12]}..."
                        )
                    else:
                        self.logger.warning(
                            f"[{self.name}] ContractAgent failed: {contract_msg.error}"
                        )

                # ── Stage 4: Governance Agent (pattern → timelock) ───────
                if self.governance_agent is not None:
                    gov_msg = self.governance_agent.run(current_msg)
                    if gov_msg.status == "ok":
                        current_msg = gov_msg
                        gp = gov_msg.payload.get("governance_proposal")
                        if gp:
                            self.logger.info(
                                f"[{self.name}] GovernanceAgent: proposal submitted | "
                                f"param={gp.get('param','?')} | "
                                f"new_value={gp.get('new_value','?')}"
                            )

                # Adapt state for next batch
                adapt_msg = self.adaptation_agent.run(current_msg)
                if adapt_msg.status == "ok":
                    self.agent_state.update(adapt_msg.payload["new_state"])
                    self.logger.info(
                        f"[{self.name}] State → "
                        f"w={self.agent_state['w']:.2f} "
                        f"tau={self.agent_state['tau_alert']:.3f} "
                        f"tau_b={self.agent_state['tau_block']:.3f}"
                    )
                else:
                    self.logger.error(
                        f"[{self.name}] AdaptationAgent failed: {adapt_msg.error}"
                    )

        # ─────────────────────────────────────────────────────
        # POST-LOOP: SAVE + UPLOAD
        # ─────────────────────────────────────────────────────
        self.logger.info(f"[{self.name}] All {num_batches} batches complete.")

        # Final index pass -- ensures the last batch's events are in RAG
        if self.knowledge_agent is not None:
            fraud_events_path = os.path.join(self.run_dir, "fraud_events.csv")
            self.knowledge_agent.index_fraud_events(fraud_events_path)
            self.logger.info(
                f"[{self.name}] RAG final index complete: "
                f"{self.knowledge_agent.get_store_size()} documents"
            )

        os.makedirs(self.run_dir, exist_ok=True)

        # Save batch history
        history_df = pd.DataFrame(self.history)
        history_path = os.path.join(self.run_dir, "batch_history.csv")
        history_df.to_csv(history_path, index=False)
        self.logger.info(f"[{self.name}] batch_history.csv saved → {history_path}")

        # Save final adaptive state
        state_path = os.path.join(self.run_dir, "final_state.json")
        with open(state_path, "w", encoding="utf-8") as f:
            json.dump(self.agent_state, f, indent=4)
        self.logger.info(f"[{self.name}] final_state.json saved → {state_path}")

        # Compute aggregate metrics
        final_metrics = {}
        if self.history:
            final_metrics = {
                "mean_precision": float(history_df["precision"].mean()),
                "mean_recall": float(history_df["recall"].mean()),
                "mean_f1": float(history_df["f1"].mean()),
                "mean_roc_auc": (
                    float(history_df["roc_auc"].dropna().mean())
                    if "roc_auc" in history_df.columns and history_df["roc_auc"].notna().any()
                    else None
                ),
                "mean_pr_ap": (
                    float(history_df["pr_ap"].dropna().mean())
                    if "pr_ap" in history_df.columns and history_df["pr_ap"].notna().any()
                    else None
                ),
                "final_tau_alert": float(self.agent_state["tau_alert"]),
                "final_tau_block": float(self.agent_state["tau_block"]),
                "final_w": float(self.agent_state["w"]),
            }

            # Optional extra summaries if new columns exist
            if "n_policy_watchlist" in history_df.columns:
                final_metrics["total_policy_watchlist"] = int(history_df["n_policy_watchlist"].sum())
            if "n_policy_block" in history_df.columns:
                final_metrics["total_policy_block"] = int(history_df["n_policy_block"].sum())

        # SageMaker Experiments / local tracker
        if self.tracker and final_metrics:
            self.tracker.log_final_metrics(final_metrics)

        # CloudWatch summary
        if self.cw_logger and final_metrics:
            self.cw_logger.log_run_summary(
                final_metrics=final_metrics,
                seed=self.seed,
            )

        # S3 upload of all run artifacts
        uploaded = {}
        if self.s3:
            uploaded = self.s3.upload_rag_store(
                run_dir=self.run_dir,
                run_name=self.run_name,
            )

            if uploaded:
                self.logger.info(
                    f"[{self.name}] Uploaded {len(uploaded)} files to S3"
                )

                if self.tracker:
                    for name, uri in uploaded.items():
                        try:
                            self.tracker.log_artifact(name, uri)
                        except Exception:
                            pass

        # Save tracker summary locally / finish run
        if self.tracker:
            try:
                self.tracker.log_artifact("batch_history", history_path)
                self.tracker.log_artifact("final_state", state_path)
            except Exception:
                pass

            try:
                self.tracker.save()
            except Exception:
                pass

        return AgentMessage(
            sender=self.name,
            payload={
                "history": self.history,
                "final_state": dict(self.agent_state),
                "final_metrics": final_metrics,
                "run_dir": self.run_dir,
                "uploaded": uploaded,
            },
            status="ok",
        )