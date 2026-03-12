"""
agents/response_agent.py
========================
AGENT 7: ResponseAgent

ROLE:
  Executes autonomous response after PolicyAgent decides:
    ALLOW / WATCHLIST / BLOCK

WHAT IT DOES:
  - writes fraud_events.csv
  - writes attack_log.json
  - passes data onward to MonitoringAgent

NO HUMAN INTERVENTION.
"""

import os
import json
from datetime import datetime
import numpy as np
import pandas as pd

from agents.base_agent import BaseAgent, AgentMessage


class ResponseAgent(BaseAgent):
    def __init__(self, run_dir: str, cw_logger=None):
        super().__init__(name="ResponseAgent")
        self.run_dir = run_dir
        self.cw_logger = cw_logger
        os.makedirs(self.run_dir, exist_ok=True)

        self.fraud_events_path = os.path.join(self.run_dir, "fraud_events.csv")
        self.attack_log_path   = os.path.join(self.run_dir, "attack_log.json")

        if not os.path.exists(self.attack_log_path):
            with open(self.attack_log_path, "w", encoding="utf-8") as f:
                json.dump([], f, indent=2)

    def _append_attack_log(self, entries):
        current = []
        if os.path.exists(self.attack_log_path):
            try:
                with open(self.attack_log_path, "r", encoding="utf-8") as f:
                    current = json.load(f)
            except Exception:
                current = []
        current.extend(entries)
        with open(self.attack_log_path, "w", encoding="utf-8") as f:
            json.dump(current, f, indent=2)

    def _append_fraud_events(self, df):
        write_header = not os.path.exists(self.fraud_events_path)
        df.to_csv(self.fraud_events_path, mode="a", index=False, header=write_header)

    def _run(self, msg: AgentMessage) -> AgentMessage:
        policy_actions = np.asarray(msg.payload["policy_actions"], dtype=object)
        policy_reasons = np.asarray(msg.payload["policy_reasons"], dtype=object)
        decisions      = np.asarray(msg.payload["decisions"], dtype=object)
        risk_scores    = np.asarray(msg.payload["risk_scores"], dtype=float)
        p_rf           = np.asarray(msg.payload["p_rf"], dtype=float)
        s_if           = np.asarray(msg.payload["s_if"], dtype=float)
        y_batch        = msg.payload["y_batch"]
        batch_idx      = msg.payload["batch_idx"]
        batch_size     = msg.payload["batch_size"]
        agent_state    = msg.payload["agent_state"]
        start_time     = msg.payload["start_time"]
        tx_meta        = msg.payload["tx_meta"]

        tx_hashes   = tx_meta["tx_hash"]
        from_addrs  = tx_meta["from_address"]
        to_addrs    = tx_meta["to_address"]
        timestamps  = tx_meta["timestamp"]

        rows = []
        attack_entries = []

        for i in range(batch_size):
            is_positive = decisions[i] in ("ALERT", "AUTO-BLOCK")
            if not is_positive:
                continue

            row = {
                "event_time": datetime.utcnow().isoformat(),
                "batch": batch_idx + 1,
                "tx_hash": str(tx_hashes[i]),
                "from_address": str(from_addrs[i]),
                "to_address": str(to_addrs[i]),
                "timestamp": str(timestamps[i]),
                "decision": str(decisions[i]),
                "policy_action": str(policy_actions[i]),
                "policy_reason": str(policy_reasons[i]),
                "risk_score": float(risk_scores[i]),
                "p_rf": float(p_rf[i]),
                "s_if": float(s_if[i]),
                "y_true": int(y_batch.iloc[i]) if hasattr(y_batch, "iloc") else int(y_batch[i]),
                "w": float(agent_state["w"]),
                "tau_alert": float(agent_state["tau_alert"]),
                "tau_block": float(agent_state["tau_block"]),
            }
            rows.append(row)
            attack_entries.append(row)

            if self.cw_logger:
                try:
                    self.cw_logger.log_event(
                        event_type="FRAUD_EVENT_DETECTED",
                        payload=row,
                    )
                except Exception:
                    pass

        if rows:
            fraud_df = pd.DataFrame(rows)
            self._append_fraud_events(fraud_df)
            self._append_attack_log(attack_entries)

        response_report = {
            "batch": batch_idx + 1,
            "fraud_events_logged": int(len(rows)),
            "fraud_events_path": self.fraud_events_path,
            "attack_log_path": self.attack_log_path,
        }

        self.logger.info(
            f"[{self.name}] Batch {batch_idx+1} | logged={len(rows)} suspicious txs"
        )

        return AgentMessage(
            sender=self.name,
            payload={
                "response_report": response_report,
                "policy_actions": policy_actions,
                "policy_reasons": policy_reasons,
                "decisions": decisions,
                "risk_scores": risk_scores,
                "p_rf": p_rf,
                "s_if": s_if,
                "y_batch": y_batch,
                "batch_idx": batch_idx,
                "batch_size": batch_size,
                "agent_state": agent_state,
                "start_time": start_time,
                "tx_meta": tx_meta,
            },
            status="ok",
        )