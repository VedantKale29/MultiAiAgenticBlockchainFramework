"""
monitoring_agent.py
===================
AGENT 6: MonitoringAgent

WHAT IS MONITORING IN THE PAPER?
-----------------------------------
The paper says:
  "The monitoring module evaluates detection quality and operational
   metrics in real time. It tracks precision, recall, F1-score,
   ROC-AUC, and PR-AP, as well as decision latency. Monitoring
   signals feed back into the learning and adaptation block,
   triggering threshold recalibration or weight adjustments when
   deviations from performance targets are observed."

WHAT DOES THIS AGENT DO?
-------------------------
After each batch is processed and decisions are made, MonitoringAgent:
  1. Computes TP, FP, FN, TN from decisions vs ground truth
  2. Computes Precision, Recall, F1-score
  3. Computes ROC-AUC and PR-AP (threshold-independent metrics)
  4. Measures per-transaction latency
  5. Extracts TP-side RF and IF scores (needed by AdaptationAgent)
  6. Packages everything into a batch log record

WHY IS MONITORING AN AGENT?
-----------------------------
In the old code, metrics were computed inline inside batch_runner.py.
By making it an agent:
  - Monitoring is clearly responsible for ONE thing: measuring quality
  - You can add new metrics (e.g., drift detection) without touching
    the detection or adaptation logic
  - The monitoring output is a clean, structured message

This also matches the paper's architecture: monitoring provides the
FEEDBACK signal to adaptation.

INPUT  (AgentMessage payload):
  - "action_report" : dict        — from ActionAgent
  - "decisions"     : np.ndarray  — CLEAR/ALERT/AUTO-BLOCK
  - "risk_scores"   : np.ndarray  — hybrid S(z)
  - "p_rf"          : np.ndarray  — RF probabilities
  - "s_if"          : np.ndarray  — IF anomaly scores
  - "y_batch"       : pd.Series   — true labels
  - "batch_idx"     : int         — batch number
  - "batch_size"    : int         — transactions in batch
  - "agent_state"   : dict        — current {w, tau_alert, tau_block}
  - "start_time"    : float       — batch start timestamp

OUTPUT (AgentMessage payload):
  - "batch_log"   : dict        — all metrics for this batch
  - "p_rf_tp"     : np.ndarray  — RF scores where decision=TP
  - "s_if_tp"     : np.ndarray  — IF scores where decision=TP
  - "tp/fp/fn"    : int         — for AdaptationAgent
  - "prec/rec"    : float       — for AdaptationAgent
  - "batch_idx"   : int         — passed through
  - "agent_state" : dict        — passed through
"""

import time
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score

from agents.base_agent import BaseAgent, AgentMessage


def _decisions_to_binary(decisions: np.ndarray) -> np.ndarray:
    """Both ALERT and AUTO-BLOCK count as predicted positive (fraud)."""
    return np.array(
        [1 if d in ("ALERT", "AUTO-BLOCK") else 0 for d in decisions],
        dtype=int,
    )


class MonitoringAgent(BaseAgent):

    def __init__(self):
        super().__init__(name="MonitoringAgent")

    def _run(self, msg: AgentMessage) -> AgentMessage:
        """
        Compute all metrics and build the batch log record.
        """
        action_report: dict    = msg.payload["action_report"]
        decisions: np.ndarray  = msg.payload["decisions"]
        risk_scores: np.ndarray = msg.payload["risk_scores"]
        p_rf: np.ndarray       = msg.payload["p_rf"]
        s_if: np.ndarray       = msg.payload["s_if"]
        y_batch: pd.Series     = msg.payload["y_batch"]
        batch_idx: int         = msg.payload["batch_idx"]
        batch_size: int        = msg.payload["batch_size"]
        agent_state: dict      = msg.payload["agent_state"]
        start_time: float      = msg.payload["start_time"]

        end_time = time.time()

        # ── Step 1: Binary predictions from decisions ──────────────
        y_true = np.asarray(y_batch.values).astype(int)
        y_pred = _decisions_to_binary(decisions)

        # ── Step 2: Confusion Matrix ───────────────────────────────
        TP = int(np.sum((y_true == 1) & (y_pred == 1)))
        FP = int(np.sum((y_true == 0) & (y_pred == 1)))
        FN = int(np.sum((y_true == 1) & (y_pred == 0)))
        TN = int(np.sum((y_true == 0) & (y_pred == 0)))

        # ── Step 3: Precision / Recall / F1 ───────────────────────
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
        recall    = TP / (TP + FN) if (TP + FN) > 0 else 0.0
        f1        = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0 else 0.0
        )

        # ── Step 4: ROC-AUC and PR-AP ─────────────────────────────
        if len(np.unique(y_true)) > 1:
            roc_auc = float(roc_auc_score(y_true, risk_scores))
            pr_ap   = float(average_precision_score(y_true, risk_scores))
        else:
            roc_auc = float("nan")
            pr_ap   = float("nan")

        # ── Step 5: Latency ────────────────────────────────────────
        total_time = max(0.0, end_time - start_time)
        latency_per_tx = total_time / batch_size if batch_size > 0 else 0.0

        # ── Step 6: TP-side scores (for AdaptationAgent) ──────────
        tp_mask = (y_true == 1) & (y_pred == 1)
        p_rf_tp = p_rf[tp_mask]   # RF scores only for True Positives
        s_if_tp = s_if[tp_mask]   # IF scores only for True Positives

        # ── Step 7: Build the batch log record ────────────────────
        batch_log = {
            "batch"                : batch_idx + 1,
            "w"                    : agent_state["w"],
            "tau_alert"            : agent_state["tau_alert"],
            "tau_block"            : agent_state["tau_block"],
            "precision"            : round(precision, 6),
            "recall"               : round(recall, 6),
            "f1"                   : round(f1, 6),
            "roc_auc"              : round(roc_auc, 6) if not np.isnan(roc_auc) else None,
            "pr_ap"                : round(pr_ap, 6)   if not np.isnan(pr_ap)   else None,
            "tp"                   : TP,
            "fp"                   : FP,
            "fn"                   : FN,
            "tn"                   : TN,
            "latency_per_tx_sec"   : latency_per_tx,
            "batch_total_time_sec" : total_time,
            **action_report,       # merge in action counts
        }

        self.logger.info(
            f"[{self.name}] Batch {batch_idx+1} | "
            f"TP={TP} FP={FP} FN={FN} TN={TN} | "
            f"P={precision:.3f} R={recall:.3f} F1={f1:.3f} | "
            f"ROC={roc_auc:.4f} PR={pr_ap:.4f} | "
            f"latency={latency_per_tx:.6f}s/tx"
        )

        return AgentMessage(
            sender=self.name,
            payload={
                "batch_log"   : batch_log,
                "p_rf_tp"     : p_rf_tp,
                "s_if_tp"     : s_if_tp,
                "tp"          : TP,
                "fp"          : FP,
                "fn"          : FN,
                "prec"        : precision,
                "rec"         : recall,
                "batch_idx"   : batch_idx,
                "agent_state" : agent_state,
            },
            status="ok",
        )
