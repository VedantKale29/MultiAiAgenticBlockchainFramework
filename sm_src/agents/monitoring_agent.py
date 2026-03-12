"""
agents/monitoring_agent.py
==========================
AGENT 6: MonitoringAgent

ROLE IN PAPER:
  "The monitoring module evaluates detection quality and operational
   metrics in real time. Monitoring signals feed back into the
   learning and adaptation block."

WHAT IT DOES:
  1. Computes TP, FP, FN, TN, Precision, Recall, F1
  2. Computes ROC-AUC, PR-AP
  3. Measures per-transaction latency
  4. Extracts TP-side RF/IF scores (needed by AdaptationAgent)
  5. ─── AWS INTEGRATION ───────────────────────────────────
     Logs batch metrics to CloudWatch (structured, searchable)
     Logs batch metrics to SageMaker Experiments (charts in UI)

WHY MONITORING IS THE RIGHT PLACE FOR AWS LOGGING:
  MonitoringAgent is the only agent that has the COMPLETE picture
  of one batch: decisions, metrics, scores — all computed.
  It is the natural place to push data to AWS before adaptation happens.

RECEIVES from ActionAgent:
  action_report, decisions, risk_scores, p_rf, s_if,
  y_batch, batch_idx, batch_size, agent_state, start_time

SENDS to AdaptationAgent:
  batch_log, p_rf_tp, s_if_tp, tp, fp, fn, prec, rec,
  batch_idx, agent_state
"""

import time
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score

from agents.base_agent import BaseAgent, AgentMessage


def _to_binary(decisions):
    """ALERT and AUTO-BLOCK = predicted positive (fraud). CLEAR = negative."""
    return np.array([1 if d in ("ALERT", "AUTO-BLOCK") else 0 for d in decisions], dtype=int)


class MonitoringAgent(BaseAgent):

    def __init__(self, cw_logger=None, tracker=None):
        """
        Parameters
        ----------
        cw_logger : CloudWatchLogger | None
            AWS CloudWatch logger. If None, only local logging happens.
        tracker : ExperimentTracker | None
            SageMaker Experiments tracker. If None, only local logging happens.
        """
        super().__init__(name="MonitoringAgent")
        self.cw_logger = cw_logger   # CloudWatchLogger (AWS)
        self.tracker   = tracker     # ExperimentTracker (AWS)

    def _run(self, msg: AgentMessage) -> AgentMessage:
        action_report = msg.payload["action_report"]
        decisions     = msg.payload["decisions"]
        risk_scores   = msg.payload["risk_scores"]
        p_rf          = msg.payload["p_rf"]
        s_if          = msg.payload["s_if"]
        y_batch       = msg.payload["y_batch"]
        batch_idx     = msg.payload["batch_idx"]
        batch_size    = msg.payload["batch_size"]
        agent_state   = msg.payload["agent_state"]
        start_time    = msg.payload["start_time"]

        end_time = time.time()

        # ── Confusion matrix ──────────────────────────────────────
        y_true = np.asarray(y_batch.values).astype(int)
        y_pred = _to_binary(decisions)

        TP = int(np.sum((y_true == 1) & (y_pred == 1)))
        FP = int(np.sum((y_true == 0) & (y_pred == 1)))
        FN = int(np.sum((y_true == 1) & (y_pred == 0)))
        TN = int(np.sum((y_true == 0) & (y_pred == 0)))

        # ── Precision / Recall / F1 ───────────────────────────────
        prec = TP / (TP + FP) if (TP + FP) > 0 else 0.0
        rec  = TP / (TP + FN) if (TP + FN) > 0 else 0.0
        f1   = 2*prec*rec / (prec+rec) if (prec+rec) > 0 else 0.0

        # ── ROC-AUC / PR-AP ───────────────────────────────────────
        if len(np.unique(y_true)) > 1:
            roc_auc = float(roc_auc_score(y_true, risk_scores))
            pr_ap   = float(average_precision_score(y_true, risk_scores))
        else:
            roc_auc = float("nan")
            pr_ap   = float("nan")

        # ── Latency ───────────────────────────────────────────────
        total_time     = max(0.0, end_time - start_time)
        latency_per_tx = total_time / batch_size if batch_size > 0 else 0.0

        # ── TP-side scores (for AdaptationAgent weight update) ────
        tp_mask  = (y_true == 1) & (y_pred == 1)
        p_rf_tp  = p_rf[tp_mask]
        s_if_tp  = s_if[tp_mask]

        # ── Local log ─────────────────────────────────────────────
        self.logger.info(
            f"[{self.name}] Batch {batch_idx+1} | "
            f"TP={TP} FP={FP} FN={FN} TN={TN} | "
            f"P={prec:.3f} R={rec:.3f} F1={f1:.3f} | "
            f"ROC={roc_auc:.4f} PR={pr_ap:.4f} | "
            f"lat={latency_per_tx:.6f}s/tx"
        )

        # ── Build batch log record ────────────────────────────────
        batch_log = {
            "batch"               : batch_idx + 1,
            "w"                   : agent_state["w"],
            "tau_alert"           : agent_state["tau_alert"],
            "tau_block"           : agent_state["tau_block"],
            "precision"           : round(prec, 6),
            "recall"              : round(rec,  6),
            "f1"                  : round(f1,   6),
            "roc_auc"             : round(roc_auc, 6) if not np.isnan(roc_auc) else None,
            "pr_ap"               : round(pr_ap,   6) if not np.isnan(pr_ap)   else None,
            "tp": TP, "fp": FP, "fn": FN, "tn": TN,
            "latency_per_tx_sec"  : latency_per_tx,
            "batch_total_time_sec": total_time,
            **action_report,
        }

        # ════════════════════════════════════════════════════════
        # AWS INTEGRATION POINT 1 — CloudWatch structured logging
        # ════════════════════════════════════════════════════════
        # This logs the batch line to CloudWatch in a searchable format.
        # In SageMaker, Python stdout is automatically captured by CloudWatch,
        # so the self.logger.info() above already goes to CloudWatch.
        # The cw_logger sends ADDITIONAL structured JSON entries.
        if self.cw_logger:
            self.cw_logger.log_batch(
                batch_idx=batch_idx,
                metrics={
                    "precision" : prec,
                    "recall"    : rec,
                    "f1"        : f1,
                    "tau_alert" : agent_state["tau_alert"],
                    "w"         : agent_state["w"],
                    "tp"        : TP, "fp": FP, "fn": FN, "tn": TN,
                    "roc_auc"   : roc_auc if not np.isnan(roc_auc) else None,
                    "pr_ap"     : pr_ap   if not np.isnan(pr_ap)   else None,
                },
            )

        # ════════════════════════════════════════════════════════
        # AWS INTEGRATION POINT 2 — SageMaker Experiments metrics
        # ════════════════════════════════════════════════════════
        # This logs the metrics as a time-series data point in SM Experiments.
        # Each batch = one point on the precision/recall chart in the SM UI.
        if self.tracker:
            self.tracker.log_batch_metrics(
                batch     = batch_idx + 1,
                precision = prec,
                recall    = rec,
                f1        = f1,
                tau_alert = agent_state["tau_alert"],
                w         = agent_state["w"],
                tp=TP, fp=FP, fn=FN, tn=TN,
                roc_auc   = roc_auc if not np.isnan(roc_auc) else None,
                pr_ap     = pr_ap   if not np.isnan(pr_ap)   else None,
            )

        return AgentMessage(
            sender=self.name,
            payload={
                "batch_log"  : batch_log,
                "p_rf_tp"    : p_rf_tp,
                "s_if_tp"    : s_if_tp,
                "tp": TP, "fp": FP, "fn": FN,
                "prec"       : prec,
                "rec"        : rec,
                "batch_idx"  : batch_idx,
                "agent_state": agent_state,
            },
            status="ok",
        )