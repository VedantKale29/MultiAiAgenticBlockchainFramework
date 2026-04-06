"""
agents/monitoring_agent.py
===========================
AGENT 8 — MonitoringAgent  (base pipeline, batch metrics)

ROLE:
  Receives the completed batch payload from ResponseAgent.
  Computes per-batch classification metrics (Precision, Recall, F1,
  ROC-AUC, PR-AP, latency) and builds the batch_log dict that
  CoordinatorAgent appends to batch_history.csv.

  Optionally logs metrics to CloudWatch and SageMaker Experiments.

WHAT IT IS NOT:
  This is NOT the real-time blockchain monitor.
  That is MonitorAgent in agents/monitor_agent.py (Stage 2, Gap 6).
  Different name, different file, different role:
    MonitoringAgent  → batch CSV metrics  (this file)
    MonitorAgent     → live Hardhat events (monitor_agent.py)

INPUT PAYLOAD (from ResponseAgent):
  decisions       — list of "CLEAR" / "ALERT" / "AUTO-BLOCK"
  policy_actions  — list of "ALLOW" / "WATCHLIST" / "BLOCK"
  risk_scores     — np.ndarray of S(z) hybrid scores
  p_rf            — np.ndarray of RF probabilities
  s_if            — np.ndarray of IF anomaly scores
  y_batch         — pd.Series of ground-truth labels (0/1)
  y_true          — same as y_batch (alias)
  batch_idx       — int batch number (0-indexed)
  agent_state     — dict {w, tau_alert, tau_block}
  start_time      — float perf_counter timestamp from batch start

OUTPUT PAYLOAD (adds to existing payload):
  batch_log       — dict with all 19 batch_history.csv columns:
    batch, w, tau_alert, tau_block,
    precision, recall, f1, roc_auc, pr_ap, latency_per_tx,
    tp, fp, fn, tn,
    n_pred_positive, n_alert, n_auto_block,
    n_policy_watchlist, n_policy_block
"""

import time
import numpy as np

from agents.base_agent import BaseAgent, AgentMessage
from metrics import (
    compute_batch_metrics,
    compute_global_metrics,
    compute_latency,
    decisions_to_binary,
)


class MonitoringAgent(BaseAgent):
    """
    Batch metrics agent — Agent #8 in the base pipeline.

    Accepts optional cw_logger and tracker for AWS integration;
    both default to None so the agent works fully offline.
    """

    def __init__(self, cw_logger=None, tracker=None):
        # NOTE: cw_logger and tracker are stored here, NOT passed to
        # BaseAgent.__init__() — BaseAgent only accepts name=.
        super().__init__(name="MonitoringAgent")
        self.cw_logger = cw_logger
        self.tracker   = tracker

    # ── Main logic ────────────────────────────────────────────────

    def _run(self, msg: AgentMessage) -> AgentMessage:
        payload = msg.payload

        # ── Read pipeline inputs ───────────────────────────────────
        decisions      = list(payload.get("decisions",      []))
        policy_actions = list(payload.get("policy_actions", []))
        risk_scores    = np.asarray(payload.get("risk_scores", []), dtype=float)
        p_rf           = np.asarray(payload.get("p_rf",   risk_scores), dtype=float)
        s_if           = np.asarray(payload.get("s_if",   1 - risk_scores), dtype=float)

        # y_true: accept either key name used by different pipeline stages
        y_batch = payload.get("y_batch", payload.get("y_true", None))
        if y_batch is None:
            y_true = np.zeros(len(decisions), dtype=int)
        else:
            y_true = np.asarray(y_batch, dtype=int)

        batch_idx   = int(payload.get("batch_idx",  0))
        agent_state = payload.get("agent_state", {})
        start_time  = payload.get("start_time",  time.perf_counter())

        w         = float(agent_state.get("w",         0.70))
        tau_alert = float(agent_state.get("tau_alert", 0.487))
        tau_block = float(agent_state.get("tau_block", 0.587))

        batch_size = len(decisions)

        # ── Classification metrics ─────────────────────────────────
        if batch_size > 0 and len(y_true) == batch_size:
            tp, fp, fn, tn, precision, recall, f1 = compute_batch_metrics(
                y_true, decisions
            )
        else:
            tp = fp = fn = tn = 0
            precision = recall = f1 = 0.0

        # ── ROC-AUC / PR-AP (need both classes present) ───────────
        roc_auc = None
        pr_ap   = None
        if (
            len(risk_scores) == len(y_true)
            and len(np.unique(y_true)) > 1
        ):
            try:
                roc_auc, pr_ap = compute_global_metrics(y_true, risk_scores)
                if np.isnan(roc_auc):
                    roc_auc = None
                if np.isnan(pr_ap):
                    pr_ap = None
            except Exception:
                pass

        # ── Latency ───────────────────────────────────────────────
        end_time       = time.perf_counter()
        latency_per_tx = compute_latency(start_time, end_time, max(batch_size, 1))

        # ── Decision / policy counts ──────────────────────────────
        n_pred_positive    = int(np.sum(decisions_to_binary(decisions)))
        n_alert            = int(decisions.count("ALERT"))
        n_auto_block       = int(decisions.count("AUTO-BLOCK"))
        n_policy_watchlist = int(policy_actions.count("WATCHLIST"))
        n_policy_block     = int(policy_actions.count("BLOCK"))

        # ── batch_log — exactly matches batch_history.csv columns ─
        batch_log = {
            "batch":             batch_idx + 1,   # 1-indexed for display
            "w":                 w,
            "tau_alert":         tau_alert,
            "tau_block":         tau_block,
            "precision":         float(precision),
            "recall":            float(recall),
            "f1":                float(f1),
            "roc_auc":           float(roc_auc) if roc_auc is not None else None,
            "pr_ap":             float(pr_ap)   if pr_ap   is not None else None,
            "latency_per_tx":    float(latency_per_tx),
            "tp":                tp,
            "fp":                fp,
            "fn":                fn,
            "tn":                tn,
            "n_pred_positive":   n_pred_positive,
            "n_alert":           n_alert,
            "n_auto_block":      n_auto_block,
            "n_policy_watchlist": n_policy_watchlist,
            "n_policy_block":    n_policy_block,
        }

        self.logger.info(
            f"[{self.name}] Batch {batch_idx + 1}: "
            f"P={precision:.3f} R={recall:.3f} F1={f1:.3f} "
            f"TP={tp} FP={fp} FN={fn} "
            f"alert={n_alert} block={n_auto_block} "
            f"lat={latency_per_tx*1000:.2f}ms/tx"
        )

        # ── Optional AWS integrations ─────────────────────────────
        if self.cw_logger is not None:
            try:
                self.cw_logger.log_batch(
                    batch_idx=batch_idx,
                    metrics=batch_log,
                )
            except Exception as e:
                self.logger.warning(
                    f"[{self.name}] CloudWatch log failed: {e}"
                )

        if self.tracker is not None:
            try:
                self.tracker.log_batch_metrics(
                    batch=batch_idx + 1,
                    precision=float(precision),
                    recall=float(recall),
                    f1=float(f1),
                    tau_alert=tau_alert,
                    w=w,
                    tp=tp, fp=fp, fn=fn, tn=tn,
                    roc_auc=float(roc_auc) if roc_auc is not None else None,
                    pr_ap=float(pr_ap)     if pr_ap   is not None else None,
                )
            except Exception as e:
                self.logger.warning(
                    f"[{self.name}] Tracker log failed: {e}"
                )

        # ── Return enriched payload ────────────────────────────────
        return AgentMessage(
            sender=self.name,
            payload={
                **payload,
                "batch_log":  batch_log,
                "tp":         tp,
                "fp":         fp,
                "fn":         fn,
                "tn":         tn,
                "prec":       precision,
                "rec":        recall,
                # p_rf_tp / s_if_tp needed by AdaptationAgent
                "p_rf_tp":    p_rf[
                    (y_true == 1) & (decisions_to_binary(decisions) == 1)
                ] if batch_size > 0 else np.array([]),
                "s_if_tp":    s_if[
                    (y_true == 1) & (decisions_to_binary(decisions) == 1)
                ] if batch_size > 0 else np.array([]),
            },
            status="ok",
        )