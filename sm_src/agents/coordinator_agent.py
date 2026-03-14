"""
agents/coordinator_agent.py
============================
AGENT 10: CoordinatorAgent — THE MASTER ORCHESTRATOR

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

        self.history = []

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
                        f"{current_msg.error} — skipping batch"
                    )
                    break

            else:
                # Save monitoring log
                batch_log = current_msg.payload["batch_log"]
                self.history.append(batch_log)

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
            uploaded = self.s3.upload_run_results(
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