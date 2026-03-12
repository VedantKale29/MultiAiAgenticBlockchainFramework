"""
agents/coordinator_agent.py
============================
AGENT 8: CoordinatorAgent — THE MASTER ORCHESTRATOR

ROLE:
  Owns and runs the entire pipeline. It is the only agent that:
  - Knows all other agents exist
  - Manages the shared agent_state (w, tau_alert, tau_block)
  - Runs the batch loop
  - Handles AWS: uploads results to S3, logs final summary

PIPELINE ORDER (exactly matches your diagram):
  PerceptionAgent  → validates state vector z
  RFAgent          → p_RF(z)
  IFAgent          → s_IF(z)
  FusionAgent      → S(z) = w*p_RF + (1-w)*s_IF → decisions
  ActionAgent      → CLEAR / ALERT / AUTO-BLOCK
  MonitoringAgent  → metrics + CloudWatch + SM Experiments  ← AWS
  AdaptationAgent  → update tau, w + CloudWatch event log   ← AWS
  (repeat for next batch with updated state)

After ALL batches:
  Save batch_history.csv to run_dir
  Upload all results to S3                                   ← AWS
  Log final run summary to CloudWatch                        ← AWS
  Log final metrics to SM Experiments                        ← AWS

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
from agents.monitoring_agent  import MonitoringAgent
from agents.adaptation_agent  import AdaptationAgent

import  config as config


class CoordinatorAgent(BaseAgent):

    def __init__(
        self,
        rf_model,
        if_model,
        expected_features : list,
        run_dir           : str,
        run_name          : str,
        seed              : int,
        cw_logger         = None,   # CloudWatchLogger (optional)
        tracker           = None,   # ExperimentTracker (optional)
        s3                = None,   # S3Manager (optional)
    ):
        super().__init__(name="CoordinatorAgent")

        self.run_dir  = run_dir
        self.run_name = run_name
        self.seed     = seed
        self.tracker  = tracker
        self.cw_logger= cw_logger
        self.s3       = s3

        # ── Shared agent state (the adaptive "memory") ────────────
        self.agent_state = {
            "w"         : config.INITIAL_WEIGHT_W0,
            "tau_alert" : config.INITIAL_THRESHOLD_TAU0,
            "tau_block" : float(np.clip(
                config.INITIAL_THRESHOLD_TAU0 + config.BLOCK_MARGIN_DELTA, 0.0, 1.0
            )),
        }
        self.logger.info(
            f"[{self.name}] Initial state: {self.agent_state}"
        )

        # ── Instantiate all 7 worker agents ───────────────────────
        # MonitoringAgent and AdaptationAgent receive AWS components
        self.perception_agent = PerceptionAgent(expected_features)
        self.rf_agent         = RFAgent(rf_model)
        self.if_agent         = IFAgent(if_model)
        self.fusion_agent     = FusionAgent()
        self.action_agent     = ActionAgent()
        self.monitoring_agent = MonitoringAgent(
            cw_logger = cw_logger,   # ← CloudWatch logging
            tracker   = tracker,     # ← SageMaker Experiments logging
        )
        self.adaptation_agent = AdaptationAgent(
            cw_logger = cw_logger,   # ← CloudWatch adaptation events
        )

        self.history = []  # accumulates batch_log dicts

    def _run(self, msg: AgentMessage) -> AgentMessage:
        """
        Called once with X_test and y_test.
        Internally loops over all batches.
        """
        X_test = msg.payload["X_test"]
        y_test = msg.payload["y_test"]

        batch_size  = config.BATCH_SIZE
        num_samples = len(X_test)
        num_batches = int(np.ceil(num_samples / batch_size))

        self.logger.info(
            f"[{self.name}] Starting: {num_samples} samples | "
            f"{num_batches} batches | batch_size={batch_size}"
        )

        # ════════════════════════════════════════════════════════
        # MAIN BATCH LOOP
        # ════════════════════════════════════════════════════════
        for batch_idx in range(num_batches):

            start_idx = batch_idx * batch_size
            end_idx   = min(start_idx + batch_size, num_samples)
            X_batch   = X_test.iloc[start_idx:end_idx]
            y_batch   = y_test.iloc[start_idx:end_idx]

            self.logger.info(
                f"\n{'='*50}\n"
                f"[{self.name}] BATCH {batch_idx+1}/{num_batches}\n"
                f"{'='*50}"
            )

            # Build the initial message for this batch
            current_msg = AgentMessage(
                sender="CoordinatorAgent",
                payload={
                    "X_batch"    : X_batch,
                    "y_batch"    : y_batch,
                    "batch_idx"  : batch_idx,
                    "start_time" : time.time(),
                    "agent_state": dict(self.agent_state),  # inject current state
                },
            )

            # ── Run agents in order ────────────────────────────
            for agent in [
                self.perception_agent,
                self.rf_agent,
                self.if_agent,
                self.fusion_agent,
                self.action_agent,
                self.monitoring_agent,
            ]:
                current_msg = agent.run(current_msg)
                if current_msg.status == "error":
                    self.logger.error(
                        f"[{self.name}] {agent.name} failed: {current_msg.error} — skipping batch"
                    )
                    break
            else:
                # ── Save batch log ─────────────────────────────
                batch_log = current_msg.payload["batch_log"]
                self.history.append(batch_log)

                # ── Run adaptation ─────────────────────────────
                adapt_msg = self.adaptation_agent.run(current_msg)
                if adapt_msg.status == "ok":
                    # Update shared state for next batch
                    self.agent_state.update(adapt_msg.payload["new_state"])
                    self.logger.info(
                        f"[{self.name}] State → "
                        f"w={self.agent_state['w']:.2f} "
                        f"tau={self.agent_state['tau_alert']:.3f} "
                        f"tau_b={self.agent_state['tau_block']:.3f}"
                    )

        # ════════════════════════════════════════════════════════
        # POST-LOOP: SAVE + UPLOAD
        # ════════════════════════════════════════════════════════
        self.logger.info(
            f"[{self.name}] All {num_batches} batches complete."
        )

        os.makedirs(self.run_dir, exist_ok=True)

        # ── Save batch history CSV ─────────────────────────────
        history_df    = pd.DataFrame(self.history)
        history_path  = os.path.join(self.run_dir, "batch_history.csv")
        history_df.to_csv(history_path, index=False)
        self.logger.info(f"[{self.name}] batch_history.csv saved → {history_path}")

        # ── Save final agent state JSON ────────────────────────
        state_path = os.path.join(self.run_dir, "final_state.json")
        with open(state_path, "w") as f:
            json.dump(self.agent_state, f, indent=4)
        self.logger.info(f"[{self.name}] final_state.json saved → {state_path}")

        # ── Compute final aggregate metrics ───────────────────
        final_metrics = {}
        if self.history:
            final_metrics = {
                "mean_precision"  : float(history_df["precision"].mean()),
                "mean_recall"     : float(history_df["recall"].mean()),
                "mean_f1"         : float(history_df["f1"].mean()),
                "mean_roc_auc"    : float(history_df["roc_auc"].dropna().mean()) if "roc_auc" in history_df else None,
                "mean_pr_ap"      : float(history_df["pr_ap"].dropna().mean())   if "pr_ap"   in history_df else None,
                "final_tau_alert" : float(self.agent_state["tau_alert"]),
                "final_tau_block" : float(self.agent_state["tau_block"]),
                "final_w"         : float(self.agent_state["w"]),
            }

        # ══════════════════════════════════════════════════════
        # AWS INTEGRATION POINT 3 — Log final metrics to SM Experiments
        # ══════════════════════════════════════════════════════
        if self.tracker and final_metrics:
            self.tracker.log_final_metrics(final_metrics)

        # ══════════════════════════════════════════════════════
        # AWS INTEGRATION POINT 4 — Log run summary to CloudWatch
        # ══════════════════════════════════════════════════════
        if self.cw_logger and final_metrics:
            self.cw_logger.log_run_summary(
                final_metrics=final_metrics,
                seed=self.seed,
            )

        # ══════════════════════════════════════════════════════
        # AWS INTEGRATION POINT 5 — Upload run results to S3
        # ══════════════════════════════════════════════════════
        # Uploads: batch_history.csv, final_state.json, config.json,
        #          and any other files saved in run_dir
        if self.s3 and self.s3.is_available():
            self.logger.info(f"[{self.name}] Uploading run results to S3...")
            uploaded = self.s3.upload_run_results(
                run_dir  = self.run_dir,
                run_name = self.run_name,
            )
            # Log S3 artifact URIs to SM Experiments so you can click them in UI
            if self.tracker:
                for fname, uri in uploaded.items():
                    self.tracker.log_artifact(fname.split(".")[0], uri)
            self.logger.info(f"[{self.name}] {len(uploaded)} files uploaded to S3")

        # ── Finalize tracker (saves local JSON + closes SM run) ─
        if self.tracker:
            self.tracker.finish()

        return AgentMessage(
            sender=self.name,
            payload={
                "history"    : self.history,
                "final_state": self.agent_state,
                "final_metrics": final_metrics,
            },
            status="ok",
        )