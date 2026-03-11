"""
coordinator_agent.py
====================
AGENT 8: CoordinatorAgent  ← THE MASTER ORCHESTRATOR

WHAT IS THE COORDINATOR IN THE PAPER?
---------------------------------------
The paper describes an "autonomous orchestration mechanism" that:
  "coordinates various elements, integrates explainability into action,
   and fine-tunes decisions based on cost-benefit trade-offs"

The CoordinatorAgent IS that mechanism. It is the brain that:
  - Knows which agents exist
  - Knows in what ORDER to call them
  - Manages the SHARED STATE (w, tau_alert, tau_block)
  - Runs the batch loop
  - Collects all batch logs into the history

WHY IS THE COORDINATOR SPECIAL?
---------------------------------
All other agents are WORKERS — they do ONE specific job.
The CoordinatorAgent is the MANAGER — it:
  1. Passes data from one agent to the next (message routing)
  2. Updates the shared agent_state after each batch
  3. Stops processing if any agent fails (error handling)
  4. Builds the final history across all batches

AGENT PIPELINE ORDER (matches your diagram exactly):
  PerceptionAgent   ->> cleans state vector z
       ↓
  RFAgent           ->> computes p_RF(z)
       ↓
  IFAgent           ->> computes s_IF(z)
       ↓
  FusionAgent       ->> S(z) = w*p_RF + (1-w)*s_IF  ->> decisions
       ↓
  ActionAgent       ->> enforces CLEAR/ALERT/AUTO-BLOCK
       ↓
  MonitoringAgent   ->> computes metrics + batch log
       ↓
  AdaptationAgent   ->> updates tau, tau_block, w
       ↓
  (back to top for next batch with updated state)

HOW SHARED STATE WORKS:
-------------------------
The agent_state dict is the "memory" of the system:
  agent_state = {
      "w"         : 0.70,     ← RF weight (changes via AdaptationAgent)
      "tau_alert" : 0.487,    ← alert threshold (changes via AdaptationAgent)
      "tau_block" : 0.587,    ← block threshold = tau + delta
  }

CoordinatorAgent injects this into EVERY batch message so FusionAgent
can always read the CURRENT values.
"""

import time
import numpy as np
import pandas as pd

from  base_agent import BaseAgent, AgentMessage
from  perception_agent  import PerceptionAgent
from  rf_agent           import RFAgent
from  if_agent           import IFAgent
from  fusion_agent       import FusionAgent
from  action_agent       import ActionAgent
from  monitoring_agent   import MonitoringAgent
from  adaptation_agent   import AdaptationAgent

import config


class CoordinatorAgent(BaseAgent):
    """
    Master orchestrator for the agentic fraud detection pipeline.

    Parameters
    ----------
    rf_model : RFModel       — trained Random Forest model
    if_model : IFModel       — trained Isolation Forest model
    expected_features : list — list of feature column names
    """

    def __init__(self, rf_model, if_model, expected_features: list):
        super().__init__(name="CoordinatorAgent")

        # ── Initialize all 7 worker agents ────────────────────────
        self.perception_agent  = PerceptionAgent(expected_features)
        self.rf_agent          = RFAgent(rf_model)
        self.if_agent          = IFAgent(if_model)
        self.fusion_agent      = FusionAgent()
        self.action_agent      = ActionAgent()
        self.monitoring_agent  = MonitoringAgent()
        self.adaptation_agent  = AdaptationAgent()

        # ── Shared Agent State (the "memory") ─────────────────────
        # This is the ONLY mutable state in the system.
        # All agents READ from it; only AdaptationAgent WRITES to it
        # (via the Coordinator updating it after each batch).
        self.agent_state = {
            "w"          : config.INITIAL_WEIGHT_W0,
            "tau_alert"  : config.INITIAL_THRESHOLD_TAU0,
            "tau_block"  : float(np.clip(
                config.INITIAL_THRESHOLD_TAU0 + config.BLOCK_MARGIN_DELTA,
                0.0, 1.0
            )),
        }

        self.logger.info(
            f"[{self.name}] Initial agent state: {self.agent_state}"
        )

        # ── History accumulator ────────────────────────────────────
        self.history = []

    def _run(self, msg: AgentMessage) -> AgentMessage:
        """
        CoordinatorAgent's _run is called ONCE with X_test, y_test.
        It internally loops over all batches, calling the agent pipeline
        for each batch.

        INPUT msg payload:
          - "X_test" : pd.DataFrame
          - "y_test" : pd.Series

        OUTPUT msg payload:
          - "history"      : list of batch log dicts
          - "final_state"  : final {w, tau_alert, tau_block}
        """
        X_test: pd.DataFrame = msg.payload["X_test"]
        y_test: pd.Series    = msg.payload["y_test"]

        batch_size  = config.BATCH_SIZE
        num_samples = len(X_test)
        num_batches = int(np.ceil(num_samples / batch_size))

        self.logger.info(
            f"[{self.name}] Starting streaming pipeline: "
            f"{num_samples} samples | {num_batches} batches | "
            f"batch_size={batch_size}"
        )

        # ════════════════════════════════════════════════════════════
        # MAIN BATCH LOOP
        # ════════════════════════════════════════════════════════════
        for batch_idx in range(num_batches):

            start_idx = batch_idx * batch_size
            end_idx   = min(start_idx + batch_size, num_samples)

            X_batch = X_test.iloc[start_idx:end_idx]
            y_batch = y_test.iloc[start_idx:end_idx]

            self.logger.info(
                f"\n{'='*55}\n"
                f"[{self.name}] BATCH {batch_idx+1}/{num_batches}\n"
                f"{'='*55}"
            )

            batch_start_time = time.time()

            # ── 1. Build the initial message ───────────────────────
            # This is the "envelope" we pass through all  
            # Each agent ADDS their output to it.
            current_msg = AgentMessage(
                sender="CoordinatorAgent",
                payload={
                    "X_batch"    : X_batch,
                    "y_batch"    : y_batch,
                    "batch_idx"  : batch_idx,
                    "start_time" : batch_start_time,
                    # Inject current shared state so all agents can read it
                    "agent_state": dict(self.agent_state),
                },
                status="ok",
            )

            # ── 2. Run agents in pipeline order ───────────────────
            #    Each agent returns a new AgentMessage.
            #    We check status after each step.

            # STEP 1: Perception — validate and clean features
            current_msg = self.perception_agent.run(current_msg)
            if current_msg.status == "error":
                self.logger.error(f"PerceptionAgent failed: {current_msg.error}")
                continue

            # STEP 2: RF — compute p_RF(z)
            current_msg = self.rf_agent.run(current_msg)
            if current_msg.status == "error":
                self.logger.error(f"RFAgent failed: {current_msg.error}")
                continue

            # STEP 3: IF — compute s_IF(z)
            current_msg = self.if_agent.run(current_msg)
            if current_msg.status == "error":
                self.logger.error(f"IFAgent failed: {current_msg.error}")
                continue

            # STEP 4: Fusion — S(z) = w*p_RF + (1-w)*s_IF  ->> decisions
            current_msg = self.fusion_agent.run(current_msg)
            if current_msg.status == "error":
                self.logger.error(f"FusionAgent failed: {current_msg.error}")
                continue

            # STEP 5: Action — enforce CLEAR/ALERT/AUTO-BLOCK
            current_msg = self.action_agent.run(current_msg)
            if current_msg.status == "error":
                self.logger.error(f"ActionAgent failed: {current_msg.error}")
                continue

            # STEP 6: Monitoring — compute metrics + batch log
            current_msg = self.monitoring_agent.run(current_msg)
            if current_msg.status == "error":
                self.logger.error(f"MonitoringAgent failed: {current_msg.error}")
                continue

            # ── 3. Save batch log to history ──────────────────────
            batch_log = current_msg.payload["batch_log"]
            self.history.append(batch_log)

            # STEP 7: Adaptation — update tau and w
            current_msg = self.adaptation_agent.run(current_msg)
            if current_msg.status == "error":
                self.logger.error(f"AdaptationAgent failed: {current_msg.error}")
                # Even if adaptation fails, we continue with old state
            else:
                # ── 4. Update shared agent state ──────────────────
                # This is the KEY step: the new state from AdaptationAgent
                # becomes the shared state for the NEXT batch's FusionAgent
                new_state = current_msg.payload["new_state"]
                self.agent_state.update(new_state)

                self.logger.info(
                    f"[{self.name}] State updated  ->> "
                    f"w={self.agent_state['w']:.2f} "
                    f"tau_alert={self.agent_state['tau_alert']:.3f} "
                    f"tau_block={self.agent_state['tau_block']:.3f}"
                )

        # ════════════════════════════════════════════════════════════
        # END OF ALL BATCHES
        # ════════════════════════════════════════════════════════════
        self.logger.info(
            f"\n[{self.name}] All {num_batches} batches complete.\n"
            f"Final agent state: {self.agent_state}"
        )

        return AgentMessage(
            sender=self.name,
            payload={
                "history"    : self.history,
                "final_state": self.agent_state,
            },
            status="ok",
        )
