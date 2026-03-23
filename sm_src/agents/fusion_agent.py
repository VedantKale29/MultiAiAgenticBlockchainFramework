"""
agents/fusion_agent.py
======================
AGENT 4: FusionAgent — Cognition Layer

ROLE IN PAPER:
  S(z) = w * p_RF(z) + (1-w) * s_IF(z)
  Then apply dual thresholds:
    S >= tau_block  → AUTO-BLOCK
    S >= tau_alert  → ALERT
    else            → CLEAR

    # w = weight on RF vs IF = 0.7 in paper
    # tau_alert = threshold for ALERT = 0.5 in paper
    # tau_block = threshold for AUTO-BLOCK = 0.9 in paper
NO AWS CALLS. Pure computation using current agent_state.

READS from msg.payload:                                │   
  │   p_rf  = msg.payload['p_rf']   → [0.92, 0.05, 0.78] │  where 0.92 is RF probability for transaction 1, 0.05 for transaction 2, 0.78 for transaction 3, etc.
  │   s_if  = msg.payload['s_if']   → [0.88, 0.03, 0.71] 

  INPUT  (AgentMessage payload):
    - "p_rf"       : np.ndarray — RF probabilities for current batch
    - "s_if"       : np.ndarray — IF scores for current batch
    - "y_batch"    : pd.Series  — true labels for current batch (for
    - "tx_meta"    : pd.DataFrame — transaction metadata for current batch
    - "batch_idx"  : int        — batch number
    - "batch_size" : int        — number of transactions in current batch
    - "agent_state" : dict       — current {w, tau_alert, tau_block

OUTPUT (AgentMessage payload):
  - "risk_scores" : np.ndarray — final risk scores S(z) for current batch
  - "decisions"   : np.ndarray — final decisions ("CLEAR", "ALERT", "AUTO-BLOCK") for current batch
  - "p_rf"        : np.ndarray — passed through from input (for transparency and debugging)
  - "s_if"        : np.ndarray — passed through from input (for transparency and debugging)
  - "y_batch"     : pd.Series  — passed through from input
  - "tx_meta"     : pd.DataFrame
"""

import numpy as np
import pandas as pd
from  agents.base_agent import BaseAgent, AgentMessage


class FusionAgent(BaseAgent):
    def __init__(self):
        super().__init__(name="FusionAgent")

    def _run(self, msg: AgentMessage) -> AgentMessage:
        p_rf        = msg.payload["p_rf"]
        s_if        = msg.payload["s_if"]
        y_batch     = msg.payload["y_batch"]
        tx_meta     = msg.payload["tx_meta"]
        batch_idx   = msg.payload["batch_idx"]
        batch_size  = msg.payload["batch_size"]
        agent_state = msg.payload["agent_state"]
        start_time  = msg.payload["start_time"]

        w         = agent_state["w"]
        tau_alert = agent_state["tau_alert"]
        tau_block = agent_state["tau_block"]

        self.logger.info(
            f"[{self.name}] Batch {batch_idx+1} | "
            f"w={w:.2f} tau_alert={tau_alert:.3f} tau_block={tau_block:.3f}"
        )

        # Paper formula: S(z) = w * p_RF(z) + (1 - w) * s_IF(z)
        risk_scores = np.clip((w * p_rf) + ((1.0 - w) * s_if), 0.0, 1.0)

        # Dual-threshold decisions
        decisions = np.full(risk_scores.shape, "CLEAR", dtype=object)
        decisions[(risk_scores >= tau_alert) & (risk_scores < tau_block)] = "ALERT"
        decisions[risk_scores >= tau_block] = "AUTO-BLOCK"

        # n_clear = int(np.sum(decisions == "CLEAR"))
        # n_alert = int(np.sum(decisions == "ALERT"))
        # n_block = int(np.sum(decisions == "AUTO-BLOCK"))

        # self.logger.info(
        #     f"[{self.name}] Decisions  ->> CLEAR={n_clear} ALERT={n_alert} AUTO-BLOCK={n_block}"
        # )

        return AgentMessage(
            sender=self.name,
            payload={
                "risk_scores" : risk_scores,
                "decisions"   : decisions,
                "p_rf"        : p_rf,             # for transparency and debugging
                "s_if"        : s_if,              # for transparency and debugging
                "y_batch"     : y_batch,
                "tx_meta"     : tx_meta,
                "batch_idx"   : batch_idx,
                "batch_size"  : batch_size,
                "agent_state" : agent_state,
                "start_time"  : start_time,
            },
            status="ok",
        )
