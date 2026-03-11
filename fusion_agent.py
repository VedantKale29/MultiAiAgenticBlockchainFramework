"""
fusion_agent.py — AGENT 4: FusionAgent (Cognition Layer)
Computes S(z) = w*p_RF + (1-w)*s_IF, applies dual-threshold decisions.
"""

import numpy as np
import pandas as pd
from agents.base_agent import BaseAgent, AgentMessage


class FusionAgent(BaseAgent):
    def __init__(self):
        super().__init__(name="FusionAgent")

    def _run(self, msg: AgentMessage) -> AgentMessage:
        p_rf        = msg.payload["p_rf"]
        s_if        = msg.payload["s_if"]
        y_batch     = msg.payload["y_batch"]
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

        n_clear = int(np.sum(decisions == "CLEAR"))
        n_alert = int(np.sum(decisions == "ALERT"))
        n_block = int(np.sum(decisions == "AUTO-BLOCK"))

        self.logger.info(
            f"[{self.name}] Decisions → CLEAR={n_clear} ALERT={n_alert} AUTO-BLOCK={n_block}"
        )

        return AgentMessage(
            sender=self.name,
            payload={
                "risk_scores" : risk_scores,
                "decisions"   : decisions,
                "p_rf"        : p_rf,
                "s_if"        : s_if,
                "y_batch"     : y_batch,
                "batch_idx"   : batch_idx,
                "batch_size"  : batch_size,
                "agent_state" : agent_state,
                "start_time"  : start_time,
            },
            status="ok",
        )
