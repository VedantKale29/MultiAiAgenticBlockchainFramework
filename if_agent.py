"""
if_agent.py — AGENT 3: IFAgent
Unsupervised anomaly detector that outputs s_IF(z) scaled to [0,1].
"""

import numpy as np
import pandas as pd
from agents.base_agent import BaseAgent, AgentMessage


class IFAgent(BaseAgent):
    def __init__(self, if_model):
        super().__init__(name="IFAgent")
        self.if_model = if_model

    def _run(self, msg: AgentMessage) -> AgentMessage:
        p_rf        = msg.payload["p_rf"]
        X_batch     = msg.payload["X_batch"]
        y_batch     = msg.payload["y_batch"]
        batch_idx   = msg.payload["batch_idx"]
        batch_size  = msg.payload["batch_size"]
        agent_state = msg.payload["agent_state"]
        start_time  = msg.payload["start_time"]

        # MinMax-scaled anomaly scores in [0, 1]
        s_if: np.ndarray = self.if_model.score(X_batch)

        self.logger.info(
            f"[{self.name}] Batch {batch_idx+1} | "
            f"s_IF: min={s_if.min():.3f} max={s_if.max():.3f} mean={s_if.mean():.3f}"
        )

        return AgentMessage(
            sender=self.name,
            payload={
                "p_rf"       : p_rf,
                "s_if"       : s_if,
                "X_batch"    : X_batch,
                "y_batch"    : y_batch,
                "batch_idx"  : batch_idx,
                "batch_size" : batch_size,
                "agent_state": agent_state,
                "start_time" : start_time,
            },
            status="ok",
        )
