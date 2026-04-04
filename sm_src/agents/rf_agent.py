"""
agents/rf_agent.py
==================
AGENT 2: RFAgent -- supervised detector → p_RF(z)

ROLE IN PAPER:
  "Random Forest outputs the probability of a transaction being fraudulent."
  p_RF(z) = average of T=250 tree outputs

NO AWS CALLS. Pure inference.

Recived 
"""
import numpy as np
import pandas as pd
from  agents.base_agent import BaseAgent, AgentMessage


class RFAgent(BaseAgent):
    def __init__(self, rf_model):
        super().__init__(name="RFAgent")
        self.rf_model = rf_model

    def _run(self, msg: AgentMessage) -> AgentMessage:
        X_batch     = msg.payload["X_batch"]
        y_batch     = msg.payload["y_batch"]
        tx_meta     = msg.payload["tx_meta"]
        batch_idx   = msg.payload["batch_idx"]
        batch_size  = msg.payload["batch_size"]
        agent_state = msg.payload["agent_state"]
        start_time  = msg.payload["start_time"]

        # RF fraud probabilities
        p_rf: np.ndarray = self.rf_model.predict_proba(X_batch)

        self.logger.info(
            f"[{self.name}] Batch {batch_idx+1} | "
            f"p_RF: min={p_rf.min():.3f} max={p_rf.max():.3f} mean={p_rf.mean():.3f}"
        )

        return AgentMessage(
            sender=self.name,
            payload={
                "p_rf"       : p_rf,
                "X_batch"    : X_batch,
                "y_batch"    : y_batch,
                "tx_meta"    : tx_meta,
                "batch_idx"  : batch_idx,
                "batch_size" : batch_size,
                "agent_state": agent_state,
                "start_time" : start_time,
            },
            status="ok",
        )
