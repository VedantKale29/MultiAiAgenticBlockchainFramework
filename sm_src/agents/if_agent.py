"""
agents/if_agent.py
==================
AGENT 3: IFAgent -- unsupervised anomaly detector → s_IF(z)

if agent called by rf agent, receives:
- p_rf: RF probability of fraud (from RFAgent)
- X_batch: feature matrix for current batch
- y_batch: true labels for current batch (for evaluation only)
- tx_meta: transaction metadata (e.g. timestamps, amounts)
- batch_idx: index of current batch
- batch_size: number of transactions in current batch
- agent_state: current state of the agent (e.g. "initial", "running", "final")
- start_time: timestamp when processing of current batch started    

ROLE IN PAPER:
  "Isolation Forest produces an anomaly score reflecting deviation
   from normal behaviour." Scaled to [0, 1] via MinMaxScaler.

NO AWS CALLS. Pure inference.
INPUT  (AgentMessage payload):
  - p_rf: RF probability of fraud (from RFAgent)
  - X_batch: feature matrix for current batch
  - y_batch: true labels for current batch (for evaluation only)
  - tx_meta: transaction metadata (e.g. timestamps, amounts)
  - batch_idx: index of current batch
  - batch_size: number of transactions in current batch
  - agent_state: current state of the agent (e.g. "initial", "running", "final")
  - start_time: timestamp when processing of current batch started
  

OUTPUT (AgentMessage payload):
  - "p_rf"       : passed through from input
  - "s_if"       : IF anomaly score in [0, 1]
  - "X_batch"    : passed through from input
  - "y_batch"    : passed through from input
  - "tx_meta"    : passed through from input
  - "batch_idx"  : passed through from input
  - "batch_size" : passed through from input
  - "agent_state": passed through from input
  - "start_time" : passed through from input
"""

import numpy as np
import pandas as pd
from  agents.base_agent import BaseAgent, AgentMessage


class IFAgent(BaseAgent):
    def __init__(self, if_model):
        super().__init__(name="IFAgent")
        self.if_model = if_model

    def _run(self, msg: AgentMessage) -> AgentMessage:
        p_rf        = msg.payload["p_rf"]
        X_batch     = msg.payload["X_batch"]
        y_batch     = msg.payload["y_batch"]
        tx_meta     = msg.payload["tx_meta"]
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
                "tx_meta"    : tx_meta,
                "batch_idx"  : batch_idx,
                "batch_size" : batch_size,
                "agent_state": agent_state,
                "start_time" : start_time,
            },
            status="ok",
        )
