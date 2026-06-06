"""
agents/action_agent.py
======================
AGENT 5: ActionAgent -- enforces CLEAR / ALERT / AUTO-BLOCK


ROLE IN PAPER:
  "The action module enforces the decisions made by the cognition layer."
  Action agent receives the final decisions and risk scores for each transaction in the batch, along with all relevant metadata and agent state information. It then executes the appropriate actions (e.g. clear, alert, auto-block) based on the decisions, and generates a summary report of the actions taken for the batch.
   How it is diffrent from Fusion agent: Action agent is responsible for enforcing the decisions (e.g. by calling AWS APIs to block transactions, send alerts, etc.) and generating a summary report of actions taken. The Fusion agent is responsible for aggregating and summarizing information from multiple agents to provide a holistic view of the system state.
NO AWS CALLS. Pure decision enforcement and summary reporting.
INPUT  (AgentMessage payload):
  - "decisions"   : np.ndarray -- final decisions ("CLEAR", "ALERT", "AUTO-BLOCK") for current batch
  - "risk_scores" : np.ndarray -- final risk scores S(z) for current batch
  - "p_rf"        : np.ndarray -- passed through from input (for transparency and debugging)
  - "s_if"        : np.ndarray -- passed through from input (for transparency and debugging)
  - "y_batch"     : pd.Series  -- passed through from input
  - "tx_meta"     : pd.DataFrame -- passed through from input
  - "batch_idx"   : int        -- batch number
  - "batch_size"  : int        -- number of transactions in current batch
  - "agent_state" : dict       -- current {w, tau_alert, tau_block
  - "start_time"  : float      -- timestamp when processing of current batch started

OUTPUT (AgentMessage payload):
  - "action_report": dict       -- summary of actions taken for current batch (e.g. number of transactions cleared, alerted, auto-blocked, average risk score, etc.)
  - "decisions"    : np.ndarray -- passed through from input (for transparency and debugging)
  - "risk_scores"  : np.ndarray -- passed through from input (for transparency and debugging)
  - "p_rf"         : np.ndarray -- passed through from input (for transparency and debugging)
  - "s_if"         : np.ndarray -- passed through from input (for transparency and debugging)
  - "y_batch"      : pd.Series  -- passed through from input
  - "tx_meta"      : pd.DataFrame -- passed through from input
  - "batch_idx"    : int        -- batch number
  - "batch_size"   : int        -- number of transactions in current batch
  - "agent_state"  : dict       -- current {w, tau_alert, tau_block
  - "start_time"   : float      -- timestamp when processing of current batch started
"""


import numpy as np
from agents.base_agent import BaseAgent, AgentMessage


class ActionAgent(BaseAgent):
    def __init__(self):
        super().__init__(name="ActionAgent")

    def _run(self, msg: AgentMessage) -> AgentMessage:
        decisions   = msg.payload["decisions"]
        risk_scores = msg.payload["risk_scores"]
        p_rf        = msg.payload["p_rf"]
        s_if        = msg.payload["s_if"]
        y_batch     = msg.payload["y_batch"]
        tx_meta     = msg.payload["tx_meta"]
        batch_idx   = msg.payload["batch_idx"]
        batch_size  = msg.payload["batch_size"]
        agent_state = msg.payload["agent_state"]
        start_time  = msg.payload["start_time"]

        n_clear = int(np.sum(decisions == "CLEAR"))       # number of transactions cleared
        n_alert = int(np.sum(decisions == "ALERT"))
        n_block = int(np.sum(decisions == "AUTO-BLOCK"))

        action_report = {
            "batch"             : batch_idx + 1,
            "total_transactions": batch_size,
            "cleared"           : n_clear,
            "alerted"           : n_alert,
            "auto_blocked"      : n_block,
            "avg_risk_score"    : float(np.mean(risk_scores)),
            "max_risk_score"    : float(np.max(risk_scores)),
        }
        
        # Action agent only generates the action report and logs the summary of actions taken for the batch. 
        # It does not perform any actual enforcement (e.g. no AWS calls to block transactions, etc.) as per the requirement of being a pure decision enforcement and summary reporting agent.
        
        """Example of Action Report:
        {
            "batch": 1,
            "total_transactions": 1000,
            "cleared": 950,
            "alerted": 30,
            "auto_blocked": 20,
            "avg_risk_score": 0.15,
            "max_risk_score": 0.95
        }
        """
        self.logger.info(
            f"[{self.name}] Batch {batch_idx+1} | "
            f"CLEARED={n_clear} ALERTED={n_alert} BLOCKED={n_block}"
        )

        return AgentMessage(
            sender=self.name,
            payload={
                "action_report": action_report,
                "decisions"    : decisions,
                "risk_scores"  : risk_scores,
                "p_rf"         : p_rf,
                "s_if"         : s_if,
                "y_batch"      : y_batch,
                "tx_meta"      : tx_meta,
                "batch_idx"    : batch_idx,
                "batch_size"   : batch_size,
                "agent_state"  : agent_state,
                "start_time"   : start_time,
            },
            status="ok",
        )
