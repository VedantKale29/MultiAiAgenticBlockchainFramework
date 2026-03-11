"""
base_agent.py
=============
The BASE AGENT — the blueprint that every other agent inherits from.

WHY DO WE NEED THIS?
--------------------
In a multi-agent system, every agent (PerceptionAgent, RFAgent, etc.)
shares a common identity:
  - It has a NAME
  - It can RECEIVE a message (input dictionary)
  - It can SEND a message (output dictionary)
  - It logs what it does

Instead of copy-pasting these behaviors into every agent, we define
them ONCE here in BaseAgent, and every agent just calls super().__init__()
to get all of this for free.

WHAT IS AN "AgentMessage"?
--------------------------
Think of agents as workers in an office. When one worker finishes a task,
they put results in a labeled envelope (AgentMessage) and pass it to the
next worker. The envelope always contains:
  - sender: who created it
  - payload: the actual data (dict)
  - status: did it succeed or fail?
  - error: if failed, what went wrong?
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional
from logger import logging


# ─────────────────────────────────────────────────────────────
# AgentMessage — the "envelope" passed between agents
# ─────────────────────────────────────────────────────────────
@dataclass
class AgentMessage:
    """
    A structured message passed between agents.

    Example:
        msg = AgentMessage(
            sender="RFAgent",
            payload={"p_rf": array([0.9, 0.1, 0.7])},
            status="ok"
        )
    """
    sender: str                          # which agent created this
    payload: Dict[str, Any] = field(default_factory=dict)  # the actual data
    status: str = "ok"                   # "ok" or "error"
    error: Optional[str] = None          # error message if status == "error"


# ─────────────────────────────────────────────────────────────
# BaseAgent — the parent class every agent inherits from
# ─────────────────────────────────────────────────────────────
class BaseAgent:
    """
    All agents in this system extend BaseAgent.

    What you get for FREE by inheriting BaseAgent:
      - self.name          → agent's identity
      - self.logger        → pre-configured logger
      - self.run(msg)      → standard entry point (calls self._run internally)
      - error handling     → if _run crashes, BaseAgent catches it and
                             returns an error AgentMessage automatically

    What YOU must implement in each subclass:
      - _run(msg: AgentMessage) -> AgentMessage
        (this is where your agent's logic goes)
    """

    def __init__(self, name: str):
        self.name = name
        self.logger = logging
        self.logger.info(f"[{self.name}] Initialized")

    def run(self, msg: AgentMessage) -> AgentMessage:
        """
        PUBLIC entry point. Called by CoordinatorAgent.
        Wraps _run() with error handling so one agent crashing
        doesn't take down the whole pipeline.
        """
        try:
            self.logger.info(f"[{self.name}] Received message from '{msg.sender}'")
            result = self._run(msg)
            self.logger.info(f"[{self.name}] Done → status={result.status}")
            return result
        except Exception as e:
            self.logger.error(f"[{self.name}] CRASHED: {e}")
            return AgentMessage(
                sender=self.name,
                payload={},
                status="error",
                error=str(e),
            )

    def _run(self, msg: AgentMessage) -> AgentMessage:
        """
        OVERRIDE THIS in every subclass.
        This is where the agent's actual logic lives.
        """
        raise NotImplementedError(f"{self.name} must implement _run()")
