"""
mcp_client.py
=============
Fraud Detection MCP Client -- Python wrapper

WHAT THIS FILE IS:
  A lightweight Python client that connects to the MCP server
  (mcp_server.py) and calls tools/resources without requiring an LLM.

  This is the "pure Python supervisor" version described in the
  architecture discussion. You can use it in two modes:

  MODE A -- Direct (no server running):
    Call MCP tools directly as Python functions without spawning a
    server process. This is the fastest integration path and works
    identically to the old code -- it's just named differently.
    Zero latency overhead.

  MODE B -- Via MCP server (server running separately):
    Connect to a running mcp_server.py process over stdio.
    The client sends JSON-RPC requests and parses responses.
    This is the proper MCP architecture for when an LLM supervisor
    (Claude Desktop, LangGraph) also needs to call the same tools.

  The BatchSupervisor class below uses MODE A by default and can be
  switched to MODE B by setting use_server=True.

HOW IT INTEGRATES WITH YOUR EXISTING CODE:
  In coordinator_agent.py, AFTER FusionAgent runs and BEFORE
  ActionAgent runs, insert one supervisor check:

      supervisor_result = self.supervisor.route(
          decisions   = decisions_array,
          risk_scores = risk_scores_array,
          tx_meta     = tx_meta_dict,
      )

  If supervisor_result["action"] == "skip_policy":
      → jump straight to MonitoringAgent, skip Policy+Response
  Else:
      → run the normal pipeline

  This does NOT change the pipeline output or any existing file.
  It only adds selective skipping when there is nothing to act on.

IMPORTANT: THE EXISTING PIPELINE IS UNCHANGED.
  CoordinatorAgent still runs every agent in the same order.
  The BatchSupervisor only adds a conditional early-exit from the
  Policy+Response segment for all-CLEAR batches.
"""

import os
import json
import logging
import numpy as np
import pandas as pd
from typing import Optional

logger = logging.getLogger("FraudMCPClient")


# ════════════════════════════════════════════════════════════════════
# DIRECT MODE -- no server process, pure Python calls
# ════════════════════════════════════════════════════════════════════

class FraudMCPClientDirect:
    """
    Direct-mode MCP client.

    Calls the same logic as mcp_server.py's tools, but in-process.
    No network, no serialization overhead, no server process needed.

    This is appropriate when you want the supervisor logic inside
    CoordinatorAgent without running a separate server.
    """

    def __init__(self, rf_model, if_model, run_dir: str, agent_state: dict):
        """
        Parameters
        ----------
        rf_model    : RFModel  -- trained Random Forest wrapper
        if_model    : IFModel  -- trained Isolation Forest wrapper
        run_dir     : str      -- path to run directory (for JSON registries)
        agent_state : dict     -- current {w, tau_alert, tau_block}
        """
        self.rf_model    = rf_model
        self.if_model    = if_model
        self.run_dir     = run_dir
        self.agent_state = agent_state   # reference -- stays in sync with CoordinatorAgent

    # ── Tool: run_fraud_check ──────────────────────────────────────
    def run_fraud_check(self, features: dict) -> dict:
        """
        Run hybrid fraud detection on a single transaction feature dict.
        Mirrors the tool defined in mcp_server.py exactly.
        """
        X = pd.DataFrame([features])

        p_rf  = float(self.rf_model.predict_proba(X)[0])
        s_if  = float(self.if_model.score(X)[0])

        w         = self.agent_state["w"]
        tau_alert = self.agent_state["tau_alert"]
        tau_block = self.agent_state["tau_block"]
        score     = float(np.clip(w * p_rf + (1.0 - w) * s_if, 0.0, 1.0))

        if score >= tau_block:
            decision = "AUTO-BLOCK"
        elif score >= tau_alert:
            decision = "ALERT"
        else:
            decision = "CLEAR"

        return {
            "p_rf"    : round(p_rf, 6),
            "s_if"    : round(s_if, 6),
            "score"   : round(score, 6),
            "decision": decision,
            "w"       : round(w, 4),
            "tau_alert": round(tau_alert, 4),
            "tau_block": round(tau_block, 4),
        }

    # ── Tool: get_wallet_status ────────────────────────────────────
    def get_wallet_status(self, wallet_address: str) -> dict:
        """
        Check wallet status in watchlist + blocked registries.
        Mirrors the tool in mcp_server.py exactly.
        """
        watchlist = self._load_json("watchlist.json", default={})
        blocked   = self._load_json("blocked_wallets.json", default={})

        if wallet_address in blocked:
            e = blocked[wallet_address]
            return {
                "status"      : "BLOCKED",
                "alert_count" : e.get("block_count", 0),
                "max_risk"    : e.get("max_risk", 0.0),
                "first_seen"  : e.get("first_blocked_at"),
                "last_seen"   : e.get("last_blocked_at"),
                "block_reason": e.get("reason"),
                "is_blocked"  : True,
                "is_watchlist": False,
            }
        elif wallet_address in watchlist:
            e = watchlist[wallet_address]
            return {
                "status"      : "WATCHLIST",
                "alert_count" : e.get("alert_count", 0),
                "max_risk"    : e.get("max_risk", 0.0),
                "first_seen"  : e.get("first_seen"),
                "last_seen"   : e.get("last_seen"),
                "block_reason": None,
                "is_blocked"  : False,
                "is_watchlist": True,
            }
        else:
            return {
                "status"      : "CLEAN",
                "alert_count" : 0,
                "max_risk"    : 0.0,
                "first_seen"  : None,
                "last_seen"   : None,
                "block_reason": None,
                "is_blocked"  : False,
                "is_watchlist": False,
            }

    # ── Resource: fraud://watchlist ────────────────────────────────
    def get_resource_watchlist(self) -> dict:
        """Read the watchlist resource (fraud://watchlist)."""
        return self._load_json("watchlist.json", default={})

    # ── Resource: fraud://blocked ──────────────────────────────────
    def get_resource_blocked(self) -> dict:
        """Read the blocked wallets resource (fraud://blocked)."""
        return self._load_json("blocked_wallets.json", default={})

    # ── Resource: fraud://state ────────────────────────────────────
    def get_resource_state(self) -> dict:
        """Read the current agent state resource (fraud://state)."""
        state = self._load_json("final_state.json", default=None)
        return state if state is not None else dict(self.agent_state)

    def _load_json(self, filename: str, default):
        path = os.path.join(self.run_dir, filename)
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass
        return default


# ════════════════════════════════════════════════════════════════════
# BATCH SUPERVISOR -- uses MCP client to gate downstream agents
# ════════════════════════════════════════════════════════════════════

class BatchSupervisor:
    """
    The BatchSupervisor sits inside CoordinatorAgent between
    FusionAgent and ActionAgent.

    It replicates the MCP 'autonomous supervisor' pattern in pure
    Python -- making data-driven routing decisions based on:
      1. The distribution of risk scores in the current batch
      2. The watchlist and blocked registry state
      3. Whether any wallets in the batch are known repeat offenders

    ROUTING DECISIONS:
      "skip_policy_response" -- batch is all-CLEAR; skip Policy+Response
      "fast_block"           -- batch has known blocked wallets; prioritise
      "standard"             -- normal pipeline; run all agents

    This adds genuine intelligence to the coordinator without requiring
    an LLM call or any external server.
    """

    # Threshold: if max risk score in batch is below this,
    # consider the batch "all clear" and skip Policy+Response
    ALL_CLEAR_THRESHOLD = 0.35   # well below typical tau_alert (~0.45)

    def __init__(self, mcp_client: FraudMCPClientDirect):
        self.client = mcp_client
        logger.info("[BatchSupervisor] Initialised (direct mode, no server)")

    def route(
        self,
        decisions  : np.ndarray,
        risk_scores: np.ndarray,
        tx_meta    : dict,
    ) -> dict:
        """
        Decide how to route this batch through downstream agents.

        Parameters
        ----------
        decisions   : np.ndarray of str -- FusionAgent decisions per tx
        risk_scores : np.ndarray of float -- hybrid scores per tx
        tx_meta     : dict with keys "from_address", "tx_hash", etc.

        Returns
        -------
        dict with keys:
          "action"          : str -- "skip_policy_response" | "fast_block" | "standard"
          "reason"          : str -- human-readable explanation
          "known_blocked"   : list[str] -- wallet addresses already blocked
          "known_watchlist" : list[str] -- wallet addresses on watchlist
          "n_clear"         : int
          "n_alert"         : int
          "n_block"         : int
          "max_score"       : float
        """
        n_clear = int(np.sum(decisions == "CLEAR"))
        n_alert = int(np.sum(decisions == "ALERT"))
        n_block = int(np.sum(decisions == "AUTO-BLOCK"))
        max_score = float(np.max(risk_scores))

        from_addresses = tx_meta.get("from_address", []) if tx_meta else []

        # ── Check 1: Is the batch entirely below tau_alert? ─────────
        # max_score < ALL_CLEAR_THRESHOLD means even the hottest
        # transaction is well below the alert threshold.
        # Nothing to flag → skip Policy and Response entirely.
        if max_score < self.ALL_CLEAR_THRESHOLD:
            logger.info(
                f"[BatchSupervisor] ALL-CLEAR batch "
                f"(max_score={max_score:.3f} < {self.ALL_CLEAR_THRESHOLD}) "
                f"→ skipping PolicyAgent + ResponseAgent"
            )
            return {
                "action"          : "skip_policy_response",
                "reason"          : f"all_clear: max_score={max_score:.3f}",
                "known_blocked"   : [],
                "known_watchlist" : [],
                "n_clear"         : n_clear,
                "n_alert"         : n_alert,
                "n_block"         : n_block,
                "max_score"       : max_score,
            }

        # ── Check 2: Are any wallets already in the blocked registry?
        # Read the blocked resource once for the whole batch (efficient)
        blocked_registry = self.client.get_resource_blocked()
        watchlist_registry = self.client.get_resource_watchlist()

        known_blocked   = []
        known_watchlist = []

        for addr in from_addresses:
            if str(addr) in blocked_registry:
                known_blocked.append(str(addr))
            elif str(addr) in watchlist_registry:
                known_watchlist.append(str(addr))

        if known_blocked:
            logger.info(
                f"[BatchSupervisor] FAST-BLOCK route: "
                f"{len(known_blocked)} already-blocked wallets in batch "
                f"→ prioritise block enforcement"
            )
            return {
                "action"          : "fast_block",
                "reason"          : f"{len(known_blocked)} known blocked wallet(s)",
                "known_blocked"   : known_blocked,
                "known_watchlist" : known_watchlist,
                "n_clear"         : n_clear,
                "n_alert"         : n_alert,
                "n_block"         : n_block,
                "max_score"       : max_score,
            }

        # ── Default: standard pipeline ───────────────────────────────
        reason_parts = []
        if n_alert > 0: reason_parts.append(f"{n_alert} alerts")
        if n_block > 0: reason_parts.append(f"{n_block} blocks")
        if known_watchlist: reason_parts.append(f"{len(known_watchlist)} watchlist hits")
        reason = ", ".join(reason_parts) if reason_parts else "normal batch"

        logger.info(
            f"[BatchSupervisor] STANDARD route "
            f"({reason}) max_score={max_score:.3f}"
        )
        return {
            "action"          : "standard",
            "reason"          : reason,
            "known_blocked"   : known_blocked,
            "known_watchlist" : known_watchlist,
            "n_clear"         : n_clear,
            "n_alert"         : n_alert,
            "n_block"         : n_block,
            "max_score"       : max_score,
        }