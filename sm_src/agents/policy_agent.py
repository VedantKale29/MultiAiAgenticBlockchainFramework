"""
agents/policy_agent.py
======================
AGENT 6: PolicyAgent

ROLE:
  Converts model decisions into autonomous policy actions.

MODEL decision != POLICY action

Input decision:
  CLEAR / ALERT / AUTO-BLOCK

Policy output:
  ALLOW / WATCHLIST / BLOCK

RULES:
  1. If wallet already blocked      -> BLOCK immediately
  2. If decision == AUTO-BLOCK      -> BLOCK
  3. If decision == ALERT           -> WATCHLIST
  4. If wallet gets repeated ALERTs -> escalate to BLOCK
"""

import os
import json
from datetime import datetime
import numpy as np
import pandas as pd

from agents.base_agent import BaseAgent, AgentMessage
import config


class PolicyAgent(BaseAgent):
    def __init__(self, run_dir: str):
        super().__init__(name="PolicyAgent")
        self.run_dir = run_dir
        os.makedirs(self.run_dir, exist_ok=True)

        self.watchlist_path = os.path.join(self.run_dir, "watchlist.json")
        self.blocked_path   = os.path.join(self.run_dir, "blocked_wallets.json")

        self.watchlist = self._load_json(self.watchlist_path, default={})
        self.blocked   = self._load_json(self.blocked_path, default={})

    def _load_json(self, path, default):
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                self.logger.warning(f"[{self.name}] Could not parse {path}, resetting.")
        return default

    def _save_json(self, path, obj):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(obj, f, indent=2)

    def _touch_watchlist(self, wallet, tx_hash, risk_score):
        now = datetime.utcnow().isoformat()
        entry = self.watchlist.get(wallet, {
            "alert_count": 0,
            "recent_tx_hashes": [],
            "first_seen": now,
            "last_seen": now,
            "max_risk": 0.0,
            "status": "WATCHLIST",
        })

        entry["alert_count"] += 1
        entry["last_seen"] = now
        entry["max_risk"] = float(max(entry.get("max_risk", 0.0), risk_score))
        entry["status"] = "WATCHLIST"

        hashes = entry.get("recent_tx_hashes", [])
        hashes.append(tx_hash)
        entry["recent_tx_hashes"] = hashes[-10:]  # keep last 10

        self.watchlist[wallet] = entry
        return entry

    def _block_wallet(self, wallet, tx_hash, risk_score, reason):
        now = datetime.utcnow().isoformat()
        entry = self.blocked.get(wallet, {
            "first_blocked_at": now,
            "last_blocked_at": now,
            "block_count": 0,
            "max_risk": 0.0,
            "reason": reason,
            "recent_tx_hashes": [],
            "status": "BLOCKED",
        })

        entry["last_blocked_at"] = now
        entry["block_count"] = int(entry.get("block_count", 0)) + 1
        entry["max_risk"] = float(max(entry.get("max_risk", 0.0), risk_score))
        entry["reason"] = reason
        entry["status"] = "BLOCKED"

        hashes = entry.get("recent_tx_hashes", [])
        hashes.append(tx_hash)
        entry["recent_tx_hashes"] = hashes[-10:]

        self.blocked[wallet] = entry

    def _run(self, msg: AgentMessage) -> AgentMessage:
        decisions   = np.asarray(msg.payload["decisions"], dtype=object)
        risk_scores = np.asarray(msg.payload["risk_scores"], dtype=float)
        p_rf        = np.asarray(msg.payload["p_rf"], dtype=float)
        s_if        = np.asarray(msg.payload["s_if"], dtype=float)
        y_batch     = msg.payload["y_batch"]
        batch_idx   = msg.payload["batch_idx"]
        batch_size  = msg.payload["batch_size"]
        agent_state = msg.payload["agent_state"]
        start_time  = msg.payload["start_time"]
        tx_meta     = msg.payload["tx_meta"]

        tx_hashes   = tx_meta["tx_hash"]
        from_addrs  = tx_meta["from_address"]
        to_addrs    = tx_meta["to_address"]
        timestamps  = tx_meta["timestamp"]

        policy_actions = []
        policy_reasons = []
        escalated_count = 0
        blocked_now = 0
        watchlisted_now = 0

        repeat_alert_threshold = config.POLICY_ALERT_ESCALATION_THRESHOLD

        for i in range(batch_size):
            wallet   = str(from_addrs[i])
            tx_hash  = str(tx_hashes[i])
            score    = float(risk_scores[i])
            decision = str(decisions[i])

            # Rule 1: previously blocked wallet => block immediately
            if wallet in self.blocked:
                policy_actions.append("BLOCK")
                policy_reasons.append("previously_blocked")
                blocked_now += 1
                self._block_wallet(wallet, tx_hash, score, reason="previously_blocked")
                continue

            # Rule 2: model says AUTO-BLOCK => block
            if decision == "AUTO-BLOCK":
                policy_actions.append("BLOCK")
                policy_reasons.append("model_auto_block")
                blocked_now += 1
                self._block_wallet(wallet, tx_hash, score, reason="model_auto_block")
                continue

            # Rule 3: model says ALERT => watchlist
            if decision == "ALERT":
                entry = self._touch_watchlist(wallet, tx_hash, score)
                alert_count = int(entry["alert_count"])

                # Rule 4: repeated alerts => escalate to block
                if alert_count >= repeat_alert_threshold:
                    policy_actions.append("BLOCK")
                    policy_reasons.append("repeat_alert_escalation")
                    escalated_count += 1
                    blocked_now += 1
                    self._block_wallet(wallet, tx_hash, score, reason="repeat_alert_escalation")
                else:
                    policy_actions.append("WATCHLIST")
                    policy_reasons.append("alert_watchlist")
                    watchlisted_now += 1
                continue

            # Rule 5: CLEAR => allow
            policy_actions.append("ALLOW")
            policy_reasons.append("clear")
        
        self._save_json(self.watchlist_path, self.watchlist)
        self._save_json(self.blocked_path, self.blocked)

        policy_report = {
            "batch": batch_idx + 1,
            "blocked_now": int(blocked_now),
            "watchlisted_now": int(watchlisted_now),
            "escalated_now": int(escalated_count),
            "blocked_registry_size": int(len(self.blocked)),
            "watchlist_registry_size": int(len(self.watchlist)),
        }

        self.logger.info(
            f"[{self.name}] Batch {batch_idx+1} | "
            f"BLOCK={blocked_now} WATCHLIST={watchlisted_now} ESCALATED={escalated_count}"
        )

        return AgentMessage(
            sender=self.name,
            payload={
                "policy_actions": np.asarray(policy_actions, dtype=object),
                "policy_reasons": np.asarray(policy_reasons, dtype=object),
                "policy_report": policy_report,
                "decisions": decisions,
                "risk_scores": risk_scores,
                "p_rf": p_rf,
                "s_if": s_if,
                "y_batch": y_batch,
                "batch_idx": batch_idx,
                "batch_size": batch_size,
                "agent_state": agent_state,
                "start_time": start_time,
                "tx_meta": tx_meta,
            },
            status="ok",
        )