"""
perception_agent.py
===================
AGENT 1: PerceptionAgent

WHAT IS PERCEPTION IN THE PAPER?
---------------------------------
The paper says:
  "The perception module acts as the system's interface between raw
   features and higher-level reasoning. It consolidates the extracted
   features into a decision-ready state vector z."

In simple words:
  Raw CSV data  ->> cleaned, validated, normalized feature vector z

WHAT DOES THIS AGENT DO?
-------------------------
When given a batch of raw transactions (X_batch), this agent:
  1. Validates that all expected feature columns are present
  2. Checks for and fills any NaN values that might appear mid-stream
  3. Reports the batch shape and any anomalies found
  4. Passes the clean feature matrix forward as the "state vector z"

INPUT  (AgentMessage payload):
  - "X_batch"     : pd.DataFrame  — raw feature batch
  - "y_batch"     : pd.Series     — true labels (for monitoring)
  - "batch_idx"   : int           — which batch number we're on
  - "agent_state" : dict          — current {w, tau_alert, tau_block}
  - "start_time"  : float         — batch start timestamp

OUTPUT (AgentMessage payload):
  - "X_batch"     : pd.DataFrame  — validated, clean feature matrix (state z)
  - "y_batch"     : pd.Series     — true labels passed through
  - "batch_idx"   : int           — batch number passed through
  - "batch_size"  : int           — number of transactions in this batch
  - "agent_state" : dict          — passed through
  - "start_time"  : float         — passed through
"""

import numpy as np
import pandas as pd

from  agents.base_agent import BaseAgent, AgentMessage


class PerceptionAgent(BaseAgent):

    def __init__(self, expected_features: list):
        super().__init__(name="PerceptionAgent")
        self.expected_features = expected_features

    def _run(self, msg: AgentMessage) -> AgentMessage:
        X_batch: pd.DataFrame = msg.payload["X_batch"]
        y_batch: pd.Series    = msg.payload["y_batch"]
        batch_idx: int        = msg.payload["batch_idx"]
        agent_state: dict     = msg.payload["agent_state"]
        start_time: float     = msg.payload["start_time"]
        raw_meta              = msg.payload.get("tx_meta", None)


        # ── Step 1: check for missing columns ──────────────────────
        missing_cols = [c for c in self.expected_features if c not in X_batch.columns]
        if missing_cols:
            return AgentMessage(
                sender=self.name,
                payload={},
                status="error",
                error=f"Missing feature columns in batch {batch_idx}: {missing_cols}",
            )

        # ── Step 2: check and fix NaNs ─────────────────────────────
        nan_count = int(X_batch.isna().sum().sum())
        if nan_count > 0:
            self.logger.warning(
                f"[{self.name}] Batch {batch_idx+1}: found {nan_count} NaNs  ->> filling with 0"
            )
            X_batch = X_batch.fillna(0)

        # ── Step 3: report batch info ──────────────────────────────
        batch_size = len(X_batch)
        fraud_count = int(y_batch.sum())

        # Build metadata for policy/response layer
        if raw_meta is None:
            raw_meta = pd.DataFrame(index=X_batch.index)

        tx_meta = {
            "tx_hash": raw_meta["tx_hash"].astype(str).tolist()
                if "tx_hash" in raw_meta.columns
                else [f"tx_{batch_idx+1}_{i}" for i in range(batch_size)],
            "from_address": raw_meta["from_address"].astype(str).tolist()
                if "from_address" in raw_meta.columns
                else [f"wallet_from_{idx}" for idx in X_batch.index],
            "to_address": raw_meta["to_address"].astype(str).tolist()
                if "to_address" in raw_meta.columns
                else [f"wallet_to_{idx}" for idx in X_batch.index],
            "timestamp": raw_meta["timestamp"].astype(str).tolist()
                if "timestamp" in raw_meta.columns
                else [f"batch_{batch_idx+1}" for _ in range(batch_size)],
        }

        self.logger.info(
            f"[{self.name}] Batch {batch_idx+1} | "
            f"size={batch_size} | "
            f"frauds_in_batch={fraud_count} | "
            f"fraud_rate={fraud_count/batch_size:.3f}"
        )

        # ── Step 4: pass clean state vector forward ────────────────
        return AgentMessage(
            sender=self.name,
            payload={
                "X_batch"    : X_batch,
                "y_batch"    : y_batch,
                "batch_idx"  : batch_idx,
                "tx_meta"    : tx_meta,
                "batch_size" : batch_size,
                "agent_state": agent_state,
                "start_time" : start_time,
            },
            status="ok",
        )
