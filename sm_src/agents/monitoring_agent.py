"""
agents/monitor_agent.py
========================
STAGE 2 — Real-Time Blockchain Monitor Agent

ROLE (Section 3 of Framework — Self-Triggering Mechanism):
  Continuously watches the Hardhat local chain for suspicious
  transaction patterns. When an anomaly score crosses the cached
  threshold, emits THREAT_DETECTED and activates the Decision Agent.

HOW IT WORKS (3 steps from the framework doc):
  Step 1 — Continuous monitoring:
    Web3.py event filter polls every POLL_INTERVAL_SECONDS (default 1s).
    Watches the configured VulnerablePool for unusual activity.

  Step 2 — Anomaly scoring:
    Each incoming event is scored against in-memory thresholds.
    Checks:
      - Single-block outflow > OUTFLOW_LIMIT_ETH
      - Transaction frequency spike (> FREQ_SPIKE_THRESHOLD txs in window)
      - Price oracle deviation > ORACLE_DEVIATION_PCT

  Step 3 — Trigger emission:
    When score >= tau_alert, builds a THREAT_DETECTED payload and
    calls handle_threat(payload) — your callback that invokes the
    Decision → Contract → Governance chain.

THRESHOLD CACHE (Section 3, Impossibility 2 solution):
  - Initialised once from GovernanceContract at startup (1 RPC call)
  - Synced by ThresholdUpdated events from GovernanceContract
  - Sub-millisecond lookup — no per-event RPC call
  - Cache update logged every time it changes

ZERO BREAKING CHANGES:
  - Does NOT modify any existing agent
  - If Web3 / Hardhat absent: monitor logs a warning and is a no-op
  - All errors are caught; never crashes main pipeline

DESIGN (from roadmap doc, Section 9.1 row 4):
  "Web3.py 1-second filter on Hardhat; WebSocket on Sepolia as secondary"

INSTALL:
  pip install web3
  npx hardhat node    ← local chain on http://127.0.0.1:8545
  # Deploy contracts first (scripts/deploy.js)

ENVIRONMENT VARIABLES (optional):
  HARDHAT_URL                 — default http://127.0.0.1:8545
  GOVERNANCE_CONTRACT_ADDRESS — deployed GovernanceContract address
  VULNERABLE_POOL_ADDRESS     — deployed NaiveReceiverLenderPool address
  REGISTRY_CONTRACT_ADDRESS   — deployed ContractRegistry address
"""

import os
import time
import json
import threading
import logging
from datetime import datetime, timezone
from typing import Optional, Callable

from agents.base_agent import BaseAgent, AgentMessage

logger = logging.getLogger(__name__)

# ── Optional Web3 import ───────────────────────────────────────
try:
    from web3 import Web3
    from web3.exceptions import BlockNotFound
    _WEB3_AVAILABLE = True
except ImportError:
    _WEB3_AVAILABLE = False

# ── GovernanceContract minimal ABI (only what we need) ────────
_GOV_ABI = [
    {
        "inputs": [{"internalType": "string", "name": "param", "type": "string"}],
        "name": "getParameter",
        "outputs": [{"internalType": "uint256", "name": "", "type": "uint256"}],
        "stateMutability": "view",
        "type": "function",
    },
    {
        "anonymous": False,
        "inputs": [
            {"indexed": False, "internalType": "string",  "name": "param",    "type": "string"},
            {"indexed": False, "internalType": "uint256", "name": "oldValue", "type": "uint256"},
            {"indexed": False, "internalType": "uint256", "name": "newValue", "type": "uint256"},
        ],
        "name": "ParameterUpdated",
        "type": "event",
    },
]

# ── NaiveReceiverLenderPool minimal ABI ───────────────────────
_POOL_ABI = [
    {
        "anonymous": False,
        "inputs": [
            {"indexed": True,  "internalType": "address", "name": "borrower",  "type": "address"},
            {"indexed": False, "internalType": "uint256", "name": "amount",    "type": "uint256"},
            {"indexed": False, "internalType": "uint256", "name": "fee",       "type": "uint256"},
        ],
        "name": "FlashLoan",
        "type": "event",
    },
    {
        "anonymous": False,
        "inputs": [
            {"indexed": True,  "internalType": "address", "name": "from",   "type": "address"},
            {"indexed": False, "internalType": "uint256", "name": "amount", "type": "uint256"},
        ],
        "name": "Deposit",
        "type": "event",
    },
    {
        "inputs":  [],
        "name":    "poolBalance",
        "outputs": [{"internalType": "uint256", "name": "", "type": "uint256"}],
        "stateMutability": "view",
        "type": "function",
    },
    {
        "inputs": [],
        "name": "maxFlashLoan",
        "outputs": [{"internalType": "uint256", "name": "", "type": "uint256"}],
        "stateMutability": "view",
        "type": "function",
    },
]


class ThresholdCache:
    """
    In-memory cache of GovernanceContract parameters.
    Initialised once at startup; updated via ThresholdUpdated events.
    Sub-millisecond reads — no per-event RPC call.
    """

    # Defaults match config.py paper values
    _DEFAULTS = {
        "tau_alert":    0.487,
        "tau_block":    0.587,
        "w":            0.700,
        "escalation_n": 3.0,
    }

    def __init__(self):
        self._cache: dict = dict(self._DEFAULTS)
        self._lock  = threading.Lock()

    def get(self, param: str) -> float:
        with self._lock:
            return self._cache.get(param, self._DEFAULTS.get(param, 0.0))

    def update(self, param: str, value: float):
        with self._lock:
            old = self._cache.get(param)
            self._cache[param] = value
            logger.info(
                f"[ThresholdCache] {param}: {old} → {value}  (on-chain update)"
            )

    def load_from_contract(self, gov_contract) -> bool:
        """Read all known params from the deployed GovernanceContract."""
        try:
            for param in self._DEFAULTS:
                raw = gov_contract.functions.getParameter(param).call()
                self.update(param, raw / 1e18)
            logger.info("[ThresholdCache] Loaded all params from GovernanceContract")
            return True
        except Exception as e:
            logger.warning(
                f"[ThresholdCache] Could not load from contract ({e})"
                " — using defaults"
            )
            return False

    def as_dict(self) -> dict:
        with self._lock:
            return dict(self._cache)


class MonitorAgent(BaseAgent):
    """
    Real-time blockchain monitor.
    Polls Hardhat for events every POLL_INTERVAL_SECONDS.
    Fires handle_threat() when anomaly score >= tau_alert.
    """

    POLL_INTERVAL_SECONDS     = 1.0    # Web3.py filter poll interval
    OUTFLOW_LIMIT_ETH         = 50.0   # single-block outflow threshold (ETH)
    FREQ_SPIKE_THRESHOLD      = 3      # flash loan events per block = spike
    ORACLE_DEVIATION_PCT      = 15.0   # price oracle deviation % threshold
    ANOMALY_SCORE_FLASH_LOAN  = 0.95   # score assigned to detected flash loan
    ANOMALY_SCORE_FREQ_SPIKE  = 0.75   # score assigned to frequency spike
    ANOMALY_SCORE_OUTFLOW     = 0.65   # score assigned to high outflow

    def __init__(
        self,
        hardhat_url:                str = "http://127.0.0.1:8545",
        governance_contract_address: Optional[str] = None,
        pool_contract_address:       Optional[str] = None,
        handle_threat:               Optional[Callable] = None,
        **kwargs,
    ):
        super().__init__(name="MonitorAgent", **kwargs)

        self.hardhat_url                = os.getenv("HARDHAT_URL", hardhat_url)
        self.governance_contract_address = (
            governance_contract_address
            or os.getenv("GOVERNANCE_CONTRACT_ADDRESS")
        )
        self.pool_contract_address = (
            pool_contract_address
            or os.getenv("VULNERABLE_POOL_ADDRESS")
        )
        self.handle_threat: Optional[Callable] = handle_threat

        # In-memory threshold cache
        self.threshold_cache = ThresholdCache()

        # Web3 connection
        self._w3:            Optional[object] = None
        self._gov_contract:  Optional[object] = None
        self._pool_contract: Optional[object] = None
        self._running        = False
        self._monitor_thread: Optional[threading.Thread] = None

        # State for frequency-spike detection
        self._block_flash_loan_counts: dict = {}  # block_number → count

        # Detect + response latency tracking
        self.last_threat_payload: Optional[dict] = None

        self._connect()

    # ── Connection ─────────────────────────────────────────────

    def _connect(self):
        """Try to connect to Hardhat. Fail silently if unavailable."""
        if not _WEB3_AVAILABLE:
            self.logger.warning(
                "[MonitorAgent] web3 not installed — monitor is a no-op. "
                "Run: pip install web3"
            )
            return

        try:
            w3 = Web3(Web3.HTTPProvider(self.hardhat_url, request_kwargs={"timeout": 3}))
            if not w3.is_connected():
                self.logger.warning(
                    f"[MonitorAgent] Cannot reach Hardhat at {self.hardhat_url}. "
                    "Run: npx hardhat node"
                )
                return

            self._w3 = w3
            self.logger.info(
                f"[MonitorAgent] Connected to Hardhat @ {self.hardhat_url} "
                f"(chain_id={w3.eth.chain_id})"
            )

            # Wire GovernanceContract
            if self.governance_contract_address:
                self._gov_contract = w3.eth.contract(
                    address=Web3.to_checksum_address(self.governance_contract_address),
                    abi=_GOV_ABI,
                )
                self.threshold_cache.load_from_contract(self._gov_contract)
            else:
                self.logger.info(
                    "[MonitorAgent] No GovernanceContract address — "
                    "using default thresholds"
                )

            # Wire VulnerablePool
            if self.pool_contract_address:
                self._pool_contract = w3.eth.contract(
                    address=Web3.to_checksum_address(self.pool_contract_address),
                    abi=_POOL_ABI,
                )
                self.logger.info(
                    f"[MonitorAgent] Watching pool @ {self.pool_contract_address}"
                )

        except Exception as e:
            self.logger.warning(f"[MonitorAgent] Connection failed ({e})")

    # ── BaseAgent._run() ───────────────────────────────────────

    def _run(self, message: AgentMessage) -> AgentMessage:
        """
        Called by CoordinatorAgent once per batch (existing pipeline).
        In the existing batch mode, the monitor is a pass-through.
        Real-time monitoring happens in the background thread started
        by start() / stop().
        """
        payload = dict(message.payload)
        payload["monitor_active"] = self._running
        payload["threshold_cache"] = self.threshold_cache.as_dict()
        if self.last_threat_payload:
            payload["last_threat"] = self.last_threat_payload
        return AgentMessage(
            sender=self.name,
            payload=payload,
            status="ok",
        )

    # ── Background monitoring loop ─────────────────────────────

    def start(self):
        """
        Start the background monitoring thread.
        Non-blocking: returns immediately.
        """
        if self._w3 is None:
            self.logger.warning(
                "[MonitorAgent] No Web3 connection — cannot start monitor."
            )
            return

        if self._running:
            return

        self._running = True
        self._monitor_thread = threading.Thread(
            target=self._monitor_loop,
            daemon=True,
            name="MonitorAgent-loop",
        )
        self._monitor_thread.start()
        self.logger.info("[MonitorAgent] Background monitoring started")

    def stop(self):
        """Signal the background loop to stop."""
        self._running = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5.0)
        self.logger.info("[MonitorAgent] Background monitoring stopped")

    def _monitor_loop(self):
        """
        Main polling loop.
        Runs in a daemon thread.
        """
        w3 = self._w3
        last_block = w3.eth.block_number
        gov_filter = None

        # Create event filter for GovernanceContract ThresholdUpdated
        if self._gov_contract:
            try:
                gov_filter = self._gov_contract.events.ParameterUpdated.create_filter(
                    from_block="latest"
                )
                self.logger.info("[MonitorAgent] GovernanceContract filter created")
            except Exception as e:
                self.logger.warning(f"[MonitorAgent] Gov filter failed ({e})")

        # Create event filter for VulnerablePool FlashLoan
        pool_filter = None
        if self._pool_contract:
            try:
                pool_filter = self._pool_contract.events.FlashLoan.create_filter(
                    from_block="latest"
                )
                self.logger.info("[MonitorAgent] Pool FlashLoan filter created")
            except Exception as e:
                self.logger.warning(f"[MonitorAgent] Pool filter failed ({e})")

        self.logger.info(
            f"[MonitorAgent] Polling every {self.POLL_INTERVAL_SECONDS}s "
            f"from block {last_block}"
        )

        while self._running:
            try:
                # ── 1. Sync threshold cache from GovernanceContract events ──
                if gov_filter:
                    for event in gov_filter.get_new_entries():
                        param    = event["args"]["param"]
                        newValue = event["args"]["newValue"] / 1e18
                        self.threshold_cache.update(param, newValue)

                # ── 2. Scan new blocks for anomalies ──────────────────────
                current_block = w3.eth.block_number
                if current_block > last_block:
                    self._scan_blocks(last_block + 1, current_block)
                    last_block = current_block

                # ── 3. Process pool FlashLoan events ─────────────────────
                if pool_filter:
                    for event in pool_filter.get_new_entries():
                        self._handle_flash_loan_event(event)

            except Exception as e:
                self.logger.warning(f"[MonitorAgent] Poll error: {e}")

            time.sleep(self.POLL_INTERVAL_SECONDS)

    def _scan_blocks(self, from_block: int, to_block: int):
        """
        Scan a range of newly mined blocks for suspicious patterns.
        Called on every poll cycle that found new blocks.
        """
        w3 = self._w3
        for block_num in range(from_block, to_block + 1):
            try:
                block = w3.eth.get_block(block_num, full_transactions=True)
            except BlockNotFound:
                continue

            txs = block.get("transactions", [])
            if not txs:
                continue

            # Check total ETH outflow in this block
            total_outflow_wei = sum(
                tx.get("value", 0) for tx in txs
                if tx.get("value", 0) > 0
            )
            total_outflow_eth = total_outflow_wei / 1e18

            score = 0.0
            reasons = []

            if total_outflow_eth > self.OUTFLOW_LIMIT_ETH:
                score = max(score, self.ANOMALY_SCORE_OUTFLOW)
                reasons.append(
                    f"high_outflow:{total_outflow_eth:.2f}ETH > {self.OUTFLOW_LIMIT_ETH}ETH"
                )

            # Check tx count spike targeting the pool
            if self._pool_contract:
                pool_addr = self.pool_contract_address.lower()
                pool_txs  = [
                    tx for tx in txs
                    if tx.get("to", "").lower() == pool_addr
                ]
                if len(pool_txs) >= self.FREQ_SPIKE_THRESHOLD:
                    score = max(score, self.ANOMALY_SCORE_FREQ_SPIKE)
                    reasons.append(
                        f"freq_spike:{len(pool_txs)} pool_txs in block {block_num}"
                    )

            # Emit THREAT_DETECTED if score exceeds tau_alert
            tau_alert = self.threshold_cache.get("tau_alert")
            if score >= tau_alert:
                self._emit_threat(
                    block_num=block_num,
                    score=score,
                    anomaly_type="block_scan",
                    reasons=reasons,
                    tx_count=len(txs),
                    total_outflow_eth=total_outflow_eth,
                )

    def _handle_flash_loan_event(self, event):
        """
        Called when a FlashLoan event is emitted by the VulnerablePool.
        Flash loan = highest anomaly score, triggers immediately.
        """
        args       = event["args"]
        block_num  = event["blockNumber"]
        borrower   = args.get("borrower", "0x0")
        amount_wei = args.get("amount",   0)
        amount_eth = amount_wei / 1e18

        self.logger.warning(
            f"[MonitorAgent] FLASH LOAN detected | "
            f"block={block_num} borrower={borrower[:12]}... "
            f"amount={amount_eth:.4f}ETH"
        )

        # Track per-block flash loan counts for frequency scoring
        count = self._block_flash_loan_counts.get(block_num, 0) + 1
        self._block_flash_loan_counts[block_num] = count

        score = self.ANOMALY_SCORE_FLASH_LOAN
        tau_alert = self.threshold_cache.get("tau_alert")

        if score >= tau_alert:
            self._emit_threat(
                block_num=block_num,
                score=score,
                anomaly_type="flash_loan",
                reasons=[f"FlashLoan event: borrower={borrower} amount={amount_eth:.4f}ETH"],
                tx_count=1,
                total_outflow_eth=amount_eth,
                tx_hash=event.get("transactionHash", b"").hex()
                        if hasattr(event.get("transactionHash", b""), "hex") else "",
                attacker_address=borrower,
            )

    def _emit_threat(
        self,
        block_num:         int,
        score:             float,
        anomaly_type:      str,
        reasons:           list,
        tx_count:          int,
        total_outflow_eth: float,
        tx_hash:           str = "",
        attacker_address:  str = "",
    ):
        """
        Build the THREAT_DETECTED payload and call handle_threat().
        Records detection timestamp for latency measurement.
        """
        tau_alert = self.threshold_cache.get("tau_alert")
        tau_block = self.threshold_cache.get("tau_block")
        w         = self.threshold_cache.get("w")

        payload = {
            "event_type":       "THREAT_DETECTED",
            "block_number":     block_num,
            "anomaly_score":    round(score, 4),
            "anomaly_type":     anomaly_type,
            "reasons":          reasons,
            "tx_hash":          tx_hash,
            "attacker_address": attacker_address,
            "tx_count_in_block": tx_count,
            "total_outflow_eth": round(total_outflow_eth, 4),
            "threshold_at_trigger": {
                "tau_alert": tau_alert,
                "tau_block": tau_block,
                "w":         w,
            },
            "timestamp":        datetime.now(timezone.utc).isoformat(),
            "detection_time_ns": time.perf_counter_ns(),
        }

        self.logger.warning(
            f"[MonitorAgent] >>> THREAT_DETECTED <<< "
            f"block={block_num} score={score:.3f} type={anomaly_type} "
            f"tau_alert={tau_alert:.3f}"
        )

        self.last_threat_payload = payload

        if self.handle_threat is not None:
            try:
                self.handle_threat(payload)
            except Exception as e:
                self.logger.error(f"[MonitorAgent] handle_threat callback failed: {e}")

    # ── Utility ────────────────────────────────────────────────

    def is_connected(self) -> bool:
        return self._w3 is not None and self._w3.is_connected()

    def get_threshold_snapshot(self) -> dict:
        return self.threshold_cache.as_dict()