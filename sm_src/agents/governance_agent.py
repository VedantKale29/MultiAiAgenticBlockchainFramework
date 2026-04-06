"""
agents/governance_agent.py
===========================
STAGE 4 -- GovernanceAgent

ROLE IN FRAMEWORK (Section 4.4 -- revised):
  After N consecutive batches with the SAME threat type, proposes
  a governance update to the on-chain GovernanceContract using the
  OpenZeppelin TimelockController propose-timelock-execute pattern.

  The AI proposes.  The blockchain executes after the grace period.
  Human cancel is available during the time-lock window.

MAPS TO:
  RO5 -- 'on-chain self-governance with time-lock: AI-proposed
         governance changes are executed by the blockchain autonomously
         after a grace period, making AI self-modification fully
         auditable and reversible.'

DESIGN (from roadmap doc, Section 9.1 row 6):
  - AI submits a PROPOSAL transaction (no privileged key required).
  - TimelockController queues it with a configurable delay
    (60 seconds on Hardhat = represents 24 hours in production).
  - Anyone can call execute() after the delay.
  - Human can call cancel() during the window.
  - All proposals are logged to governance_proposals.json.
  - If Web3 / Hardhat absent: proposals are logged locally only
    (simulation mode) -- pipeline never crashes.

GOVERNANCE PARAMETERS that can be updated:
  - tau_alert     : detection threshold
  - tau_block     : auto-block threshold
  - w             : fusion weight
  - escalation_n  : repeat-alert count before escalation

TRIGGER LOGIC:
  GovernanceAgent tracks a rolling window of the last
  GOVERNANCE_WINDOW batches. If the same threat_type appears in
  >= GOVERNANCE_CONSECUTIVE_THRESHOLD batches AND the current
  tau_alert is above the GOVERNANCE_TAU_FLOOR, it proposes
  lowering tau_alert by GOVERNANCE_TAU_STEP.

INSTALL:
  pip install web3
  Run: npx hardhat node   (Hardhat on http://127.0.0.1:8545)
  Deploy GovernanceContract.sol (see contracts/GovernanceContract.sol)
"""

import os
import json
import hashlib
from datetime import datetime
from collections import deque
from typing import Optional

from agents.base_agent import BaseAgent, AgentMessage
import config


# ── Governance constants (overridable via config) ──────────────
GOVERNANCE_WINDOW                = int(getattr(config, "GOVERNANCE_WINDOW",          3))
GOVERNANCE_CONSECUTIVE_THRESHOLD = int(getattr(config, "GOVERNANCE_CONSECUTIVE",     2))
GOVERNANCE_TAU_STEP              = float(getattr(config, "GOVERNANCE_TAU_STEP",     0.02))
GOVERNANCE_TAU_FLOOR             = float(getattr(config, "GOVERNANCE_TAU_FLOOR",    0.30))
GOVERNANCE_TIMELOCK_DELAY_S      = int(getattr(config, "GOVERNANCE_TIMELOCK_DELAY", 60))

# ── GovernanceContract ABI (propose function only) ─────────────
GOVERNANCE_ABI = [
    {
        "inputs": [
            {"name": "param",    "type": "string"},
            {"name": "newValue", "type": "uint256"},
            {"name": "reason",   "type": "string"},
        ],
        "name": "propose",
        "outputs": [{"name": "proposalId", "type": "bytes32"}],
        "stateMutability": "nonpayable",
        "type": "function",
    },
    {
        "inputs": [{"name": "proposalId", "type": "bytes32"}],
        "name": "cancel",
        "outputs": [],
        "stateMutability": "nonpayable",
        "type": "function",
    },
    {
        "anonymous": False,
        "inputs": [
            {"indexed": True,  "name": "proposalId", "type": "bytes32"},
            {"indexed": False, "name": "param",      "type": "string"},
            {"indexed": False, "name": "newValue",   "type": "uint256"},
            {"indexed": False, "name": "eta",        "type": "uint256"},
        ],
        "name": "ProposalCreated",
        "type": "event",
    },
]


class GovernanceAgent(BaseAgent):
    """
    Monitors consecutive threat patterns and proposes on-chain
    governance updates via the TimelockController pattern.
    """

    def __init__(
        self,
        run_dir: str,
        hardhat_url: str = "http://127.0.0.1:8545",
        governance_address: Optional[str] = None,
        deployer_key: Optional[str] = None,
    ):
        super().__init__(name="GovernanceAgent")
        self.run_dir            = run_dir
        self.hardhat_url        = hardhat_url
        self.governance_address = governance_address
        self.deployer_key       = deployer_key or os.getenv("HARDHAT_DEPLOYER_KEY", "")

        # Rolling window of (batch_idx, threat_type) for pattern detection
        self._threat_window: deque = deque(maxlen=GOVERNANCE_WINDOW)
        # Proposals already submitted this run (avoid duplicates)
        self._submitted_proposals: set = set()

        self._w3             = None
        self._governance     = None
        self._account        = None
        self._web3_available = False

        self._proposals_path = os.path.join(run_dir, "governance_proposals.json")
        os.makedirs(run_dir, exist_ok=True)
        if not os.path.exists(self._proposals_path):
            with open(self._proposals_path, "w") as f:
                json.dump([], f, indent=2)

        self._init_web3()

    # ═══════════════════════════════════════════════════════════
    # WEB3 INIT (safe)
    # ═══════════════════════════════════════════════════════════
    def _init_web3(self):
        try:
            from web3 import Web3
            w3 = Web3(Web3.HTTPProvider(self.hardhat_url))
            if not w3.is_connected():
                self.logger.warning(
                    f"[{self.name}] Hardhat not reachable -- governance proposals "
                    f"will be logged locally only (simulation mode)."
                )
                return
            self._w3 = w3
            if self.deployer_key:
                self._account = w3.eth.account.from_key(self.deployer_key)
            else:
                accounts = w3.eth.accounts
                if accounts:
                    self._account = type("A", (), {"address": accounts[0]})()
            if self.governance_address and self._account:
                self._governance = w3.eth.contract(
                    address=Web3.to_checksum_address(self.governance_address),
                    abi=GOVERNANCE_ABI,
                )
            self._web3_available = bool(self._account)
            self.logger.info(
                f"[{self.name}] Web3 ready | "
                f"governance={'connected' if self._governance else 'not configured'}"
            )
        except ImportError:
            self.logger.info(
                f"[{self.name}] web3 not installed -- simulation mode. "
                f"Install: pip install web3"
            )
        except Exception as e:
            self.logger.warning(f"[{self.name}] Web3 init failed ({e})")

    # ═══════════════════════════════════════════════════════════
    # PATTERN DETECTION
    # ═══════════════════════════════════════════════════════════
    def _check_consecutive_pattern(self) -> Optional[str]:
        """
        Returns the threat_type if it appears in >=GOVERNANCE_CONSECUTIVE_THRESHOLD
        of the last GOVERNANCE_WINDOW batches. Returns None otherwise.
        """
        if len(self._threat_window) < GOVERNANCE_CONSECUTIVE_THRESHOLD:
            return None
        from collections import Counter
        counts = Counter(t for _, t in self._threat_window if t and t != "unknown")
        if not counts:
            return None
        top_type, top_count = counts.most_common(1)[0]
        if top_count >= GOVERNANCE_CONSECUTIVE_THRESHOLD:
            return top_type
        return None

    # ═══════════════════════════════════════════════════════════
    # PROPOSAL SUBMISSION
    # ═══════════════════════════════════════════════════════════
    def _submit_proposal(
        self,
        param: str,
        old_value: float,
        new_value: float,
        reason: str,
        batch_idx: int,
    ) -> dict:
        """
        Submit proposal on-chain (or simulate). Returns proposal record.
        """
        proposal_id = hashlib.sha256(
            f"{param}:{new_value}:{batch_idx}".encode()
        ).hexdigest()[:12]

        # Convert float threshold to uint256 (scaled by 1e18 for Solidity)
        new_value_uint = int(new_value * 1e18)

        tx_hash     = None
        on_chain    = False
        simulated   = True
        eta_seconds = None

        if self._web3_available and self._governance and self._w3:
            try:
                deployer_addr = self._account.address
                w3 = self._w3
                propose_fn = self._governance.functions.propose(
                    param, new_value_uint, reason
                )
                # Estimate gas with a safe floor of 500K
                # (200K was too low — GovernanceContract string storage costs ~221K)
                try:
                    estimated = propose_fn.estimate_gas({"from": deployer_addr})
                    gas_limit = max(int(estimated * 2), 500_000)
                except Exception:
                    gas_limit = 500_000
                tx = propose_fn.build_transaction({
                    "from":     deployer_addr,
                    "gas":      gas_limit,
                    "gasPrice": w3.to_wei("1", "gwei"),
                    "nonce":    w3.eth.get_transaction_count(deployer_addr),
                    "chainId":  31337,
                })
                if self.deployer_key:
                    signed = w3.eth.account.sign_transaction(tx, self.deployer_key)
                    txh = w3.eth.send_raw_transaction(signed.raw_transaction)
                else:
                    txh = w3.eth.send_transaction(tx)
                receipt = w3.eth.wait_for_transaction_receipt(txh, timeout=30)
                tx_hash   = txh.hex()
                on_chain  = True
                simulated = False
                eta_seconds = GOVERNANCE_TIMELOCK_DELAY_S
                self.logger.info(
                    f"[{self.name}] Governance proposal submitted on-chain | "
                    f"param={param} new={new_value:.4f} | "
                    f"tx={tx_hash[:12]}... | "
                    f"timelock={GOVERNANCE_TIMELOCK_DELAY_S}s"
                )
            except Exception as e:
                self.logger.warning(
                    f"[{self.name}] On-chain proposal failed ({e}) -- simulating."
                )

        if simulated:
            tx_hash     = "0x" + hashlib.sha256(f"sim:{proposal_id}".encode()).hexdigest()[:64]
            eta_seconds = GOVERNANCE_TIMELOCK_DELAY_S
            self.logger.info(
                f"[{self.name}] Governance proposal SIMULATED | "
                f"param={param} {old_value:.4f}→{new_value:.4f} | "
                f"reason={reason}"
            )

        record = {
            "proposal_id":       proposal_id,
            "batch":             batch_idx + 1,
            "timestamp":         datetime.utcnow().isoformat(),
            "param":             param,
            "old_value":         old_value,
            "new_value":         new_value,
            "reason":            reason,
            "tx_hash":           tx_hash,
            "on_chain":          on_chain,
            "simulated":         simulated,
            "timelock_delay_s":  eta_seconds,
            "status":            "PENDING",
        }
        return record

    def _save_proposal(self, record: dict):
        proposals = []
        if os.path.exists(self._proposals_path):
            try:
                with open(self._proposals_path) as f:
                    proposals = json.load(f)
            except Exception:
                proposals = []
        proposals.append(record)
        with open(self._proposals_path, "w") as f:
            json.dump(proposals, f, indent=2, default=str)

    # ═══════════════════════════════════════════════════════════
    # MAIN _run
    # ═══════════════════════════════════════════════════════════
    def _run(self, msg: AgentMessage) -> AgentMessage:
        action_plan  = msg.payload.get("action_plan")
        agent_state  = msg.payload.get("agent_state", {})
        batch_idx    = msg.payload.get("batch_idx", 0)

        proposal_record = None

        # Update rolling window with this batch's threat type
        threat_type = action_plan.get("threat_type", "unknown") if action_plan else "unknown"
        self._threat_window.append((batch_idx, threat_type))

        # Check for consecutive pattern
        pattern = self._check_consecutive_pattern()
        if pattern:
            current_tau = float(agent_state.get("tau_alert", config.INITIAL_THRESHOLD_TAU0))
            proposal_key = f"{pattern}:tau:{batch_idx}"

            if (
                proposal_key not in self._submitted_proposals
                and current_tau > GOVERNANCE_TAU_FLOOR
            ):
                new_tau = max(GOVERNANCE_TAU_FLOOR, current_tau - GOVERNANCE_TAU_STEP)
                reason  = (
                    f"Consecutive {pattern} threats detected in "
                    f"{GOVERNANCE_CONSECUTIVE_THRESHOLD}+ of last "
                    f"{GOVERNANCE_WINDOW} batches. "
                    f"Proposing lower detection threshold to improve recall."
                )
                proposal_record = self._submit_proposal(
                    param="tau_alert",
                    old_value=current_tau,
                    new_value=new_tau,
                    reason=reason,
                    batch_idx=batch_idx,
                )
                self._save_proposal(proposal_record)
                self._submitted_proposals.add(proposal_key)

                self.logger.info(
                    f"[{self.name}] Batch {batch_idx+1} | "
                    f"pattern={pattern} | "
                    f"proposed tau_alert: {current_tau:.4f}→{new_tau:.4f}"
                )
            else:
                self.logger.info(
                    f"[{self.name}] Batch {batch_idx+1} | "
                    f"pattern={pattern} detected but proposal already submitted "
                    f"or tau_alert at floor ({GOVERNANCE_TAU_FLOOR})"
                )
        else:
            self.logger.info(
                f"[{self.name}] Batch {batch_idx+1} | "
                f"no governance trigger | "
                f"window={list(self._threat_window)}"
            )

        return AgentMessage(
            sender=self.name,
            payload={**msg.payload, "governance_proposal": proposal_record},
            status="ok",
        )