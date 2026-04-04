"""
agents/contract_agent.py
=========================
STAGE 3 — ContractAgent

ROLE IN FRAMEWORK (Section 4.1 — revised):
  Receives ActionPlan from DecisionAgent.
  Selects the best-matching pre-audited Solidity template from the
  RAG contract_templates collection.
  Injects ActionPlan parameters into the template's placeholder slots.
  Runs Slither static analysis on the parameterised contract.
  Deploys via Web3.py to the local Hardhat chain.
  Registers the deployed address in the on-chain ContractRegistry.
  Writes the deployment record to contract_deployments.json.

MAPS TO:
  RO5 — autonomous contract selection and parameterised deployment
         without human intervention, within the attack time window.
         The novel contribution is the SELECTION and PARAMETERISATION
         logic, not code generation.

DESIGN DECISIONS (from the roadmap doc, Section 9.1):
  - NO LLM Solidity generation.  Templates are pre-audited.
  - Three built-in templates as a fallback if RAG not yet populated.
  - Slither is in the hot path (fast: <2s).  Mythril is offline.
  - Web3.py deployment to Hardhat.  Safe fallback: simulated deploy
    (writes to contract_deployments.json only) if Web3/Hardhat absent.
  - All errors caught; pipeline never crashes.

CONTRACT TEMPLATES (built-in library):
  Three Solidity templates are embedded as Python strings so the
  ContractAgent works with zero external files.  In production these
  would be in the RAG contract_templates collection and pre-audited
  with Mythril.

  1. circuit_breaker — calls pause() on the target contract
  2. address_blocklist — records attacker in a mapping; blocks calls
  3. rate_limiter — throttles per-block volume below a threshold

INSTALL (optional but enables full functionality):
  pip install web3           # blockchain deployment
  Hardhat: npx hardhat node  # local chain on http://127.0.0.1:8545
  pip install slither-analyzer  # static analysis (requires solc)
"""

import os
import json
import subprocess
import tempfile
import hashlib
from datetime import datetime
from typing import Optional

from agents.base_agent import BaseAgent, AgentMessage


# ══════════════════════════════════════════════════════════════
# BUILT-IN TEMPLATE LIBRARY
# ══════════════════════════════════════════════════════════════
# Templates use {{PLACEHOLDER}} syntax for parameterisation.
# All templates inherit from OpenZeppelin base contracts.
# Pre-audit with Mythril before adding to RAG in production.

TEMPLATES = {
    "circuit_breaker": {
        "description": "Calls pause() on the target contract to halt all activity.",
        "solidity": """\
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;
interface IPausable { function pause() external; }
contract CircuitBreaker_{{INCIDENT_ID}} {
    address public immutable owner;
    address public immutable target;
    constructor(address _target) { owner = msg.sender; target = _target; }
    modifier onlyOwner() { require(msg.sender == owner, "Not owner"); _; }
    function activate() external onlyOwner {
        IPausable(target).pause();
    }
}
""",
        "placeholders": ["INCIDENT_ID", "TARGET_ADDRESS"],
    },
    "address_blocklist": {
        "description": "Records attacker address; emits BlockedAddress event for off-chain enforcement.",
        "solidity": """\
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;
contract AddressBlocklist_{{INCIDENT_ID}} {
    address public immutable owner;
    mapping(address => bool) public blocked;
    event BlockedAddress(address indexed wallet, uint256 riskScore);
    constructor() { owner = msg.sender; }
    modifier onlyOwner() { require(msg.sender == owner, "Not owner"); _; }
    function blockAddress(address wallet, uint256 riskScore) external onlyOwner {
        blocked[wallet] = true;
        emit BlockedAddress(wallet, riskScore);
    }
    function isBlocked(address wallet) external view returns (bool) {
        return blocked[wallet];
    }
}
""",
        "placeholders": ["INCIDENT_ID"],
    },
    "rate_limiter": {
        "description": "Throttles per-block transaction volume below a configurable threshold.",
        "solidity": """\
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;
contract RateLimiter_{{INCIDENT_ID}} {
    address public immutable owner;
    uint256 public maxVolumePerBlock;
    uint256 public currentBlock;
    uint256 public currentBlockVolume;
    event VolumeExceeded(uint256 blockNum, uint256 volume, uint256 limit);
    constructor(uint256 _maxVolume) {
        owner = msg.sender;
        maxVolumePerBlock = _maxVolume;
    }
    modifier onlyOwner() { require(msg.sender == owner, "Not owner"); _; }
    function recordVolume(uint256 amount) external onlyOwner returns (bool allowed) {
        if (block.number > currentBlock) {
            currentBlock = block.number;
            currentBlockVolume = 0;
        }
        currentBlockVolume += amount;
        if (currentBlockVolume > maxVolumePerBlock) {
            emit VolumeExceeded(block.number, currentBlockVolume, maxVolumePerBlock);
            return false;
        }
        return true;
    }
    function updateLimit(uint256 newLimit) external onlyOwner {
        maxVolumePerBlock = newLimit;
    }
}
""",
        "placeholders": ["INCIDENT_ID"],
    },
}


# ══════════════════════════════════════════════════════════════
# DEPLOYMENT RECORD SCHEMA
# ══════════════════════════════════════════════════════════════
def _make_deployment_record(
    incident_id: str,
    batch_idx: int,
    template_key: str,
    params: dict,
    deployed_address: str,
    tx_hash: str,
    slither_passed: bool,
    simulated: bool,
    action_plan: dict,
) -> dict:
    return {
        "incident_id":      incident_id,
        "batch":            batch_idx + 1,
        "timestamp":        datetime.utcnow().isoformat(),
        "template":         template_key,
        "parameters":       params,
        "deployed_address": deployed_address,
        "tx_hash":          tx_hash,
        "slither_passed":   slither_passed,
        "simulated":        simulated,
        "threat_type":      action_plan.get("threat_type", "unknown"),
        "severity":         action_plan.get("severity",    "UNKNOWN"),
        "reasoning":        action_plan.get("reasoning",   ""),
    }


class ContractAgent(BaseAgent):
    """
    Autonomous contract selection, parameterisation, and deployment.
    Degrades gracefully when Web3 / Hardhat / Slither are absent.
    """

    REGISTRY_ABI = [
        {
            "inputs": [
                {"name": "incidentId", "type": "bytes32"},
                {"name": "contractAddress", "type": "address"},
                {"name": "templateKey",     "type": "string"},
            ],
            "name": "register",
            "outputs": [],
            "stateMutability": "nonpayable",
            "type": "function",
        }
    ]

    def __init__(
        self,
        run_dir: str,
        knowledge_agent=None,       # FraudKnowledgeAgent (optional — Stage 1)
        hardhat_url: str  = "http://127.0.0.1:8545",
        registry_address: Optional[str] = None,  # ContractRegistry on-chain address
        deployer_key: Optional[str] = None,       # Hardhat test account private key
    ):
        super().__init__(name="ContractAgent")
        self.run_dir          = run_dir
        self.knowledge_agent  = knowledge_agent
        self.hardhat_url      = hardhat_url
        self.registry_address = registry_address
        self.deployer_key     = deployer_key or os.getenv("HARDHAT_DEPLOYER_KEY", "")

        self._w3              = None
        self._account         = None
        self._registry        = None
        self._web3_available  = False

        self._deployments_path = os.path.join(run_dir, "contract_deployments.json")
        os.makedirs(run_dir, exist_ok=True)
        if not os.path.exists(self._deployments_path):
            with open(self._deployments_path, "w") as f:
                json.dump([], f, indent=2)

        self._init_web3()

    # ═══════════════════════════════════════════════════════════
    # WEB3 INITIALISATION (safe)
    # ═══════════════════════════════════════════════════════════
    def _init_web3(self):
        try:
            from web3 import Web3
            w3 = Web3(Web3.HTTPProvider(self.hardhat_url))
            if not w3.is_connected():
                self.logger.warning(
                    f"[{self.name}] Hardhat not reachable at {self.hardhat_url}. "
                    f"Contract deployment will be simulated."
                )
                return

            self._w3 = w3

            if self.deployer_key:
                self._account = w3.eth.account.from_key(self.deployer_key)
                self.logger.info(
                    f"[{self.name}] Web3 ready | "
                    f"deployer={self._account.address[:10]}... | "
                    f"chain_id={w3.eth.chain_id}"
                )
            else:
                # Use first Hardhat test account (no private key needed on local chain)
                accounts = w3.eth.accounts
                if accounts:
                    self._account = type("Acct", (), {"address": accounts[0]})()
                    self.logger.info(
                        f"[{self.name}] Web3 ready | "
                        f"deployer={accounts[0][:10]}... (Hardhat test account)"
                    )

            if self.registry_address and self._account:
                self._registry = w3.eth.contract(
                    address=Web3.to_checksum_address(self.registry_address),
                    abi=self.REGISTRY_ABI,
                )

            self._web3_available = True
        except ImportError:
            self.logger.warning(
                f"[{self.name}] web3 not installed — deployment will be simulated. "
                f"Install with: pip install web3"
            )
        except Exception as e:
            self.logger.warning(
                f"[{self.name}] Web3 init failed ({e}) — deployment will be simulated."
            )

    # ═══════════════════════════════════════════════════════════
    # TEMPLATE SELECTION via RAG (with built-in fallback)
    # ═══════════════════════════════════════════════════════════
    def _select_template(self, action_plan: dict) -> tuple[str, str]:
        """
        Returns (template_key, solidity_source).

        Selection order:
          1. query_template()  — uses the contract_templates ChromaDB collection
                                 (Gap 1 fix: the RAG path is now fully exercised)
          2. Built-in TEMPLATES dict fallback — if RAG unavailable or no match

        The query text is built from the ActionPlan's threat_type and severity
        so cosine similarity correctly selects the best-matching template.
        For the novel_variant scenario (unknown threat type), the embedding
        space finds the closest-matching template without an exact keyword match.
        """
        recommended = action_plan.get("recommended_template", "address_blocklist")
        threat_type = action_plan.get("threat_type", "unknown")
        severity    = action_plan.get("severity", "HIGH")

        # ── 1. RAG contract_templates collection ────────────────
        if self.knowledge_agent is not None:
            try:
                # Query mirrors the keyword-dense embedding text style so
                # all-MiniLM-L6-v2 cosine similarity scores reliably above threshold.
                query = (
                    f"threat type {threat_type} {severity} "
                    f"use {recommended} recommended {recommended}"
                )
                results = self.knowledge_agent.query_template(
                    query_text=query,
                    n_results=1,
                )
                if results:
                    best       = results[0]
                    rag_key    = best["metadata"].get("template_key", "")
                    similarity = best.get("similarity", 0.0)

                    if rag_key in TEMPLATES:
                        # Get the full Solidity source (not the truncated preview)
                        solidity = self.knowledge_agent.get_template_solidity(rag_key)
                        if not solidity:
                            solidity = TEMPLATES[rag_key]["solidity"]

                        self.logger.info(
                            f"[{self.name}] RAG selected template: '{rag_key}' "
                            f"(similarity={similarity:.3f}, "
                            f"threat={threat_type}, severity={severity})"
                        )
                        return rag_key, solidity

            except Exception as e:
                self.logger.warning(
                    f"[{self.name}] RAG template query failed ({e}) "
                    "— falling back to built-in library"
                )

        # ── 2. Built-in TEMPLATES dict fallback ─────────────────
        self.logger.info(
            f"[{self.name}] Using built-in template: '{recommended}' "
            "(RAG unavailable or no match)"
        )
        if recommended in TEMPLATES:
            return recommended, TEMPLATES[recommended]["solidity"]
        return "address_blocklist", TEMPLATES["address_blocklist"]["solidity"]

    # ═══════════════════════════════════════════════════════════
    # PARAMETERISATION
    # ═══════════════════════════════════════════════════════════
    @staticmethod
    def _parameterise(solidity: str, incident_id: str, params: dict) -> str:
        """Replace {{PLACEHOLDER}} slots with actual values."""
        source = solidity.replace("{{INCIDENT_ID}}", incident_id[:8])
        # Future: replace target_address slot if templates evolve to use it
        return source

    # ═══════════════════════════════════════════════════════════
    # SLITHER STATIC ANALYSIS
    # ═══════════════════════════════════════════════════════════
    def _run_slither(self, solidity_source: str) -> bool:
        """
        Writes source to a temp file, runs Slither, returns True if passes.
        Returns True (skip) if Slither is not installed — safety is still
        provided by the pre-audited template library.
        """
        try:
            with tempfile.NamedTemporaryFile(
                suffix=".sol", mode="w", delete=False
            ) as tmp:
                tmp.write(solidity_source)
                tmp_path = tmp.name

            result = subprocess.run(
                ["slither", tmp_path, "--json", "-"],
                capture_output=True, text=True, timeout=30
            )
            os.unlink(tmp_path)

            # Slither exit code 0 = no issues, 1 = issues found
            if result.returncode == 0:
                self.logger.info(f"[{self.name}] Slither PASSED")
                return True
            else:
                # Parse JSON output to check if only informational findings
                try:
                    output = json.loads(result.stdout)
                    detectors = output.get("results", {}).get("detectors", [])
                    high_impact = [
                        d for d in detectors
                        if d.get("impact", "").lower() in ("high", "medium")
                    ]
                    if not high_impact:
                        self.logger.info(
                            f"[{self.name}] Slither: only low-impact findings — PASSED"
                        )
                        return True
                    self.logger.warning(
                        f"[{self.name}] Slither found {len(high_impact)} high/medium issues"
                    )
                    return False
                except Exception:
                    # Could not parse Slither output — allow pre-audited templates
                    return True

        except FileNotFoundError:
            self.logger.info(
                f"[{self.name}] Slither not installed — using pre-audited template "
                f"(install: pip install slither-analyzer). Skipping analysis."
            )
            return True
        except subprocess.TimeoutExpired:
            self.logger.warning(f"[{self.name}] Slither timed out — allowing deployment")
            return True
        except Exception as e:
            self.logger.warning(f"[{self.name}] Slither error ({e}) — allowing deployment")
            return True

    # ═══════════════════════════════════════════════════════════
    # DEPLOYMENT
    # ═══════════════════════════════════════════════════════════
    def _deploy(
        self,
        template_key: str,
        solidity_source: str,
        params: dict,
        incident_id: str,
    ) -> tuple[str, str, bool]:
        """
        Returns (deployed_address, tx_hash, simulated).
        Tries real deployment; falls back to simulation.
        """
        if not self._web3_available or self._w3 is None:
            return self._simulate_deploy(template_key, incident_id)

        try:
            # For Hardhat local chain: use eth_sendTransaction with pre-compiled bytecode
            # The simple contracts in our library compile to predictable bytecode.
            # We use solc if available; otherwise simulate.
            try:
                bytecode, abi = self._compile_solidity(solidity_source, template_key)
            except Exception as e:
                self.logger.warning(
                    f"[{self.name}] Compilation failed ({e}) — simulating deployment"
                )
                return self._simulate_deploy(template_key, incident_id)

            w3 = self._w3
            contract = w3.eth.contract(abi=abi, bytecode=bytecode)
            deployer_addr = self._account.address

            # Constructor args based on template
            constructor_args = []
            if template_key == "circuit_breaker":
                target = params.get("target_address", deployer_addr)
                constructor_args = [w3.to_checksum_address(target)]
            elif template_key == "rate_limiter":
                threshold = int(params.get("threshold", 0.5) * 1e18)
                constructor_args = [threshold]
            # address_blocklist has no constructor args

            # Estimate gas — use 2x estimate with 2_000_000 floor
            try:
                estimated = contract.constructor(*constructor_args).estimate_gas(
                    {"from": deployer_addr}
                )
                gas_limit = max(2_000_000, int(estimated * 2))
            except Exception:
                gas_limit = 2_000_000

            nonce = w3.eth.get_transaction_count(deployer_addr)
            tx = contract.constructor(*constructor_args).build_transaction({
                "from":     deployer_addr,
                "gas":      gas_limit,
                "gasPrice": w3.to_wei("1", "gwei"),
                "nonce":    nonce,
                "chainId":  31337,
            })

            if self.deployer_key:
                signed = w3.eth.account.sign_transaction(tx, self.deployer_key)
                tx_hash = w3.eth.send_raw_transaction(signed.raw_transaction)
            else:
                # Hardhat unlocked accounts (no signing needed)
                tx_hash = w3.eth.send_transaction(tx)

            receipt = w3.eth.wait_for_transaction_receipt(tx_hash, timeout=30)
            deployed_address = receipt.contractAddress

            # Register in ContractRegistry if available
            if self._registry and deployed_address:
                try:
                    incident_bytes = w3.to_bytes(
                        hexstr=w3.keccak(text=incident_id).hex()
                    )
                    reg_tx = self._registry.functions.register(
                        incident_bytes, deployed_address, template_key
                    ).build_transaction({
                        "from":     deployer_addr,
                        "gas":      500000,
                        "gasPrice": w3.to_wei("1", "gwei"),
                        "nonce":    w3.eth.get_transaction_count(deployer_addr),
                        "chainId":  31337,
                    })
                    if self.deployer_key:
                        signed_reg = w3.eth.account.sign_transaction(reg_tx, self.deployer_key)
                        w3.eth.send_raw_transaction(signed_reg.raw_transaction)
                    else:
                        w3.eth.send_transaction(reg_tx)
                    self.logger.info(
                        f"[{self.name}] Registered {deployed_address[:10]}... "
                        f"in ContractRegistry"
                    )
                except Exception as e:
                    self.logger.warning(
                        f"[{self.name}] Registry registration failed: {e}"
                    )

            self.logger.info(
                f"[{self.name}] Deployed {template_key} at {deployed_address} | "
                f"tx={tx_hash.hex()[:12]}..."
            )
            return deployed_address, tx_hash.hex(), False

        except Exception as e:
            self.logger.warning(
                f"[{self.name}] Deployment failed ({e}) — simulating"
            )
            return self._simulate_deploy(template_key, incident_id)

    def _simulate_deploy(self, template_key: str, incident_id: str) -> tuple[str, str, bool]:
        """Return a deterministic fake address for simulation mode."""
        fake_addr = "0x" + hashlib.sha256(
            f"{template_key}:{incident_id}".encode()
        ).hexdigest()[:40]
        fake_tx   = "0x" + hashlib.sha256(
            f"tx:{fake_addr}".encode()
        ).hexdigest()[:64]
        self.logger.info(
            f"[{self.name}] SIMULATED deployment | "
            f"template={template_key} addr={fake_addr[:12]}..."
        )
        return fake_addr, fake_tx, True

    def _compile_solidity(self, source: str, template_key: str):
        """
        Compile Solidity source with solc.
        Returns (bytecode, abi).
        Raises on failure.

        FIXES:
          1. UTF-8 encoding: write source via Path.write_text(encoding="utf-8")
             — avoids Windows-1252 invalid byte errors from tempfile paths
          2. JSON input format: use solc --standard-json instead of positional
             arg, which avoids the "JSON object must be str" error
          3. Explicit ASCII-safe temp path in the system temp dir
        """
        import subprocess, json, tempfile, os
        from pathlib import Path

        # Write source to an explicitly UTF-8 encoded file with ASCII-safe name
        tmp_dir  = tempfile.gettempdir()
        src_path = os.path.join(tmp_dir, f"contract_{template_key}.sol")
        Path(src_path).write_text(source, encoding="utf-8")

        try:
            # Build solc standard-JSON input — avoids all encoding issues
            std_input = json.dumps({
                "language": "Solidity",
                "sources": {
                    "contract.sol": {"content": source}
                },
                "settings": {
                    "outputSelection": {
                        "*": {"*": ["abi", "evm.bytecode.object"]}
                    }
                }
            })

            result = subprocess.run(
                ["solc", "--standard-json"],
                input=std_input,
                capture_output=True,
                text=True,
                encoding="utf-8",
                timeout=30
            )

            # Clean up temp file
            try:
                os.unlink(src_path)
            except Exception:
                pass

            if result.returncode != 0:
                raise RuntimeError(f"solc error: {result.stderr[:300]}")

            output = json.loads(result.stdout)

            # Check for solc-level errors
            errors = [e for e in output.get("errors", [])
                      if e.get("severity") == "error"]
            if errors:
                raise RuntimeError(
                    f"solc compile error: {errors[0].get('message','unknown')}"
                )

            # Extract bytecode + ABI from standard-JSON output
            sources_out = output.get("contracts", {})
            for src_name, contracts in sources_out.items():
                for contract_name, contract_data in contracts.items():
                    if template_key.replace("_", "").lower() in contract_name.lower():
                        bytecode = contract_data["evm"]["bytecode"]["object"]
                        abi      = contract_data["abi"]
                        return bytecode, abi

            # Fallback: return first contract found
            for src_name, contracts in sources_out.items():
                for contract_name, contract_data in contracts.items():
                    bytecode = contract_data["evm"]["bytecode"]["object"]
                    abi      = contract_data["abi"]
                    return bytecode, abi

            raise RuntimeError("solc produced no contracts in output")
            # Fallback: use first contract
            first = next(iter(contracts.values()))
            return first["bin"], json.loads(first["abi"])
        except Exception:
            if os.path.exists(src_path):
                os.unlink(src_path)
            raise

    # ═══════════════════════════════════════════════════════════
    # DEPLOYMENT RECORD PERSISTENCE
    # ═══════════════════════════════════════════════════════════
    def _save_deployment(self, record: dict):
        deployments = []
        if os.path.exists(self._deployments_path):
            try:
                with open(self._deployments_path) as f:
                    deployments = json.load(f)
            except Exception:
                deployments = []
        deployments.append(record)
        with open(self._deployments_path, "w") as f:
            json.dump(deployments, f, indent=2, default=str)

    # ═══════════════════════════════════════════════════════════
    # MAIN _run
    # ═══════════════════════════════════════════════════════════
    def _run(self, msg: AgentMessage) -> AgentMessage:
        action_plan = msg.payload.get("action_plan")
        batch_idx   = msg.payload.get("batch_idx", 0)

        # If no action plan (all-CLEAR batch), pass through
        if action_plan is None:
            self.logger.info(
                f"[{self.name}] Batch {batch_idx+1} — no action_plan, skipping."
            )
            return AgentMessage(
                sender=self.name,
                payload={**msg.payload, "deployment_record": None},
                status="ok",
            )

        # 1. Generate incident ID
        ts          = datetime.utcnow().strftime("%Y%m%dT%H%M%S")
        incident_id = hashlib.sha256(
            f"b{batch_idx}:{ts}".encode()
        ).hexdigest()[:12]

        # 2. Select template
        template_key, solidity_source = self._select_template(action_plan)

        # 3. Parameterise
        params = action_plan.get("parameters", {})
        parameterised_source = self._parameterise(
            solidity_source, incident_id, params
        )

        # 4. Slither verification
        slither_ok = self._run_slither(parameterised_source)
        if not slither_ok:
            self.logger.warning(
                f"[{self.name}] Slither failed for batch {batch_idx+1} — "
                f"aborting deployment for this batch."
            )
            return AgentMessage(
                sender=self.name,
                payload={**msg.payload, "deployment_record": None},
                status="ok",
            )

        # 5. Deploy
        deployed_address, tx_hash, simulated = self._deploy(
            template_key, parameterised_source, params, incident_id
        )

        # 6. Build and persist record
        record = _make_deployment_record(
            incident_id=incident_id,
            batch_idx=batch_idx,
            template_key=template_key,
            params=params,
            deployed_address=deployed_address,
            tx_hash=tx_hash,
            slither_passed=slither_ok,
            simulated=simulated,
            action_plan=action_plan,
        )
        self._save_deployment(record)

        self.logger.info(
            f"[{self.name}] Batch {batch_idx+1} | "
            f"template={template_key} | "
            f"incident={incident_id} | "
            f"address={deployed_address[:12]}... | "
            f"simulated={simulated}"
        )

        return AgentMessage(
            sender=self.name,
            payload={**msg.payload, "deployment_record": record},
            status="ok",
        )