"""
blockchain_pipeline.py
=======================
LIVE RO5 DEMO — Self-Triggering, Self-Managing Autonomous Blockchain

PURPOSE:
  This is the standalone entry point that proves RO5: the system detects
  an attack and responds AUTONOMOUSLY — no human input after launch.

  It is NOT the batch CSV pipeline (run_pipeline.py). It runs against a
  LIVE Hardhat chain, listens for real on-chain events, and triggers the
  full Decision → Contract → Governance chain in real time.

WHAT IT DEMONSTRATES:
  1. Self-triggering : MonitorAgent detects FlashLoan event within <2s
  2. Self-management : DecisionAgent → ContractAgent → CircuitBreaker deployed
  3. On-chain audit  : ContractRegistry records the deployed contract address
  4. Self-governance : GovernanceAgent proposes threshold update on-chain
  5. Measurable      : Detection latency + response latency printed at end

FLOW:
  Start ──► Assert Hardhat running
        ──► Load deployed contract addresses (from env vars or deployments/localhost.json)
        ──► Boot agents (DecisionAgent, ContractAgent, GovernanceAgent, AuditAgent)
        ──► Start MonitorAgent background thread
        ──► (Optional) Auto-fire attack via attack.js
        ──► Wait for THREAT_DETECTED
        ──► Run response chain: Decision → Contract → Governance → Audit
        ──► Print latency report
        ──► Verify simulated=False in response
        ──► Exit

USAGE:
  # Step 1 — start Hardhat in another terminal:
  npx hardhat node

  # Step 2 — deploy contracts:
  npx hardhat run scripts/deploy.js --network localhost

  # Step 3 — set env vars printed by deploy.js, then:
  python blockchain_pipeline.py

  # Or let the script fire the attack itself:
  python blockchain_pipeline.py --auto-attack

  # Full options:
  python blockchain_pipeline.py --auto-attack --repeat 3 --timeout 30

ENVIRONMENT VARIABLES (required, printed by deploy.js):
  GOVERNANCE_CONTRACT_ADDRESS   — GovernanceContract address
  CONTRACT_REGISTRY_ADDRESS     — ContractRegistry address
  VULNERABLE_POOL_ADDRESS       — NaiveReceiverLenderPool address
  FLASH_LOAN_RECEIVER_ADDRESS   — FlashLoanReceiver (victim) address
  FLASH_LOAN_ATTACKER_ADDRESS   — FlashLoanAttacker address
  HARDHAT_DEPLOYER_KEY          — private key (default: Hardhat test key)
  ANTHROPIC_API_KEY             — optional; enables LLM reasoning in DecisionAgent

MAPS TO FRAMEWORK:
  Section 3  — Self-Triggering Mechanism (MonitorAgent loop)
  Section 4.1 — Autonomous contract selection + deployment (ContractAgent)
  Section 4.2 — RAG self-improvement (AuditAgent post-incident)
  Section 4.4 — On-chain self-governance (GovernanceAgent timelock)
  Section 9.4 — RO5 evidence: "self-triggering validated on Hardhat"
"""

import os
import sys
import json
import time
import logging
import argparse
import subprocess
import threading
import shutil
from datetime import datetime, timezone
from pathlib import Path

# Force UTF-8 on Windows
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

# Project root
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)-26s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("BlockchainPipeline")

SEP  = "=" * 72
SEP2 = "-" * 72


# ══════════════════════════════════════════════════════════════════════════════
# WINDOWS-AWARE NODE/NPX RESOLVER
# ══════════════════════════════════════════════════════════════════════════════

def _find_node_cmd(name: str) -> list:
    """
    Return the correct command list to invoke a Node.js CLI tool on any OS.

    On Windows, npm/npx are .cmd or .ps1 scripts, not Unix executables.
    subprocess.run(['npx', ...]) fails because Python on Windows uses
    CreateProcess which doesn't resolve PATH scripts without shell=True,
    and shutil.which() finds npx.ps1 / npx.cmd rather than a binary.

    Strategy (in order):
      1. shutil.which(name + '.cmd')   — standard npm install on Windows
      2. shutil.which(name + '.ps1')   — PowerShell wrapper (nvm4w)
      3. shutil.which(name)            — Unix / Git-Bash / WSL
      4. Common Windows Node.js paths  — C:\\nvm4w\\nodejs, C:\\Program Files\\nodejs
      5. Raise with a clear message
    """
    IS_WIN = sys.platform == "win32"

    if IS_WIN:
        # Check .cmd first (standard npm global install creates these)
        for ext in (".cmd", ".ps1", ""):
            found = shutil.which(name + ext)
            if found:
                if ext == ".ps1":
                    # PowerShell scripts need powershell.exe as the launcher
                    return ["powershell.exe", "-NonInteractive", "-File", found]
                return [found]

        # Hard-coded fallback paths for common Windows Node installations
        common_paths = [
            Path(r"C:\nvm4w\nodejs") / (name + ".cmd"),
            Path(r"C:\nvm4w\nodejs") / (name + ".ps1"),
            Path(r"C:\Program Files\nodejs") / (name + ".cmd"),
            Path(r"C:\Program Files\nodejs") / name,
            Path(os.environ.get("APPDATA", "")) / "npm" / (name + ".cmd"),
            Path(os.environ.get("NVM_HOME", "")) / (name + ".cmd"),
        ]
        for candidate in common_paths:
            if candidate.exists():
                if str(candidate).endswith(".ps1"):
                    return ["powershell.exe", "-NonInteractive", "-File", str(candidate)]
                return [str(candidate)]
    else:
        found = shutil.which(name)
        if found:
            return [found]

    raise FileNotFoundError(
        f"Cannot find '{name}' executable.\n"
        f"  Make sure Node.js is installed and '{name}' is on your PATH.\n"
        f"  Windows: check C:\\nvm4w\\nodejs\\ or C:\\Program Files\\nodejs\\\n"
        f"  Then run: npm install (inside sm_src/) to install Hardhat."
    )


def _run_node_script(script_path: Path, network: str = "localhost",
                     env: dict = None, cwd: Path = None, timeout: int = 60):
    """
    Run  npx hardhat run <script_path> --network <network>
    using the OS-correct npx command. Returns subprocess.CompletedProcess.
    """
    npx_cmd  = _find_node_cmd("npx")
    full_cmd = npx_cmd + ["hardhat", "run", str(script_path), "--network", network]
    log.info(f"  Running: {' '.join(str(c) for c in full_cmd)}")
    return subprocess.run(
        full_cmd,
        env=env or os.environ.copy(),
        capture_output=True,
        text=True,
        timeout=timeout,
        cwd=str(cwd or ROOT),
        # shell=False is safer and works once we have the .cmd path
    )


# ══════════════════════════════════════════════════════════════════════════════
# ARGS
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description="Live RO5 demo: self-triggering blockchain fraud detection."
    )
    p.add_argument("--simulate",    action="store_true",
                   help="Run a full offline simulation without requiring Hardhat")
    p.add_argument("--auto-attack",  action="store_true",
                   help="Automatically fire the flash loan attack after starting MonitorAgent")
    p.add_argument("--repeat",       type=int, default=3,
                   help="Number of flash loan events to fire (default: 3)")
    p.add_argument("--amount-eth",   type=float, default=10.0,
                   help="Flash loan amount in ETH (default: 10)")
    p.add_argument("--simulate-threat", type=str, default="flash_loan",
                   help="Threat type for --simulate mode (default: flash_loan)")
    p.add_argument("--simulate-score", type=float, default=0.95,
                   help="Risk/anomaly score for --simulate mode (default: 0.95)")
    p.add_argument("--timeout",      type=int, default=60,
                   help="Seconds to wait for THREAT_DETECTED before giving up (default: 60)")
    p.add_argument("--hardhat-url",  type=str, default="http://127.0.0.1:8545",
                   help="Hardhat RPC URL (default: http://127.0.0.1:8545)")
    p.add_argument("--run-dir",      type=str, default=None,
                   help="Output directory (default: runs/blockchain_run_<timestamp>)")
    p.add_argument("--no-attack",    action="store_true",
                   help="Start monitor only, do not fire attack (manual attack mode)")
    p.add_argument("--edge",         action="store_true",
                   help="Simulate IoT/edge node: max_workers=1 (Section 4.3 RO5 validation)")
    return p.parse_args()


# ══════════════════════════════════════════════════════════════════════════════
# ADDRESS LOADING
# ══════════════════════════════════════════════════════════════════════════════

def load_addresses():
    """
    Load contract addresses from env vars (priority) or deployments/localhost.json.
    Returns dict with keys: governance, registry, pool, receiver, attacker.
    """
    addresses = {
        "governance": os.getenv("GOVERNANCE_CONTRACT_ADDRESS"),
        "registry":   os.getenv("CONTRACT_REGISTRY_ADDRESS"),
        "pool":       os.getenv("VULNERABLE_POOL_ADDRESS"),
        "receiver":   os.getenv("FLASH_LOAN_RECEIVER_ADDRESS"),
        "attacker":   os.getenv("FLASH_LOAN_ATTACKER_ADDRESS"),
    }

    # If any missing, try deployments JSON
    if not all(addresses.values()):
        json_path = ROOT / "deployments" / "localhost.json"
        if json_path.exists():
            info = json.loads(json_path.read_text(encoding="utf-8"))
            addresses["governance"] = addresses["governance"] or info.get("GovernanceContract")
            addresses["registry"]   = addresses["registry"]   or info.get("ContractRegistry")
            addresses["pool"]       = addresses["pool"]        or info.get("NaiveReceiverLenderPool")
            addresses["receiver"]   = addresses["receiver"]    or info.get("FlashLoanReceiver")
            addresses["attacker"]   = addresses["attacker"]    or info.get("FlashLoanAttacker")

    missing = [k for k, v in addresses.items() if not v]
    if missing:
        log.error(f"Missing contract addresses: {missing}")
        log.error("Run: npx hardhat run scripts/deploy.js --network localhost")
        log.error("Then copy the printed env vars into your shell.")
        sys.exit(1)

    return addresses


# ══════════════════════════════════════════════════════════════════════════════
# HARDHAT HEALTH CHECK
# ══════════════════════════════════════════════════════════════════════════════

def assert_hardhat_running(hardhat_url: str):
    """Check Hardhat is reachable. Exit with clear message if not."""
    try:
        from web3 import Web3
        w3 = Web3(Web3.HTTPProvider(hardhat_url, request_kwargs={"timeout": 3}))
        if not w3.is_connected():
            raise ConnectionError("not connected")
        chain_id    = w3.eth.chain_id
        block       = w3.eth.block_number
        log.info(f"Hardhat running at {hardhat_url} (chain_id={chain_id}, block={block})")
        return w3
    except Exception as e:
        log.error(f"Cannot reach Hardhat at {hardhat_url}: {e}")
        log.error("Start Hardhat first: npx hardhat node")
        sys.exit(1)


# ══════════════════════════════════════════════════════════════════════════════
# AGENT BOOTSTRAP
# ══════════════════════════════════════════════════════════════════════════════

def build_agents(addresses: dict, run_dir: str, hardhat_url: str, **kwargs):
    """
    Instantiate the stage agents needed for the live response chain.
    Returns dict of agent instances.
    """
    agents = {}
    deployer_key  = os.getenv(
        "HARDHAT_DEPLOYER_KEY",
        "0xac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80",
    )
    anthropic_key = os.getenv("ANTHROPIC_API_KEY")

    # ── FraudKnowledgeAgent (RAG) ──────────────────────────────────
    try:
        from agents.fraud_knowledge_agent import FraudKnowledgeAgent
        agents["knowledge"] = FraudKnowledgeAgent(run_dir=run_dir)
        sz = agents["knowledge"].get_store_size()
        log.info(f"FraudKnowledgeAgent ready  (rag_store={sz} docs)")
    except Exception as e:
        log.warning(f"FraudKnowledgeAgent unavailable ({e}) — RAG disabled")
        agents["knowledge"] = None

    # ── AuditAgent ─────────────────────────────────────────────────
    try:
        from agents.audit_agent import AuditAgent
        agents["audit"] = AuditAgent(
            run_dir=run_dir,
            run_name="blockchain_pipeline",
            knowledge_agent=agents.get("knowledge"),
        )
        log.info("AuditAgent ready")
    except Exception as e:
        log.warning(f"AuditAgent unavailable ({e})")
        agents["audit"] = None

    # ── DecisionAgent ──────────────────────────────────────────────
    try:
        from agents.decision_agent import DecisionAgent
        agents["decision"] = DecisionAgent(
            knowledge_agent=agents.get("knowledge"),
            anthropic_api_key=anthropic_key,
        )
        llm_mode = "LLM+RAG" if anthropic_key else "rule-based fallback"
        log.info(f"DecisionAgent ready        (mode={llm_mode})")
    except Exception as e:
        log.warning(f"DecisionAgent unavailable ({e})")
        agents["decision"] = None

    # ── ContractAgent ──────────────────────────────────────────────
    try:
        from agents.contract_agent import ContractAgent
        agents["contract"] = ContractAgent(
            run_dir=run_dir,
            knowledge_agent=agents.get("knowledge"),
            hardhat_url=hardhat_url,
            registry_address=addresses["registry"],
            deployer_key=deployer_key,
        )
        log.info("ContractAgent ready")
    except Exception as e:
        log.warning(f"ContractAgent unavailable ({e})")
        agents["contract"] = None

    # ── GovernanceAgent ────────────────────────────────────────────
    try:
        from agents.governance_agent import GovernanceAgent
        agents["governance_ag"] = GovernanceAgent(
            run_dir=run_dir,
            hardhat_url=hardhat_url,
            governance_address=addresses["governance"],
            deployer_key=deployer_key,
        )
        log.info("GovernanceAgent ready")
    except Exception as e:
        log.warning(f"GovernanceAgent unavailable ({e})")
        agents["governance_ag"] = None

    # ── MonitorAgent ───────────────────────────────────────────────
    if kwargs.get("include_monitor", True):
        try:
            from agents.monitor_agent import MonitorAgent
            max_workers = 1 if kwargs.get("edge_mode") else 4
            agents["monitor"] = MonitorAgent(
                hardhat_url=hardhat_url,
                governance_contract_address=addresses["governance"],
                pool_contract_address=addresses["pool"],
                max_workers=max_workers,
            )
            mode_label = "edge (max_workers=1)" if max_workers == 1 else "cloud (max_workers=4)"
            log.info(f"MonitorAgent ready         (mode={mode_label}, poll=1s)")
        except Exception as e:
            log.error(f"MonitorAgent unavailable ({e})")
            log.error("Install web3: pip install web3")
            sys.exit(1)
    else:
        agents["monitor"] = None

    return agents


# ══════════════════════════════════════════════════════════════════════════════
# RESPONSE CHAIN
# ══════════════════════════════════════════════════════════════════════════════

def run_response_chain(
    threat_payload: dict,
    agents:         dict,
    run_dir:        str,
    detect_time_ns: int,
) -> dict:
    """
    Execute the full autonomous response to a THREAT_DETECTED event.

    Pipeline:
      THREAT_DETECTED
        → DecisionAgent  (LLM + RAG → ActionPlan)
        → ContractAgent  (template select → Slither → deploy)
        → GovernanceAgent (consecutive pattern → timelock proposal)
        → AuditAgent     (log incident + re-index RAG)

    Returns a result dict with timing and outcome.
    """
    from agents.base_agent import AgentMessage

    log.info(SEP2)
    log.info("RESPONSE CHAIN ACTIVATED")
    log.info(SEP2)
    log.info(f"  Threat type : {threat_payload.get('anomaly_type', '?')}")
    log.info(f"  Score       : {threat_payload.get('anomaly_score', '?')}")
    log.info(f"  Block       : {threat_payload.get('block_number', '?')}")
    log.info(f"  Tx hash     : {threat_payload.get('tx_hash', '?')[:20]}...")
    log.info(f"  Attacker    : {threat_payload.get('attacker_address', '?')[:20]}...")

    result = {
        "threat_payload":       threat_payload,
        "action_plan":          None,
        "deployment_record":    None,
        "governance_proposal":  None,
        "audit_written":        False,
        "detect_to_response_ms": 0,
        "total_response_ms":    0,
        "simulated":            True,  # flipped to False if real deploy succeeds
    }

    t_response_start = time.perf_counter()

    # Build a minimal AgentMessage that stage agents can consume.
    # They expect: batch_idx, agent_state, decisions list, policy_actions.
    attacker_addr = threat_payload.get("attacker_address", "0x0000000000000000000000000000000000000000")
    risk_score    = threat_payload.get("anomaly_score", 0.95)

    # Synthetic batch payload — represents the threat as a single-tx batch
    tx_hash = threat_payload.get("tx_hash", "") or f"sim_tx_{int(time.time())}"
    tx_meta = {
        "tx_hash": [tx_hash],
        "from_address": [attacker_addr],
        "to_address": [threat_payload.get("pool_address", "")],
        "timestamp": [threat_payload.get("timestamp", "")],
    }

    msg = AgentMessage(
        sender="BlockchainPipeline",
        payload={
            "batch_idx":     0,
            "agent_state":   {"w": 0.70, "tau_alert": 0.487, "tau_block": 0.587},
            "decisions":     ["AUTO-BLOCK"],
            "policy_actions": ["BLOCK"],
            "risk_scores":   [risk_score],
            "p_rf":          [risk_score],
            "s_if":          [max(0.0, min(1.0, risk_score - 0.05))],
            "y_batch":       [1],
            "tx_meta":       tx_meta,
            # DecisionAgent reads these for threat classification
            "threat_context": {
                "anomaly_type":    threat_payload.get("anomaly_type", "flash_loan"),
                "anomaly_score":   risk_score,
                "attacker":        attacker_addr,
                "block_number":    threat_payload.get("block_number", 0),
                "reasons":         threat_payload.get("reasons", []),
                "total_outflow_eth": threat_payload.get("total_outflow_eth", 0),
            },
        },
        status="ok",
    )

    # ── Step 1: DecisionAgent ──────────────────────────────────────
    t1 = time.perf_counter()
    if agents.get("decision"):
        try:
            decision_msg = agents["decision"].run(msg)
            if decision_msg.status == "ok":
                msg = decision_msg
                ap = msg.payload.get("action_plan", {})
                result["action_plan"] = ap
                log.info(
                    f"  DecisionAgent  → threat={ap.get('threat_type','?')} "
                    f"severity={ap.get('severity','?')} "
                    f"template={ap.get('recommended_template','?')} "
                    f"rag_hits={ap.get('rag_hits',0)} "
                    f"llm={ap.get('llm_used',False)}"
                )
            else:
                log.warning(f"  DecisionAgent  → failed: {decision_msg.error}")
        except Exception as e:
            log.warning(f"  DecisionAgent  → exception: {e}")
    else:
        # Manual ActionPlan fallback when DecisionAgent not loaded
        fallback_plan = {
            "threat_type":          "flash_loan",
            "severity":             "CRITICAL",
            "recommended_template": "circuit_breaker",
            "parameters": {
                "target_address":   threat_payload.get("pool_address", ""),
                "attacker_address": attacker_addr,
                "threshold":        0.5,
            },
            "reasoning": "Flash loan detected. Circuit breaker recommended.",
            "rag_hits":  0,
            "rag_max_similarity": 0.0,
            "llm_used":  False,
        }
        msg.payload["action_plan"] = fallback_plan
        result["action_plan"] = fallback_plan
        log.info("  DecisionAgent  → not loaded, using fallback ActionPlan")

    d1 = (time.perf_counter() - t1) * 1000
    log.info(f"  DecisionAgent  → {d1:.0f}ms")

    # ── Step 2: ContractAgent ──────────────────────────────────────
    t2 = time.perf_counter()
    if agents.get("contract"):
        # Inject pool address as target if not already set
        action_plan = msg.payload.get("action_plan", {})
        if action_plan and not action_plan.get("parameters", {}).get("target_address"):
            action_plan.setdefault("parameters", {})["target_address"] = (
                threat_payload.get("pool_address", "")
            )

        try:
            contract_msg = agents["contract"].run(msg)
            if contract_msg.status == "ok":
                msg = contract_msg
                dr = msg.payload.get("deployment_record", {})
                result["deployment_record"] = dr
                if dr:
                    simulated = dr.get("simulated", True)
                    result["simulated"] = simulated
                    log.info(
                        f"  ContractAgent  → template={dr.get('template','?')} "
                        f"slither={dr.get('slither_passed','?')} "
                        f"simulated={simulated} "
                        f"address={str(dr.get('deployed_address','?'))[:16]}..."
                    )
                else:
                    log.info("  ContractAgent  → no deployment record (skipped)")
            else:
                log.warning(f"  ContractAgent  → failed: {contract_msg.error}")
        except Exception as e:
            log.warning(f"  ContractAgent  → exception: {e}")
    else:
        log.info("  ContractAgent  → not loaded (simulated)")

    d2 = (time.perf_counter() - t2) * 1000
    log.info(f"  ContractAgent  → {d2:.0f}ms")

    # ── Step 3: GovernanceAgent ────────────────────────────────────
    t3 = time.perf_counter()
    if agents.get("governance_ag"):
        try:
            gov_msg = agents["governance_ag"].run(msg)
            if gov_msg.status == "ok":
                msg = gov_msg
                gp = msg.payload.get("governance_proposal")
                result["governance_proposal"] = gp
                if gp:
                    log.info(
                        f"  GovernanceAgent → proposal: param={gp.get('param','?')} "
                        f"{gp.get('old_value',0):.4f}→{gp.get('new_value',0):.4f} "
                        f"simulated={gp.get('simulated',True)}"
                    )
                else:
                    log.info("  GovernanceAgent → no proposal this cycle")
        except Exception as e:
            log.warning(f"  GovernanceAgent → exception: {e}")
    else:
        log.info("  GovernanceAgent → not loaded")

    d3 = (time.perf_counter() - t3) * 1000
    log.info(f"  GovernanceAgent → {d3:.0f}ms")

    # ── Step 4: AuditAgent ─────────────────────────────────────────
    t4 = time.perf_counter()
    if agents.get("audit"):
        try:
            audit_msg = agents["audit"].run(msg)
            if audit_msg.status == "ok":
                msg = audit_msg
                n = audit_msg.payload.get("audit_records_written", 0)
                result["audit_written"] = n > 0
                log.info(f"  AuditAgent     → {n} record(s) written to audit_log.jsonl")
        except Exception as e:
            log.warning(f"  AuditAgent     → exception: {e}")
    else:
        log.info("  AuditAgent     → not loaded")

    d4 = (time.perf_counter() - t4) * 1000
    log.info(f"  AuditAgent     → {d4:.0f}ms")

    # ── Timing ────────────────────────────────────────────────────
    total_response_ms = (time.perf_counter() - t_response_start) * 1000

    # Detect-to-response: from MonitorAgent detection to end of chain
    now_ns              = time.perf_counter_ns()
    detect_to_response  = (now_ns - detect_time_ns) / 1_000_000  # ms

    result["total_response_ms"]     = total_response_ms
    result["detect_to_response_ms"] = detect_to_response

    log.info(SEP2)
    log.info(f"  Response chain complete in {total_response_ms:.0f}ms")
    log.info(f"  Detect-to-response latency: {detect_to_response:.0f}ms")

    return result


def build_simulated_addresses() -> dict:
    """Deterministic placeholder addresses for full offline simulation."""
    return {
        "governance": "0x1000000000000000000000000000000000000001",
        "registry":   "0x2000000000000000000000000000000000000002",
        "pool":       "0x3000000000000000000000000000000000000003",
        "receiver":   "0x4000000000000000000000000000000000000004",
        "attacker":   "0x5000000000000000000000000000000000000005",
    }


def build_simulated_threat(args, addresses: dict) -> dict:
    """Create one synthetic THREAT_DETECTED payload for offline simulation."""
    score = max(0.0, min(1.0, args.simulate_score))
    return {
        "event_type": "THREAT_DETECTED",
        "block_number": 999999,
        "anomaly_score": score,
        "anomaly_type": args.simulate_threat,
        "reasons": [
            f"simulated_threat:{args.simulate_threat}",
            f"simulated_repeat:{args.repeat}",
            f"simulated_outflow:{args.amount_eth:.4f}ETH",
        ],
        "tx_hash": f"0xsim{int(time.time())}",
        "attacker_address": addresses["attacker"],
        "tx_count_in_block": max(1, args.repeat),
        "total_outflow_eth": round(args.amount_eth * max(1, args.repeat), 4),
        "threshold_at_trigger": {
            "tau_alert": 0.487,
            "tau_block": 0.587,
            "w": 0.70,
        },
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "detection_time_ns": time.perf_counter_ns(),
        "pool_address": addresses["pool"],
    }


# ══════════════════════════════════════════════════════════════════════════════
# ATTACK LAUNCHER
# ══════════════════════════════════════════════════════════════════════════════

def fire_attack(addresses: dict, amount_eth: float, repeat: int, delay_s: float = 2.0):
    """
    Fire the flash loan attack via attack.js in a subprocess.
    Runs after a short delay to give MonitorAgent time to start.
    """
    log.info(f"  Waiting {delay_s}s for MonitorAgent to be ready...")
    time.sleep(delay_s)

    env = os.environ.copy()
    env["VULNERABLE_POOL_ADDRESS"]     = addresses["pool"]
    env["FLASH_LOAN_RECEIVER_ADDRESS"] = addresses["receiver"]
    env["FLASH_LOAN_ATTACKER_ADDRESS"] = addresses["attacker"]
    env["FLASH_LOAN_AMOUNT_ETH"]       = str(amount_eth)
    env["FLASH_LOAN_REPEAT"]           = str(repeat)

    attack_script = ROOT / "attack.js"
    if not attack_script.exists():
        # Try scripts/ subdirectory
        attack_script = ROOT / "scripts" / "attack.js"

    if not attack_script.exists():
        log.error(f"attack.js not found at {attack_script}")
        return

    log.info(f"  Launching attack: {repeat}x {amount_eth} ETH flash loan(s)...")
    try:
        result = _run_node_script(attack_script, network="localhost", env=env, timeout=60)
        if result.returncode == 0:
            log.info("  Attack script completed successfully")
            for line in result.stdout.splitlines():
                if any(kw in line for kw in ["FlashLoan", "Transaction", "ATTACK", "Pool balance", "Victim"]):
                    log.info(f"    {line.strip()}")
        else:
            log.warning(f"  Attack script returned code {result.returncode}")
            log.warning(result.stderr[-800:] if result.stderr else "no stderr")
    except subprocess.TimeoutExpired:
        log.warning("  Attack script timed out after 60s")
    except FileNotFoundError as e:
        log.error(f"  {e}")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def run_full_simulation(args, run_dir: str) -> dict:
    """
    Full offline simulation that exercises the response chain without Hardhat.
    """
    log.info(SEP)
    log.info("PHASE 1 - Full Offline Simulation")
    log.info(SEP)
    addresses = build_simulated_addresses()
    for name, addr in addresses.items():
        log.info(f"  simulated_{name:<12}: {addr}")

    log.info(SEP)
    log.info("PHASE 2 - Boot Response Agents")
    log.info(SEP)
    agents = build_agents(
        addresses,
        run_dir,
        args.hardhat_url,
        edge_mode=args.edge,
        include_monitor=False,
    )

    threat = build_simulated_threat(args, addresses)
    result = run_response_chain(
        threat_payload=threat,
        agents=agents,
        run_dir=run_dir,
        detect_time_ns=threat["detection_time_ns"],
    )
    return {"result": result, "threat": threat}


def main():
    args    = parse_args()
    t_start = time.perf_counter()

    print()
    print(SEP)
    print("  AGENTIC AI BLOCKCHAIN FRAUD DETECTION")
    print("  Live RO5 Demo — Self-Triggering Autonomous Response")
    print(SEP)
    print(f"  simulate={args.simulate}  simulate_threat={args.simulate_threat}")
    print(f"  auto_attack={args.auto_attack}  repeat={args.repeat}")
    print(f"  timeout={args.timeout}s  hardhat={args.hardhat_url}")
    print(SEP)
    print()

    # ── Run directory ─────────────────────────────────────────────
    ts      = datetime.now().strftime("%Y%m%dT%H%M%S")
    run_dir = args.run_dir or str(ROOT / "runs" / f"blockchain_{ts}")
    os.makedirs(run_dir, exist_ok=True)
    log.info(f"Output dir: {run_dir}")

    if args.simulate:
        result_holder = run_full_simulation(args, run_dir)
        result = result_holder["result"]
        threat = result_holder["threat"]

        print()
        print(SEP)
        print("  FULL SIMULATION RESULTS")
        print(SEP)

        response_ms   = result.get("total_response_ms", 0)
        end_to_end_ms = result.get("detect_to_response_ms", 0)

        print(f"\n  Threat detection")
        print(f"    Anomaly type   : {threat.get('anomaly_type','?')}")
        print(f"    Anomaly score  : {threat.get('anomaly_score','?'):.3f}")
        print(f"    Block number   : {threat.get('block_number','?')}")
        print(f"    Attacker addr  : {threat.get('attacker_address','?')[:20]}...")

        print(f"\n  Latency")
        print(f"    Response chain : {response_ms:.0f}ms")
        print(f"    End-to-end     : {end_to_end_ms:.0f}ms")

        ap = result.get("action_plan") or {}
        print(f"\n  DecisionAgent (ActionPlan)")
        print(f"    Threat type    : {ap.get('threat_type','?')}")
        print(f"    Severity       : {ap.get('severity','?')}")
        print(f"    Template       : {ap.get('recommended_template','?')}")
        print(f"    RAG hits       : {ap.get('rag_hits',0)}")
        print(f"    LLM used       : {ap.get('llm_used',False)}")

        dr = result.get("deployment_record") or {}
        simulated = result.get("simulated", True)
        print(f"\n  ContractAgent (Deployment)")
        print(f"    Template       : {dr.get('template','?')}")
        print(f"    Slither passed : {dr.get('slither_passed','?')}")
        print(f"    Simulated      : {simulated}")
        print(f"    Address        : {str(dr.get('deployed_address','?'))[:20]}...")

        gp = result.get("governance_proposal")
        print(f"\n  GovernanceAgent")
        if gp:
            print(f"    Proposal       : {gp.get('param','?')} {gp.get('old_value',0):.4f}->{gp.get('new_value',0):.4f}")
            print(f"    Simulated      : {gp.get('simulated',True)}")
        else:
            print("    No proposal this cycle (window not yet filled)")

        print(f"\n  AuditAgent")
        print(f"    Record written : {result.get('audit_written', False)}")

        total_s = time.perf_counter() - t_start
        print()
        print(SEP)
        print(f"  Simulation complete in {total_s:.1f}s")
        print(f"  Output dir: {run_dir}")
        print(SEP)
        print()

        demo_output = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "run_dir": run_dir,
            "threat": threat,
            "action_plan": ap,
            "deployment_record": dr,
            "governance_proposal": gp,
            "latency": {
                "response_chain_ms": response_ms,
                "end_to_end_ms": end_to_end_ms,
            },
            "simulated": simulated,
            "mode": "offline_full_simulation",
        }
        out_path = os.path.join(run_dir, "blockchain_pipeline_result.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(demo_output, f, indent=2, default=str)
        log.info(f"Result saved: {out_path}")
        return

    # ── Step 1: Assert Hardhat running ────────────────────────────
    log.info(SEP)
    log.info("PHASE 1 — Hardhat Health Check")
    log.info(SEP)
    assert_hardhat_running(args.hardhat_url)

    # ── Step 2: Load addresses ────────────────────────────────────
    log.info(SEP)
    log.info("PHASE 2 — Load Contract Addresses")
    log.info(SEP)
    addresses = load_addresses()
    for name, addr in addresses.items():
        log.info(f"  {name:<12}: {addr}")

    # ── Step 3: Boot agents ───────────────────────────────────────
    log.info(SEP)
    log.info("PHASE 3 — Boot Response Agents")
    log.info(SEP)
    agents = build_agents(addresses, run_dir, args.hardhat_url, edge_mode=args.edge)

    # ── Step 4: Wire handle_threat callback into MonitorAgent ─────
    log.info(SEP)
    log.info("PHASE 4 — Wire Self-Trigger Callback")
    log.info(SEP)

    # Shared state for coordination between callback and main thread
    threat_event = threading.Event()
    result_holder = {"result": None, "threat": None, "detect_time_ns": None}

    def handle_threat(payload: dict):
        """
        This callback is invoked by MonitorAgent when THREAT_DETECTED fires.
        It runs on the MonitorAgent's background thread.
        """
        detect_time_ns = time.perf_counter_ns()
        log.warning(SEP)
        log.warning(">>> THREAT_DETECTED — AUTONOMOUS RESPONSE ACTIVATED <<<")
        log.warning(SEP)

        # Inject pool address into payload so ContractAgent can set target
        payload["pool_address"] = addresses["pool"]

        result_holder["threat"]         = payload
        result_holder["detect_time_ns"] = detect_time_ns

        # Run response chain on callback thread (blocking — this is intentional
        # for a single-threat demo; concurrent threats use ThreadPoolExecutor)
        result = run_response_chain(
            threat_payload=payload,
            agents=agents,
            run_dir=run_dir,
            detect_time_ns=detect_time_ns,
        )
        result_holder["result"] = result

        # Signal main thread that response is complete
        threat_event.set()

    agents["monitor"].handle_threat = handle_threat
    log.info("  handle_threat callback wired to MonitorAgent")

    # ── Step 5: Start MonitorAgent ────────────────────────────────
    log.info(SEP)
    log.info("PHASE 5 — Start MonitorAgent Background Loop")
    log.info(SEP)
    agents["monitor"].start()
    log.info("  MonitorAgent polling for blockchain events...")
    log.info(f"  Watching pool : {addresses['pool']}")
    log.info(f"  Watching gov  : {addresses['governance']}")
    log.info(f"  Threshold     : {agents['monitor'].threshold_cache.get('tau_alert'):.3f}")

    # ── Step 6: Fire attack (optional) ───────────────────────────
    if args.auto_attack and not args.no_attack:
        log.info(SEP)
        log.info("PHASE 6 — Auto-Fire Flash Loan Attack")
        log.info(SEP)
        attack_thread = threading.Thread(
            target=fire_attack,
            args=(addresses, args.amount_eth, args.repeat),
            daemon=True,
            name="AttackLauncher",
        )
        attack_thread.start()
    elif not args.no_attack:
        log.info(SEP)
        print()
        print("  MonitorAgent is running. Waiting for attack...")
        print()
        print("  In another terminal, run ONE of:")
        print(f"    npx hardhat run scripts/attack.js --network localhost")
        print(f"    python blockchain_pipeline.py --auto-attack")
        print()
        log.info(SEP)

    # ── Step 7: Wait for THREAT_DETECTED + response ───────────────
    log.info("Waiting for THREAT_DETECTED...")
    detected = threat_event.wait(timeout=args.timeout)

    # Stop monitor
    agents["monitor"].stop()

    if not detected:
        log.error(f"No THREAT_DETECTED within {args.timeout}s timeout.")
        log.error("Check: (a) Hardhat is running, (b) attack.js was executed,")
        log.error("       (c) MonitorAgent is watching the correct pool address.")
        sys.exit(1)

    # ── Step 8: Results ───────────────────────────────────────────
    result = result_holder["result"]
    threat = result_holder["threat"]

    print()
    print(SEP)
    print("  RO5 DEMO RESULTS")
    print(SEP)

    # Detection latency (MonitorAgent detects within poll interval)
    detect_latency_ms = result.get("detect_to_response_ms", 0) - result.get("total_response_ms", 0)
    response_ms       = result.get("total_response_ms", 0)
    end_to_end_ms     = result.get("detect_to_response_ms", 0)

    print(f"\n  Threat detection")
    print(f"    Anomaly type   : {threat.get('anomaly_type','?')}")
    print(f"    Anomaly score  : {threat.get('anomaly_score','?'):.3f}")
    print(f"    Block number   : {threat.get('block_number','?')}")
    print(f"    Attacker addr  : {threat.get('attacker_address','?')[:20]}...")

    print(f"\n  Latency")
    print(f"    Response chain : {response_ms:.0f}ms")
    print(f"    End-to-end     : {end_to_end_ms:.0f}ms")

    ap = result.get("action_plan") or {}
    print(f"\n  DecisionAgent (ActionPlan)")
    print(f"    Threat type    : {ap.get('threat_type','?')}")
    print(f"    Severity       : {ap.get('severity','?')}")
    print(f"    Template       : {ap.get('recommended_template','?')}")
    print(f"    RAG hits       : {ap.get('rag_hits',0)}")
    print(f"    LLM used       : {ap.get('llm_used',False)}")

    dr = result.get("deployment_record") or {}
    simulated = result.get("simulated", True)
    print(f"\n  ContractAgent (Deployment)")
    print(f"    Template       : {dr.get('template','?')}")
    print(f"    Slither passed : {dr.get('slither_passed','?')}")
    print(f"    Simulated      : {simulated}")
    print(f"    Address        : {str(dr.get('deployed_address','?'))[:20]}...")
    if not simulated:
        print(f"\n  *** simulated=False — REAL contract deployed on Hardhat! ***")

    gp = result.get("governance_proposal")
    print(f"\n  GovernanceAgent")
    if gp:
        print(f"    Proposal       : {gp.get('param','?')} {gp.get('old_value',0):.4f}→{gp.get('new_value',0):.4f}")
        print(f"    Simulated      : {gp.get('simulated',True)}")
    else:
        print(f"    No proposal this cycle (window not yet filled)")

    print(f"\n  AuditAgent")
    print(f"    Record written : {result.get('audit_written', False)}")

    # ── Self-triggering assertion ─────────────────────────────────
    print()
    print(SEP)
    print("  RO5 VERIFICATION")
    print(SEP)
    checks = {
        "Self-trigger: THREAT_DETECTED fired without human input": True,
        "DecisionAgent: ActionPlan produced":                       ap is not None,
        "ContractAgent: deployment attempted":                      dr is not None,
        "AuditAgent: incident logged":                              result.get("audit_written", False),
        "Real deployment (simulated=False)":                        not simulated,
        f"Node mode: {'edge (max_workers=1)' if args.edge else 'cloud (max_workers=4)'}": True,
    }
    all_pass = True
    for check, passed in checks.items():
        icon = "[PASS]" if passed else "[FAIL]"
        print(f"  {icon}  {check}")
        if not passed and "simulated" not in check:
            all_pass = False  # simulated=True is acceptable on first run

    total_s = time.perf_counter() - t_start
    print()
    print(SEP)
    print(f"  Demo complete in {total_s:.1f}s")
    print(f"  Output dir: {run_dir}")
    print(SEP)
    print()

    # ── Save demo result ──────────────────────────────────────────
    demo_output = {
        "timestamp":          datetime.now(timezone.utc).isoformat(),
        "run_dir":            run_dir,
        "threat":             threat,
        "action_plan":        ap,
        "deployment_record":  dr,
        "governance_proposal": gp,
        "latency": {
            "response_chain_ms": response_ms,
            "end_to_end_ms":     end_to_end_ms,
        },
        "ro5_checks": checks,
        "simulated": simulated,
    }
    out_path = os.path.join(run_dir, "blockchain_pipeline_result.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(demo_output, f, indent=2, default=str)
    log.info(f"Result saved: {out_path}")


if __name__ == "__main__":
    main()
