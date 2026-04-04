"""
scripts/debug_simulated.py
===========================
Run this from sm_src/ to diagnose exactly why simulated=True persists.
Prints the actual env vars seen by Python and tests ContractAgent init.

RUN:
    python scripts/debug_simulated.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

print("\n" + "="*60)
print("  SIMULATED=TRUE DIAGNOSTIC")
print("="*60)

# ── 1. Env vars ──────────────────────────────────────────
print("\n[1] Environment variables seen by Python right now:")
gov  = os.getenv("GOVERNANCE_CONTRACT_ADDRESS", "NOT SET")
reg  = os.getenv("CONTRACT_REGISTRY_ADDRESS",   "NOT SET")
key  = os.getenv("HARDHAT_DEPLOYER_KEY",        "NOT SET")
url  = os.getenv("HARDHAT_URL", "http://127.0.0.1:8545")

print(f"  GOVERNANCE_CONTRACT_ADDRESS = {gov}")
print(f"  CONTRACT_REGISTRY_ADDRESS   = {reg}")
print(f"  HARDHAT_DEPLOYER_KEY        = {key[:20]}..." if key != "NOT SET" else f"  HARDHAT_DEPLOYER_KEY        = NOT SET")
print(f"  HARDHAT_URL                 = {url}")

if "NOT SET" in [gov, reg, key]:
    print("\n   PROBLEM: env vars not set in this Python session.")
    print("  Fix: set them in the SAME PowerShell window before running pipeline.")
    print("  $env:GOVERNANCE_CONTRACT_ADDRESS = '0x...'")
    sys.exit(1)
print("   All 3 env vars present")

# ── 2. run_pipeline.py has the fix ───────────────────────
print("\n[2] Checking run_pipeline.py has the registry_address fix:")
rp_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "run_pipeline.py")
if os.path.exists(rp_path):
    with open(rp_path, "r", encoding="utf-8", errors="ignore") as f:
        content = f.read()
    has_fix = 'registry_address' in content and 'CONTRACT_REGISTRY_ADDRESS' in content
    print(f"  File: {rp_path}")
    print(f"  Has registry_address fix: {'YES' if has_fix else 'NO — copy fixed run_pipeline.py'}")
    if not has_fix:
        print("\n  PROBLEM: run_pipeline.py is the OLD version.")
        sys.exit(1)
else:
    print(f"  run_pipeline.py not found at {rp_path}")
    sys.exit(1)

# ── 3. Web3 connects ─────────────────────────────────────
print("\n[3] Testing Web3 connection:")
try:
    from web3 import Web3
    w3 = Web3(Web3.HTTPProvider(url, request_kwargs={"timeout": 5}))
    connected = w3.is_connected()
    print(f"  Connected: {' YES' if connected else ' NO — start npx hardhat node'}")
    if not connected:
        sys.exit(1)
    print(f"  Chain ID: {w3.eth.chain_id}")
    gov_code = w3.eth.get_code(Web3.to_checksum_address(gov))
    reg_code = w3.eth.get_code(Web3.to_checksum_address(reg))
    print(f"  GovernanceContract code: {' ' + str(len(gov_code)) + ' bytes' if len(gov_code) > 2 else ' 0 bytes — run deploy.js first'}")
    print(f"  ContractRegistry code:   {' ' + str(len(reg_code)) + ' bytes' if len(reg_code) > 2 else ' 0 bytes — run deploy.js first'}")
    if len(gov_code) <= 2 or len(reg_code) <= 2:
        print("\n  PROBLEM: Contracts not on chain. Run:")
        print("  npx hardhat run scripts/deploy.js --network localhost")
        sys.exit(1)
except Exception as e:
    print(f"   Web3 error: {e}")
    sys.exit(1)

# ── 4. ContractAgent init with env vars ──────────────────
print("\n[4] Testing ContractAgent initialises with real Web3:")
try:
    import tempfile
    tmp = tempfile.mkdtemp()
    from agents.contract_agent import ContractAgent
    agent = ContractAgent(
        run_dir           = tmp,
        hardhat_url       = url,
        registry_address  = reg,
        deployer_key      = key,
    )
    print(f"  _web3_available : {' True' if agent._web3_available else ' False'}")
    print(f"  _account        : {' ' + agent._account.address[:16] + '...' if agent._account else ' None'}")
    print(f"  _registry       : {' ContractRegistry wired' if agent._registry else ' None — wrong registry address?'}")
    if not agent._web3_available:
        print("\n  PROBLEM: ContractAgent._web3_available=False even with env vars.")
        print("  This means Web3 init failed. Check hardhat node and deployer key.")
        sys.exit(1)
except Exception as e:
    print(f"   ContractAgent init error: {e}")
    sys.exit(1)

# ── 5. Test one real deployment ───────────────────────────
print("\n[5] Testing one real contract deployment:")
try:
    from agents.base_agent import AgentMessage
    test_payload = {
        "batch_idx": 99,
        "action_plan": {
            "threat_type": "phishing",
            "severity": "HIGH",
            "recommended_template": "address_blocklist",
            "parameters": {"target_address": "0xf39Fd6e51aad88F6F4ce6aB8827279cffFb92266", "threshold": 0.85, "attacker_address": "0xf39Fd6e51aad88F6F4ce6aB8827279cffFb92266"},
            "reasoning": "diagnostic test",
            "rag_hits": 0,
            "rag_max_similarity": 0.0,
            "llm_used": False,
        },
        "decisions": ["AUTO-BLOCK"] * 3,
        "risk_scores": [0.92, 0.88, 0.91],
        "p_rf": [0.90, 0.85, 0.89],
        "s_if": [0.87, 0.82, 0.86],
        "policy_actions": ["BLOCK", "BLOCK", "BLOCK"],
        "tx_meta": {"from_address": ["0xabc"] * 3, "to_address": ["0xdef"] * 3},
        "agent_state": {"w": 0.70, "tau_alert": 0.487, "tau_block": 0.587},
    }
    msg = AgentMessage(sender="Diagnostic", payload=test_payload)
    result = agent.run(msg)
    dr = result.payload.get("deployment_record")
    if dr:
        simulated = dr.get("simulated", True)
        addr = dr.get("deployed_address", "")
        print(f"  simulated       : {' True — compilation or tx failed' if simulated else ' False — REAL deployment!'}")
        print(f"  deployed_address: {addr}")
        if simulated:
            print("\n  PROBLEM: Deployment fell back to simulation.")
            print("  This means compilation failed. Check solc is on PATH:")
            print("  solc --version")
        else:
            print("\n   REAL DEPLOYMENT WORKS. Run the pipeline now.")
    else:
        print("   No deployment record returned")
except Exception as e:
    print(f"   Deployment test error: {e}")
    import traceback; traceback.print_exc()

print("\n" + "="*60 + "\n")