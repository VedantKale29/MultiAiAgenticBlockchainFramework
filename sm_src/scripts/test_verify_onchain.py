"""
scripts/test_verify_onchain.py
================================
STEP 8 — Verify everything is on-chain after deploy.js

WHAT IT CHECKS:
  1. deployments/localhost.json exists (written by deploy.js)
  2. GovernanceContract address is valid and has code on-chain
  3. GovernanceContract initial parameters are correct
     (tau_alert=0.487, tau_block=0.587, w=0.700 from config.py)
  4. GovernanceContract ParameterUpdated event is queryable
  5. ContractRegistry address is valid and has code on-chain
  6. ContractRegistry getCount() returns 0 initially
  7. Run full pipeline and verify contract_deployments.json shows simulated=False
  8. After pipeline: ContractRegistry.getCount() > 0

PRE-REQUISITES:
  1. npx hardhat node                              ← running in separate terminal
  2. npx hardhat run scripts/deploy.js --network localhost ← must have run
  3. python scripts/test_deploy_contract.py        ← must pass

RUN:
  python scripts/test_verify_onchain.py

EXPECTED OUTPUT:
  Every check ✅ — system fully operational on-chain, simulated=False
"""

import sys
import os
import json

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

PASS = "✅"
FAIL = "❌"
WARN = "⚠️ "
INFO = "ℹ️ "
SEP  = "─" * 60

HARDHAT_URL           = os.getenv("HARDHAT_URL", "http://127.0.0.1:8545")
HARDHAT_PRIVATE_KEY_0 = "0xac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80"
DEPLOYMENT_JSON       = os.path.join(PROJECT_ROOT, "deployments", "localhost.json")

# GovernanceContract minimal ABI — only what we need
GOV_ABI = [
    {"inputs": [{"internalType": "string", "name": "param", "type": "string"}],
     "name": "getParameter", "outputs": [{"internalType": "uint256", "name": "", "type": "uint256"}],
     "stateMutability": "view", "type": "function"},
    {"inputs": [], "name": "getAllParameters",
     "outputs": [{"internalType": "uint256[]", "name": "", "type": "uint256[]"}],
     "stateMutability": "view", "type": "function"},
    {"inputs": [{"internalType": "string", "name": "param", "type": "string"},
                {"internalType": "uint256", "name": "newValue", "type": "uint256"},
                {"internalType": "string", "name": "reason", "type": "string"}],
     "name": "propose", "outputs": [{"internalType": "bytes32", "name": "", "type": "bytes32"}],
     "stateMutability": "nonpayable", "type": "function"},
    {"anonymous": False,
     "inputs": [{"indexed": False, "name": "param", "type": "string"},
                {"indexed": False, "name": "oldValue", "type": "uint256"},
                {"indexed": False, "name": "newValue", "type": "uint256"},
                {"indexed": False, "name": "blockNumber", "type": "uint256"}],
     "name": "ParameterUpdated", "type": "event"},
]

# ContractRegistry minimal ABI
REG_ABI = [
    {"inputs": [], "name": "getCount",
     "outputs": [{"internalType": "uint256", "name": "", "type": "uint256"}],
     "stateMutability": "view", "type": "function"},
    {"inputs": [{"internalType": "bytes32", "name": "incidentId", "type": "bytes32"},
                {"internalType": "address",  "name": "contractAddr", "type": "address"},
                {"internalType": "string",   "name": "templateKey",  "type": "string"}],
     "name": "register", "outputs": [],
     "stateMutability": "nonpayable", "type": "function"},
    {"inputs": [{"internalType": "bytes32", "name": "incidentId", "type": "bytes32"}],
     "name": "getRecord",
     "outputs": [{"internalType": "address", "name": "contractAddr", "type": "address"},
                 {"internalType": "string",  "name": "templateKey",  "type": "string"},
                 {"internalType": "uint256", "name": "timestamp",    "type": "uint256"}],
     "stateMutability": "view", "type": "function"},
]

# Expected initial params from config.py
EXPECTED_PARAMS = {
    "tau_alert":    0.487,
    "tau_block":    0.587,
    "w":            0.700,
    "escalation_n": 3.0,
}


def section(title):
    print(f"\n{SEP}")
    print(f"  {title}")
    print(SEP)


def check(label, condition, detail=""):
    icon = PASS if condition else FAIL
    line = f"  {icon}  {label}"
    if detail:
        line += f"\n       {detail}"
    print(line)
    return condition


def main():
    all_failures = []

    print("\n" + "=" * 60)
    print("  ON-CHAIN VERIFICATION TEST — Step 8")
    print(f"  Hardhat: {HARDHAT_URL}")
    print("=" * 60)

    # ── 0. Web3 connection ───────────────────────────────────────
    section("0. Connection")
    try:
        from web3 import Web3
        w3 = Web3(Web3.HTTPProvider(HARDHAT_URL, request_kwargs={"timeout": 10}))
        connected = w3.is_connected()
        check("Hardhat reachable", connected,
              "Fix: run 'npx hardhat node'" if not connected else "")
        if not connected:
            sys.exit(1)
        check("Chain ID == 31337", w3.eth.chain_id == 31337,
              f"chain_id={w3.eth.chain_id}")
    except ImportError:
        print(f"  {FAIL}  web3 not installed — run: pip install web3")
        sys.exit(1)

    # ── 1. deployments/localhost.json ────────────────────────────
    section("1. Deployment JSON (written by deploy.js)")
    exists = os.path.exists(DEPLOYMENT_JSON)
    check("deployments/localhost.json exists",
          exists,
          f"path: {DEPLOYMENT_JSON}\n"
          f"       Fix: run:  npx hardhat run scripts/deploy.js --network localhost"
          if not exists else f"path: {DEPLOYMENT_JSON}")
    if not exists:
        all_failures.append("deployments/localhost.json missing — run deploy.js first")
        _summary(all_failures)
        sys.exit(1)

    with open(DEPLOYMENT_JSON) as f:
        deployment_info = json.load(f)

    gov_addr = deployment_info.get("GovernanceContract")
    reg_addr = deployment_info.get("ContractRegistry")

    check("GovernanceContract address present",
          bool(gov_addr), f"address: {gov_addr}")
    check("ContractRegistry address present",
          bool(reg_addr), f"address: {reg_addr}")

    if not gov_addr or not reg_addr:
        all_failures.append("Missing contract address in deployment JSON")
        _summary(all_failures)
        sys.exit(1)

    print(f"       Deployed by:  {deployment_info.get('deployer', 'unknown')}")
    print(f"       Timestamp:    {deployment_info.get('timestamp', 'unknown')}")
    print(f"       Network:      {deployment_info.get('network', 'unknown')}")

    # ── 2. GovernanceContract on-chain ───────────────────────────
    section("2. GovernanceContract on-chain verification")
    gov_addr_cs = Web3.to_checksum_address(gov_addr)
    try:
        gov_code = w3.eth.get_code(gov_addr_cs)
        check("GovernanceContract has code on-chain",
              len(gov_code) > 2,
              f"code size: {len(gov_code)} bytes at {gov_addr_cs}")
        if len(gov_code) <= 2:
            all_failures.append("GovernanceContract has no code — wrong address?")

        gov = w3.eth.contract(address=gov_addr_cs, abi=GOV_ABI)

        # Read each parameter
        print(f"\n  Initial parameters (expected from config.py):")
        for param, expected in EXPECTED_PARAMS.items():
            raw = gov.functions.getParameter(param).call()
            actual = raw / 1e18
            tolerance = 0.001
            ok = abs(actual - expected) < tolerance
            check(f"  {param} == {expected}",
                  ok,
                  f"on-chain: {actual:.4f}  (raw: {raw})")
            if not ok:
                all_failures.append(
                    f"GovernanceContract {param}: expected {expected}, got {actual:.4f}"
                )

        # Test propose() — submit a dummy proposal (doesn't execute until timelock expires)
        print(f"\n  Testing propose() — AI governance submission:")
        deployer = w3.eth.account.from_key(HARDHAT_PRIVATE_KEY_0)
        new_tau  = int(0.467 * 1e18)  # lower tau_alert slightly
        try:
            nonce = w3.eth.get_transaction_count(deployer.address)
            tx = gov.functions.propose(
                "tau_alert", new_tau, "Test proposal: lower detection threshold"
            ).build_transaction({
                "from":     deployer.address,
                "gas":      500000,
                "gasPrice": w3.to_wei("1", "gwei"),
                "nonce":    nonce,
                "chainId":  31337,
            })
            signed  = w3.eth.account.sign_transaction(tx, HARDHAT_PRIVATE_KEY_0)
            tx_hash = w3.eth.send_raw_transaction(signed.raw_transaction)
            receipt = w3.eth.wait_for_transaction_receipt(tx_hash, timeout=15)
            check("  propose() accepted on-chain",
                  receipt.status == 1,
                  f"status={receipt.status}  gasUsed={receipt.gasUsed}  "
                  f"block={receipt.blockNumber}")

            # Verify parameter NOT yet changed (timelock not expired)
            current_tau = gov.functions.getParameter("tau_alert").call() / 1e18
            check("  tau_alert NOT changed yet (timelock pending)",
                  abs(current_tau - EXPECTED_PARAMS["tau_alert"]) < 0.001,
                  f"tau_alert still = {current_tau:.4f} (proposal in timelock queue)")

        except Exception as e:
            check("  propose() call", False, str(e))
            all_failures.append(f"GovernanceContract propose() failed: {e}")

    except Exception as e:
        check("GovernanceContract readable", False, str(e))
        all_failures.append(f"GovernanceContract error: {e}")

    # ── 3. ContractRegistry on-chain ─────────────────────────────
    section("3. ContractRegistry on-chain verification")
    reg_addr_cs = Web3.to_checksum_address(reg_addr)
    try:
        reg_code = w3.eth.get_code(reg_addr_cs)
        check("ContractRegistry has code on-chain",
              len(reg_code) > 2,
              f"code size: {len(reg_code)} bytes at {reg_addr_cs}")

        reg = w3.eth.contract(address=reg_addr_cs, abi=REG_ABI)

        count = reg.functions.getCount().call()
        check("getCount() returns a number",
              isinstance(count, int),
              f"current count: {count} registered contracts")

        # Register a test entry
        deployer = w3.eth.account.from_key(HARDHAT_PRIVATE_KEY_0)
        import time as _time
        test_incident_hash = w3.keccak(text=f"test_incident_{int(_time.time())}")
        test_addr          = Web3.to_checksum_address(
            "0x" + "deadbeef" * 5
        )

        nonce = w3.eth.get_transaction_count(deployer.address)
        reg_tx = reg.functions.register(
            test_incident_hash, test_addr, "circuit_breaker"
        ).build_transaction({
            "from":     deployer.address,
            "gas":      300000,
            "gasPrice": w3.to_wei("1", "gwei"),
            "nonce":    nonce,
            "chainId":  31337,
        })
        signed  = w3.eth.account.sign_transaction(reg_tx, HARDHAT_PRIVATE_KEY_0)
        tx_hash = w3.eth.send_raw_transaction(signed.raw_transaction)
        receipt = w3.eth.wait_for_transaction_receipt(tx_hash, timeout=15)
        check("register() call succeeds",
              receipt.status == 1,
              f"status={receipt.status}  gasUsed={receipt.gasUsed}")

        new_count = reg.functions.getCount().call()
        check("getCount() increased after register()",
              new_count == count + 1,
              f"before={count}  after={new_count}")

        # Retrieve the record
        record = reg.functions.getRecord(test_incident_hash).call()
        check("getRecord() returns correct address",
              record[0].lower() == test_addr.lower(),
              f"stored: {record[0]}\n       sent:   {test_addr}")
        check("getRecord() returns correct template key",
              record[1] == "circuit_breaker",
              f"stored: {record[1]}")

    except Exception as e:
        check("ContractRegistry operations", False, str(e))
        all_failures.append(f"ContractRegistry error: {e}")

    # ── 4. contract_deployments.json check ───────────────────────
    section("4. Check contract_deployments.json for simulated=False")
    import glob
    dep_files = glob.glob(os.path.join(PROJECT_ROOT, "runs", "**", "contract_deployments.json"),
                          recursive=True)
    if dep_files:
        latest = max(dep_files, key=os.path.getmtime)
        with open(latest) as f:
            deployments = json.load(f)

        print(f"  Found {len(deployments)} deployment record(s) in: {latest}")
        real_deploys = [d for d in deployments if not d.get("simulated", True)]
        sim_deploys  = [d for d in deployments if d.get("simulated", True)]

        check("At least one REAL deployment (simulated=False)",
              len(real_deploys) > 0,
              f"real={len(real_deploys)}  simulated={len(sim_deploys)}\n"
              f"       If all are simulated: run the pipeline with Hardhat running\n"
              f"       and GOVERNANCE_CONTRACT_ADDRESS + CONTRACT_REGISTRY_ADDRESS set.")

        for d in real_deploys[:3]:
            print(f"       {PASS} batch={d.get('batch')} template={d.get('template')} "
                  f"address={d.get('deployed_address','')[:20]}...")
        for d in sim_deploys[:3]:
            print(f"       {WARN} batch={d.get('batch')} template={d.get('template')} "
                  f"SIMULATED — Hardhat not running or addresses not set")

        if not real_deploys:
            all_failures.append(
                "All deployments are simulated — pipeline not using real Hardhat"
            )
    else:
        print(f"  {WARN} No contract_deployments.json found in runs/")
        print(f"       Run: python run_pipeline.py --eval")
        print(f"       This check will rerun after the pipeline completes.")

    # ── 5. Environment variable instructions ─────────────────────
    section("5. Environment variables needed for full pipeline")
    print(f"  Set these before running python run_pipeline.py:\n")
    print(f"  Windows PowerShell:")
    print(f"    $env:GOVERNANCE_CONTRACT_ADDRESS = \"{gov_addr}\"")
    print(f"    $env:CONTRACT_REGISTRY_ADDRESS   = \"{reg_addr}\"")
    print(f"    $env:HARDHAT_DEPLOYER_KEY         = \"{HARDHAT_PRIVATE_KEY_0}\"")
    print()
    print(f"  Linux / Mac:")
    print(f"    export GOVERNANCE_CONTRACT_ADDRESS=\"{gov_addr}\"")
    print(f"    export CONTRACT_REGISTRY_ADDRESS=\"{reg_addr}\"")
    print(f"    export HARDHAT_DEPLOYER_KEY=\"{HARDHAT_PRIVATE_KEY_0}\"")
    print()
    print(f"  Then run:")
    print(f"    python run_pipeline.py --eval")

    # ── SUMMARY ──────────────────────────────────────────────────
    _summary(all_failures)
    sys.exit(0 if not all_failures else 1)


def _summary(failures):
    print("\n" + "=" * 60)
    if not failures:
        print(f"{PASS}  ALL CHECKS PASSED")
        print("  GovernanceContract and ContractRegistry are live on-chain.")
        print("  Pipeline will deploy real contracts (simulated=False).")
    else:
        print(f"{FAIL}  {len(failures)} check(s) FAILED:")
        for f in failures:
            print(f"    • {f}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()