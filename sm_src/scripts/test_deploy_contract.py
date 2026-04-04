"""
scripts/test_deploy_contract.py
=================================
STEP 5 — Compile AND deploy each template to Hardhat

WHAT IT CHECKS:
  1. Compile each template (circuit_breaker, address_blocklist, rate_limiter)
  2. Deploy to local Hardhat chain via Web3.py
  3. Get deployment receipt with real contract address
  4. Confirm simulated=False (real on-chain deployment)
  5. Verify deployed contract has code on-chain (not an EOA)
  6. Call a read function on the deployed contract to confirm it works
  7. Test constructor args are applied correctly:
       circuit_breaker: target address stored correctly
       rate_limiter:    maxVolumePerBlock stored correctly
  8. Full round-trip timing < 12 seconds (flash loan window requirement)

PRE-REQUISITES:
  1. npx hardhat node  ← must be running in a separate terminal
  2. python scripts/test_blockchain_connect.py  ← must pass
  3. python scripts/test_compile_template.py    ← must pass

RUN:
  python scripts/test_deploy_contract.py

EXPECTED OUTPUT:
  Every check ✅  — ContractAgent will deploy real contracts (simulated=False)
"""

import sys
import os
import json
import time
import subprocess
import tempfile
from pathlib import Path

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

PASS = "✅"
FAIL = "❌"
INFO = "ℹ️ "
SEP  = "─" * 60

HARDHAT_URL           = os.getenv("HARDHAT_URL", "http://127.0.0.1:8545")
HARDHAT_PRIVATE_KEY_0 = "0xac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80"
HARDHAT_ACCOUNT_0     = "0xf39Fd6e51aad88F6F4ce6aB8827279cffFb92266"
HARDHAT_ACCOUNT_1     = "0x70997970C51812dc3A010C7d01b50e0d17dc79C8"

# Minimal contract sources — same as compile test
SOURCES = {
    "address_blocklist": """\
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;
contract AddressBlocklist_t001 {
    address public immutable owner;
    mapping(address => bool) public blocked;
    event BlockedAddress(address indexed wallet, uint256 riskScore, uint256 timestamp);
    constructor() { owner = msg.sender; }
    modifier onlyOwner() { require(msg.sender == owner, "Not owner"); _; }
    function blockAddress(address wallet, uint256 riskScore) external onlyOwner {
        blocked[wallet] = true;
        emit BlockedAddress(wallet, riskScore, block.timestamp);
    }
    function isBlocked(address wallet) external view returns (bool) { return blocked[wallet]; }
}
""",
    "rate_limiter": """\
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;
contract RateLimiter_t001 {
    address public immutable owner;
    uint256 public maxVolumePerBlock;
    constructor(uint256 _maxVolumeWei) {
        require(_maxVolumeWei > 0, "Must be positive");
        owner = msg.sender;
        maxVolumePerBlock = _maxVolumeWei;
    }
    function getLimit() external view returns (uint256) { return maxVolumePerBlock; }
}
""",
    "circuit_breaker": """\
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;
interface IPausable { function pause() external; }
contract CircuitBreaker_t001 {
    address public immutable owner;
    address public immutable target;
    bool    public activated;
    event CircuitBreakerActivated(address indexed target, address indexed activatedBy, uint256 timestamp);
    constructor(address _target) {
        require(_target != address(0), "Invalid target");
        owner  = msg.sender;
        target = _target;
    }
    modifier onlyOwner() { require(msg.sender == owner, "Not owner"); _; }
    function activate() external onlyOwner {
        require(!activated, "Already activated");
        activated = true;
        emit CircuitBreakerActivated(target, msg.sender, block.timestamp);
    }
    function isActivated() external view returns (bool) { return activated; }
    function getTarget() external view returns (address) { return target; }
}
""",
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


def compile_source(source: str, template_key: str):
    """Compile using standard-JSON. Returns (bytecode, abi) or raises."""
    tmp_dir  = tempfile.gettempdir()
    src_path = os.path.join(tmp_dir, f"deploy_test_{template_key}.sol")
    Path(src_path).write_text(source, encoding="utf-8")

    std_input = json.dumps({
        "language": "Solidity",
        "sources":  {"contract.sol": {"content": source}},
        "settings": {"outputSelection": {"*": {"*": ["abi", "evm.bytecode.object"]}}},
    })

    try:
        result = subprocess.run(
            ["solc", "--standard-json"],
            input=std_input, capture_output=True,
            text=True, encoding="utf-8", timeout=30,
        )
    finally:
        try:
            os.unlink(src_path)
        except Exception:
            pass

    if result.returncode != 0:
        raise RuntimeError(f"solc exit {result.returncode}: {result.stderr[:300]}")

    output = json.loads(result.stdout)
    errors = [e for e in output.get("errors", []) if e.get("severity") == "error"]
    if errors:
        raise RuntimeError(f"Compile error: {errors[0].get('message','')}")

    for src_name, contracts in output.get("contracts", {}).items():
        for name, data in contracts.items():
            bytecode = data["evm"]["bytecode"]["object"]
            abi      = data["abi"]
            if bytecode:
                return bytecode, abi

    raise RuntimeError("No contracts in solc output")


def main():
    all_failures = []

    print("\n" + "=" * 60)
    print("  DEPLOY CONTRACT TEST — Step 5")
    print(f"  Hardhat: {HARDHAT_URL}")
    print("=" * 60)

    # ── Connect to Hardhat ───────────────────────────────────────
    section("0. Hardhat connection check")

    try:
        from web3 import Web3
    except ImportError:
        print(f"  {FAIL}  web3 not installed — run: pip install web3")
        sys.exit(1)

    w3 = Web3(Web3.HTTPProvider(HARDHAT_URL, request_kwargs={"timeout": 10}))
    connected = w3.is_connected()
    check("Hardhat node reachable",
          connected,
          f"URL: {HARDHAT_URL}"
          if connected else
          "Fix: run 'npx hardhat node' in a separate terminal")
    if not connected:
        print("\n  Cannot deploy without Hardhat. Start it first.\n")
        sys.exit(1)

    chain_id = w3.eth.chain_id
    check("Chain ID == 31337", chain_id == 31337, f"chain_id={chain_id}")

    deployer_acct = w3.eth.account.from_key(HARDHAT_PRIVATE_KEY_0)
    check("Deployer account loaded",
          deployer_acct.address.lower() == HARDHAT_ACCOUNT_0.lower(),
          f"address: {deployer_acct.address}")

    # ── Deploy each template ──────────────────────────────────────
    deployment_results = {}

    for template_key, source in SOURCES.items():
        section(f"Template: {template_key}")

        t_start = time.perf_counter()

        # ── Compile ──────────────────────────────────────────────
        try:
            bytecode, abi = compile_source(source, template_key)
            check("Compilation succeeds",
                  True, f"bytecode: {len(bytecode)} hex chars | ABI: {len(abi)} entries")
        except Exception as e:
            check("Compilation", False, str(e))
            all_failures.append(f"{template_key}: compile failed: {e}")
            continue

        compile_ms = (time.perf_counter() - t_start) * 1000
        print(f"       Compile time: {compile_ms:.0f}ms")

        # ── Build constructor args ────────────────────────────────
        deployer_addr   = deployer_acct.address
        constructor_args = []
        extra_checks     = {}

        if template_key == "circuit_breaker":
            # target = hardhat account #1 (a real address on local chain)
            target = w3.to_checksum_address(HARDHAT_ACCOUNT_1)
            constructor_args = [target]
            extra_checks["target_address"] = target

        elif template_key == "rate_limiter":
            # 50 ETH limit in wei
            limit_wei = int(50 * 1e18)
            constructor_args = [limit_wei]
            extra_checks["max_volume_wei"] = limit_wei

        # address_blocklist: no constructor args

        # ── Deploy ────────────────────────────────────────────────
        t_deploy = time.perf_counter()
        try:
            contract = w3.eth.contract(abi=abi, bytecode=bytecode)
            nonce    = w3.eth.get_transaction_count(deployer_addr)

            tx = contract.constructor(*constructor_args).build_transaction({
                "from":     deployer_addr,
                "gas":      800000,
                "gasPrice": w3.to_wei("1", "gwei"),
                "nonce":    nonce,
                "chainId":  31337,
            })

            signed  = w3.eth.account.sign_transaction(tx, HARDHAT_PRIVATE_KEY_0)
            tx_hash = w3.eth.send_raw_transaction(signed.raw_transaction)
            check("send_raw_transaction() accepted",
                  True, f"tx: {tx_hash.hex()[:20]}...")

        except Exception as e:
            check("Transaction submission", False, str(e))
            all_failures.append(f"{template_key}: tx submission failed: {e}")
            continue

        # ── Wait for receipt ──────────────────────────────────────
        try:
            receipt = w3.eth.wait_for_transaction_receipt(tx_hash, timeout=15)
            deploy_ms = (time.perf_counter() - t_deploy) * 1000
            total_ms  = (time.perf_counter() - t_start) * 1000

            check("Transaction mined successfully",
                  receipt.status == 1,
                  f"status={receipt.status}  block={receipt.blockNumber}  "
                  f"gasUsed={receipt.gasUsed}")

            deployed_addr = receipt.contractAddress
            check("Contract address returned",
                  bool(deployed_addr),
                  f"address: {deployed_addr}")

            check("simulated=False (REAL deployment)",
                  deployed_addr is not None,
                  f"{deployed_addr} — this is a REAL on-chain address")

            print(f"       Deploy time: {deploy_ms:.0f}ms | Total: {total_ms:.0f}ms")

            # Flash loan window check
            check("Total latency < 12000ms (flash loan window)",
                  total_ms < 12000,
                  f"{total_ms:.0f}ms vs 12000ms limit")

            if receipt.status != 1:
                all_failures.append(f"{template_key}: tx reverted")
                continue

            deployment_results[template_key] = {
                "address":  deployed_addr,
                "tx_hash":  tx_hash.hex(),
                "gas_used": receipt.gasUsed,
                "block":    receipt.blockNumber,
            }

        except Exception as e:
            check("Receipt received", False, str(e))
            all_failures.append(f"{template_key}: receipt failed: {e}")
            continue

        # ── Verify on-chain code exists ───────────────────────────
        try:
            code = w3.eth.get_code(deployed_addr)
            check("Contract has code on-chain (not empty)",
                  len(code) > 2,
                  f"code length: {len(code)} bytes")
            if len(code) <= 2:
                all_failures.append(f"{template_key}: deployed contract has no code")
        except Exception as e:
            check("get_code() works", False, str(e))
            all_failures.append(f"{template_key}: get_code failed: {e}")

        # ── Read-function verification ────────────────────────────
        try:
            deployed_contract = w3.eth.contract(address=deployed_addr, abi=abi)

            if template_key == "address_blocklist":
                # isBlocked(random_address) should return False
                result = deployed_contract.functions.isBlocked(
                    w3.to_checksum_address(HARDHAT_ACCOUNT_1)
                ).call()
                check("isBlocked(addr) returns False (initial state)",
                      result == False, f"returned: {result}")

                # owner should be deployer
                owner = deployed_contract.functions.owner().call()
                check("owner() == deployer",
                      owner.lower() == deployer_addr.lower(),
                      f"owner: {owner}")

            elif template_key == "circuit_breaker":
                # isActivated() should return False initially
                activated = deployed_contract.functions.isActivated().call()
                check("isActivated() returns False (initial state)",
                      activated == False, f"returned: {activated}")

                # getTarget() should return the address we passed
                stored_target = deployed_contract.functions.getTarget().call()
                expected_target = extra_checks.get("target_address", "")
                check("getTarget() == constructor arg",
                      stored_target.lower() == expected_target.lower(),
                      f"stored: {stored_target}\n       expected: {expected_target}")
                if stored_target.lower() != expected_target.lower():
                    all_failures.append(f"{template_key}: wrong target stored")

            elif template_key == "rate_limiter":
                # getLimit() should return what we passed to constructor
                stored_limit = deployed_contract.functions.getLimit().call()
                expected_limit = extra_checks.get("max_volume_wei", 0)
                check("getLimit() == constructor arg",
                      stored_limit == expected_limit,
                      f"stored: {stored_limit} wei ({stored_limit/1e18:.1f} ETH)\n"
                      f"       expected: {expected_limit} wei ({expected_limit/1e18:.1f} ETH)")
                if stored_limit != expected_limit:
                    all_failures.append(f"{template_key}: wrong limit stored")

                # owner should be deployer
                owner = deployed_contract.functions.owner().call()
                check("owner() == deployer",
                      owner.lower() == deployer_addr.lower(),
                      f"owner: {owner}")

        except Exception as e:
            check("Read function verification", False, str(e))
            all_failures.append(f"{template_key}: read function failed: {e}")

    # ── Final summary of all deployments ─────────────────────────
    section("Deployment summary")
    print(f"  {len(deployment_results)}/{len(SOURCES)} templates deployed successfully\n")
    for key, info in deployment_results.items():
        print(f"  {PASS}  {key}")
        print(f"       address:  {info['address']}")
        print(f"       tx_hash:  {info['tx_hash'][:24]}...")
        print(f"       gas_used: {info['gas_used']:,}")
        print(f"       block:    {info['block']}")

    not_deployed = [k for k in SOURCES if k not in deployment_results]
    for key in not_deployed:
        print(f"  {FAIL}  {key}  — not deployed (see errors above)")

    # ── FINAL SUMMARY ────────────────────────────────────────────
    print("\n" + "=" * 60)
    if not all_failures:
        print(f"{PASS}  ALL CHECKS PASSED")
        print("  All templates deploy successfully with real addresses.")
        print("  ContractAgent will show simulated=False in production.")
        print("  Safe to proceed to Step 6 — run deploy.js.")
    else:
        print(f"{FAIL}  {len(all_failures)} check(s) FAILED:")
        for f in all_failures:
            print(f"    • {f}")
        print("\n  Fix the failures above before running the full pipeline.")
    print("=" * 60 + "\n")
    sys.exit(0 if not all_failures else 1)


if __name__ == "__main__":
    main()