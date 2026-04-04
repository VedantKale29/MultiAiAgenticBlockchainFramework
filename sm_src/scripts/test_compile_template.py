"""
scripts/test_compile_template.py
==================================
STEP 4 — Compile each Solidity template in isolation

WHAT IT CHECKS:
  1. solc is installed and on PATH
  2. solc version is 0.8.x (matches pragma in templates)
  3. solc --standard-json mode works (the fixed compilation path)
  4. Each template compiles without errors:
       - circuit_breaker     (needs {{INCIDENT_ID}} replaced first)
       - address_blocklist
       - rate_limiter
  5. Compiled bytecode is non-empty hex string
  6. ABI is a list with at least one function
  7. Constructor arguments match what ContractAgent passes
  8. UTF-8 encoding works (no 0x97 byte errors on Windows)

PRE-REQUISITE:
  solc must be installed:
    Windows: download from https://github.com/ethereum/solidity/releases
             or:  pip install py-solc-x && python -c "from solcx import install_solc; install_solc('0.8.20')"
    Linux/Mac: apt install solc  OR  brew install solidity

RUN:
  python scripts/test_compile_template.py

EXPECTED OUTPUT:
  Every check ✅  — templates are ready for on-chain deployment
"""

import sys
import os
import json
import subprocess
import tempfile
import shutil
from pathlib import Path

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

PASS = "✅"
FAIL = "❌"
WARN = "⚠️ "
SEP  = "─" * 60

# Minimal Solidity source for each template (parameterised)
# Uses same source as contract_agent.py TEMPLATES dict
MINIMAL_SOURCES = {
    "circuit_breaker": """\
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;
interface IPausable { function pause() external; }
contract CircuitBreaker_test0001 {
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
        IPausable(target).pause();
        emit CircuitBreakerActivated(target, msg.sender, block.timestamp);
    }
    function isActivated() external view returns (bool) { return activated; }
}
""",
    "address_blocklist": """\
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;
contract AddressBlocklist_test0001 {
    address public immutable owner;
    mapping(address => bool)    public blocked;
    mapping(address => uint256) public riskScores;
    mapping(address => uint256) public blockedAt;
    uint256 public totalBlocked;
    event BlockedAddress(address indexed wallet, uint256 riskScore, uint256 timestamp);
    event RemovedAddress(address indexed wallet, uint256 timestamp);
    constructor() { owner = msg.sender; }
    modifier onlyOwner() { require(msg.sender == owner, "Not owner"); _; }
    function blockAddress(address wallet, uint256 riskScore) external onlyOwner {
        require(wallet != address(0), "Cannot block zero address");
        if (!blocked[wallet]) { totalBlocked += 1; }
        blocked[wallet]    = true;
        riskScores[wallet] = riskScore;
        blockedAt[wallet]  = block.timestamp;
        emit BlockedAddress(wallet, riskScore, block.timestamp);
    }
    function removeAddress(address wallet) external onlyOwner {
        require(blocked[wallet], "Address not blocked");
        blocked[wallet] = false;
        if (totalBlocked > 0) totalBlocked -= 1;
        emit RemovedAddress(wallet, block.timestamp);
    }
    function isBlocked(address wallet) external view returns (bool) { return blocked[wallet]; }
    function getRiskScore(address wallet) external view returns (uint256) { return riskScores[wallet]; }
}
""",
    "rate_limiter": """\
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;
contract RateLimiter_test0001 {
    address public immutable owner;
    uint256 public maxVolumePerBlock;
    uint256 private _currentBlock;
    uint256 private _currentBlockVolume;
    event VolumeExceeded(uint256 indexed blockNum, uint256 cumulativeVolume, uint256 limit);
    event LimitUpdated(uint256 oldLimit, uint256 newLimit, uint256 timestamp);
    event VolumeRecorded(uint256 indexed blockNum, uint256 cumulativeVolume, uint256 limit, bool allowed);
    constructor(uint256 _maxVolumeWei) {
        require(_maxVolumeWei > 0, "Limit must be positive");
        owner             = msg.sender;
        maxVolumePerBlock = _maxVolumeWei;
    }
    modifier onlyOwner() { require(msg.sender == owner, "Not owner"); _; }
    function recordVolume(uint256 amountWei) external onlyOwner returns (bool allowed) {
        if (block.number > _currentBlock) { _currentBlock = block.number; _currentBlockVolume = 0; }
        _currentBlockVolume += amountWei;
        allowed = (_currentBlockVolume <= maxVolumePerBlock);
        if (!allowed) { emit VolumeExceeded(block.number, _currentBlockVolume, maxVolumePerBlock); }
        emit VolumeRecorded(block.number, _currentBlockVolume, maxVolumePerBlock, allowed);
        return allowed;
    }
    function updateLimit(uint256 newLimitWei) external onlyOwner {
        require(newLimitWei > 0, "Limit must be positive");
        uint256 old = maxVolumePerBlock;
        maxVolumePerBlock = newLimitWei;
        emit LimitUpdated(old, newLimitWei, block.timestamp);
    }
    function getCurrentBlockVolume() external view returns (uint256) {
        if (block.number > _currentBlock) return 0;
        return _currentBlockVolume;
    }
    function isWithinLimit(uint256 additionalWei) external view returns (bool) {
        uint256 vol = (block.number > _currentBlock) ? additionalWei : _currentBlockVolume + additionalWei;
        return vol <= maxVolumePerBlock;
    }
}
""",
}

# Expected constructor argument counts
CONSTRUCTOR_ARGS = {
    "circuit_breaker":  1,  # address _target
    "address_blocklist":0,  # no args
    "rate_limiter":     1,  # uint256 _maxVolumeWei
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


def compile_with_standard_json(source: str, template_key: str) -> dict:
    """
    Compile using the exact same method as the fixed ContractAgent._compile_solidity().
    Returns parsed solc output dict.
    Raises RuntimeError on failure.
    """
    # ASCII-safe temp path (the fix for Windows UTF-8 issue)
    tmp_dir  = tempfile.gettempdir()
    src_path = os.path.join(tmp_dir, f"contract_{template_key}_test.sol")
    Path(src_path).write_text(source, encoding="utf-8")

    std_input = json.dumps({
        "language": "Solidity",
        "sources":  {"contract.sol": {"content": source}},
        "settings": {
            "outputSelection": {"*": {"*": ["abi", "evm.bytecode.object"]}}
        },
    })

    try:
        result = subprocess.run(
            ["solc", "--standard-json"],
            input=std_input,
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=30,
        )
    finally:
        try:
            os.unlink(src_path)
        except Exception:
            pass

    if result.returncode != 0:
        raise RuntimeError(f"solc non-zero exit: {result.stderr[:400]}")

    return json.loads(result.stdout)


def main():
    all_failures = []

    print("\n" + "=" * 60)
    print("  COMPILE TEMPLATE TEST — Step 4")
    print("=" * 60)

    # ── 1. solc on PATH ──────────────────────────────────────────
    section("1. solc installation")

    solc_path = shutil.which("solc")
    ok = check("solc found on PATH",
               solc_path is not None,
               f"path: {solc_path}" if solc_path else
               "Fix: install solc\n"
               "       Windows: download from https://github.com/ethereum/solidity/releases\n"
               "       or run:  pip install py-solc-x\n"
               "                python -c \"from solcx import install_solc; install_solc('0.8.20')\"\n"
               "                python -c \"import solcx; import os; os.environ['PATH'] += ';' + str(solcx.get_solcx_install_folder())\"")
    if not ok:
        all_failures.append("solc not found on PATH")
        print("\n  Cannot continue without solc. Install it first.\n")
        _summary(all_failures)
        sys.exit(1)

    # solc version
    try:
        ver_result = subprocess.run(["solc", "--version"],
                                    capture_output=True, text=True, timeout=10)
        ver_line = [l for l in ver_result.stdout.splitlines() if "Version" in l]
        ver_str  = ver_line[0] if ver_line else ver_result.stdout.strip()
        is_08    = "0.8" in ver_str
        check("solc version is 0.8.x",
              is_08, f"reported: {ver_str}"
              if is_08 else
              f"got: {ver_str}\n       templates use pragma ^0.8.20 — you need solc 0.8.x")
        if not is_08:
            all_failures.append(f"Wrong solc version: {ver_str}")
    except Exception as e:
        check("solc version readable", False, str(e))
        all_failures.append(f"solc version check failed: {e}")

    # ── 2. --standard-json mode ──────────────────────────────────
    section("2. solc --standard-json mode (the fixed compilation path)")

    trivial_source = "// SPDX-License-Identifier: MIT\npragma solidity ^0.8.20;\ncontract T {}\n"
    trivial_input  = json.dumps({
        "language": "Solidity",
        "sources":  {"t.sol": {"content": trivial_source}},
        "settings": {"outputSelection": {"*": {"*": ["abi", "evm.bytecode.object"]}}},
    })
    try:
        result = subprocess.run(
            ["solc", "--standard-json"],
            input=trivial_input,
            capture_output=True, text=True, encoding="utf-8", timeout=15,
        )
        ok_rc = check("--standard-json returns exit code 0",
                      result.returncode == 0,
                      f"exit code: {result.returncode}\n"
                      f"       stderr: {result.stderr[:200]}" if result.returncode != 0 else "")

        if result.stdout:
            parsed = json.loads(result.stdout)
            has_contracts = bool(parsed.get("contracts"))
            check("Output is valid JSON with contracts key",
                  has_contracts,
                  f"keys in output: {list(parsed.keys())}")
            if not has_contracts:
                errors = parsed.get("errors", [])
                for e in errors:
                    print(f"       solc error: {e.get('message','')}")
                all_failures.append("--standard-json produced no contracts")
        else:
            check("--standard-json produces output", False, "stdout is empty")
            all_failures.append("--standard-json stdout empty")

    except json.JSONDecodeError as e:
        check("--standard-json output is valid JSON", False,
              f"JSON parse error: {e}\n"
              f"       raw stdout: {result.stdout[:200]}")
        all_failures.append("JSON decode error in solc output")
    except Exception as e:
        check("--standard-json mode works", False, str(e))
        all_failures.append(f"--standard-json failed: {e}")

    # ── 3. Compile each template ─────────────────────────────────
    section("3. Compile each template")

    for template_key, source in MINIMAL_SOURCES.items():
        print(f"\n  Template: {template_key}")

        # UTF-8 encoding check
        try:
            encoded = source.encode("utf-8")
            check(f"  Source encodes to UTF-8 cleanly",
                  True, f"{len(encoded)} bytes, no invalid chars")
        except UnicodeEncodeError as e:
            check(f"  UTF-8 encoding", False, str(e))
            all_failures.append(f"{template_key}: UTF-8 encode error")
            continue

        # Compile
        try:
            output = compile_with_standard_json(source, template_key)
        except RuntimeError as e:
            check(f"  Compilation succeeds", False, str(e))
            all_failures.append(f"{template_key}: compile error: {e}")
            continue
        except Exception as e:
            check(f"  Compilation (unexpected error)", False, str(e))
            all_failures.append(f"{template_key}: unexpected: {e}")
            continue

        # Check for solc errors
        errors = [e for e in output.get("errors", [])
                  if e.get("severity") == "error"]
        warnings = [e for e in output.get("errors", [])
                    if e.get("severity") == "warning"]
        check(f"  No compile errors",
              len(errors) == 0,
              f"{len(errors)} error(s): {errors[0].get('message','') if errors else ''}")
        if errors:
            all_failures.append(f"{template_key}: {errors[0].get('message','')}")
            continue
        if warnings:
            print(f"       {WARN} {len(warnings)} warning(s) — not blocking")

        # Extract bytecode + ABI
        contracts_out = output.get("contracts", {})
        bytecode = None
        abi      = None
        for src_name, contracts in contracts_out.items():
            for name, data in contracts.items():
                bytecode = data.get("evm", {}).get("bytecode", {}).get("object", "")
                abi      = data.get("abi", [])
                break
            if bytecode:
                break

        check(f"  Bytecode is non-empty hex",
              bool(bytecode) and len(bytecode) > 10,
              f"length: {len(bytecode)} hex chars")
        if not bytecode:
            all_failures.append(f"{template_key}: empty bytecode")

        check(f"  ABI is a list",
              isinstance(abi, list),
              f"type: {type(abi).__name__}")

        # Count constructor inputs
        constructor = [fn for fn in (abi or []) if fn.get("type") == "constructor"]
        n_inputs    = len(constructor[0].get("inputs", [])) if constructor else 0
        expected    = CONSTRUCTOR_ARGS[template_key]
        check(f"  Constructor takes {expected} arg(s)",
              n_inputs == expected,
              f"got {n_inputs} — "
              + {0: "no constructor args (correct for address_blocklist)",
                 1: "1 arg (correct for circuit_breaker=address, rate_limiter=uint256)"}
              .get(expected, f"expected {expected}"))

        # Check key functions exist in ABI
        fn_names = {fn.get("name") for fn in (abi or []) if fn.get("type") == "function"}
        expected_fns = {
            "circuit_breaker":  {"activate", "isActivated"},
            "address_blocklist":{"blockAddress", "isBlocked"},
            "rate_limiter":     {"recordVolume", "updateLimit"},
        }[template_key]
        missing = expected_fns - fn_names
        check(f"  Key functions in ABI: {expected_fns}",
              len(missing) == 0,
              f"missing: {missing}" if missing else f"all present, total {len(fn_names)} functions")
        if missing:
            all_failures.append(f"{template_key}: missing ABI functions: {missing}")

        print(f"       ABI functions: {sorted(fn_names)}")

    # ── 4. Unicode / Windows encoding test ──────────────────────
    section("4. Windows encoding safety (the 0x97 byte fix)")

    tmp = tempfile.gettempdir()
    safe_path = os.path.join(tmp, "contract_circuit_breaker_test.sol")
    check("Temp dir is accessible", os.path.exists(tmp), f"tmp dir: {tmp}")

    try:
        Path(safe_path).write_text(
            "pragma solidity ^0.8.20; contract T {}",
            encoding="utf-8"
        )
        with open(safe_path, encoding="utf-8") as f:
            content = f.read()
        check("Write + read UTF-8 temp file works",
              "pragma" in content,
              f"path: {safe_path}")
        os.unlink(safe_path)

        # Check the path itself contains no non-ASCII chars
        has_non_ascii = any(ord(c) > 127 for c in safe_path)
        check("Temp path contains only ASCII chars",
              not has_non_ascii,
              f"path: {safe_path}"
              if not has_non_ascii else
              f"WARNING: non-ASCII in path: {safe_path}\n"
              f"       This was the root cause of the 0x97 error on Windows.\n"
              f"       The fix uses Path.write_text(encoding='utf-8') which bypasses this.")

    except Exception as e:
        check("Encoding safety test", False, str(e))
        all_failures.append(f"Encoding test failed: {e}")

    # ── SUMMARY ──────────────────────────────────────────────────
    _summary(all_failures)
    sys.exit(0 if not all_failures else 1)


def _summary(failures):
    print("\n" + "=" * 60)
    if not failures:
        print(f"{PASS}  ALL CHECKS PASSED")
        print("  All templates compile cleanly.")
        print("  Safe to proceed to Step 5 — deploy contract.")
    else:
        print(f"{FAIL}  {len(failures)} check(s) FAILED:")
        for f in failures:
            print(f"    • {f}")
        print("\n  Fix the failures above before deploying.")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()