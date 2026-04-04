"""
scripts/test_blockchain_connect.py
====================================
STEP 3 — Verify Web3.py connects to Hardhat

WHAT IT CHECKS:
  1. web3 package is installed
  2. Hardhat node is reachable at http://127.0.0.1:8545
  3. Chain ID is 31337 (Hardhat local chain)
  4. At least 10 unlocked accounts available
  5. Account balances are > 0 ETH (confirms Hardhat funded them)
  6. Can send a simple ETH transfer (proves write access works)
  7. Block number advances after a transaction (proves mining works)

PRE-REQUISITE:
  npx hardhat node   ← must be running in a separate terminal

RUN:
  python scripts/test_blockchain_connect.py

EXPECTED OUTPUT:
  Every check ✅  — Web3 is ready for contract deployment
"""

import sys
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

PASS = "✅"
FAIL = "❌"
INFO = "ℹ️ "
SEP  = "─" * 60
HARDHAT_URL = os.getenv("HARDHAT_URL", "http://127.0.0.1:8545")

# Hardhat default test account #0
HARDHAT_ACCOUNT_0     = "0xf39Fd6e51aad88F6F4ce6aB8827279cffFb92266"
HARDHAT_PRIVATE_KEY_0 = "0xac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80"
HARDHAT_ACCOUNT_1     = "0x70997970C51812dc3A010C7d01b50e0d17dc79C8"


def section(title):
    print(f"\n{SEP}")
    print(f"  {title}")
    print(SEP)


def check(label, condition, detail="", fatal=False):
    icon = PASS if condition else FAIL
    line = f"  {icon}  {label}"
    if detail:
        line += f"\n       {detail}"
    print(line)
    if not condition and fatal:
        print(f"\n  FATAL — cannot continue without this. Fix it first.\n")
        sys.exit(1)
    return condition


def main():
    print("\n" + "=" * 60)
    print("  BLOCKCHAIN CONNECT TEST — Step 3")
    print("  Target:", HARDHAT_URL)
    print("=" * 60)

    failures = []

    # ── 1. web3 installed ────────────────────────────────────────
    section("1. Package installation")
    try:
        from web3 import Web3
        import web3
        check("web3 installed", True, f"version: {web3.__version__}")
    except ImportError as e:
        check("web3 installed", False,
              f"Error: {e}\nFix: pip install web3", fatal=True)

    # ── 2. Hardhat reachable ─────────────────────────────────────
    section("2. Hardhat node connectivity")
    from web3 import Web3

    w3 = Web3(Web3.HTTPProvider(HARDHAT_URL, request_kwargs={"timeout": 5}))

    connected = w3.is_connected()
    check("Hardhat node reachable",  connected,
          f"URL: {HARDHAT_URL}\n"
          f"       Fix: run 'npx hardhat node' in a separate terminal",
          fatal=True)
    if not connected:
        failures.append("Hardhat not reachable")

    # ── 3. Chain ID ──────────────────────────────────────────────
    section("3. Chain verification")
    try:
        chain_id = w3.eth.chain_id
        ok = chain_id == 31337
        check("Chain ID == 31337 (Hardhat local)",
              ok, f"got chain_id={chain_id}")
        if not ok:
            failures.append(f"Wrong chain_id: {chain_id}")

        block = w3.eth.block_number
        check("Block number readable", True, f"current block: {block}")

        gas_price = w3.eth.gas_price
        check("Gas price readable", True,
              f"{w3.from_wei(gas_price, 'gwei'):.2f} gwei")

    except Exception as e:
        check("Chain info readable", False, str(e))
        failures.append(str(e))

    # ── 4. Accounts ──────────────────────────────────────────────
    section("4. Unlocked accounts")
    try:
        accounts = w3.eth.accounts
        check("At least 10 accounts available",
              len(accounts) >= 10, f"found {len(accounts)} accounts")

        check("Account #0 matches Hardhat default",
              len(accounts) > 0 and
              accounts[0].lower() == HARDHAT_ACCOUNT_0.lower(),
              f"expected: {HARDHAT_ACCOUNT_0}\n"
              f"       got:      {accounts[0] if accounts else 'none'}")

        for i, addr in enumerate(accounts[:3]):
            bal_wei = w3.eth.get_balance(addr)
            bal_eth = w3.from_wei(bal_wei, "ether")
            check(f"Account #{i} has ETH balance > 0",
                  bal_eth > 0,
                  f"{addr[:16]}... = {bal_eth:.0f} ETH")

    except Exception as e:
        check("Accounts readable", False, str(e))
        failures.append(str(e))

    # ── 5. Send a simple ETH transfer ────────────────────────────
    section("5. Write access — simple ETH transfer")
    try:
        sender   = accounts[0]
        receiver = accounts[1]
        amount   = w3.to_wei("0.001", "ether")

        bal_before = w3.eth.get_balance(receiver)
        block_before = w3.eth.block_number

        tx_hash = w3.eth.send_transaction({
            "from":  sender,
            "to":    receiver,
            "value": amount,
            "gas":   21000,
        })
        check("send_transaction() accepted", True,
              f"tx: {tx_hash.hex()[:20]}...")

        receipt = w3.eth.wait_for_transaction_receipt(tx_hash, timeout=15)
        check("Transaction mined (receipt received)",
              receipt.status == 1,
              f"status={receipt.status}  block={receipt.blockNumber}  "
              f"gasUsed={receipt.gasUsed}")

        bal_after = w3.eth.get_balance(receiver)
        check("Receiver balance increased",
              bal_after > bal_before,
              f"before={w3.from_wei(bal_before,'ether'):.6f} ETH  "
              f"after={w3.from_wei(bal_after,'ether'):.6f} ETH")

        block_after = w3.eth.block_number
        check("Block number advanced after tx",
              block_after > block_before,
              f"before={block_before}  after={block_after}")

    except Exception as e:
        check("Simple ETH transfer works", False, str(e))
        failures.append(f"ETH transfer failed: {e}")

    # ── 6. Private key account ───────────────────────────────────
    section("6. Private key account (needed for contract signing)")
    try:
        acct = w3.eth.account.from_key(HARDHAT_PRIVATE_KEY_0)
        check("Private key #0 loads correctly",
              acct.address.lower() == HARDHAT_ACCOUNT_0.lower(),
              f"derived address: {acct.address}\n"
              f"       expected:  {HARDHAT_ACCOUNT_0}")

        # Sign a dummy transaction (not sent — just proves signing works)
        nonce = w3.eth.get_transaction_count(acct.address)
        dummy_tx = {
            "to":       HARDHAT_ACCOUNT_1,
            "value":    w3.to_wei("0.0001", "ether"),
            "gas":      21000,
            "gasPrice": w3.to_wei("1", "gwei"),
            "nonce":    nonce,
            "chainId":  31337,
        }
        signed = w3.eth.account.sign_transaction(dummy_tx, HARDHAT_PRIVATE_KEY_0)
        check("sign_transaction() works",
              len(signed.raw_transaction) > 0,
              f"raw_transaction length: {len(signed.raw_transaction)} bytes")

    except Exception as e:
        check("Private key operations work", False, str(e))
        failures.append(f"Private key error: {e}")

    # ── SUMMARY ──────────────────────────────────────────────────
    print("\n" + "=" * 60)
    if not failures:
        print(f"{PASS}  ALL CHECKS PASSED")
        print("  Web3.py is fully connected to Hardhat.")
        print("  Safe to proceed to Step 4 — compile template.")
    else:
        print(f"{FAIL}  {len(failures)} check(s) FAILED:")
        for f in failures:
            print(f"    • {f}")
        print("\n  Fix the failures above before proceeding to Step 4.")
    print("=" * 60 + "\n")
    sys.exit(0 if not failures else 1)


if __name__ == "__main__":
    main()