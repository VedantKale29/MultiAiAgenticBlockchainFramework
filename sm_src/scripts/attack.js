/**
 * scripts/attack.js
 * ==================
 * Executes a simulated flash loan attack against NaiveReceiverLenderPool.
 *
 * PURPOSE:
 *   Triggers a real on-chain FlashLoan event that MonitorAgent detects
 *   within its 1-second polling interval. This is the entry point for
 *   the RO5 self-triggering demo.
 *
 * WHAT HAPPENS:
 *   1. FlashLoanAttacker.attack(pool, victim, amount, repeat) is called
 *   2. NaiveReceiverLenderPool executes the flash loan
 *   3. FlashLoan event is emitted on-chain
 *   4. MonitorAgent (running in background) detects the event in <2s
 *   5. THREAT_DETECTED payload fires → Decision → Contract → Governance chain
 *
 * USAGE (standalone — after deploy.js):
 *   npx hardhat run scripts/attack.js --network localhost
 *
 * USAGE (with env vars set by deploy.js or blockchain_pipeline.py):
 *   VULNERABLE_POOL_ADDRESS=0x... \
 *   FLASH_LOAN_RECEIVER_ADDRESS=0x... \
 *   FLASH_LOAN_ATTACKER_ADDRESS=0x... \
 *   npx hardhat run scripts/attack.js --network localhost
 *
 * ENVIRONMENT VARIABLES (read from shell or deployments/localhost.json):
 *   VULNERABLE_POOL_ADDRESS         — NaiveReceiverLenderPool address
 *   FLASH_LOAN_RECEIVER_ADDRESS     — FlashLoanReceiver (victim) address
 *   FLASH_LOAN_ATTACKER_ADDRESS     — FlashLoanAttacker address
 *   FLASH_LOAN_AMOUNT_ETH           — amount to borrow (default: 10 ETH)
 *   FLASH_LOAN_REPEAT               — how many loans to fire (default: 3)
 *
 * DEFAULT ADDRESSES (fallback if env vars not set):
 *   Reads from deployments/localhost.json written by deploy.js
 */

const hre = require("hardhat");
const fs  = require("fs");
const path = require("path");

// ── Minimal ABIs (only what we need) ──────────────────────────────
const POOL_ABI = [
  "function poolBalance() external view returns (uint256)",
  "function maxFlashLoan() external view returns (uint256)",
  "function isPaused() external view returns (bool)",
  "event FlashLoan(address indexed borrower, uint256 amount, uint256 fee)",
];

const RECEIVER_ABI = [
  "function getBalance() external view returns (uint256)",
];

const ATTACKER_ABI = [
  "function attack(address pool, address victim, uint256 amount, uint256 repeat) external",
  "event AttackLaunched(address indexed pool, address indexed victim, uint256 amount, uint256 timestamp)",
];

// ── Load deployment addresses ──────────────────────────────────────
function loadAddresses(network) {
  // Priority: env vars > deployments JSON
  const envPool     = process.env.VULNERABLE_POOL_ADDRESS;
  const envReceiver = process.env.FLASH_LOAN_RECEIVER_ADDRESS;
  const envAttacker = process.env.FLASH_LOAN_ATTACKER_ADDRESS;

  if (envPool && envReceiver && envAttacker) {
    return { poolAddr: envPool, receiverAddr: envReceiver, attackerAddr: envAttacker };
  }

  // Fall back to deployments JSON
  const jsonPath = path.join(__dirname, "..", "deployments", `${network}.json`);
  if (fs.existsSync(jsonPath)) {
    const info = JSON.parse(fs.readFileSync(jsonPath, "utf8"));
    if (info.NaiveReceiverLenderPool && info.FlashLoanReceiver && info.FlashLoanAttacker) {
      return {
        poolAddr:     info.NaiveReceiverLenderPool,
        receiverAddr: info.FlashLoanReceiver,
        attackerAddr: info.FlashLoanAttacker,
      };
    }
  }

  throw new Error(
    "Cannot find contract addresses.\n" +
    "Either set env vars (VULNERABLE_POOL_ADDRESS, FLASH_LOAN_RECEIVER_ADDRESS, FLASH_LOAN_ATTACKER_ADDRESS)\n" +
    "or run: npx hardhat run scripts/deploy.js --network localhost"
  );
}

async function main() {
  const [deployer] = await hre.ethers.getSigners();
  const network = hre.network.name;

  console.log("=".repeat(60));
  console.log("Flash Loan Attack — RO5 Self-Triggering Demo");
  console.log("=".repeat(60));
  console.log("Network  :", network);
  console.log("Attacker :", deployer.address);

  // ── Load addresses ───────────────────────────────────────────────
  const { poolAddr, receiverAddr, attackerAddr } = loadAddresses(network);
  console.log("\nContract addresses:");
  console.log("  Pool     :", poolAddr);
  console.log("  Victim   :", receiverAddr);
  console.log("  Attacker :", attackerAddr);

  // ── Parameters ───────────────────────────────────────────────────
  const amountEth = parseFloat(process.env.FLASH_LOAN_AMOUNT_ETH || "10");
  const repeat    = parseInt(process.env.FLASH_LOAN_REPEAT || "3");
  const amount    = hre.ethers.parseEther(amountEth.toString());

  console.log(`\nAttack parameters:`);
  console.log(`  Flash loan amount : ${amountEth} ETH`);
  console.log(`  Repeat count      : ${repeat}  (each fires a FlashLoan event)`);

  // ── Connect to contracts ─────────────────────────────────────────
  const pool     = new hre.ethers.Contract(poolAddr,     POOL_ABI,     deployer);
  const receiver = new hre.ethers.Contract(receiverAddr, RECEIVER_ABI, deployer);
  const attacker = new hre.ethers.Contract(attackerAddr, ATTACKER_ABI, deployer);

  // ── Pre-attack state ─────────────────────────────────────────────
  console.log("\n--- Pre-attack state ---");
  const poolBalBefore     = await pool.poolBalance();
  const victimBalBefore   = await receiver.getBalance();
  const paused            = await pool.isPaused();
  const maxLoan           = await pool.maxFlashLoan();

  console.log(`  Pool balance      : ${hre.ethers.formatEther(poolBalBefore)} ETH`);
  console.log(`  Victim balance    : ${hre.ethers.formatEther(victimBalBefore)} ETH`);
  console.log(`  Pool paused       : ${paused}`);
  console.log(`  Max flash loan    : ${hre.ethers.formatEther(maxLoan)} ETH`);

  if (paused) {
    console.log("\n  [!] Pool is already paused — cannot execute attack.");
    console.log("      This means a CircuitBreaker was already deployed.");
    console.log("      Restart Hardhat (npx hardhat node) and redeploy to reset.");
    process.exit(0);
  }

  if (poolBalBefore < amount) {
    console.log(`\n  [!] Pool balance (${hre.ethers.formatEther(poolBalBefore)} ETH)`);
    console.log(`      is less than requested loan (${amountEth} ETH).`);
    console.log(`      Adjusting loan amount to pool balance...`);
    // Use 80% of pool balance to leave room for fees
    const adjusted = (poolBalBefore * 80n) / 100n;
    console.log(`      Adjusted amount: ${hre.ethers.formatEther(adjusted)} ETH`);
  }

  // ── Execute attack ────────────────────────────────────────────────
  console.log("\n--- Launching attack ---");
  console.log(`  Firing ${repeat} flash loan(s)...`);
  console.log("  [MonitorAgent should detect FlashLoan events within 1-2 seconds]\n");

  const t0 = Date.now();

  // Listen for events before sending tx
  const detectedEvents = [];
  pool.on("FlashLoan", (borrower, amount, fee, event) => {
    const elapsed = Date.now() - t0;
    detectedEvents.push({ borrower, amount, fee, block: event.log.blockNumber });
    console.log(`  [+] FlashLoan event detected at +${elapsed}ms`);
    console.log(`      borrower : ${borrower}`);
    console.log(`      amount   : ${hre.ethers.formatEther(amount)} ETH`);
    console.log(`      fee      : ${hre.ethers.formatEther(fee)} ETH`);
    console.log(`      block    : ${event.log.blockNumber}`);
  });

  let txHash;
  try {
    const tx = await attacker.attack(poolAddr, receiverAddr, amount, repeat);
    txHash = tx.hash;
    console.log(`  Transaction submitted: ${txHash}`);
    const receipt = await tx.wait();
    const elapsed = Date.now() - t0;
    console.log(`  Transaction confirmed: block=${receipt.blockNumber} (${elapsed}ms)`);
    console.log(`  Gas used: ${receipt.gasUsed.toString()}`);
  } catch (err) {
    console.error("\n  [!] Attack transaction failed:", err.message);
    console.error("      Make sure pool has enough liquidity and victim has ETH.");
    process.exit(1);
  }

  // Give event listeners a moment to fire
  await new Promise(r => setTimeout(r, 500));
  pool.removeAllListeners();

  // ── Post-attack state ─────────────────────────────────────────────
  console.log("\n--- Post-attack state ---");
  const poolBalAfter   = await pool.poolBalance();
  const victimBalAfter = await receiver.getBalance();
  const pausedAfter    = await pool.isPaused();

  const poolDelta   = poolBalAfter - poolBalBefore;
  const victimDelta = victimBalAfter - victimBalBefore;

  console.log(`  Pool balance      : ${hre.ethers.formatEther(poolBalAfter)} ETH`);
  console.log(`  Victim balance    : ${hre.ethers.formatEther(victimBalAfter)} ETH`);
  console.log(`  Pool delta        : ${hre.ethers.formatEther(poolDelta)} ETH`);
  console.log(`  Victim drained    : ${hre.ethers.formatEther(-victimDelta)} ETH`);
  console.log(`  Pool paused       : ${pausedAfter}`);

  if (pausedAfter) {
    console.log(`\n  *** Pool was PAUSED during attack! ***`);
    console.log(`  This means MonitorAgent detected the flash loan and`);
    console.log(`  ContractAgent deployed a CircuitBreaker that called pause().`);
  }

  // ── Summary ───────────────────────────────────────────────────────
  const totalElapsed = Date.now() - t0;
  console.log("\n" + "=".repeat(60));
  console.log("ATTACK COMPLETE");
  console.log("=".repeat(60));
  console.log(`  FlashLoan events fired : ${repeat}`);
  console.log(`  Transaction hash       : ${txHash}`);
  console.log(`  Total elapsed          : ${totalElapsed}ms`);
  console.log(`  Victim ETH drained     : ${hre.ethers.formatEther(-victimDelta)} ETH`);
  console.log("\nMonitorAgent should have fired THREAT_DETECTED.");
  console.log("Check blockchain_pipeline.py output for response chain.");
  console.log("=".repeat(60));

  // ── Write attack record ───────────────────────────────────────────
  const record = {
    timestamp:        new Date().toISOString(),
    network:          network,
    attacker:         deployer.address,
    txHash:           txHash,
    poolAddress:      poolAddr,
    receiverAddress:  receiverAddr,
    attackerAddress:  attackerAddr,
    amountEth:        amountEth,
    repeat:           repeat,
    elapsedMs:        totalElapsed,
    victimDrainedEth: parseFloat(hre.ethers.formatEther(-victimDelta)),
    poolPausedAfter:  pausedAfter,
    flashLoanEvents:  detectedEvents.length,
  };

  const recPath = "deployments/last_attack.json";
  fs.writeFileSync(recPath, JSON.stringify(record, null, 2));
  console.log(`\nAttack record saved to: ${recPath}`);
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});