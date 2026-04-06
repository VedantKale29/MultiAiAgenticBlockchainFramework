/**
 * scripts/deploy.js
 * ==================
 * Deploys ALL contracts needed for the full agentic fraud detection demo.
 *
 * DEPLOYMENT ORDER:
 *   1. GovernanceContract      — on-chain parameter governance (TimelockController)
 *   2. ContractRegistry        — tamper-proof registry of autonomously deployed contracts
 *   3. NaiveReceiverLenderPool — vulnerable flash loan pool (MonitorAgent watches this)
 *   4. FlashLoanReceiver       — victim contract (holds ETH, gets drained via pool fee)
 *   5. FlashLoanAttacker       — attack contract (called by attack.js to trigger detection)
 *
 * USAGE:
 *   npx hardhat run scripts/deploy.js --network localhost
 *
 * OUTPUT:
 *   - Prints all env vars to copy into your shell
 *   - Saves deployments/localhost.json with all addresses
 *
 * AFTER RUNNING:
 *   Copy the printed export lines into your shell, then:
 *     python blockchain_pipeline.py       ← live RO5 demo
 *     python run_pipeline.py              ← batch pipeline with real contracts
 */

const hre = require("hardhat");
const fs  = require("fs");

async function main() {
  const [deployer] = await hre.ethers.getSigners();

  console.log("=".repeat(60));
  console.log("Agentic AI Blockchain Fraud Detection");
  console.log("Deploying all contracts");
  console.log("=".repeat(60));
  console.log("Network  :", hre.network.name);
  console.log("Deployer :", deployer.address);

  const balance = await deployer.provider.getBalance(deployer.address);
  console.log("Balance  :", hre.ethers.formatEther(balance), "ETH");
  console.log("=".repeat(60));

  // ── 1. GovernanceContract ─────────────────────────────────────────
  const timelockDelay =
    hre.network.name === "localhost" || hre.network.name === "hardhat"
      ? 60      // 60s on Hardhat  (represents 24h in production)
      : 86400;  // 24h on testnets / mainnet

  console.log(`\n[1/5] Deploying GovernanceContract (timelock=${timelockDelay}s)...`);
  const GovernanceContract = await hre.ethers.getContractFactory("GovernanceContract");
  const governance = await GovernanceContract.deploy(timelockDelay);
  await governance.waitForDeployment();
  const governanceAddr = await governance.getAddress();
  console.log(`      GovernanceContract : ${governanceAddr}`);

  // Verify initial parameters written by constructor
  const params = await governance.getAllParameters();
  console.log(`      Initial params     :`);
  console.log(`        tau_alert = ${Number(params[0]) / 1e18}`);
  console.log(`        tau_block = ${Number(params[1]) / 1e18}`);
  console.log(`        w         = ${Number(params[2]) / 1e18}`);

  // ── 2. ContractRegistry ───────────────────────────────────────────
  console.log(`\n[2/5] Deploying ContractRegistry...`);
  const ContractRegistry = await hre.ethers.getContractFactory("ContractRegistry");
  const registry = await ContractRegistry.deploy();
  await registry.waitForDeployment();
  const registryAddr = await registry.getAddress();
  console.log(`      ContractRegistry   : ${registryAddr}`);

  // ── 3. NaiveReceiverLenderPool ────────────────────────────────────
  // Fund pool with 100 ETH so flash loans can execute
  const poolFunding = hre.ethers.parseEther("100");
  console.log(`\n[3/5] Deploying NaiveReceiverLenderPool (funding: 100 ETH)...`);
  const NaiveReceiverLenderPool = await hre.ethers.getContractFactory("NaiveReceiverLenderPool");
  const pool = await NaiveReceiverLenderPool.deploy({ value: poolFunding });
  await pool.waitForDeployment();
  const poolAddr = await pool.getAddress();
  const poolBalance = await pool.poolBalance();
  console.log(`      NaiveReceiverLenderPool : ${poolAddr}`);
  console.log(`      Pool balance            : ${hre.ethers.formatEther(poolBalance)} ETH`);

  // ── 4. FlashLoanReceiver (victim) ─────────────────────────────────
  // Fund victim with 10 ETH — this is what gets drained by the attack fee
  const victimFunding = hre.ethers.parseEther("10");
  console.log(`\n[4/5] Deploying FlashLoanReceiver / victim (funding: 10 ETH)...`);
  const FlashLoanReceiver = await hre.ethers.getContractFactory("FlashLoanReceiver");
  const receiver = await FlashLoanReceiver.deploy(poolAddr, { value: victimFunding });
  await receiver.waitForDeployment();
  const receiverAddr = await receiver.getAddress();
  const receiverBalance = await receiver.getBalance();
  console.log(`      FlashLoanReceiver : ${receiverAddr}`);
  console.log(`      Victim balance    : ${hre.ethers.formatEther(receiverBalance)} ETH`);

  // ── 5. FlashLoanAttacker ──────────────────────────────────────────
  console.log(`\n[5/5] Deploying FlashLoanAttacker...`);
  const FlashLoanAttacker = await hre.ethers.getContractFactory("FlashLoanAttacker");
  const attacker = await FlashLoanAttacker.deploy();
  await attacker.waitForDeployment();
  const attackerAddr = await attacker.getAddress();
  console.log(`      FlashLoanAttacker : ${attackerAddr}`);

  // ── Summary ───────────────────────────────────────────────────────
  console.log("\n" + "=".repeat(60));
  console.log("DEPLOYMENT COMPLETE — 5 contracts deployed");
  console.log("=".repeat(60));

  console.log("\nCopy these into your shell before running Python:\n");
  console.log(`export GOVERNANCE_CONTRACT_ADDRESS="${governanceAddr}"`);
  console.log(`export CONTRACT_REGISTRY_ADDRESS="${registryAddr}"`);
  console.log(`export VULNERABLE_POOL_ADDRESS="${poolAddr}"`);
  console.log(`export FLASH_LOAN_RECEIVER_ADDRESS="${receiverAddr}"`);
  console.log(`export FLASH_LOAN_ATTACKER_ADDRESS="${attackerAddr}"`);
  console.log(`export HARDHAT_DEPLOYER_KEY="0xac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80"`);

  console.log("\nThen run the live RO5 demo:");
  console.log("  python blockchain_pipeline.py");
  console.log("\nOr the full batch pipeline:");
  console.log("  python run_pipeline.py");
  console.log("=".repeat(60));

  // ── Save JSON ─────────────────────────────────────────────────────
  const deploymentInfo = {
    network:                  hre.network.name,
    deployer:                 deployer.address,
    timestamp:                new Date().toISOString(),
    timelockDelay:            timelockDelay,
    GovernanceContract:       governanceAddr,
    ContractRegistry:         registryAddr,
    NaiveReceiverLenderPool:  poolAddr,
    FlashLoanReceiver:        receiverAddr,
    FlashLoanAttacker:        attackerAddr,
  };

  if (!fs.existsSync("deployments")) {
    fs.mkdirSync("deployments");
  }

  const outPath = `deployments/${hre.network.name}.json`;
  fs.writeFileSync(outPath, JSON.stringify(deploymentInfo, null, 2));
  console.log(`\nDeployment info saved to: ${outPath}`);
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});