/**
 * scripts/deploy.js (ESM FIXED)
 */

import hre from "hardhat";
import fs from "fs";

async function main() {
  const [deployer] = await hre.ethers.getSigners();

  console.log("=".repeat(60));
  console.log("Deploying fraud detection governance contracts");
  console.log("Network:", hre.network.name);
  console.log("Deployer:", deployer.address);

  const balance = await deployer.provider.getBalance(deployer.address);
  console.log("Balance:", hre.ethers.formatEther(balance), "ETH");
  console.log("=".repeat(60));

  // ── 1. Deploy GovernanceContract ────────────────────────────────
  const timelockDelay =
    hre.network.name === "localhost" || hre.network.name === "hardhat"
      ? 60
      : 86400;

  console.log(`\nDeploying GovernanceContract (timelock=${timelockDelay}s)...`);
  const GovernanceContract = await hre.ethers.getContractFactory("GovernanceContract");
  const governance = await GovernanceContract.deploy(timelockDelay);
  await governance.waitForDeployment();

  const governanceAddr = await governance.getAddress();
  console.log(`  GovernanceContract deployed: ${governanceAddr}`);

  const params = await governance.getAllParameters();
  console.log(`  Initial params:`);
  console.log(`    tau_alert = ${Number(params[0]) / 1e18}`);
  console.log(`    tau_block = ${Number(params[1]) / 1e18}`);
  console.log(`    w         = ${Number(params[2]) / 1e18}`);

  // ── 2. Deploy ContractRegistry ───────────────────────────────────
  console.log(`\nDeploying ContractRegistry...`);
  const ContractRegistry = await hre.ethers.getContractFactory("ContractRegistry");
  const registry = await ContractRegistry.deploy();
  await registry.waitForDeployment();

  const registryAddr = await registry.getAddress();
  console.log(`  ContractRegistry deployed: ${registryAddr}`);

  // ── 3. Print env vars ────────────────────────────────────────────
  console.log("\n" + "=".repeat(60));
  console.log("DEPLOYMENT COMPLETE");
  console.log("=".repeat(60));

  console.log("\nAdd these to your environment before running main.py:\n");
  console.log(`export GOVERNANCE_CONTRACT_ADDRESS="${governanceAddr}"`);
  console.log(`export CONTRACT_REGISTRY_ADDRESS="${registryAddr}"`);
  console.log(
    `export HARDHAT_DEPLOYER_KEY="0xac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80"`
  );

  console.log("\nThen run:");
  console.log(`  ANTHROPIC_API_KEY=sk-ant-... python main.py --run_evaluation`);
  console.log("=".repeat(60));

  // ── 4. Save JSON ────────────────────────────────────────────────
  const deploymentInfo = {
    network: hre.network.name,
    deployer: deployer.address,
    timestamp: new Date().toISOString(),
    timelockDelay: timelockDelay,
    GovernanceContract: governanceAddr,
    ContractRegistry: registryAddr,
  };

  if (!fs.existsSync("deployments")) {
    fs.mkdirSync("deployments");
  }

  const outPath = `deployments/${hre.network.name}.json`;
  fs.writeFileSync(outPath, JSON.stringify(deploymentInfo, null, 2));

  console.log(`\nDeployment info saved to ${outPath}`);
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});