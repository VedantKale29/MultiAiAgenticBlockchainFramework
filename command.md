aws s3 cp s3://multiagenticblockchain/agentic-fraud/runs/fraud-seedv1242-agentic/fraud-seedv1242-agentic/output/output.tar.gz "C:/Users/DEEPA KALE/Music/MultiAiAgenticBlockchain/sm_src/output.tar.gz"


## Artifact
aws s3 sync s3://multiagenticblockchain/agentic-fraud/runs/run_seed42_v1 sm_src/runs/run_seed42_v1


cd "C:/Users/DEEPA KALE/Music/MultiAiAgenticBlockchain/sm_src"
tar -xzf "output.tar.gz" -C "runs/run_seed42_v1"


python .\sm_src\aws\sm_launcher.py --seed 42 --run_mode AGENTIC --wait


$env:GOVERNANCE_CONTRACT_ADDRESS = "0x59b670e9fA9D0A427751Af201D676719a970857b"
    $env:CONTRACT_REGISTRY_ADDRESS   = "0x4ed7c70F96B99c776995fB64377f0d4aB3B0e1C1"
    $env:HARDHAT_DEPLOYER_KEY         = "0xac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80"