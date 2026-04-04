aws s3 cp s3://multiagenticblockchain/agentic-fraud/runs/fraud-seedv1242-agentic/fraud-seedv1242-agentic/output/output.tar.gz "C:/Users/DEEPA KALE/Music/MultiAiAgenticBlockchain/sm_src/output.tar.gz"


## Artifact
aws s3 sync s3://multiagenticblockchain/agentic-fraud/runs/run_seed42_v1 sm_src/runs/run_seed42_v1


cd "C:/Users/DEEPA KALE/Music/MultiAiAgenticBlockchain/sm_src"
tar -xzf "output.tar.gz" -C "runs/run_seed42_v1"


python .\sm_src\aws\sm_launcher.py --seed 42 --run_mode AGENTIC --wait