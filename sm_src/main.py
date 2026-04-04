"""
main.py
=======
UNIFIED ENTRY POINT — works locally AND inside SageMaker.

WHAT THIS FILE DOES:
  1. Loads dataset (from local disk or SageMaker /opt/ml/input)
  2. Trains RF + IF models (one time, before streaming)
  3. Creates all AWS components (S3Manager, CloudWatchLogger, ExperimentTracker)
     → If not in AWS, these degrade gracefully to local-only mode
  4. Creates CoordinatorAgent with all 8 agents wired together
  5. CoordinatorAgent runs the full streaming batch pipeline
  6. Results are saved locally AND uploaded to S3 (if in AWS)

HOW TO RUN:

  Locally (same as before — nothing changed):
      python main.py

  In SageMaker (via sm_launcher.py from your laptop):
      python aws/sm_launcher.py --seed 42

  Multiple seeds in parallel on SageMaker:
      python aws/sm_launcher.py --multi

GRACEFUL DEGRADATION:
  If boto3 / sagemaker SDK are not installed:
    - S3Manager      → skips all uploads silently
    - CloudWatchLogger → only logs to terminal (same as before)
    - ExperimentTracker → saves local JSON (same as LocalExperimentTracker)
  Your local workflow is 100% unchanged.
"""
import subprocess
import sys

required_packages = [
    "chromadb",
    "sentence-transformers",
    "anthropic",      # Stage 3 — LLM reasoning in DecisionAgent
    "web3",           # Stage 3 — contract deployment via Hardhat
]

for pkg in required_packages:
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", pkg, "--quiet"])
    except Exception as e:
        # Never crash pipeline if optional package fails
        print(f"[main] Warning: could not install {pkg}: {e}")

import os
import json
import time
import argparse
import pandas as pd

import  config as config
from  logger import logging
from data_loader import load_and_clean_data, get_train_test_split
from  rf_model import RFModel
from  if_model import IFModel

# AWS components (graceful fallback if not available)
from aws.s3_manager         import S3Manager
from aws.cloudwatch_logger  import CloudWatchLogger
from aws.experiment_tracker import ExperimentTracker

# Multi-agent system
from agents.coordinator_agent import CoordinatorAgent
from agents.base_agent        import AgentMessage


def parse_args():
    """Accept command-line args when launched by SageMaker."""
    p = argparse.ArgumentParser()
    p.add_argument("--seed",      type=int, default=config.BASE_SEED)
    p.add_argument("--run_mode",  type=str, default=None)
    p.add_argument("--s3_bucket", type=str, default=None)
    args, _ = p.parse_known_args()
    return args


def save_config_json(run_dir: str) -> str:
    """Save all config values to a JSON file in the run directory."""
    cfg = {k: v for k, v in vars(config).items()
           if not k.startswith("__") and isinstance(v, (int, float, str, bool, list))}
    path = os.path.join(run_dir, "config.json")
    with open(path, "w") as f:
        json.dump(cfg, f, indent=4)
    return path


def main():
    args = parse_args()

    # Allow command-line overrides (SageMaker passes these as args)
    seed     = args.seed
    run_mode = args.run_mode or config.RUN_MODE
    if args.s3_bucket:
        config.S3_BUCKET = args.s3_bucket

    logging.info("=" * 60)
    logging.info("AGENTIC AI FRAUD DETECTION — Multi-Agent + AWS")
    logging.info(f"seed={seed} | mode={run_mode} | sagemaker={config.IS_SAGEMAKER}")
    logging.info("=" * 60)

    # ──────────────────────────────────────────────────────────
    # STEP 1: LOAD DATASET
    # config.get_dataset_path() returns the right path for
    # local or SageMaker automatically
    # ──────────────────────────────────────────────────────────
    dataset_path = config.get_dataset_path()
    logging.info(f"Dataset path: {dataset_path}")

    # Read raw dataset separately so metadata columns are preserved
    raw_df = pd.read_csv(dataset_path)

    # Exact metadata columns from your dataset
    label_col = "FLAG"
    wallet_col = "Address"
    row_id_col = "Index"
    fallback_row_id_col = "Unnamed: 0"

    # Build metadata table from raw dataset
    df_meta = pd.DataFrame(index=raw_df.index)

    # Wallet identity
    df_meta["from_address"] = raw_df[wallet_col].astype(str)

    # Synthetic tx_hash (because dataset has no real tx_hash column)
    # tx_hash = "tx_" + Index or Unnamed: 0 or row number
    # Example: Index=123 → tx_hash=tx_123
    if row_id_col in raw_df.columns:
        df_meta["tx_hash"] = raw_df[row_id_col].apply(lambda x: f"tx_{x}").astype(str)
    elif fallback_row_id_col in raw_df.columns:
        df_meta["tx_hash"] = raw_df[fallback_row_id_col].apply(lambda x: f"tx_{x}").astype(str)
    else:
        df_meta["tx_hash"] = [f"tx_{i}" for i in range(len(raw_df))]

    # Dataset has no explicit receiver or timestamp
    df_meta["to_address"] = "unknown_to"
    df_meta["timestamp"] = "not_available"

    if not os.path.exists(dataset_path) and not dataset_path.startswith("s3://"):
        logging.error(f"Dataset not found: {dataset_path}")
        return

    df = load_and_clean_data(dataset_path)

    # ──────────────────────────────────────────────────────────
    # STEP 2: SETUP RUN DIRECTORIES
    # ──────────────────────────────────────────────────────────
    run_name = config.make_run_name(seed)

    # In SageMaker: /opt/ml/output/data (auto-uploaded to S3 after job)
    # Locally:      runs/run1/
    if config.IS_SAGEMAKER:
        run_dir = config.SM_OUTPUT_DIR
    else:
        run_dir = os.path.join(config.LOCAL_RESULTS_DIR, f"run_{seed}")

    os.makedirs(run_dir, exist_ok=True)
    logging.info(f"Run: {run_name} | Output dir: {run_dir}")

    # ──────────────────────────────────────────────────────────
    # STEP 3: INITIALIZE AWS COMPONENTS
    # All three degrade gracefully if AWS is not available
    # ──────────────────────────────────────────────────────────
    s3         = S3Manager()
    cw_logger  = CloudWatchLogger(run_name=run_name)
    tracker    = ExperimentTracker(run_name=run_name, run_dir=run_dir, seed=seed)

    # Log all hyperparameters to SM Experiments
    tracker.log_params({
        "seed"          : seed,
        "run_mode"      : run_mode,
        "batch_size"    : config.BATCH_SIZE,
        "rf_trees"      : config.RF_N_ESTIMATORS,
        "if_trees"      : config.IF_N_ESTIMATORS,
        "initial_w"     : config.INITIAL_WEIGHT_W0,
        "initial_tau"   : config.INITIAL_THRESHOLD_TAU0,
        "delta"         : config.BLOCK_MARGIN_DELTA,
        "target_prec"   : config.TARGET_PRECISION,
        "target_rec"    : config.TARGET_RECALL,
        "step_tau"      : config.STEP_SIZE_TAU,
        "step_w"        : config.STEP_SIZE_W,
        "data_split"    : config.DATA_SPLIT_RATIO,
    })

    # Save config to run_dir (will be uploaded to S3 by CoordinatorAgent)
    save_config_json(run_dir)

    # ──────────────────────────────────────────────────────────
    # STEP 4: TRAIN MODELS (one time before streaming loop)
    # ──────────────────────────────────────────────────────────
    logging.info("\n--- Training Phase ---")
    X_train, X_test, y_train, y_test = get_train_test_split(df, seed=seed)
    # Align metadata with test rows using preserved indices
    X_test_meta = df_meta.loc[X_test.index].copy()

    rf_model = RFModel(seed=seed)
    t0 = time.time()
    rf_model.train(X_train, y_train)
    rf_train_time = time.time() - t0
    logging.info(f"RF trained in {rf_train_time:.2f}s")

    # Save feature importance CSV (CoordinatorAgent will upload it)
    rf_imp_path = os.path.join(run_dir, "rf_feature_importance.csv")
    rf_model.log_feature_importance(X_train, output_path=rf_imp_path)

    if_model = IFModel(seed=seed, y_train=y_train)
    t0 = time.time()
    if_model.train(X_train, y_train)
    if_train_time = time.time() - t0
    logging.info(f"IF trained in {if_train_time:.2f}s")

    tracker.log_params({
        "rf_train_time_sec": round(rf_train_time, 4),
        "if_train_time_sec": round(if_train_time, 4),
    })

    # ──────────────────────────────────────────────────────────
    # STEP 5: BUILD MULTI-AGENT SYSTEM
    # CoordinatorAgent wires all 8 agents together and receives
    # the AWS components (s3, cw_logger, tracker)
    # ──────────────────────────────────────────────────────────
    logging.info("\n--- Building Multi-Agent System ---")
    logging.info("  [1] PerceptionAgent  — validates z")
    logging.info("  [2] RFAgent          — p_RF(z)")
    logging.info("  [3] IFAgent          — s_IF(z)")
    logging.info("  [4] FusionAgent      — S(z), decisions")
    logging.info("  [5] ActionAgent      — CLEAR/ALERT/BLOCK")
    logging.info("  [6] MonitoringAgent  — metrics + CloudWatch + SM Experiments")
    logging.info("  [7] AdaptationAgent  — tau, w update + CloudWatch events")
    logging.info("  [8] CoordinatorAgent — orchestrates + S3 upload")

    coordinator = CoordinatorAgent(
        rf_model           = rf_model,
        if_model           = if_model,
        expected_features  = list(X_train.columns),
        run_dir            = run_dir,
        run_name           = run_name,
        seed               = seed,
        cw_logger          = cw_logger,                        # ← AWS CloudWatch
        tracker            = tracker,                          # ← AWS SM Experiments
        s3                 = s3,                               # ← AWS S3
        anthropic_api_key  = os.getenv("ANTHROPIC_API_KEY"),                    # ← Stage 3 LLM
        hardhat_url        = os.getenv("HARDHAT_URL", "http://127.0.0.1:8545"), # ← Stage 3
        registry_address   = os.getenv("CONTRACT_REGISTRY_ADDRESS"),            # ← Stage 3 registry
        governance_address = os.getenv("GOVERNANCE_CONTRACT_ADDRESS"),          # ← Stage 4 governance
        deployer_key       = os.getenv("HARDHAT_DEPLOYER_KEY"),                 # ← Stage 3/4 signing
    )

    # ──────────────────────────────────────────────────────────
    # STEP 6: RUN STREAMING EVALUATION
    # One call → CoordinatorAgent runs all batches internally
    # ──────────────────────────────────────────────────────────
    logging.info("\n--- Streaming Evaluation (agentic batch loop) ---")

    result = coordinator.run(
        AgentMessage(
            sender="main",
            payload={
                "X_test": X_test,
                "y_test": y_test,
                "X_test_meta": X_test_meta,
            },
        )
    )

    if result.status == "error":
        logging.error(f"Pipeline failed: {result.error}")
        return

    # ──────────────────────────────────────────────────────────
    # STEP 7: PRINT SUMMARY
    # ──────────────────────────────────────────────────────────
    history = result.payload["history"]
    final   = result.payload["final_metrics"]

    print("\n" + "="*75)
    print(f"{'RESULTS — ' + run_name:^75}")
    print("="*75)
    print(f"{'Batch':>5} | {'w':>4} | {'tau':>7} | {'P':>7} | {'R':>7} | {'F1':>7} | {'TP':>4} | {'FP':>3} | {'FN':>4}")
    print("-"*75)
    for row in history[:5]:
        print(
            f"{row['batch']:>5} | {row['w']:>4.2f} | "
            f"{row['tau_alert']:>7.3f} | {row['precision']:>7.3f} | "
            f"{row['recall']:>7.3f} | {row['f1']:>7.3f} | "
            f"{row['tp']:>4} | {row['fp']:>3} | {row['fn']:>4}"
        )
    print("="*75)
    if final:
        print(f"mean_P={final.get('mean_precision',0):.4f}  "
              f"mean_R={final.get('mean_recall',0):.4f}  "
              f"mean_F1={final.get('mean_f1',0):.4f}")
        print(f"final_tau={final.get('final_tau_alert',0):.3f}  "
              f"final_w={final.get('final_w',0):.2f}")
    print("="*75 + "\n")

    logging.info("Done.")


if __name__ == "__main__":
    main()