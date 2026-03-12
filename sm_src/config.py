"""
config.py
=========
SINGLE SOURCE OF TRUTH for all project settings.

Two sections:
  1. ML Hyperparameters  — unchanged from the paper
  2. AWS Settings        — S3 bucket, SageMaker role, region

HOW ENVIRONMENT DETECTION WORKS:
  When your code runs inside a SageMaker Training Job container,
  SageMaker creates the folder /opt/ml/input automatically.
  We check for that folder to know if we're in AWS or local.

  IS_SAGEMAKER = True  → running in SageMaker → read dataset from /opt/ml/input
  IS_SAGEMAKER = False → running locally       → read dataset from current folder
"""

import os

# ══════════════════════════════════════════════════════
# ENVIRONMENT DETECTION
# ══════════════════════════════════════════════════════

# True when running inside a SageMaker Training Job container
IS_SAGEMAKER = os.path.exists("/opt/ml/input")

# ══════════════════════════════════════════════════════
# ML HYPERPARAMETERS  (paper values — do not change)
# ══════════════════════════════════════════════════════

DATA_SPLIT_RATIO = 0.75
BATCH_SIZE       = 600
BASE_SEED        = 42
N_RUNS           = int(os.getenv("N_RUNS", 1))

EXPERIMENT_SEEDS = [BASE_SEED for _ in range(N_RUNS)]

# Random Forest
RF_N_ESTIMATORS = 250
RF_CLASS_WEIGHT = "balanced"

# Isolation Forest
IF_N_ESTIMATORS = 200
IF_MAX_SAMPLES  = 2048

# Agent initial state
INITIAL_WEIGHT_W0      = 0.70
INITIAL_THRESHOLD_TAU0 = 0.487
BLOCK_MARGIN_DELTA     = 0.10

# Adaptation targets and step sizes
TARGET_PRECISION = 0.80
TARGET_RECALL    = 0.80
STEP_SIZE_TAU    = 0.02
STEP_SIZE_W      = 0.05

# Baseline model settings (XGBoost, KMeans, SVM)
XGB_N_ESTIMATORS    = 250
XGB_MAX_DEPTH       = 6
XGB_LEARNING_RATE   = 0.05
XGB_SUBSAMPLE       = 0.8
XGB_COLSAMPLE_BYTREE= 0.8
XGB_EVAL_METRIC     = "logloss"
KMEANS_N_CLUSTERS       = 2
KMEANS_THRESHOLD_QUANTILE = 0.95
SVM_KERNEL       = "rbf"
SVM_C            = 1.0
SVM_GAMMA        = "scale"
SVM_CLASS_WEIGHT = "balanced"
SVM_PROBABILITY  = True

# ══════════════════════════════════════════════════════
# PATHS
# ══════════════════════════════════════════════════════

DATASET_FILE_NAME = "transaction_dataset.csv"
LOCAL_RESULTS_DIR = "runs/"
LOG_DIR           = "logs/"
RUN_MODE          = os.getenv("RUN_MODE", "AGENTIC")
SAVE_HISTORY      = True

# ══════════════════════════════════════════════════════
# AWS SETTINGS
# ══════════════════════════════════════════════════════

# ── S3 ────────────────────────────────────────────────
# Change S3_BUCKET to your real bucket name before running on AWS
S3_BUCKET  = os.getenv("S3_BUCKET", "multiagenticblockchain")
S3_PREFIX  = os.getenv("S3_PREFIX", "agentic-fraud")

# Derived S3 paths (do not change these)
S3_DATA_PREFIX = f"{S3_PREFIX}/data"
S3_RUNS_PREFIX = f"{S3_PREFIX}/runs"
S3_LOGS_PREFIX = f"{S3_PREFIX}/logs"
DATASET_URI    = f"s3://{S3_BUCKET}/{S3_DATA_PREFIX}/{DATASET_FILE_NAME}"

# ── SageMaker ─────────────────────────────────────────
# Change SAGEMAKER_ROLE to your real IAM role ARN
SAGEMAKER_ROLE     = os.getenv(
    "SAGEMAKER_ROLE",
    "arn:aws:iam::551222650550:role/MultiAgenticBlockchain"
)
AWS_REGION         = os.getenv("AWS_REGION", "us-east-1")
INSTANCE_TYPE      = os.getenv("SAGEMAKER_INSTANCE", "ml.t3.large")
EXPERIMENT_NAME    = os.getenv("EXPERIMENT_NAME", "multi-agentic-fraud-detection")

# ── SageMaker container paths (do not change) ─────────
SM_INPUT_DIR  = "/opt/ml/input/data/training"
SM_OUTPUT_DIR = "/opt/ml/output/data"

# ── CloudWatch ────────────────────────────────────────
CLOUDWATCH_LOG_GROUP = "/fraud-detection/agentic-runs"

# ══════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ══════════════════════════════════════════════════════

def get_dataset_path() -> str:
    """
    Returns the correct dataset path for the current environment.

    Local:      transaction_dataset.csv  (current directory)
    SageMaker:  /opt/ml/input/data/training/transaction_dataset.csv
    """
    if IS_SAGEMAKER:
        return f"{SM_INPUT_DIR}/{DATASET_FILE_NAME}"
    if os.path.exists(DATASET_FILE_NAME):
        return DATASET_FILE_NAME
    return DATASET_URI  # fallback to S3 URI


def get_output_dir() -> str:
    """
    Returns the correct output directory for the current environment.

    Local:      runs/run1/
    SageMaker:  /opt/ml/output/data/   (auto-uploaded to S3 by SageMaker)
    """
    if IS_SAGEMAKER:
        return SM_OUTPUT_DIR
    return LOCAL_RESULTS_DIR


def make_run_name(seed: int, version: int = 1) -> str:
    """
    Consistent run name used in both local folders and SageMaker Experiments.
    Example: make_run_name(42) → "run_seed42_v1"
    """
    return f"run_seed{seed}_v{version}"