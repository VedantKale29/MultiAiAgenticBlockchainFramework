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

# Original fixed step sizes (kept for reference / paper reproduction mode)
STEP_SIZE_TAU = 0.02
STEP_SIZE_W   = 0.05
 
TARGET_PRECISION = 0.80
TARGET_RECALL    = 0.80
 
# ══════════════════════════════════════════════════════
# PI CONTROLLER SETTINGS  (Extension #8 — v2)
# ══════════════════════════════════════════════════════
#
# Set USE_PI_ADAPTATION = False to revert to the paper's fixed steps.
USE_PI_ADAPTATION = True
 
# ── TAU gains ────────────────────────────────────────────────────
#
# K_p derivation: K_p ≈ old_fixed_step / typical_gap
#   old_fixed = 0.02,  typical recall gap ≈ 0.06
#   → K_p = 0.02 / 0.06 ≈ 0.33  →  use 0.30 (conservative)
#
# v1 used 0.10 → produced steps of only 0.005 (4× too small)
# v2 uses 0.30 → produces steps of ~0.018 (≈ paper's 0.02)
#
PI_KP_TAU = float(os.getenv("PI_KP_TAU", 0.30))   # ← v2 CHANGED (was 0.10)
PI_KI_TAU = float(os.getenv("PI_KI_TAU", 0.02))   # ← v2 CHANGED (was 0.01)
 
# ── WEIGHT gains ─────────────────────────────────────────────────
PI_KP_W = float(os.getenv("PI_KP_W", 0.15))        # ← v2 CHANGED (was 0.12)
PI_KI_W = float(os.getenv("PI_KI_W", 0.02))        # ← v2 CHANGED (was 0.01)
 
# ── Anti-windup ───────────────────────────────────────────────────
PI_MAX_INTEGRAL = float(os.getenv("PI_MAX_INTEGRAL", 1.0))
 
# ── Tau step bounds ───────────────────────────────────────────────
#
# MIN_TAU_STEP = 0.015: floor ensures controller never stalls near
#   target. Paper used 0.02 fixed; 0.015 gives headroom.
# MAX_TAU_STEP = 0.10: allows slightly faster recovery on first batch
#   where recall may be very low.
#
PI_MIN_TAU_STEP = float(os.getenv("PI_MIN_TAU_STEP", 0.015))  # ← v2 CHANGED (was 0.005)
PI_MAX_TAU_STEP = float(os.getenv("PI_MAX_TAU_STEP", 0.10))   # ← v2 CHANGED (was 0.08)
 
# ── Weight movement threshold ─────────────────────────────────────
#
# Gap difference must exceed this before w shifts.
# 0.03 (lowered from 0.05): weight now responds when gaps differ by 3%,
# giving more opportunity to leverage IF's recall sensitivity.
#
PI_WEIGHT_MOVE_THRESHOLD = float(os.getenv("PI_WEIGHT_MOVE_THRESHOLD", 0.03))  # ← v2 CHANGED (was 0.05)
 
# ── Recall priority bias ──────────────────────────────────────────
#
# Encodes the asymmetric cost of FN vs FP in fraud detection.
# In AML/blockchain: missing a fraud (FN) is far more costly than a
# false alarm (FP). We encode a 1.5:1 cost ratio by default.
#
# How it works:
#   rec_gap_w  = PI_RECALL_WEIGHT × rec_gap   →  bigger steps for recall
#   prec_gap_w = PI_PREC_WEIGHT   × prec_gap  →  normal steps for precision
#
# The weighted gaps feed into BOTH tau and weight update logic.
#
# Tuning guide:
#   Set both to 1.0 → symmetric mode (paper-equivalent behaviour)
#   Increase RECALL_WEIGHT → more aggressive recall recovery
#   Decrease RECALL_WEIGHT → more balanced precision/recall treatment
#
# Typical values by use case:
#   AML / financial fraud detection:  1.5 – 3.0  (FN very costly)
#   Spam filtering:                   1.0 – 1.2  (near symmetric)
#   Medical diagnosis:                2.0 – 5.0  (missing disease = catastrophic)
#
PI_RECALL_WEIGHT = float(os.getenv("PI_RECALL_WEIGHT", 1.5))  # ← v2 NEW
PI_PREC_WEIGHT   = float(os.getenv("PI_PREC_WEIGHT",   1.0))  # ← v2 NEW

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

# Policy / response settings
POLICY_ALERT_ESCALATION_THRESHOLD = int(os.getenv("POLICY_ALERT_ESCALATION_THRESHOLD", 3))
SAVE_WATCHLIST = True
SAVE_BLOCKLIST = True
SAVE_FRAUD_EVENTS = True