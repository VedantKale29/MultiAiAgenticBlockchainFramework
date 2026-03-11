"""
config.py
Single source of truth for all hyperparameters
for Agentic AI Fraud Detection paper reproduction
"""

# =========================
# Execution
# =========================

DATA_SPLIT_RATIO = 0.75
BATCH_SIZE = 600

import os

N_RUNS = int(os.getenv("N_RUNS", 1))

BASE_SEED = 42

EXPERIMENT_SEEDS = [
    BASE_SEED
    for _ in range(N_RUNS)
]

# EXPERIMENT_SEEDS = [42,44,46,48,40]
# MAX_RUNS = len(EXPERIMENT_SEEDS)

RUN_NAME = "agentic_fraud_exp"

DATASET_FILE_NAME = "transaction_dataset.csv"
# =========================
# Baseline models
# =========================

RF_N_ESTIMATORS = 250
RF_CLASS_WEIGHT = "balanced"

IF_N_ESTIMATORS = 200
IF_MAX_SAMPLES = 2048


# =========================
# Agent initial state
# =========================

INITIAL_WEIGHT_W0 = 0.70
INITIAL_THRESHOLD_TAU0 = 0.487
BLOCK_MARGIN_DELTA = 0.10


# =========================
# Adaptation
# =========================

TARGET_PRECISION = 0.80
TARGET_RECALL = 0.80

STEP_SIZE_TAU = 0.02    # THRESHOLD_STEP
STEP_SIZE_W = 0.05      #  WEIGHT_STEP 


# =========================
# Paths
# =========================

LOCAL_RESULTS_DIR = "runs/"
LOG_DIR = "logs/"

DATASET_FILE_NAME = "transaction_dataset.csv"

S3_BUCKET = "your-s3-bucket-name"
DATASET_URI = f"s3://{S3_BUCKET}/data/ethereum_fraud_dataset.csv"


# =========================
# Flags
# =========================

SAVE_HISTORY = True

RUN_MODE = os.getenv("RUN_MODE", "AGENTIC")

# options:
# AGENTIC
# BASELINE_RF
# BASELINE_IF
# BASELINE_SVM
# BASELINE_XGBOOST
# BASELINE_KMEANS

# =========================================================
# XGBoost BASELINE CONFIG
# =========================================================

XGB_N_ESTIMATORS = 250
XGB_MAX_DEPTH = 6
XGB_LEARNING_RATE = 0.05
XGB_SUBSAMPLE = 0.8
XGB_COLSAMPLE_BYTREE = 0.8
XGB_EVAL_METRIC = "logloss"

# =========================================================
# KMEANS BASELINE CONFIG
# =========================================================

KMEANS_N_CLUSTERS = 2
KMEANS_THRESHOLD_QUANTILE = 0.95

# =========================================================
# SVM BASELINE CONFIG
# =========================================================

SVM_KERNEL = "rbf"
SVM_C = 1.0
SVM_GAMMA = "scale"
SVM_CLASS_WEIGHT = "balanced"
SVM_PROBABILITY = True