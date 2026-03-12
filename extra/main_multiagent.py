"""
main_multiagent.py
==================
ENTRY POINT for the Multi-Agent Fraud Detection System

This replaces the old batch_runner.py approach.

HOW IT WORKS:
--------------
Old approach (batch_runner.py):
  - ONE big function that did everything
  - Called RF, IF, fusion, metrics, adaptation all inline
  - Hard to understand, hard to extend

New approach (multi-agent):
  - main_multiagent.py sets everything up (training, data loading)
  - Creates each specialized agent
  - Hands control to CoordinatorAgent
  - CoordinatorAgent runs the batch pipeline

WHAT THIS FILE DOES:
---------------------
  1. Loads and cleans the Ethereum dataset
  2. Splits into train/test (75:25, seed=42)
  3. Trains Random Forest (one-time, before streaming)
  4. Trains Isolation Forest (one-time, before streaming)
  5. Creates CoordinatorAgent with all 7 worker agents
  6. Runs the streaming evaluation via CoordinatorAgent
  7. Saves batch_history.csv and final_state.json
  8. Prints summary comparison with paper's Table 3/5 values
"""

import os
import json
import time
import pandas as pd

from  logger import logging
import  config as config

# Data loading (unchanged from original project)
from data_loader import load_and_clean_data, get_train_test_split, PAPER_FEATURES

# Models (unchanged from original project)
from  rf_model import RFModel
from  if_model import IFModel

# ── NEW: Multi-Agent imports ───────────────────────────────────
from  agents.coordinator_agent import CoordinatorAgent
from  agents.base_agent import AgentMessage


# ──────────────────────────────────────────────────────────────
# PAPER REFERENCE VALUES (for comparison printout)
# ──────────────────────────────────────────────────────────────
PAPER_TABLE5 = [
    {"batch": 1, "w": 0.70, "tau_alert": 0.487234, "precision": 0.990385, "recall": 0.715278, "f1": 0.830645, "tp": 103, "fp": 1,  "fn": 41, "tn": 455},
    {"batch": 2, "w": 0.70, "tau_alert": 0.467234, "precision": 0.971429, "recall": 0.790698, "f1": 0.871795, "tp": 102, "fp": 3,  "fn": 27, "tn": 468},
    {"batch": 3, "w": 0.70, "tau_alert": 0.447234, "precision": 0.990476, "recall": 0.781955, "f1": 0.873950, "tp": 104, "fp": 1,  "fn": 29, "tn": 466},
    {"batch": 4, "w": 0.70, "tau_alert": 0.427234, "precision": 0.990476, "recall": 0.838710, "f1": 0.908297, "tp": 104, "fp": 1,  "fn": 20, "tn": 475},
    {"batch": 5, "w": 0.70, "tau_alert": 0.427234, "precision": 1.000000, "recall": 0.800000, "f1": 0.888889, "tp": 12,  "fp": 0,  "fn": 3,  "tn": 46},
]


def print_comparison_table(history: list):
    """
    Print a side-by-side comparison of our results vs the paper's Table 5.
    This makes it easy to verify reproduction accuracy.
    """
    print("\n" + "="*90)
    print("REPRODUCTION COMPARISON vs PAPER TABLE 5")
    print("="*90)

    header = f"{'Batch':>5} | {'w':>5} | {'tau':>8} | {'Prec':>8} | {'Rec':>8} | {'F1':>8} | {'TP':>4} | {'FP':>4} | {'FN':>4} | {'TN':>4}"
    print(f"\n{'OUR RESULTS':^90}")
    print(header)
    print("-"*90)
    for row in history[:5]:
        print(
            f"{row['batch']:>5} | "
            f"{row['w']:>5.2f} | "
            f"{row['tau_alert']:>8.6f} | "
            f"{row['precision']:>8.6f} | "
            f"{row['recall']:>8.6f} | "
            f"{row['f1']:>8.6f} | "
            f"{row['tp']:>4} | "
            f"{row['fp']:>4} | "
            f"{row['fn']:>4} | "
            f"{row['tn']:>4}"
        )

    print(f"\n{'PAPER TABLE 5':^90}")
    print(header)
    print("-"*90)
    for row in PAPER_TABLE5:
        print(
            f"{row['batch']:>5} | "
            f"{row['w']:>5.2f} | "
            f"{row['tau_alert']:>8.6f} | "
            f"{row['precision']:>8.6f} | "
            f"{row['recall']:>8.6f} | "
            f"{row['f1']:>8.6f} | "
            f"{row['tp']:>4} | "
            f"{row['fp']:>4} | "
            f"{row['fn']:>4} | "
            f"{row['tn']:>4}"
        )
    print("="*90 + "\n")


def main():
    logging.info("="*60)
    logging.info("MULTI-AGENT BLOCKCHAIN FRAUD DETECTION")
    logging.info("Paper: Agentic AI Framework with Hybrid Classifiers")
    logging.info("="*60)

    # ──────────────────────────────────────────────────────────
    # PHASE 1: DATA LOADING
    # ──────────────────────────────────────────────────────────
    dataset_path = config.DATASET_FILE_NAME
    if not os.path.exists(dataset_path):
        logging.error(f"Dataset not found: {dataset_path}")
        return

    logging.info("\n--- PHASE 1: Data Loading ---")
    df = load_and_clean_data(dataset_path)

    seed = config.BASE_SEED  # 42 (paper requirement)
    X_train, X_test, y_train, y_test = get_train_test_split(df, seed=seed)

    # Feature columns (everything except FLAG)
    expected_features = list(X_train.columns)
    logging.info(f"Features: {expected_features}")

    # ──────────────────────────────────────────────────────────
    # PHASE 2: MODEL TRAINING (happens ONCE before streaming)
    # ──────────────────────────────────────────────────────────
    logging.info("\n--- PHASE 2: Training Models ---")

    # Train Random Forest
    rf_model = RFModel(seed=seed)
    t0 = time.time()
    rf_model.train(X_train, y_train)
    rf_train_time = time.time() - t0
    logging.info(f"RF trained in {rf_train_time:.2f}s")

    # Train Isolation Forest
    if_model = IFModel(seed=seed, y_train=y_train)
    t0 = time.time()
    if_model.train(X_train, y_train)
    if_train_time = time.time() - t0
    logging.info(f"IF trained in {if_train_time:.2f}s")

    # ──────────────────────────────────────────────────────────
    # PHASE 3: BUILD MULTI-AGENT SYSTEM
    # ──────────────────────────────────────────────────────────
    logging.info("\n--- PHASE 3: Building Multi-Agent System ---")
    logging.info("Creating agents:")
    logging.info("  [1] PerceptionAgent  — validates state vector z")
    logging.info("  [2] RFAgent          — supervised detector  ->> p_RF(z)")
    logging.info("  [3] IFAgent          — anomaly detector  ->> s_IF(z)")
    logging.info("  [4] FusionAgent      — S(z) = w*p_RF + (1-w)*s_IF")
    logging.info("  [5] ActionAgent      — CLEAR / ALERT / AUTO-BLOCK")
    logging.info("  [6] MonitoringAgent  — metrics + batch log")
    logging.info("  [7] AdaptationAgent  — updates tau, tau_block, w")
    logging.info("  [8] CoordinatorAgent — orchestrates all the above")

    coordinator = CoordinatorAgent(
        rf_model=rf_model,
        if_model=if_model,
        expected_features=expected_features,
    )

    # ──────────────────────────────────────────────────────────
    # PHASE 4: RUN STREAMING EVALUATION
    # ──────────────────────────────────────────────────────────
    logging.info("\n--- PHASE 4: Running Streaming Evaluation ---")
    logging.info(f"Seed={seed} | Batches=~{len(X_test)//config.BATCH_SIZE+1} | "
                 f"Batch size={config.BATCH_SIZE}")

    # This single call triggers the entire agent pipeline
    # CoordinatorAgent handles the batch loop internally
    result_msg = coordinator.run(
        AgentMessage(
            sender="main",
            payload={
                "X_test": X_test,
                "y_test": y_test,
            },
        )
    )

    if result_msg.status == "error":
        logging.error(f"Pipeline failed: {result_msg.error}")
        return

    history     = result_msg.payload["history"]
    final_state = result_msg.payload["final_state"]

    # ──────────────────────────────────────────────────────────
    # PHASE 5: SAVE RESULTS
    # ──────────────────────────────────────────────────────────
    logging.info("\n--- PHASE 5: Saving Results ---")

    os.makedirs("runs/run_multiagent", exist_ok=True)

    # Save batch history CSV
    history_df = pd.DataFrame(history)
    history_path = "runs/run_multiagent/batch_history.csv"
    history_df.to_csv(history_path, index=False)
    logging.info(f"Batch history saved  ->> {history_path}")

    # Save final agent state
    state_path = "runs/run_multiagent/final_state.json"
    with open(state_path, "w") as f:
        json.dump(final_state, f, indent=4)
    logging.info(f"Final state saved  ->> {state_path}")

    # ──────────────────────────────────────────────────────────
    # PHASE 6: PRINT SUMMARY
    # ──────────────────────────────────────────────────────────
    logging.info("\n--- PHASE 6: Results Summary ---")

    # Print comparison with paper
    print_comparison_table(history)

    # Print mean metrics across all batches
    if history:
        mean_prec = sum(r["precision"] for r in history) / len(history)
        mean_rec  = sum(r["recall"]    for r in history) / len(history)
        mean_f1   = sum(r["f1"]        for r in history) / len(history)

        print(f"Mean Precision across all batches: {mean_prec:.6f}")
        print(f"Mean Recall    across all batches: {mean_rec:.6f}")
        print(f"Mean F1        across all batches: {mean_f1:.6f}")
        print(f"Final State: {final_state}")

    logging.info("Multi-Agent Pipeline Complete!")


if __name__ == "__main__":
    main()
