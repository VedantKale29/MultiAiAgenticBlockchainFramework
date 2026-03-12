import json
import os
import numpy as np
from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
)
from  logger import logging


def compute_latency(start_time, end_time, batch_size):
    total_time = max(0.0, end_time - start_time)
    latency = total_time / batch_size if batch_size > 0 else 0.0

    logging.info(
        f"Latency -> total={total_time:.6f}s "
        f"per_tx={latency:.6f}s "
        f"batch={batch_size}"
    )
    return latency


def save_baseline_metrics(
    y_true,
    y_pred,
    y_score,
    run_dir,
    tracker,
    model_name,
    train_time_sec,
    infer_time_sec,
):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    if y_score is not None and len(set(y_true)) > 1:
        roc_auc = roc_auc_score(y_true, y_score)
        pr_ap = average_precision_score(y_true, y_score)
    else:
        roc_auc = None
        pr_ap = None

    metrics = {
        "model": model_name,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "accuracy": float(accuracy),
        "roc_auc": None if roc_auc is None else float(roc_auc),
        "pr_ap": None if pr_ap is None else float(pr_ap),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
        "train_time_sec": float(train_time_sec),
        "infer_time_sec": float(infer_time_sec),
    }

    metrics_path = os.path.join(run_dir, f"{model_name.lower()}_metrics.json")

    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)

    tracker.log_artifact(f"{model_name.lower()}_metrics", metrics_path)
    tracker.log_final_metrics(metrics)

    roc_text = "None" if roc_auc is None else f"{roc_auc:.4f}"
    pr_text = "None" if pr_ap is None else f"{pr_ap:.4f}"

    logging.info(
        f"{model_name} metrics: "
        f"Accuracy={accuracy:.4f}, "
        f"Precision={precision:.4f}, "
        f"Recall={recall:.4f}, "
        f"F1={f1:.4f}, "
        f"ROC AUC={roc_text}, "
        f"PR AP={pr_text}"
    )

    logging.info(f"{model_name} metrics saved -> {metrics_path}")

    return metrics, metrics_path


def decisions_to_binary(decisions):
    """
    Treat ALERT and AUTO-BLOCK as positive/fraud actions.
    CLEAR is negative.
    """
    return np.array(
        [1 if d in ("ALERT", "AUTO-BLOCK") else 0 for d in decisions],
        dtype=int,
    )


def compute_batch_metrics(y_true, decisions):
    y_true = np.asarray(y_true).astype(int)
    y_pred = decisions_to_binary(decisions)

    TP = int(np.sum((y_true == 1) & (y_pred == 1)))
    FP = int(np.sum((y_true == 0) & (y_pred == 1)))
    FN = int(np.sum((y_true == 1) & (y_pred == 0)))
    TN = int(np.sum((y_true == 0) & (y_pred == 0)))

    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    logging.info(
        f"Batch metrics -> "
        f"TP={TP} FP={FP} FN={FN} TN={TN} "
        f"P={precision:.3f} R={recall:.3f} F1={f1:.3f}"
    )

    return TP, FP, FN, TN, precision, recall, f1


def extract_tp_scores(y_true, decisions, p_rf, s_if):
    y_true = np.asarray(y_true).astype(int)
    y_pred = decisions_to_binary(decisions)

    tp_mask = (y_true == 1) & (y_pred == 1)

    if np.sum(tp_mask) == 0:
        logging.info("No TP in batch")

    return p_rf[tp_mask], s_if[tp_mask]


def compute_global_metrics(y_true, risk_scores):
    y_true = np.asarray(y_true).astype(int)
    risk_scores = np.asarray(risk_scores)

    if len(np.unique(y_true)) > 1:
        roc_auc = roc_auc_score(y_true, risk_scores)
        pr_ap = average_precision_score(y_true, risk_scores)
    else:
        roc_auc = float("nan")
        pr_ap = float("nan")

    logging.info(f"Global metrics -> ROC={roc_auc} PR_AP={pr_ap}")
    return roc_auc, pr_ap