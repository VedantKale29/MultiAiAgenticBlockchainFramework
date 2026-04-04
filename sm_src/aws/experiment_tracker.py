"""
aws/experiment_tracker.py
==========================
EXPERIMENT TRACKER -- replaces your existing LocalExperimentTracker.

WHAT IT DOES:
  Tracks every run's parameters, per-batch metrics, and final results.
  Works in TWO modes depending on environment:

  Mode 1 -- SageMaker Experiments (when running in AWS):
    Metrics appear as live charts in the SageMaker Console.
    You can compare run_seed42 vs run_seed7 side-by-side visually.
    Results are stored permanently in AWS.

  Mode 2 -- Local JSON fallback (when running on your laptop):
    Same behavior as your original LocalExperimentTracker.
    Saves experiment_summary.json in the run directory.

THE KEY DIFFERENCE FROM YOUR ORIGINAL LocalExperimentTracker:
  Original → saves one JSON file, only visible if you open the file.
  New      → saves to SageMaker Experiments UI + JSON file.
             You can see all 5 seeds compared in one table in the UI.

INTERFACE IS THE SAME as your original LocalExperimentTracker:
  tracker.log_params(...)
  tracker.log_final_metrics(...)
  tracker.log_artifact(...)
  tracker.save() / tracker.finish()
So your existing main.py calls work without any changes.

CALLED BY:
  CoordinatorAgent (which replaced batch_runner.py)
"""

import os
import json
import logging
from datetime import datetime
import  config as config

try:
    import sagemaker
    from sagemaker.experiments.run import Run
    from sagemaker.session import Session
    SM_AVAILABLE = True
except ImportError:
    SM_AVAILABLE = False


class ExperimentTracker:
    """
    Unified experiment tracker.
    In SageMaker → uses SageMaker Experiments API.
    Locally       → saves to experiment_summary.json (same as before).
    """

    def __init__(self, run_name: str, run_dir: str, seed: int = 42):
        self.run_name = run_name
        self.run_dir  = run_dir
        self.seed     = seed
        self.logger   = logging.getLogger("ExperimentTracker")
        self._sm_run  = None
        self._batch_metrics = []  # stores per-batch records

        # Local data (always maintained -- this is the fallback)
        self._data = {
            "run_name"    : run_name,
            "seed"        : seed,
            "started_at"  : datetime.now().isoformat(),
            "params"      : {},
            "batch_metrics": [],
            "final_metrics": {},
            "artifacts"   : {},
        }

        # Try SageMaker Experiments (only available in SageMaker environment)
        if SM_AVAILABLE and config.IS_SAGEMAKER:
            try:
                self._sm_run = Run(
                    experiment_name=config.EXPERIMENT_NAME,
                    run_name=run_name,
                    sagemaker_session=Session(),
                )
                self.logger.info(
                    f"[ExperimentTracker] SageMaker Experiments active: "
                    f"experiment={config.EXPERIMENT_NAME} run={run_name}"
                )
            except Exception as e:
                self.logger.warning(
                    f"[ExperimentTracker] SageMaker Experiments unavailable: {e}"
                )
        else:
            self.logger.info(
                f"[ExperimentTracker] Local mode → {run_dir}/experiment_summary.json"
            )

    # ─────────────────────────────────────────────────
    # PARAMETER LOGGING
    # ─────────────────────────────────────────────────

    def log_params(self, params: dict):
        """Log hyperparameters. Called once at start of each run."""
        self._data["params"].update(params)

        if self._sm_run:
            for k, v in params.items():
                try:
                    self._sm_run.log_parameter(k, str(v))
                except Exception:
                    pass

        self.logger.info(f"[ExperimentTracker] Params: {params}")

    # ─────────────────────────────────────────────────
    # METRIC LOGGING
    # ─────────────────────────────────────────────────

    def log_batch_metrics(
        self,
        batch     : int,
        precision : float,
        recall    : float,
        f1        : float,
        tau_alert : float = None,
        w         : float = None,
        tp: int = 0, fp: int = 0, fn: int = 0, tn: int = 0,
        roc_auc: float = None,
        pr_ap:   float = None,
    ):
        """
        Log metrics for one batch.
        In SageMaker, batch number = x-axis on the metrics chart.
        So you'll see precision/recall curves over batches in the UI.
        """
        record = {
            "batch": batch, "precision": round(precision, 6),
            "recall": round(recall, 6), "f1": round(f1, 6),
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        }
        if tau_alert is not None: record["tau_alert"] = round(tau_alert, 6)
        if w         is not None: record["w"]         = round(w, 4)
        if roc_auc   is not None: record["roc_auc"]   = round(roc_auc, 6)
        if pr_ap     is not None: record["pr_ap"]     = round(pr_ap, 6)

        self._data["batch_metrics"].append(record)

        if self._sm_run:
            try:
                # batch = step → draws a time-series chart in SageMaker UI
                self._sm_run.log_metric("precision", precision, step=batch)
                self._sm_run.log_metric("recall",    recall,    step=batch)
                self._sm_run.log_metric("f1",        f1,        step=batch)
                if tau_alert is not None:
                    self._sm_run.log_metric("tau_alert", tau_alert, step=batch)
                if w is not None:
                    self._sm_run.log_metric("rf_weight", w, step=batch)
            except Exception:
                pass

    def log_final_metrics(self, metrics: dict):
        """Log aggregate metrics at end of run (mean_f1, mean_precision, etc.)"""
        self._data["final_metrics"].update(metrics)

        if self._sm_run:
            for k, v in metrics.items():
                if isinstance(v, (int, float)) and v is not None:
                    try:
                        self._sm_run.log_metric(f"final_{k}", float(v))
                    except Exception:
                        pass

        self.logger.info(f"[ExperimentTracker] Final metrics: {metrics}")

    # ─────────────────────────────────────────────────
    # ARTIFACT LOGGING
    # ─────────────────────────────────────────────────

    def log_artifact(self, name: str, path: str):
        """
        Log a file artifact (local path or S3 URI).
        In SageMaker UI, these appear as clickable links.
        """
        self._data["artifacts"][name] = path

        if self._sm_run:
            try:
                media = "text/csv" if path.endswith(".csv") else "text/plain"
                self._sm_run.log_artifact(name=name, value=path, media_type=media)
            except Exception:
                pass

    # ─────────────────────────────────────────────────
    # SAVE / FINISH
    # ─────────────────────────────────────────────────

    def finish(self):
        """
        Finalize the run. Always call at end of training.
        Saves local JSON and closes SageMaker run.
        """
        self._data["finished_at"] = datetime.now().isoformat()
        os.makedirs(self.run_dir, exist_ok=True)

        meta_path = os.path.join(self.run_dir, "experiment_summary.json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(self._data, f, indent=4)

        self.logger.info(f"[ExperimentTracker] Saved → {meta_path}")

        if self._sm_run:
            try:
                self._sm_run.__exit__(None, None, None)
            except Exception:
                pass

    # Alias for backward compatibility with old LocalExperimentTracker.save()
    def save(self):
        self.finish()

    # Old interface compatibility
    def log_note(self, text: str):
        self._data.setdefault("notes", []).append(text)