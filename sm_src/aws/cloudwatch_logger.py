"""
aws/cloudwatch_logger.py
========================
CLOUDWATCH LOGGER — logs batch metrics to AWS CloudWatch.

HOW CLOUDWATCH LOGGING WORKS IN SAGEMAKER:
  When your code runs in a SageMaker Training Job, Python's
  StreamHandler (stdout) is automatically captured by SageMaker
  and sent to CloudWatch Logs. You don't need to configure anything.

  Your existing logger.py already has a StreamHandler, so your
  current logs ALREADY appear in CloudWatch when running in SageMaker.

WHAT THIS CLASS ADDS ON TOP:
  Structured log lines with a consistent format that can be searched
  and filtered in CloudWatch. For example, you can search for:
    "precision < 0.8"   → find all batches where precision was low
    "threshold_lowered" → find all adaptation events

FORMAT OF LOG LINES (what you see in CloudWatch):
  [BATCH][run_seed42_v1][B=1] P=0.990 R=0.715 F1=0.831 tau=0.487 w=0.70 TP=103 FP=1
  [ADAPT][run_seed42_v1][B=1] threshold_lowered: 0.487 → 0.467 (recall_below_target)
  [DONE ][run_seed42_v1] mean_P=0.989 mean_R=0.779 mean_F1=0.871

GRACEFUL FALLBACK:
  If not in SageMaker or boto3 not available, logs go to the
  standard Python logger (your terminal / log file). Nothing breaks.
"""

import json
import logging
from datetime import datetime, timezone
import  config as config

try:
    import boto3
    BOTO3_AVAILABLE = True
except ImportError:
    BOTO3_AVAILABLE = False


class CloudWatchLogger:

    def __init__(self, run_name: str):
        self.run_name  = run_name
        self.logger    = logging.getLogger("CWLogger")
        self._client   = None
        self._token    = None  # CloudWatch sequence token

        # Only attempt direct CloudWatch API if boto3 is available
        if BOTO3_AVAILABLE and config.IS_SAGEMAKER:
            try:
                self._client = boto3.client("logs", region_name=config.AWS_REGION)
                self._ensure_log_group()
                logging.info(f"[CloudWatchLogger] Connected → group={config.CLOUDWATCH_LOG_GROUP}")
            except Exception as e:
                logging.info(f"[CloudWatchLogger] Direct API unavailable: {e} — using stdout only")

    # ─────────────────────────────────────────────────
    # PUBLIC METHODS (called by MonitoringAgent and AdaptationAgent)
    # ─────────────────────────────────────────────────

    def log_batch(self, batch_idx: int, metrics: dict, seed: int = 42):
        """
        Log per-batch metrics. Called by MonitoringAgent after each batch.

        What gets logged:
          Precision, Recall, F1, tau_alert, w, TP, FP, FN, TN
        """
        # ── 1. Human-readable log line (always written) ───────────
        # This appears in: local terminal, log file, AND CloudWatch
        # (SageMaker captures stdout → CloudWatch automatically)
        self.logger.info(
            f"[BATCH][{self.run_name}][B={batch_idx+1}] "
            f"P={metrics.get('precision', 0):.3f} "
            f"R={metrics.get('recall', 0):.3f} "
            f"F1={metrics.get('f1', 0):.3f} "
            f"tau={metrics.get('tau_alert', 0):.3f} "
            f"w={metrics.get('w', 0):.2f} "
            f"TP={metrics.get('tp', 0)} "
            f"FP={metrics.get('fp', 0)} "
            f"FN={metrics.get('fn', 0)} "
            f"TN={metrics.get('tn', 0)}"
        )

        # ── 2. Structured JSON to CloudWatch Logs API (optional) ──
        # This enables filtering like: {$.precision < 0.8}
        self._push(
            stream=f"{self.run_name}/batches",
            payload={
                "event"    : "batch_complete",
                "run_name" : self.run_name,
                "seed"     : seed,
                "batch"    : batch_idx + 1,
                "timestamp": self._now(),
                **metrics,
            }
        )

    def log_adaptation(
        self,
        batch_idx : int,
        event     : str,
        old_tau   : float,
        new_tau   : float,
        old_w     : float,
        new_w     : float,
        reason    : str,
    ):
        """
        Log each adaptation event. Called by AdaptationAgent.

        event  examples: "threshold_lowered", "threshold_raised", "weight_decreased"
        reason examples: "recall_below_target", "fp_dominated"
        """
        self.logger.info(
            f"[ADAPT][{self.run_name}][B={batch_idx+1}] "
            f"{event}: tau {old_tau:.3f}→{new_tau:.3f} "
            f"w {old_w:.2f}→{new_w:.2f} | {reason}"
        )

        self._push(
            stream=f"{self.run_name}/adaptation",
            payload={
                "event"    : "adaptation",
                "sub_event": event,
                "run_name" : self.run_name,
                "batch"    : batch_idx + 1,
                "timestamp": self._now(),
                "old_tau"  : round(old_tau, 4),
                "new_tau"  : round(new_tau, 4),
                "old_w"    : round(old_w, 4),
                "new_w"    : round(new_w, 4),
                "reason"   : reason,
            }
        )

    def log_run_summary(self, final_metrics: dict, seed: int):
        """
        Log the final summary at end of a run. Called by CoordinatorAgent.
        """
        self.logger.info(
            f"[DONE ][{self.run_name}] "
            f"mean_P={final_metrics.get('mean_precision', 0):.4f} "
            f"mean_R={final_metrics.get('mean_recall', 0):.4f} "
            f"mean_F1={final_metrics.get('mean_f1', 0):.4f} "
            f"final_tau={final_metrics.get('final_tau_alert', 0):.3f} "
            f"final_w={final_metrics.get('final_w', 0):.2f}"
        )

        self._push(
            stream=f"{self.run_name}/summary",
            payload={
                "event"         : "run_complete",
                "run_name"      : self.run_name,
                "seed"          : seed,
                "timestamp"     : self._now(),
                "final_metrics" : final_metrics,
            }
        )

    # ─────────────────────────────────────────────────
    # INTERNAL
    # ─────────────────────────────────────────────────

    def _push(self, stream: str, payload: dict):
        """Push structured JSON to CloudWatch Logs API. Silently skips if unavailable."""
        if not self._client:
            return
        try:
            self._ensure_stream(stream)
            kwargs = {
                "logGroupName" : config.CLOUDWATCH_LOG_GROUP,
                "logStreamName": stream,
                "logEvents"    : [{
                    "timestamp": int(datetime.now(timezone.utc).timestamp() * 1000),
                    "message"  : json.dumps(payload),
                }],
            }
            if self._token:
                kwargs["sequenceToken"] = self._token
            resp = self._client.put_log_events(**kwargs)
            self._token = resp.get("nextSequenceToken")
        except Exception:
            pass  # Never crash the pipeline due to logging

    def _ensure_log_group(self):
        try:
            self._client.create_log_group(logGroupName=config.CLOUDWATCH_LOG_GROUP)
        except Exception:
            pass

    def _ensure_stream(self, stream: str):
        try:
            self._client.create_log_stream(
                logGroupName=config.CLOUDWATCH_LOG_GROUP,
                logStreamName=stream,
            )
        except Exception:
            pass

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat()