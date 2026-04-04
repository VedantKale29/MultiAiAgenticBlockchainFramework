"""
aws/s3_manager.py
=================
S3 MANAGER -- uploads and downloads files from Amazon S3.

WHAT IT DOES IN THIS PROJECT:
  1. upload_dataset()      -- upload CSV once before training starts
  2. upload_rag_store()  -- upload everything in a run folder after training
  3. download_dataset()    -- download CSV at start of SageMaker job

WHEN EACH IS CALLED:
  From your laptop (before first run):
      python -c "from aws.s3_manager import S3Manager; S3Manager().upload_dataset('transaction_dataset.csv')"

  From CoordinatorAgent (after each run finishes):
      s3.upload_rag_store(run_dir="runs/run1", run_name="run_seed42_v1")

GRACEFUL FALLBACK:
  If boto3 is not installed or no AWS credentials are configured,
  all methods return False silently. Your pipeline still runs locally.
"""

import os
import logging
import  config as config

try:
    import boto3
    from botocore.exceptions import ClientError, NoCredentialsError
    BOTO3_AVAILABLE = True
except ImportError:
    BOTO3_AVAILABLE = False


class S3Manager:

    def __init__(self):
        self.bucket      = config.S3_BUCKET
        self.data_prefix = config.S3_DATA_PREFIX
        self.runs_prefix = config.S3_RUNS_PREFIX
        self.logs_prefix = config.S3_LOGS_PREFIX
        self._client     = None

        if BOTO3_AVAILABLE:
            try:
                self._client = boto3.client("s3", region_name=config.AWS_REGION)
                logging.info(f"[S3Manager] Connected → bucket={self.bucket}")
            except Exception as e:
                logging.warning(f"[S3Manager] Could not connect: {e} (running locally)")
        else:
            logging.info("[S3Manager] boto3 not installed -- running in local mode")

    # ─────────────────────────────────────────────────
    # UPLOAD
    # ─────────────────────────────────────────────────

    def upload_file(self, local_path: str, s3_key: str) -> bool:
        """Upload one file to S3. Returns True if successful."""
        if not self._client:
            return False
        if not os.path.exists(local_path):
            logging.warning(f"[S3Manager] File not found: {local_path}")
            return False
        try:
            self._client.upload_file(local_path, self.bucket, s3_key)
            logging.info(f"[S3Manager] ✓ {local_path} → s3://{self.bucket}/{s3_key}")
            return True
        except Exception as e:
            logging.warning(f"[S3Manager] Upload failed: {e}")
            return False

    def upload_dataset(self, local_csv: str) -> bool:
        """
        Upload the Ethereum fraud dataset to S3.
        Call this ONCE from your laptop before any training runs.
        """
        s3_key = f"{self.data_prefix}/{config.DATASET_FILE_NAME}"
        logging.info(f"[S3Manager] Uploading dataset → s3://{self.bucket}/{s3_key}")
        return self.upload_file(local_csv, s3_key)

    def upload_rag_store(self, run_dir: str, run_name: str) -> dict:
        """
        Upload all output files from a completed run to S3.
        Called by CoordinatorAgent at the end of each run.

        Uploads: .csv, .json, .log, .png, .txt files
        Returns: dict of {filename: s3_uri} for all uploaded files
        """
        uploaded = {}
        if not self._client or not os.path.isdir(run_dir):
            return uploaded

        ALLOWED = (".csv", ".json", ".log", ".png", ".txt")
        for fname in os.listdir(run_dir):
            if not fname.endswith(ALLOWED):
                continue
            local_path = os.path.join(run_dir, fname)
            if not os.path.isfile(local_path):
                continue
            s3_key = f"{self.runs_prefix}/{run_name}/{fname}"
            if self.upload_file(local_path, s3_key):
                uploaded[fname] = f"s3://{self.bucket}/{s3_key}"

        logging.info(
            f"[S3Manager] Run upload done: {len(uploaded)} files "
            f"→ s3://{self.bucket}/{self.runs_prefix}/{run_name}/"
        )
        return uploaded

    # ─────────────────────────────────────────────────
    # DOWNLOAD
    # ─────────────────────────────────────────────────

    def download_dataset(self, local_path: str) -> bool:
        """
        Download dataset from S3 to local_path.
        Called at the start of a SageMaker Training Job.
        """
        if not self._client:
            return False
        s3_key = f"{self.data_prefix}/{config.DATASET_FILE_NAME}"
        try:
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            self._client.download_file(self.bucket, s3_key, local_path)
            logging.info(f"[S3Manager] ✓ Downloaded dataset → {local_path}")
            return True
        except Exception as e:
            logging.warning(f"[S3Manager] Download failed: {e}")
            return False

    def is_available(self) -> bool:
        """Returns True if S3 is connected and accessible."""
        return self._client is not None