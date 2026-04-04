"""
aws/sm_launcher.py
==================
PRODUCTION FIX -- pins to sagemaker 2.x (last stable API before v3 broke everything)

THE PROBLEM:
  sagemaker 3.x completely restructured its module layout.
  sagemaker.estimator, sagemaker.session, sagemaker.inputs -- all moved or removed.
  ModelTrainer/ComputeConfig/TrainingJobConfig don't exist in sagemaker-train 1.6.0.
  Only 5 modules visible: ai_registry, core, mlops, serve, train -- none have Estimator.

THE INDUSTRY STANDARD FIX:
  Pin to sagemaker 2.x. AWS themselves recommend this for production workloads.
  sagemaker 2.232.2 is the last stable v2 release (Dec 2024).
  All major ML platforms (Weights & Biases, Hugging Face, fast.ai) pin to 2.x.

INSTALL (run this FIRST, before running this script):
  pip install "sagemaker>=2.200,<3.0" boto3

  OR pin exactly:
  pip install sagemaker==2.232.2 boto3

VERIFY:
  python -c "import sagemaker; print(sagemaker.__version__)"
  Should print: 2.232.2 (or similar 2.x)

YOUR SETUP:
  ECR image : 551222650550.dkr.ecr.us-east-1.amazonaws.com/fraud-sagemaker:latest
  Account   : 551222650550
  Region    : us-east-1
"""

import os
import sys
import logging
import argparse

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s")
logger = logging.getLogger("sm_launcher")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

try:
    import boto3
    import sagemaker
    from sagemaker import Session
    from sagemaker.estimator import Estimator
    from sagemaker.inputs import TrainingInput

    ver = sagemaker.__version__
    major = int(ver.split(".")[0])
    if major >= 3:
        logger.error(
            f"sagemaker {ver} detected -- v3 broke the stable API.\n"
            f"Fix: pip install 'sagemaker>=2.200,<3.0' boto3\n"
            f"Then restart your terminal and re-run."
        )
        SM_AVAILABLE = False
    else:
        logger.info(f"sagemaker {ver} -- OK (v2 stable API)")
        SM_AVAILABLE = True

except ImportError as e:
    SM_AVAILABLE = False
    logger.warning(f"Import failed: {e}")
    logger.warning("Fix: pip install 'sagemaker>=2.200,<3.0' boto3")


ECR_IMAGE_URI = "551222650550.dkr.ecr.us-east-1.amazonaws.com/fraud-sagemaker:latest"


def upload_dataset(local_csv: str = "transaction_dataset.csv"):
    from aws.s3_manager import S3Manager
    s3 = S3Manager()
    if not os.path.exists(local_csv):
        logger.error(f"Dataset not found: {local_csv}")
        return False
    success = s3.upload_dataset(local_csv)
    if success:
        logger.info(f"Dataset uploaded to {config.DATASET_URI}")
    return success


def launch_job(seed: int = 42, run_mode: str = "AGENTIC", wait: bool = False) -> str:
    if not SM_AVAILABLE:
        logger.error("sagemaker v2 not available. Run: pip install 'sagemaker>=2.200,<3.0' boto3")
        return ""

    job_name = f"fraud-seedv20{seed}-{run_mode.lower().replace('_', '-')}"
    source_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    logger.info(f"Source dir : {source_dir}")
    logger.info(f"ECR image  : {ECR_IMAGE_URI}")
    logger.info(f"Job name   : {job_name}")

    estimator = Estimator(
        image_uri         = ECR_IMAGE_URI,
        entry_point       = "main.py",
        source_dir        = source_dir,
        role              = config.SAGEMAKER_ROLE,
        instance_type     = config.INSTANCE_TYPE,
        instance_count    = 1,
        hyperparameters   = {
            "seed"      : seed,
            "run_mode"  : run_mode,
            "s3_bucket" : config.S3_BUCKET,
        },
        output_path       = f"s3://{config.S3_BUCKET}/{config.S3_RUNS_PREFIX}/{job_name}/",
        max_run           = 3600,
        sagemaker_session = Session(),
        environment       = {
            "RUN_MODE" : run_mode,
            "N_RUNS"   : "1",
        },
    )

    training_input = TrainingInput(
        s3_data      = f"s3://{config.S3_BUCKET}/{config.S3_DATA_PREFIX}/",
        content_type = "text/csv",
    )

    logger.info(f"Instance : {config.INSTANCE_TYPE}")
    logger.info(f"Output   : s3://{config.S3_BUCKET}/{config.S3_RUNS_PREFIX}/{job_name}/")
    logger.info("Submitting job...")

    estimator.fit(
        inputs   = {"training": training_input},
        job_name = job_name,
        wait     = wait,
        logs     = wait,
    )

    logger.info("Job submitted.")
    if not wait:
        logger.info(
            f"Monitor: https://console.aws.amazon.com/sagemaker/home"
            f"?region={config.AWS_REGION}#/jobs/{job_name}"
        )
    return job_name


def launch_multi(seeds=None, run_mode="AGENTIC"):
    if seeds is None:
        seeds = [42, 7, 21, 100, 999]
    logger.info(f"Launching {len(seeds)} parallel jobs: seeds={seeds}")
    jobs = [launch_job(seed=s, run_mode=run_mode, wait=False) for s in seeds]
    for j in jobs:
        logger.info(f"  submitted: {j}")
    return jobs


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--seed",        type=int,  default=42)
    p.add_argument("--run_mode",    type=str,  default="AGENTIC")
    p.add_argument("--multi",       action="store_true")
    p.add_argument("--upload_data", action="store_true")
    p.add_argument("--wait",        action="store_true")
    args = p.parse_args()

    if args.upload_data:
        upload_dataset()

    if args.multi:
        launch_multi(run_mode=args.run_mode)
    else:
        launch_job(seed=args.seed, run_mode=args.run_mode, wait=args.wait)