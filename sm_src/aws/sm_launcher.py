"""
aws/sm_launcher.py
==================
Run this from YOUR LAPTOP to launch a SageMaker training job.

USAGE:
  # Upload dataset to S3 (one time only):
  python aws/sm_launcher.py --upload_data

  # Launch single run (seed=42, paper verification):
  python aws/sm_launcher.py --seed 42

  # Launch 5 runs in parallel (robustness testing):
  python aws/sm_launcher.py --multi

PREREQUISITES:
  pip install sagemaker boto3
  aws configure   ← enter your access key + region

BEFORE RUNNING:
  Edit config.py:
    S3_BUCKET      = "your-real-bucket-name"
    SAGEMAKER_ROLE = "arn:aws:iam::YOUR_ACCOUNT:role/SageMakerExecutionRole"
"""

import os
import sys
import logging
import argparse
logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s")
logger = logging.getLogger("sm_launcher")

# Add parent directory to path so we can import config
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

try:
    import boto3
    import sagemaker
    from sagemaker.sklearn.estimator import SKLearn
    from sagemaker.session import Session
    SM_AVAILABLE = True
except ImportError:
    SM_AVAILABLE = False
    logger.warning("Install with: pip install sagemaker boto3")


def upload_dataset(local_csv: str = "transaction_dataset.csv"):
    """Upload the dataset to S3. Run this once before training."""
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from aws.s3_manager import S3Manager
    s3 = S3Manager()
    if not os.path.exists(local_csv):
        logger.error(f"Dataset not found: {local_csv}")
        return False
    success = s3.upload_dataset(local_csv)
    if success:
        logger.info(f"Dataset uploaded → {config.DATASET_URI}")
    return success


def launch_job(seed: int = 42, run_mode: str = "AGENTIC", wait: bool = False) -> str:
    """Launch a single SageMaker training job."""
    if not SM_AVAILABLE:
        logger.error("sagemaker SDK not installed")
        return ""

    # Job name must be unique in SageMaker (no underscores allowed)
    job_name = f"fraud-seddv5{seed}-{run_mode.lower().replace('_', '-')}"

    source_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    print(f"Source dir for SageMaker: {source_dir}")

    estimator = SKLearn(
        entry_point      = "main.py",         # SageMaker runs this
        source_dir       = source_dir,        # your whole project folder
        framework_version= "1.0-1",
        py_version       = "py3",
        role             = config.SAGEMAKER_ROLE,
        instance_type    = config.INSTANCE_TYPE,
        instance_count   = 1,
        hyperparameters  = {
            "seed"     : seed,
            "run_mode" : run_mode,
            "s3_bucket": config.S3_BUCKET,
        },
        output_path      = (
            f"s3://{config.S3_BUCKET}/{config.S3_RUNS_PREFIX}/{job_name}/"
        ),
        max_run          = 3600,  # 1 hour max
        sagemaker_session= Session(),
        environment      = {
            "RUN_MODE": run_mode,
            "N_RUNS"  : "1",
        },
    )

    training_input = sagemaker.inputs.TrainingInput(
        s3_data     = f"s3://{config.S3_BUCKET}/{config.S3_DATA_PREFIX}/",
        content_type= "text/csv",
    )

    logger.info(f"Launching: {job_name}")
    logger.info(f"Instance:  {config.INSTANCE_TYPE}")
    logger.info(f"Output:    s3://{config.S3_BUCKET}/{config.S3_RUNS_PREFIX}/{job_name}/")
    logger.info("About to call estimator.fit()")

    estimator.fit(
        inputs   = {"training": training_input},
        job_name = job_name,
        wait     = wait,
        logs     = wait,
    )
    logger.info("estimator.fit() returned")

    if not wait:
        logger.info(
            f"Job submitted. Monitor at:\n"
            f"https://console.aws.amazon.com/sagemaker/home"
            f"?region={config.AWS_REGION}#/jobs/{job_name}"
        )
    return job_name


def launch_multi(seeds=None, run_mode="AGENTIC"):
    """Launch multiple jobs in parallel — one per seed."""
    if seeds is None:
        seeds = [42, 7, 21, 100, 999]
    logger.info(f"Launching {len(seeds)} parallel jobs: seeds={seeds}")
    jobs = [launch_job(seed=s, run_mode=run_mode, wait=False) for s in seeds]
    logger.info("All jobs submitted:")
    for j in jobs:
        logger.info(f"  → {j}")
    return jobs


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--seed",        type=int, default=42)
    p.add_argument("--run_mode",    type=str, default="AGENTIC")
    p.add_argument("--multi",       action="store_true",
                   help="Launch 5 parallel jobs")
    p.add_argument("--upload_data", action="store_true",
                   help="Upload dataset to S3 first")
    p.add_argument("--wait",        action="store_true",
                   help="Block until job finishes")
    args = p.parse_args()

    if args.upload_data:
        upload_dataset()

    if args.multi:
        launch_multi(run_mode=args.run_mode)
    else:
        launch_job(seed=args.seed, run_mode=args.run_mode, wait=args.wait)