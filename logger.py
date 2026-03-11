import logging
import os
from datetime import datetime

# Create a logs directory if it doesn't exist locally
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

# Create a unique log file name based on the current timestamp
LOG_FILE = f"{datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}.log"
LOG_FILE_PATH = os.path.join(LOG_DIR, LOG_FILE)

# Define the logging format
# Example output: [2026-03-08 18:14:00,000: INFO: hybrid_agent]: Threshold updated to 0.467
LOG_FORMAT = "[%(asctime)s: %(levelname)s: %(module)s]: %(message)s"

logging.basicConfig(
    level=logging.INFO,
    format=LOG_FORMAT,
    handlers=[
        # 1. FileHandler: Saves logs to a local file in the /logs directory
        logging.FileHandler(LOG_FILE_PATH),
        
        # 2. StreamHandler: Prints to the VS Code terminal 
        # (and automatically syncs to AWS CloudWatch when running in SageMaker)
        logging.StreamHandler()
    ]
)

# Export the configured logger object so other files can just do:
# from logger import logging