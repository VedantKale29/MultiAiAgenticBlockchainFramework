import os
from pathlib import Path

def create_project_structure():
    # Define the base directory name
    base_dir = Path("./")

    # Define all directories to create
    directories = [
        base_dir / "runs" / "run1",
        base_dir / "runs" / "run2",
        base_dir / "runs" / "run3",
        base_dir / "runs" / "run4",
        base_dir / "runs" / "run5",
    ]

    # Define all python scripts and configs to create
    files = [
        base_dir / "config.py",
        base_dir / "data_loader.py",
        base_dir / "feature_processing.py", 
        base_dir / "rf_model.py",
        base_dir / "if_model.py",
        base_dir / "hybrid_agent.py",
        base_dir / "adaptation.py",
        base_dir / "batch_runner.py",
        base_dir / "metrics.py",
        base_dir / "logger.py",
        base_dir / "experiment_tracker.py",
        base_dir / "main.py",
        base_dir / "requirements.txt",
        base_dir / "setup.py",
        base_dir / ".gitignore",
        base_dir / "__init__.py" # Makes the directory a package for the -e . install
    ]

    # 1. Create the directories
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
        print(f"📁 Created directory: {directory}")

    # 2. Create the empty files
    for file_path in files:
        file_path.touch(exist_ok=True)
        print(f"📄 Created file: {file_path.name}")

    # 3. Auto-populate requirements.txt
    requirements_content = """pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0

matplotlib>=3.7.0
seaborn>=0.12.0

boto3>=1.28.0
sagemaker>=2.170.0

# logging / tracking
tqdm>=4.66.0

# optional but safe
joblib>=1.3.0

-e .
"""
    with open(base_dir / "requirements.txt", "w") as f:
        f.write(requirements_content)

    # 4. Auto-populate setup.py
    setup_content = """from setuptools import setup, find_packages

setup(
    name="agentic_fraud_detection",
    version="0.1",
    packages=find_packages(),
)
"""
    with open(base_dir / "setup.py", "w") as f:
        f.write(setup_content)

    # 5. Auto-populate .gitignore
    gitignore_content = """__pycache__/
*.pyc
venv/
.env
runs/
*.csv
.ipynb_checkpoints/
"""
    with open(base_dir / ".gitignore", "w") as f:
        f.write(gitignore_content)

    print(f"\n✅ Success! CD into '{base_dir}' to get started.")

if __name__ == "__main__":
    create_project_structure()