# =========================
# Base image
# =========================
FROM python:3.10-slim

# =========================
# System settings
# =========================
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

WORKDIR /opt/ml/code

# =========================
# Install system deps
# =========================
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# =========================
# Copy requirements first (cache friendly)
# =========================
COPY requirements.txt .

# =========================
# Install python deps
# =========================
RUN python -m pip install --no-cache-dir --upgrade pip \
    && python -m pip install --no-cache-dir -r requirements.txt

# =========================
# Copy project
# =========================
COPY . /opt/ml/code

# =========================
# SageMaker env
# =========================
ENV PYTHONPATH=/opt/ml/code

# =========================
# Default command
# SageMaker will override entrypoint
# =========================
ENTRYPOINT ["python", "main.py"]