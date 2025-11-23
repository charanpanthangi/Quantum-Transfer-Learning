# Simple Dockerfile for Quantum Transfer Learning demo using PennyLane
# Uses python:3.11-slim as lightweight base image
FROM python:3.11-slim

# Install system dependencies needed for PennyLane backends and basic tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set workdir inside container
WORKDIR /app

# Copy dependency list first to leverage Docker layer caching
COPY requirements.txt requirements.txt

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the full project into the image
COPY . .

# Default command runs the CLI demo
CMD ["python", "app/main.py"]
