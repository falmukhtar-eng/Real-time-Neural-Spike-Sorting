# Dockerfile for CNN-BiLSTM Spike Sorting
# CUDA 11.2, cuDNN 8.1 (Table V)

FROM nvidia/cuda:11.2.2-cudnn8-runtime-ubuntu20.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV TF_CPP_MIN_LOG_LEVEL=2

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.8 \
    python3-pip \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.8 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.8 1
RUN update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1

# Upgrade pip
RUN pip install --upgrade pip setuptools wheel

# Set working directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy source code
COPY model.py train.py preprocess.py evaluate.py baselines.py ./
COPY equivalent_accuracy.py semi_supervised.py benchmark_latency.py ./

# Create directories for data and models
RUN mkdir -p data models results

# Default command
CMD ["python", "--version"]
