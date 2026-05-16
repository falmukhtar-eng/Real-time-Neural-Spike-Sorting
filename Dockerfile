# FILE: Dockerfile
# Exact environment used to produce paper results
# Build: docker build -t cnn-bilstm:v1.0 .
# Run: docker run --gpus all -v $(pwd)/data:/app/data cnn-bilstm:v1.0

FROM nvidia/cuda:11.2.2-cudnn8-devel-ubuntu20.04

# Prevent interactive prompts during build
ENV DEBIAN_FRONTEND=noninteractive

# Install Python 3.8.13 (exact version from paper)
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
    python3.8=3.8.10-0ubuntu1~20.04.9 \
    python3.8-dev \
    python3-pip=20.0.2-5ubuntu1.9 \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Make python3.8 the default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.8 1
RUN update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1

# Upgrade pip to exact version
RUN pip install --no-cache-dir pip==22.0.4

# Set working directory
WORKDIR /app

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . .

# Set deterministic TF behavior for reproducibility
ENV TF_DETERMINISTIC_OPS=1
ENV TF_CUDNN_DETERMINISTIC=1
ENV PYTHONHASHSEED=42

# Default command
CMD ["python", "train.py"]
