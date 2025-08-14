# CPU-only base
FROM python:3.10-slim

ENV DEBIAN_FRONTEND=noninteractive

# System deps for audio I/O
RUN apt-get update && apt-get install -y --no-install-recommends \
    sox libsox-dev libsox-fmt-all libsndfile1 ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace
COPY requirements.txt /workspace/requirements.txt

# Install PyTorch CPU wheels first to avoid heavy resolution
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu \
    torch==2.3.1 torchaudio==2.3.1 && \
    pip install --no-cache-dir -r /workspace/requirements.txt

# Copy project
COPY . /workspace

# Default command prints help
CMD ["python", "train.py", "--help"]