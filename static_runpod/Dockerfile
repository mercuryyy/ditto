# RunPod Pod Dockerfile with WebSocket support for real-time streaming
# Using same base image as working serverless setup
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

# Set working directory
WORKDIR /workspace

# Install system dependencies (same as working serverless)
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    python3.10-dev \
    git \
    git-lfs \
    ffmpeg \
    build-essential \
    libsndfile1 \
    libsndfile1-dev \
    espeak \
    libespeak1 \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.10 as default (same as working serverless)
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1
RUN update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1

# Upgrade pip and install essential tools
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Install PyTorch with CUDA 11.8 support (same as working serverless)
RUN pip install --no-cache-dir torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118

# Install core dependencies with version compatibility fixes (same as working serverless)
# Pin NumPy first to prevent any package from overriding the version
RUN pip install --no-cache-dir "numpy>=1.24.0,<2.0.0"

# Create constraints file for NumPy version
RUN echo "numpy>=1.24.0,<2.0.0" > /tmp/constraints.txt

# Install other core dependencies with NumPy constraint enforced (same as working serverless)
RUN pip install --no-cache-dir \
    --constraint /tmp/constraints.txt \
    librosa \
    tqdm \
    filetype \
    imageio \
    imageio-ffmpeg \
    opencv-python-headless \
    scikit-image \
    scikit-learn \
    scipy \
    cython \
    colored \
    onnxruntime-gpu==1.16.3 \
    mediapipe \
    einops \
    gradio==4.44.0 \
    Pillow==10.3.0 \
    pydub==0.25.1 \
    requests \
    nest-asyncio

# Install WebSocket and API dependencies for Pod
RUN pip install --no-cache-dir \
    fastapi \
    uvicorn[standard] \
    websockets \
    aiohttp \
    python-multipart

# Install only essential dependencies for audio processing and image handling
RUN pip install --no-cache-dir \
    soundfile \
    aiohttp

# Install TensorRT with better error handling and fallback (same as working serverless)
RUN pip install --no-cache-dir --index-url https://pypi.nvidia.com --trusted-host pypi.nvidia.com \
    tensorrt==8.6.1 \
    cuda-python \
    polygraphy || \
    echo "WARNING: TensorRT installation failed. PyTorch model will be used as fallback."

# Copy the application code (same as working serverless)
COPY . /workspace/

# Install git lfs for model downloading
RUN git lfs install

# Create checkpoints directory
RUN mkdir -p /workspace/checkpoints

# Download model files from HuggingFace during build with retry logic and better error handling (same as working serverless)
RUN rm -rf /workspace/checkpoints && \
    echo "Starting model download from HuggingFace..." && \
    export GIT_LFS_SKIP_SMUDGE=1 && \
    git clone https://huggingface.co/digital-avatar/ditto-talkinghead /workspace/checkpoints && \
    cd /workspace/checkpoints && \
    git lfs pull && \
    echo "Models downloaded successfully from HuggingFace" && \
    ls -la /workspace/checkpoints/ && \
    find /workspace/checkpoints -name "*.bin" -o -name "*.pth" -o -name "*.onnx" | head -10

# Verify model files exist (same as working serverless)
RUN if [ ! -d "/workspace/checkpoints" ] || [ -z "$(ls -A /workspace/checkpoints)" ]; then \
        echo "ERROR: Model download failed - checkpoints directory is empty"; \
        exit 1; \
    fi

# Set environment variables (same as working serverless)
ENV PYTHONPATH=/workspace:$PYTHONPATH
ENV LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64:$LD_LIBRARY_PATH
ENV CUDA_HOME=/usr/local/cuda-11.8
ENV TORCH_CUDA_ARCH_LIST="6.0;6.1;7.0;7.5;8.0;8.6"

# Add a health check to ensure dependencies are working (same as working serverless)
RUN python -c "import torch; print('PyTorch version:', torch.__version__); print('CUDA available:', torch.cuda.is_available())" && \
    python -c "import numpy; print('NumPy version:', numpy.__version__)" && \
    python -c "import cv2; print('OpenCV version:', cv2.__version__)" && \
    python -c "import librosa; print('Librosa imported successfully')" && \
    echo "All dependencies verified successfully"

# Expose ports for WebSocket server
EXPOSE 8888

# Start the WebSocket server (instead of runpod handler)
CMD ["python", "-u", "websocket_server.py"]
