# RunPod automatically handles platform selection
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

# Set working directory
WORKDIR /workspace

# Install system dependencies
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

# Set Python 3.10 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1
RUN update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1

# Upgrade pip and install essential tools
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Install PyTorch with CUDA 11.8 support (optimized for RunPod GPUs)
RUN pip install --no-cache-dir torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118

# Copy requirements first for better caching
COPY requirements_local.txt /tmp/requirements_local.txt
COPY requirements_webui.txt /tmp/requirements_webui.txt

# Install core dependencies from requirements files with version compatibility fixes
# Pin NumPy first to prevent any package from overriding the version
RUN pip install --no-cache-dir "numpy>=1.24.0,<2.0.0"

# Create constraints file for NumPy version
RUN echo "numpy>=1.24.0,<2.0.0" > /tmp/constraints.txt

# Install other core dependencies with NumPy constraint enforced
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
    runpod \
    pyttsx3 \
    gtts \
    onnxruntime-gpu==1.16.3 \
    mediapipe \
    einops \
    gradio==4.44.0 \
    Pillow==10.3.0 \
    pydub==0.25.1 \
    requests

# Install TensorRT with better error handling and fallback
RUN pip install --no-cache-dir --index-url https://pypi.nvidia.com --trusted-host pypi.nvidia.com \
    tensorrt==8.6.1 \
    cuda-python \
    polygraphy || \
    echo "WARNING: TensorRT installation failed. PyTorch model will be used as fallback."

# Copy the application code
COPY . /workspace/

<<<<<<< HEAD
# Copy the updated real-time streaming handler with direct audio support as the default handler
COPY runpod_realtime_streaming_handler.py /workspace/runpod_handler.py

# Install only essential dependencies for audio processing
RUN pip install --no-cache-dir \
    soundfile \
    aiohttp
=======
# Copy real-time streaming handler as the default handler
COPY runpod_realtime_streaming_handler.py /workspace/runpod_handler.py
>>>>>>> 077f319a1838990c9898bb2418142f6e0614d4d1

# Install git lfs for model downloading
RUN git lfs install

# Create checkpoints directory
RUN mkdir -p /workspace/checkpoints

# Download model files from HuggingFace during build with retry logic and better error handling
RUN rm -rf /workspace/checkpoints && \
    echo "Starting model download from HuggingFace..." && \
    export GIT_LFS_SKIP_SMUDGE=1 && \
    git clone https://huggingface.co/digital-avatar/ditto-talkinghead /workspace/checkpoints && \
    cd /workspace/checkpoints && \
    git lfs pull && \
    echo "Models downloaded successfully from HuggingFace" && \
    ls -la /workspace/checkpoints/ && \
    find /workspace/checkpoints -name "*.bin" -o -name "*.pth" -o -name "*.onnx" | head -10

# Verify model files exist
RUN if [ ! -d "/workspace/checkpoints" ] || [ -z "$(ls -A /workspace/checkpoints)" ]; then \
        echo "ERROR: Model download failed - checkpoints directory is empty"; \
        exit 1; \
    fi

# Set environment variables
ENV PYTHONPATH=/workspace:$PYTHONPATH
ENV LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64:$LD_LIBRARY_PATH
ENV CUDA_HOME=/usr/local/cuda-11.8
ENV TORCH_CUDA_ARCH_LIST="6.0;6.1;7.0;7.5;8.0;8.6"

# Add a health check to ensure dependencies are working
RUN python -c "import torch; print('PyTorch version:', torch.__version__); print('CUDA available:', torch.cuda.is_available())" && \
    python -c "import numpy; print('NumPy version:', numpy.__version__)" && \
    python -c "import cv2; print('OpenCV version:', cv2.__version__)" && \
    python -c "import librosa; print('Librosa imported successfully')" && \
    echo "All dependencies verified successfully"

# Run the handler
CMD ["python", "-u", "runpod_handler.py"]
