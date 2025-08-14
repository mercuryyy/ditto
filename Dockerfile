# Target platform: linux/amd64 for RunPod GPU servers
FROM --platform=linux/amd64 nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

# Set working directory
WORKDIR /workspace

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    git \
    git-lfs \
    ffmpeg \
    build-essential \
    libsndfile1 \
    espeak \
    libespeak1 \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.10 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1
RUN update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1

# Upgrade pip
RUN pip install --upgrade pip

# Install PyTorch with CUDA 11.8 support (optimized for RunPod GPUs)
RUN pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118

# Install core dependencies
RUN pip install \
    librosa \
    tqdm \
    filetype \
    imageio \
    opencv-python-headless \
    scikit-image \
    scikit-learn \
    scipy \
    cython \
    imageio-ffmpeg \
    colored \
    "numpy<2.0" \
    runpod \
    pyttsx3 \
    gtts \
    onnxruntime-gpu==1.15.1 \
    mediapipe \
    einops

# Install TensorRT for NVIDIA GPUs (will work on RunPod)
# Using specific NVIDIA PyPI index for TensorRT
RUN pip install --index-url https://pypi.nvidia.com \
    tensorrt==8.6.1 \
    cuda-python \
    polygraphy || \
    echo "WARNING: TensorRT installation failed. PyTorch model will be used as fallback."

# Copy the application code
COPY . /workspace/

# Install git lfs for model downloading
RUN git lfs install

# Create checkpoints directory
RUN mkdir -p /workspace/checkpoints

# Download model files from HuggingFace during build
RUN git lfs install && \
    rm -rf /workspace/checkpoints && \
    git clone https://huggingface.co/digital-avatar/ditto-talkinghead /workspace/checkpoints && \
    echo "Models downloaded successfully from HuggingFace"

# Set environment variables
ENV PYTHONPATH=/workspace:$PYTHONPATH
# Add CUDA library paths to fix the missing library issue
ENV LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64:$LD_LIBRARY_PATH

# Run the handler
CMD ["python", "-u", "runpod_handler.py"]
