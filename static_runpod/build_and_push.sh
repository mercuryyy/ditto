#!/bin/bash

# Build and Push Script for RunPod Pod with Multi-platform Support
# This script builds the Docker image for linux/amd64 platform using buildx

set -e

# Configuration
IMAGE_NAME="iadsmedia/ditto-websocket-pod"
IMAGE_TAG="latest"
DOCKERFILE="./Dockerfile"

echo "🚀 Building and pushing Ditto WebSocket Pod Docker image..."
echo "📦 Image: ${IMAGE_NAME}:${IMAGE_TAG}"
echo "🏗️ Platform: linux/amd64 (for RunPod GPUs)"
echo ""

# Check if buildx is available
if ! docker buildx version > /dev/null 2>&1; then
    echo "❌ Docker buildx is not available. Please update Docker to a version that supports buildx."
    exit 1
fi

# Create builder instance if it doesn't exist
if ! docker buildx inspect multiplatform-builder > /dev/null 2>&1; then
    echo "🔨 Creating new buildx builder instance..."
    docker buildx create --name multiplatform-builder --driver docker-container --bootstrap
fi

# Use the multiplatform builder
echo "🔧 Using multiplatform-builder..."
docker buildx use multiplatform-builder

# Build and push the image for linux/amd64
echo "🏗️ Building Docker image for linux/amd64..."
docker buildx build \
    --platform linux/amd64 \
    --tag ${IMAGE_NAME}:${IMAGE_TAG} \
    --file ${DOCKERFILE} \
    --push \
    .

echo ""
echo "✅ Successfully built and pushed ${IMAGE_NAME}:${IMAGE_TAG}"
echo ""
echo "🎯 Next steps:"
echo "1. Go to RunPod Console: https://console.runpod.io/"
echo "2. Create a new Pod (not Serverless)"
echo "3. Use this container image: ${IMAGE_NAME}:${IMAGE_TAG}"
echo "4. Expose ports: 8888 (WebSocket) and 8000 (Health)"
echo "5. Select GPU: RTX 4090, A40, or A100"
echo "6. Set volume mount: /workspace (50GB recommended)"
echo ""
echo "📡 WebSocket endpoint will be: ws://POD_IP:8888/ws"
echo "🔍 Demo page will be: http://POD_IP:8888/demo"
echo "❤️ Health check: http://POD_IP:8888/health"
