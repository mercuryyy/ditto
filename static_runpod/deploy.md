# 🚀 Deployment Commands

## Step 1: Update Build Script

Edit your Docker Hub username in the build script:

```bash
cd static_runpod
nano build_and_push.sh
```

Change this line:
```bash
IMAGE_NAME="YOUR_DOCKERHUB_USERNAME/ditto-websocket-pod"
```

## Step 2: Build and Push Image

```bash
cd static_runpod
chmod +x build_and_push.sh
./build_and_push.sh
```

This will:
- 🏗️ Build for `linux/amd64` using docker buildx
- ⬆️ Push to Docker Hub automatically
- ✅ Use the exact same dependencies as your working serverless setup

## Step 3: Deploy to RunPod Pod

1. **Go to RunPod Console**: https://console.runpod.io/
2. **Click "Deploy"** → **"Pods"** (not Serverless)
3. **Container Settings**:
   ```
   Container Image: YOUR_DOCKERHUB_USERNAME/ditto-websocket-pod:latest
   Expose HTTP Ports: 8888
   Volume Mount Path: /workspace
   Volume Disk: 50 GB
   ```
4. **Select GPU**: RTX 4090 recommended ($0.69/hour)
5. **Deploy**

## Step 4: Test Your Pod

Once deployed, you'll get a URL like: `https://abcd1234-8888.proxy.runpod.net`

Test with:
```bash
# Health check
curl https://YOUR_POD_URL/health

# Demo page
open https://YOUR_POD_URL/demo
```

## Step 5: WebSocket Connection

Your WebSocket endpoint will be:
```
wss://YOUR_POD_URL/ws
```

Use this in your LiveKit agent to connect to the Pod for real-time avatar streaming!

---

**Cost**: ~$0.69/hour for RTX 4090 with true real-time WebSocket streaming (vs serverless limitations)
