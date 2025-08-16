# 🚀 RunPod Pod with Real-Time WebSocket Avatar Streaming

This directory contains everything needed to deploy a RunPod Pod with **true WebSocket streaming** for real-time Ditto avatar generation.

## 🆚 Pod vs Serverless Comparison

| Feature | RunPod Serverless | RunPod Pod (This Setup) |
|---------|-------------------|-------------------------|
| **WebSocket Support** | ❌ No | ✅ Yes |
| **Real-time Streaming** | ❌ No | ✅ Yes |
| **Response Size Limits** | ❌ ~100MB | ✅ Unlimited |
| **Persistent Connections** | ❌ No | ✅ Yes |
| **Cost Model** | Per-second billing | Hourly billing |
| **Use Case** | Batch processing | Real-time streaming |

## 📁 File Structure

```
static_runpod/
├── Dockerfile              # Multi-platform Pod image
├── websocket_server.py     # FastAPI WebSocket server
├── build_and_push.sh       # Build script for linux/amd64
├── core/                   # Ditto core modules
├── stream_pipeline_*.py    # Ditto pipeline components
└── README.md               # This file
```

## 🏗️ Build & Deploy Instructions

### **Step 1: Prepare Docker Hub**

1. Create account at [hub.docker.com](https://hub.docker.com)
2. Create a repository (e.g., `your-username/ditto-websocket-pod`)

### **Step 2: Build & Push Image**

```bash
# Navigate to static_runpod directory
cd static_runpod

# Edit build script with your Docker Hub username
nano build_and_push.sh
# Change: IMAGE_NAME="your-dockerhub-username/ditto-websocket-pod"

# Make script executable and run
chmod +x build_and_push.sh
./build_and_push.sh
```

The script will:
- ✅ Create multi-platform builder
- 🏗️ Build for `linux/amd64` (RunPod compatible)
- ⬆️ Push to Docker Hub automatically

### **Step 3: Deploy RunPod Pod**

1. **Go to RunPod Console**: https://console.runpod.io/
2. **Create New Pod** (not Serverless):
   - **Container Image**: `your-dockerhub-username/ditto-websocket-pod:latest`
   - **GPU**: RTX 4090, A40, or A100
   - **Volume**: 50GB recommended
   - **Ports**:
     - `8888`: WebSocket server
     - `8000`: Health check (optional)

3. **Template Override Settings**:
   ```
   Container Image: your-dockerhub-username/ditto-websocket-pod:latest
   Expose HTTP Ports: 8888
   Volume Mount Path: /workspace
   Volume Disk: 50 GB
   ```

4. **Deploy** - Wait for Pod to start (~2-3 minutes)

### **Step 4: Test Connection**

Once deployed, you'll get a Pod URL like: `https://abcd1234-8888.proxy.runpod.net`

**Test endpoints:**
```bash
# Health check
curl https://your-pod-url.proxy.runpod.net/health

# Service info
curl https://your-pod-url.proxy.runpod.net/

# Demo page
open https://your-pod-url.proxy.runpod.net/demo
```

## 🌐 WebSocket API Usage

### **Connection & Initialization**

```javascript
// Connect to WebSocket
const ws = new WebSocket('wss://your-pod-url.proxy.runpod.net/ws');

// Initialize with avatar
ws.onopen = () => {
    ws.send(JSON.stringify({
        type: 'init',
        avatar_image_base64: 'base64-encoded-image',
        settings: {
            max_size: 512,
            sampling_timesteps: 25,
            mouth_amplitude: 0.8,
            head_amplitude: 0.6
        }
    }));
};

// Handle ready signal
ws.onmessage = (event) => {
    const data = JSON.parse(event.data);
    
    if (data.type === 'ready') {
        console.log('Avatar session ready!');
        // Start streaming audio...
    } else if (data.type === 'frame') {
        displayFrame(data.data); // base64 JPEG frame
    }
};
```

### **Audio Streaming**

```javascript
// Stream audio chunks (Float32Array as ArrayBuffer)
const audioChunk = new Float32Array(1024); // 64ms at 16kHz
ws.send(audioChunk.buffer);
```

### **Frame Output**

Frames are streamed as JSON messages:
```json
{
  "type": "frame",
  "data": "base64-encoded-jpeg",
  "frame_number": 123,
  "timestamp": 1692123456.789
}
```

## 💰 Cost Analysis

### **Hourly Costs (as of 2024):**
- **RTX 4090**: ~$0.69/hour
- **A40**: ~$0.39/hour  
- **A100**: ~$1.10/hour

### **Break-even vs Serverless:**
- **< 3 hours/day**: Use Serverless
- **> 3 hours/day**: Pod is more cost-effective
- **24/7 production**: Pod is significantly cheaper

## 🔧 Performance Tuning

### **For Lower Latency:**
```python
settings = {
    "max_size": 512,           # Smaller = faster
    "sampling_timesteps": 15,  # Fewer steps = faster
    "smo_k_s": 3,             # Less smoothing = faster
    "overlap_v2": 3           # Less overlap = faster
}
```

### **For Higher Quality:**
```python
settings = {
    "max_size": 1280,         # Larger = better quality
    "sampling_timesteps": 50, # More steps = better quality
    "smo_k_s": 15,           # More smoothing = better quality
    "overlap_v2": 10         # More overlap = smoother
}
```

## 📡 LiveKit Integration

For LiveKit agents, connect to the Pod's WebSocket:

```python
# In your LiveKit agent
class RunPodWebSocketSession:
    def __init__(self):
        self.websocket_url = "wss://your-pod-url.proxy.runpod.net/ws"
        self.websocket = None
    
    async def connect(self, avatar_b64):
        self.websocket = await websockets.connect(self.websocket_url)
        
        # Initialize avatar
        await self.websocket.send(json.dumps({
            "type": "init",
            "avatar_image_base64": avatar_b64,
            "settings": {...}
        }))
    
    async def stream_audio(self, audio_chunk):
        # Send audio directly 
        await self.websocket.send(audio_chunk.tobytes())
    
    async def receive_frames(self):
        # Receive avatar frames
        async for message in self.websocket:
            frame_data = json.loads(message)
            if frame_data["type"] == "frame":
                yield frame_data["data"]  # base64 JPEG
```

## 🚨 Troubleshooting

### **Pod Won't Start:**
- Check Docker Hub image exists
- Verify image platform is `linux/amd64`
- Check RunPod logs for errors

### **WebSocket Connection Failed:**
- Ensure port 8888 is exposed
- Check Pod public IP/URL is correct
- Verify health endpoint responds

### **Poor Avatar Quality:**
- Increase `max_size` (512→1280)
- Increase `sampling_timesteps` (25→50)
- Check GPU memory usage

### **High Latency:**
- Decrease `max_size` (1280→512)
- Decrease `sampling_timesteps` (50→25)
- Use faster GPU (A100 > RTX 4090 > A40)

## 🎯 Production Deployment

### **Recommended Settings:**
- **GPU**: RTX 4090 (best price/performance)
- **Volume**: 50GB minimum
- **Image**: Always use specific tags (not `:latest`)
- **Health Checks**: Monitor `/health` endpoint
- **Scaling**: Deploy multiple Pods behind load balancer

### **Monitoring:**
```bash
# Check Pod health
curl https://your-pod-url.proxy.runpod.net/health

# Monitor active sessions
curl https://your-pod-url.proxy.runpod.net/ | jq '.active_sessions'
```

---

## 🎉 Ready for Real-Time Avatar Streaming!

Your RunPod Pod is now configured for **true WebSocket streaming** with sub-second latency for professional avatar applications! 🚀
