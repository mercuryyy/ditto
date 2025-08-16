"""
WebSocket Server for RunPod Pod - Real-time Ditto Avatar Streaming
Provides true WebSocket streaming for LiveKit integration
"""
import asyncio
import base64
import json
import logging
import numpy as np
import cv2
import tempfile
import os
import time
from typing import Optional, Dict, Any
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# Add the current directory to the path
import sys
sys.path.append('/workspace')

# Import Ditto components
from stream_pipeline_online import StreamSDK

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class WebSocketStreamingWriter:
    """WebSocket video writer that streams frames in real-time"""
    
    def __init__(self, websocket: WebSocket):
        self.websocket = websocket
        self.frames_streamed = 0
        self.is_active = True
        
    async def __call__(self, frame_rgb, fmt="rgb"):
        """Called by pipeline to write each frame - stream via WebSocket"""
        if not self.is_active:
            return
            
        try:
            # Convert to BGR for OpenCV
            if fmt == "rgb":
                frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            else:
                frame_bgr = frame_rgb
            
            # Encode frame as JPEG for streaming
            _, buffer = cv2.imencode('.jpg', frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
            frame_base64 = base64.b64encode(buffer).decode('utf-8')
            
            # Send frame via WebSocket
            await self.websocket.send_json({
                "type": "frame",
                "data": frame_base64,
                "frame_number": self.frames_streamed,
                "timestamp": time.time()
            })
            
            self.frames_streamed += 1
                    
        except Exception as e:
            logger.error(f"Frame streaming error: {e}")
    
    def close(self):
        """Close the streaming writer"""
        self.is_active = False


class DittoAvatarSession:
    """Manages a single avatar streaming session"""
    
    def __init__(self, session_id: str, websocket: WebSocket):
        self.session_id = session_id
        self.websocket = websocket
        self.sdk = None
        self.writer = None
        self.is_active = False
        self.audio_buffer = np.array([], dtype=np.float32)
        self.chunk_size = 6400  # 400ms at 16kHz for HuBERT
        
    async def initialize(self, avatar_image_b64: str, settings: Dict[str, Any]):
        """Initialize the Ditto SDK with avatar"""
        try:
            # Save avatar image temporarily
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_file:
                tmp_file.write(base64.b64decode(avatar_image_b64))
                avatar_path = tmp_file.name
            
            # Select model (prefer TensorRT if available)
            try:
                import tensorrt
                data_root = "/workspace/checkpoints/ditto_trt_Ampere_Plus"
                cfg_pkl = "/workspace/checkpoints/ditto_cfg/v0.4_hubert_cfg_trt_online.pkl"
                logger.info("Using TensorRT model for best performance")
            except ImportError:
                data_root = "/workspace/checkpoints/ditto_pytorch"
                cfg_pkl = "/workspace/checkpoints/ditto_cfg/v0.4_hubert_cfg_pytorch.pkl"
                logger.info("Using PyTorch model")
            
            # Initialize SDK
            self.sdk = StreamSDK(cfg_pkl, data_root)
            
            # Create WebSocket writer
            self.writer = WebSocketStreamingWriter(self.websocket)
            
            # Setup SDK with real-time settings
            setup_kwargs = {
                "online_mode": True,
                "max_size": settings.get("max_size", 1280),
                "sampling_timesteps": settings.get("sampling_timesteps", 25),
                "crop_scale": settings.get("crop_scale", 2.3),
                "crop_vx_ratio": settings.get("crop_vx_ratio", 0.0),
                "crop_vy_ratio": settings.get("crop_vy_ratio", -0.125),
                "smo_k_s": settings.get("smo_k_s", 5),
                "smo_k_d": settings.get("smo_k_d", 2),
                "overlap_v2": settings.get("overlap_v2", 5),
                "emo": settings.get("emo", 3),
                "relative_d": settings.get("relative_d", True),
                "flag_stitching": settings.get("flag_stitching", True),
            }
            
            # Apply amplitude controls
            mouth_amp = settings.get("mouth_amplitude", 0.8)
            head_amp = settings.get("head_amplitude", 0.6)  
            eye_amp = settings.get("eye_amplitude", 0.9)
            
            if mouth_amp != 1.0 or head_amp != 1.0 or eye_amp != 1.0:
                use_d_keys = {}
                if mouth_amp != 1.0:
                    for i in range(20, 40):  # Mouth movements
                        use_d_keys[i] = mouth_amp
                if head_amp != 1.0:
                    for i in range(0, 6):  # Head pose
                        use_d_keys[i] = head_amp
                if eye_amp != 1.0:
                    for i in range(40, 50):  # Eye movements
                        use_d_keys[i] = eye_amp
                setup_kwargs["use_d_keys"] = use_d_keys
            
            # Setup SDK
            self.sdk.setup(avatar_path, "/dev/null", **setup_kwargs)
            
            # Replace writer with WebSocket writer
            self.sdk.writer = self.writer
            
            self.is_active = True
            
            # Clean up temp file
            os.unlink(avatar_path)
            
            logger.info(f"Session {self.session_id} initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize session {self.session_id}: {e}")
            return False
    
    async def process_audio_chunk(self, audio_data: bytes):
        """Process incoming audio chunk"""
        if not self.is_active:
            return
        
        try:
            # Convert bytes to numpy array
            audio_chunk = np.frombuffer(audio_data, dtype=np.float32)
            
            # Add to buffer
            self.audio_buffer = np.concatenate([self.audio_buffer, audio_chunk])
            
            # Process when we have enough audio for HuBERT
            while len(self.audio_buffer) >= self.chunk_size:
                # Extract chunk
                chunk = self.audio_buffer[:self.chunk_size]
                self.audio_buffer = self.audio_buffer[self.chunk_size:]
                
                # Process chunk through Ditto pipeline
                self.sdk.run_chunk(chunk, chunksize=(3, 5, 2))
                
        except Exception as e:
            logger.error(f"Audio processing error: {e}")
    
    async def close(self):
        """Close the session and clean up resources"""
        self.is_active = False
        
        if self.sdk:
            try:
                self.sdk.close()
            except:
                pass
        
        if self.writer:
            self.writer.close()
        
        logger.info(f"Session {self.session_id} closed")


class DittoStreamingService:
    """Main service managing multiple streaming sessions"""
    
    def __init__(self):
        self.sessions: Dict[str, DittoAvatarSession] = {}
        
    async def create_session(self, websocket: WebSocket) -> DittoAvatarSession:
        """Create a new streaming session"""
        session_id = f"session_{int(time.time() * 1000)}"
        session = DittoAvatarSession(session_id, websocket)
        self.sessions[session_id] = session
        return session
    
    async def remove_session(self, session_id: str):
        """Remove and clean up a session"""
        if session_id in self.sessions:
            await self.sessions[session_id].close()
            del self.sessions[session_id]


# Initialize FastAPI app
app = FastAPI(title="Ditto Avatar WebSocket Streaming Service")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize service
service = DittoStreamingService()


@app.get("/")
async def root():
    """Root endpoint with service info"""
    return {
        "service": "Ditto Avatar WebSocket Streaming",
        "version": "1.0.0",
        "status": "running",
        "active_sessions": len(service.sessions),
        "websocket_endpoint": "/ws",
        "health_endpoint": "/health"
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "active_sessions": len(service.sessions),
        "timestamp": time.time()
    }


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time avatar streaming"""
    await websocket.accept()
    session = await service.create_session(websocket)
    
    try:
        # Wait for initialization message
        init_msg = await websocket.receive_json()
        
        if init_msg.get("type") == "init":
            # Initialize session with avatar
            avatar_b64 = init_msg.get("avatar_image_base64")
            settings = init_msg.get("settings", {})
            
            if not avatar_b64:
                await websocket.send_json({
                    "type": "error",
                    "message": "Missing avatar_image_base64"
                })
                return
            
            # Initialize Ditto SDK
            success = await session.initialize(avatar_b64, settings)
            
            if not success:
                await websocket.send_json({
                    "type": "error",
                    "message": "Failed to initialize avatar session"
                })
                return
            
            # Send ready signal
            await websocket.send_json({
                "type": "ready",
                "session_id": session.session_id,
                "message": "Avatar session ready for streaming"
            })
            
            # Process incoming audio chunks
            while True:
                # Receive data (can be JSON or binary)
                data = await websocket.receive()
                
                if "text" in data:
                    # JSON message (control messages)
                    msg = json.loads(data["text"])
                    
                    if msg.get("type") == "close":
                        break
                    elif msg.get("type") == "settings":
                        # Update settings if needed
                        pass
                        
                elif "bytes" in data:
                    # Binary audio data
                    await session.process_audio_chunk(data["bytes"])
                    
    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected for session {session.session_id}")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        await websocket.send_json({
            "type": "error",
            "message": str(e)
        })
    finally:
        # Clean up session
        await service.remove_session(session.session_id)


@app.get("/demo")
async def demo_page():
    """Serve a demo HTML page for testing"""
    return HTMLResponse(content="""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Ditto Avatar WebSocket Demo</title>
        <style>
            body { font-family: Arial; max-width: 1200px; margin: 0 auto; padding: 20px; }
            .container { display: flex; gap: 20px; }
            .controls { flex: 1; }
            .output { flex: 1; }
            video { width: 100%; max-width: 512px; border: 2px solid #333; }
            button { padding: 10px 20px; margin: 5px; font-size: 16px; }
            .status { padding: 10px; background: #f0f0f0; margin: 10px 0; }
        </style>
    </head>
    <body>
        <h1>Ditto Avatar WebSocket Streaming Demo</h1>
        
        <div class="status" id="status">Status: Disconnected</div>
        
        <div class="container">
            <div class="controls">
                <h3>Controls</h3>
                <input type="file" id="avatarFile" accept="image/*">
                <br><br>
                <button id="connectBtn">Connect</button>
                <button id="startBtn" disabled>Start Streaming</button>
                <button id="stopBtn" disabled>Stop</button>
            </div>
            
            <div class="output">
                <h3>Avatar Output</h3>
                <video id="output" autoplay></video>
                <div>Frames: <span id="frameCount">0</span></div>
            </div>
        </div>
        
        <script>
            let ws = null;
            let isStreaming = false;
            let frameCount = 0;
            let avatarImage = null;
            
            const statusEl = document.getElementById('status');
            const frameCountEl = document.getElementById('frameCount');
            const video = document.getElementById('output');
            
            document.getElementById('avatarFile').onchange = (e) => {
                const file = e.target.files[0];
                if (file) {
                    const reader = new FileReader();
                    reader.onload = (e) => {
                        avatarImage = e.target.result.split(',')[1];
                        statusEl.textContent = 'Status: Avatar loaded';
                    };
                    reader.readAsDataURL(file);
                }
            };
            
            document.getElementById('connectBtn').onclick = async () => {
                if (!avatarImage) {
                    alert('Please select an avatar image first');
                    return;
                }
                
                const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
                const wsUrl = protocol + '//' + window.location.host + '/ws';
                
                ws = new WebSocket(wsUrl);
                
                ws.onopen = () => {
                    statusEl.textContent = 'Status: Connected - Initializing...';
                    
                    // Send initialization message
                    ws.send(JSON.stringify({
                        type: 'init',
                        avatar_image_base64: avatarImage,
                        settings: {
                            max_size: 512,
                            sampling_timesteps: 25,
                            mouth_amplitude: 0.8,
                            head_amplitude: 0.6
                        }
                    }));
                };
                
                ws.onmessage = (event) => {
                    const data = JSON.parse(event.data);
                    
                    if (data.type === 'ready') {
                        statusEl.textContent = 'Status: Ready to stream';
                        document.getElementById('startBtn').disabled = false;
                        document.getElementById('connectBtn').disabled = true;
                    } else if (data.type === 'frame') {
                        displayFrame(data.data);
                        frameCount++;
                        frameCountEl.textContent = frameCount;
                    } else if (data.type === 'error') {
                        statusEl.textContent = 'Error: ' + data.message;
                    }
                };
                
                ws.onclose = () => {
                    statusEl.textContent = 'Status: Disconnected';
                    document.getElementById('connectBtn').disabled = false;
                    document.getElementById('startBtn').disabled = true;
                    document.getElementById('stopBtn').disabled = true;
                };
            };
            
            document.getElementById('startBtn').onclick = async () => {
                // Start audio capture and streaming
                const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
                const audioContext = new AudioContext({ sampleRate: 16000 });
                const source = audioContext.createMediaStreamSource(stream);
                const processor = audioContext.createScriptProcessor(1024, 1, 1);
                
                processor.onaudioprocess = (event) => {
                    if (isStreaming && ws.readyState === WebSocket.OPEN) {
                        const inputData = event.inputBuffer.getChannelData(0);
                        const buffer = new Float32Array(inputData);
                        ws.send(buffer.buffer);
                    }
                };
                
                source.connect(processor);
                processor.connect(audioContext.destination);
                
                isStreaming = true;
                document.getElementById('startBtn').disabled = true;
                document.getElementById('stopBtn').disabled = false;
                statusEl.textContent = 'Status: Streaming';
            };
            
            document.getElementById('stopBtn').onclick = () => {
                isStreaming = false;
                if (ws) {
                    ws.send(JSON.stringify({ type: 'close' }));
                    ws.close();
                }
                document.getElementById('startBtn').disabled = true;
                document.getElementById('stopBtn').disabled = true;
                document.getElementById('connectBtn').disabled = false;
            };
            
            function displayFrame(base64Data) {
                const img = new Image();
                img.onload = () => {
                    const canvas = document.createElement('canvas');
                    canvas.width = img.width;
                    canvas.height = img.height;
                    const ctx = canvas.getContext('2d');
                    ctx.drawImage(img, 0, 0);
                    
                    canvas.toBlob(blob => {
                        const url = URL.createObjectURL(blob);
                        video.src = url;
                    }, 'image/jpeg');
                };
                img.src = 'data:image/jpeg;base64,' + base64Data;
            }
        </script>
    </body>
    </html>
    """)


if __name__ == "__main__":
    # Run the server
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8888,
        log_level="info"
    )
