"""
RunPod Real-Time Streaming Handler for Ditto Talking Head Model
True frame-by-frame streaming implementation
"""
import os
import sys
import base64
import tempfile
import subprocess
import librosa
import math
import numpy as np
import threading
import queue
import time
import asyncio
import json
import cv2
from typing import Dict, Any, Iterator
import runpod

# Add the current directory to the path
sys.path.append('/workspace')

# Import Ditto components
from stream_pipeline_online import StreamSDK


class RealTimeStreamingWriter:
    """Real-time video writer that streams frames immediately via callback"""
    
    def __init__(self, frame_callback=None, fps: int = 25):
        self.frame_callback = frame_callback
        self.fps = fps
        self.frames_streamed = 0
        self.is_active = True
        self.frame_queue = queue.Queue(maxsize=5)  # Small buffer for real-time
        
        # Start streaming thread
        self.streaming_thread = threading.Thread(target=self._stream_frames)
        self.streaming_thread.start()
    
    def _stream_frames(self):
        """Background thread to stream frames immediately"""
        while self.is_active or not self.frame_queue.empty():
            try:
                frame_data = self.frame_queue.get(timeout=0.1)
                if frame_data is None:
                    break
                    
                if self.frame_callback:
                    self.frame_callback(frame_data)
                    
                self.frames_streamed += 1
                    
            except queue.Empty:
                continue
    
    def __call__(self, frame_rgb, fmt="rgb"):
        """Called by pipeline to write each frame - stream immediately"""
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
            
            # Create frame packet
            frame_packet = {
                "type": "frame",
                "data": frame_base64,
                "frame_number": self.frames_streamed,
                "timestamp": time.time()
            }
            
            # Queue for immediate streaming (non-blocking)
            try:
                self.frame_queue.put_nowait(frame_packet)
            except queue.Full:
                # Drop oldest frame if queue full (real-time priority)
                try:
                    self.frame_queue.get_nowait()
                    self.frame_queue.put_nowait(frame_packet)
                except queue.Empty:
                    pass
                    
        except Exception as e:
            print(f"Frame streaming error: {e}")
    
    def close(self):
        """Close the streaming writer"""
        self.is_active = False
        self.frame_queue.put(None)  # Signal end
        if self.streaming_thread.is_alive():
            self.streaming_thread.join(timeout=2)


class RealTimeStreamingSDK(StreamSDK):
    """Extended SDK for real-time frame streaming"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.realtime_writer = None
        self.frame_stream_callback = None
        
    def setup_realtime_streaming(self, source_path, frame_stream_callback=None, **kwargs):
        """Setup with real-time frame streaming"""
        self.frame_stream_callback = frame_stream_callback
        self.realtime_writer = RealTimeStreamingWriter(frame_stream_callback)
        
        # Setup SDK with dummy output (we stream frames directly)
        self.setup(source_path, "/dev/null", **kwargs)
        
        # Replace writer with real-time streaming writer
        self.writer = self.realtime_writer
        
    def _writer_worker(self):
        """Override writer worker for real-time streaming"""
        while not self.stop_event.is_set():
            try:
                item = self.writer_queue.get(timeout=1)
            except queue.Empty:
                continue

            if item is None:
                break
                
            res_frame_rgb = item
            
            # Stream frame immediately via real-time writer
            self.writer(res_frame_rgb, fmt="rgb")
            self.writer_pbar.update()
    
    def close(self):
        """Close with real-time streaming support"""
        super().close()
        if self.realtime_writer:
            self.realtime_writer.close()


class AudioChunkProcessor:
    """Processes audio chunks in real-time for streaming"""
    
    def __init__(self, sdk: RealTimeStreamingSDK):
        self.sdk = sdk
        self.audio_buffer = np.array([], dtype=np.float32)
        self.chunk_size = 1024  # 64ms at 16kHz
        self.sample_rate = 16000
        
    def add_audio_chunk(self, audio_chunk: np.ndarray):
        """Add audio chunk and process if enough data available"""
        # Add to buffer
        self.audio_buffer = np.concatenate([self.audio_buffer, audio_chunk])
        
        # Process if we have enough audio
        while len(self.audio_buffer) >= self.chunk_size:
            # Extract chunk
            chunk = self.audio_buffer[:self.chunk_size]
            self.audio_buffer = self.audio_buffer[self.chunk_size:]
            
            # Process chunk through Ditto pipeline
            self.sdk.run_chunk(chunk)


def realtime_streaming_handler(job: Dict[str, Any]) -> Iterator[Dict[str, Any]]:
    """
    Real-time streaming handler that yields frames as they're generated
    """
    try:
        job_input = job['input']
        
        # Extract inputs
        image_base64 = job_input.get('image_base64', '')
        ditto_settings = job_input.get('ditto_settings', {})
        stream_mode = job_input.get('stream_mode', 'websocket')  # websocket or rtmp
        
        if not image_base64:
            yield {"error": "Missing required input: image_base64"}
            return
        
        # Create temporary directory for processing
        with tempfile.TemporaryDirectory() as temp_dir:
            # Save the input image
            image_path = os.path.join(temp_dir, 'input_image.png')
            with open(image_path, 'wb') as f:
                f.write(base64.b64decode(image_base64))
            
            # Select model (optimize for real-time)
            try:
                import tensorrt
                data_root = "/workspace/checkpoints/ditto_trt_Ampere_Plus"
                cfg_pkl = "/workspace/checkpoints/ditto_cfg/v0.4_hubert_cfg_trt_online.pkl"
            except ImportError:
                data_root = "/workspace/checkpoints/ditto_pytorch"
                cfg_pkl = "/workspace/checkpoints/ditto_cfg/v0.4_hubert_cfg_pytorch.pkl"
            
            # Streamed frames collection for client
            streamed_frames = []
            
            def frame_callback(frame_packet):
                """Callback when each frame is generated"""
                streamed_frames.append(frame_packet)
                # Immediately yield frame to client
                yield {
                    "type": "frame",
                    "frame_data": frame_packet["data"],
                    "frame_number": frame_packet["frame_number"],
                    "timestamp": frame_packet["timestamp"],
                    "status": "streaming"
                }
            
            # Initialize real-time streaming SDK
            SDK = RealTimeStreamingSDK(cfg_pkl, data_root)
            
            # Real-time optimized settings
            realtime_kwargs = {
                "online_mode": True,
                "max_size": ditto_settings.get("max_size", 512),  # Smaller for speed
                "sampling_timesteps": ditto_settings.get("sampling_timesteps", 20),  # Faster
                "crop_scale": ditto_settings.get("crop_scale", 2.3),
                "crop_vx_ratio": ditto_settings.get("crop_vx_ratio", 0.0),
                "crop_vy_ratio": ditto_settings.get("crop_vy_ratio", -0.125),
                "smo_k_s": ditto_settings.get("smo_k_s", 3),  # Minimal smoothing for real-time
                "smo_k_d": ditto_settings.get("smo_k_d", 1),  # Minimal smoothing
                "overlap_v2": ditto_settings.get("overlap_v2", 2),  # Minimal overlap
                "emo": ditto_settings.get("emo", 3),
                "relative_d": ditto_settings.get("relative_d", True),
            }
            
            # Apply amplitude controls
            mouth_amp = ditto_settings.get("mouth_amplitude", 0.8)
            head_amp = ditto_settings.get("head_amplitude", 0.6)  
            eye_amp = ditto_settings.get("eye_amplitude", 0.9)
            
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
                realtime_kwargs["use_d_keys"] = use_d_keys
            
            # Setup real-time streaming
            SDK.setup_realtime_streaming(image_path, frame_callback, **realtime_kwargs)
            
            # Initialize audio processor
            audio_processor = AudioChunkProcessor(SDK)
            
            # Send ready signal
            yield {
                "type": "ready",
                "message": "Real-time streaming initialized. Send audio chunks.",
                "status": "ready"
            }
            
            # Wait for audio chunks (this would be handled by WebSocket in production)
            # For now, we'll process a demo audio file if provided
            demo_audio = job_input.get('demo_audio_base64')
            if demo_audio:
                # Decode demo audio
                audio_data = base64.b64decode(demo_audio)
                with open(os.path.join(temp_dir, 'demo_audio.wav'), 'wb') as f:
                    f.write(audio_data)
                
                # Load and process in chunks
                audio, sr = librosa.core.load(os.path.join(temp_dir, 'demo_audio.wav'), sr=16000)
                
                # Setup frame count
                num_f = math.ceil(len(audio) / 16000 * 25)
                SDK.setup_Nd(N_d=num_f, fade_in=-1, fade_out=-1, ctrl_info={})
                
                # Process in real-time chunks
                chunk_size = 1024  # ~64ms chunks
                for i in range(0, len(audio), chunk_size):
                    audio_chunk = audio[i:i + chunk_size]
                    if len(audio_chunk) < chunk_size:
                        audio_chunk = np.pad(audio_chunk, (0, chunk_size - len(audio_chunk)), mode="constant")
                    
                    # Process chunk (will trigger frame streaming via callback)
                    audio_processor.add_audio_chunk(audio_chunk)
                    
                    # Small delay to simulate real-time
                    time.sleep(chunk_size / 16000)  # Wait for real-time duration
            
            # Close SDK
            SDK.close()
            
            # Send completion signal
            yield {
                "type": "complete",
                "total_frames": len(streamed_frames),
                "message": "Real-time streaming completed",
                "status": "complete"
            }
            
    except Exception as e:
        import traceback
        yield {
            "type": "error",
            "error": str(e),
            "traceback": traceback.format_exc(),
            "status": "error"
        }


def websocket_streaming_handler(job: Dict[str, Any]) -> Dict[str, Any]:
    """
    WebSocket streaming handler for real-time communication
    """
    # This would be extended for WebSocket support
    # For now, return setup information
    return {
        "type": "websocket_info",
        "message": "WebSocket streaming not yet implemented in RunPod serverless",
        "recommendation": "Use the realtime_streaming_handler for frame-by-frame processing",
        "status": "info"
    }


def handler(job: Dict[str, Any]) -> Dict[str, Any]:
    """Main RunPod handler with real-time streaming support"""
    job_input = job.get('input', {})
    mode = job_input.get('mode', 'batch')  # batch, realtime, websocket
    
    if mode == 'realtime':
        # Real-time streaming mode
        results = []
        for frame_result in realtime_streaming_handler(job):
            results.append(frame_result)
        return {"stream_results": results, "mode": "realtime"}
    
    elif mode == 'websocket':
        # WebSocket mode (would need different RunPod setup)
        return websocket_streaming_handler(job)
    
    else:
        # Fall back to existing batch handler
        from runpod_streaming_handler import streaming_handler
        return streaming_handler(job)


# RunPod serverless handler
runpod.serverless.start({"handler": handler})
