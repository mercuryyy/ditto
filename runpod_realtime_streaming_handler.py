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
        # Use larger chunk size for HuBERT feature extraction
        # Based on official code: overlap_v2 * 640 (typically 10 * 640 = 6400)
        self.chunk_size = 6400  # 400ms at 16kHz - proper size for HuBERT
        self.sample_rate = 16000
        self.chunks_processed = 0
        
    def add_audio_chunk(self, audio_chunk: np.ndarray):
        """Add audio chunk and process if enough data available"""
        # Add to buffer
        self.audio_buffer = np.concatenate([self.audio_buffer, audio_chunk])
        
        # Process when we have enough audio for HuBERT
        while len(self.audio_buffer) >= self.chunk_size:
            # Extract chunk
            chunk = self.audio_buffer[:self.chunk_size]
            self.audio_buffer = self.audio_buffer[self.chunk_size:]
            
            # Process chunk through Ditto pipeline with default chunksize for HuBERT
            try:
                # Use the default chunksize parameter that HuBERT expects
                self.sdk.run_chunk(chunk, chunksize=(3, 5, 2))
                self.chunks_processed += 1
                print(f"Processed audio chunk {self.chunks_processed} (size: {len(chunk)})")
            except Exception as e:
                print(f"Error processing chunk: {e}")


async def realtime_streaming_handler(job: Dict[str, Any]) -> Iterator[Dict[str, Any]]:
    """
    Real-time streaming handler that yields frames as they're generated
    """
    try:
        job_input = job['input']
        
        # Extract inputs
        image_base64 = job_input.get('image_base64', '')
        audio_chunks_base64 = job_input.get('audio_chunks_base64', [])  # Direct audio chunks
        full_audio_base64 = job_input.get('full_audio_base64', '')  # Fallback for batch mode
        ditto_settings = job_input.get('ditto_settings', {})
        stream_mode = job_input.get('stream_mode', 'chunks')  # chunks, websocket, or batch
        
        if not image_base64:
            yield {"error": "Missing required input: image_base64"}
            return
        
        # Check for either chunked audio or full audio
        if not audio_chunks_base64 and not full_audio_base64:
            yield {"error": "Missing required input: audio_chunks_base64 or full_audio_base64"}
            return
        
        # Create temporary directory for processing
        with tempfile.TemporaryDirectory() as temp_dir:
            # Save the input image - decode and save properly
            image_path = os.path.join(temp_dir, 'input_image.png')
            image_data = base64.b64decode(image_base64)
            
            # Save using OpenCV to ensure proper PNG format
            import numpy as np
            from PIL import Image
            import io
            
            # Load image from bytes
            try:
                # First, try to save the raw bytes directly to file
                # This often works better than going through PIL
                with open(image_path, 'wb') as f:
                    f.write(image_data)
                
                # Now verify the file is a valid image using OpenCV
                test_img = cv2.imread(image_path)
                if test_img is None:
                    # If OpenCV can't read it, try to fix it with PIL
                    print(f"OpenCV couldn't read image, trying PIL conversion...")
                    
                    # Load with PIL and re-save
                    from PIL import Image
                    image = Image.open(image_path)
                    
                    # Convert to RGB if needed
                    if image.mode == 'RGBA':
                        image = image.convert('RGB')
                    elif image.mode != 'RGB':
                        image = image.convert('RGB')
                    
                    # Save back with proper format
                    image.save(image_path, 'PNG')
                    
                    # Try OpenCV again
                    test_img = cv2.imread(image_path)
                    if test_img is None:
                        yield {"error": f"Image saved but still cannot be read: {image_path}"}
                        return
                
                # Check if the image has a face (basic check)
                h, w = test_img.shape[:2]
                if h < 64 or w < 64:
                    yield {"error": f"Image too small: {w}x{h}, minimum 64x64 required"}
                    return
                    
                print(f"Image successfully saved to {image_path}, size: {test_img.shape}")
                
                # Additional debug info
                import filetype
                kind = filetype.guess(image_path)
                if kind is None:
                    print(f"Warning: filetype couldn't identify the file type")
                else:
                    print(f"File type detected: {kind.mime}")
                
            except Exception as e:
                import traceback
                yield {
                    "error": f"Failed to process image: {str(e)}",
                    "traceback": traceback.format_exc(),
                    "status": "error"
                }
                return
            
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
                print(f"Frame {frame_packet['frame_number']} generated and stored")
            
            # Initialize real-time streaming SDK
            SDK = RealTimeStreamingSDK(cfg_pkl, data_root)
            
            # Optimized settings for chunked audio processing with quality lip-sync
            realtime_kwargs = {
                "online_mode": True,
                "max_size": ditto_settings.get("max_size", 1280),  # Higher for quality
                "sampling_timesteps": ditto_settings.get("sampling_timesteps", 25),  # Balanced
                "crop_scale": ditto_settings.get("crop_scale", 2.3),
                "crop_vx_ratio": ditto_settings.get("crop_vx_ratio", 0.0),
                "crop_vy_ratio": ditto_settings.get("crop_vy_ratio", -0.125),
                "smo_k_s": ditto_settings.get("smo_k_s", 7),  # Higher smoothing for quality
                "smo_k_d": ditto_settings.get("smo_k_d", 3),  # More temporal smoothing
                "overlap_v2": ditto_settings.get("overlap_v2", 5),  # Higher overlap for continuity
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
                "message": "Real-time streaming initialized. Processing audio input.",
                "status": "ready"
            }
            
            # Process audio input based on mode
            if audio_chunks_base64:
                # TRUE REAL-TIME STREAMING - process chunks as they arrive
                yield {
                    "type": "processing",
                    "message": f"Starting real-time processing of {len(audio_chunks_base64)} audio chunks",
                    "status": "processing"
                }
                
                # Calculate expected frames based on total audio duration
                total_audio_samples = len(audio_chunks_base64) * 6400  # Assuming 6400 samples per chunk
                num_f = math.ceil(total_audio_samples / 16000 * 25)
                SDK.setup_Nd(N_d=num_f, fade_in=-1, fade_out=-1, ctrl_info={})
                
                # Start a background thread to monitor frame generation
                frame_monitor_active = True
                last_frame_count = 0
                
                def monitor_frames():
                    nonlocal last_frame_count
                    while frame_monitor_active:
                        current_count = len(streamed_frames)
                        if current_count > last_frame_count:
                            print(f"Frames generated: {current_count}")
                            last_frame_count = current_count
                        time.sleep(0.1)
                
                monitor_thread = threading.Thread(target=monitor_frames)
                monitor_thread.start()
                
                # Process each audio chunk in real-time
                for i, chunk_b64 in enumerate(audio_chunks_base64):
                    try:
                        # Decode audio chunk
                        chunk_bytes = base64.b64decode(chunk_b64)
                        audio_chunk = np.frombuffer(chunk_bytes, dtype=np.float32)
                        
                        # Ensure chunk is the right size (6400 samples for HuBERT)
                        if len(audio_chunk) != 6400:
                            if len(audio_chunk) < 6400:
                                audio_chunk = np.pad(audio_chunk, (0, 6400 - len(audio_chunk)), mode='constant')
                            else:
                                audio_chunk = audio_chunk[:6400]
                        
                        # Feed audio chunk to pipeline - this triggers real-time processing
                        audio_processor.add_audio_chunk(audio_chunk)
                        
                        # Small async sleep to allow frames to be generated
                        # This simulates real-time audio arrival
                        await asyncio.sleep(0.4)  # 400ms per chunk (6400 samples at 16kHz)
                        
                        # Yield intermediate progress with current frame count
                        if i % 2 == 0:  # Every 2 chunks (~800ms)
                            yield {
                                "type": "progress",
                                "processed_chunks": i + 1,
                                "total_chunks": len(audio_chunks_base64),
                                "frames_generated": len(streamed_frames),
                                "status": "streaming"
                            }
                    
                    except Exception as e:
                        print(f"Error processing chunk {i}: {e}")
                        continue
                
                # After all chunks are processed, signal end and flush
                print("All audio chunks processed, flushing pipeline...")
                SDK.close()
                
                # Wait for final frames to be generated
                await asyncio.sleep(1.0)
                
                # Stop frame monitor
                frame_monitor_active = False
                monitor_thread.join(timeout=1)
                
                yield {
                    "type": "processing_complete",
                    "message": f"Real-time streaming complete. Generated {len(streamed_frames)} frames",
                    "status": "processing"
                }
                
            elif full_audio_base64:
                # Process full audio in optimized chunks
                try:
                    # Decode full audio
                    audio_bytes = base64.b64decode(full_audio_base64)
                    
                    # Save and load audio file
                    audio_path = os.path.join(temp_dir, 'input_audio.wav')
                    with open(audio_path, 'wb') as f:
                        f.write(audio_bytes)
                    
                    # Load and resample to 16kHz mono
                    audio, sr = librosa.core.load(audio_path, sr=16000, mono=True)
                    
                    yield {
                        "type": "processing", 
                        "message": f"Processing {len(audio)/16000:.2f} seconds of audio",
                        "status": "processing"
                    }
                    
                    # Setup frame count
                    num_f = math.ceil(len(audio) / 16000 * 25)
                    SDK.setup_Nd(N_d=num_f, fade_in=-1, fade_out=-1, ctrl_info={})
                    
                    # Process in overlapped chunks for optimal lip-sync
                    chunk_size = 1024  # 64ms chunks
                    overlap_size = 256  # 25% overlap
                    step_size = chunk_size - overlap_size
                    
                    for i in range(0, len(audio) - chunk_size + 1, step_size):
                        audio_chunk = audio[i:i + chunk_size]
                        if len(audio_chunk) < chunk_size:
                            audio_chunk = np.pad(audio_chunk, (0, chunk_size - len(audio_chunk)), mode="constant")
                        
                        # Process chunk (will trigger frame streaming via callback)
                        audio_processor.add_audio_chunk(audio_chunk)
                        
                        # Real-time delay
                        await asyncio.sleep(step_size / 16000)
                        
                    # Process final chunk if any remaining audio
                    if len(audio) % step_size > 0:
                        final_chunk = audio[-(len(audio) % step_size):]
                        final_chunk = np.pad(final_chunk, (0, chunk_size - len(final_chunk)), mode="constant")
                        audio_processor.add_audio_chunk(final_chunk)
                    
                except Exception as e:
                    yield {
                        "type": "error",
                        "error": f"Failed to process full audio: {str(e)}",
                        "status": "error"
                    }
                    return
            
            else:
                yield {
                    "type": "error", 
                    "error": "No valid audio input provided",
                    "status": "error"
                }
                return
            
            # Don't close SDK here - it's already closed after processing
            
            # Send completion signal with frame count
            yield {
                "type": "complete",
                "total_frames": len(streamed_frames),
                "message": f"Real-time streaming completed. Generated {len(streamed_frames)} frames",
                "frames_generated": len(streamed_frames),
                "status": "complete"
            }
            
            # For debugging: Include first few frames in output (in production these would stream via WebSocket)
            if streamed_frames and len(streamed_frames) > 0:
                # Include a sample frame to verify generation
                sample_frames = min(3, len(streamed_frames))  # Up to 3 frames for testing
                for i in range(sample_frames):
                    yield {
                        "type": "sample_frame",
                        "frame_number": streamed_frames[i]["frame_number"],
                        "has_data": len(streamed_frames[i]["data"]) > 0,
                        "data_size": len(streamed_frames[i]["data"]),
                        "status": "frame"
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
        # Real-time streaming mode - run async handler
        async def run_streaming():
            results = []
            async for frame_result in realtime_streaming_handler(job):
                results.append(frame_result)
            return results
        
        # Check if we're already in an event loop
        import asyncio
        import nest_asyncio
        nest_asyncio.apply()  # Allow nested event loops
        
        # Run the async handler
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        results = loop.run_until_complete(run_streaming())
        loop.close()
        
        return {"stream_results": results, "mode": "realtime"}
    
    elif mode == 'websocket':
        # WebSocket mode (would need different RunPod setup)
        return websocket_streaming_handler(job)
    
    else:
        # Default to realtime mode for batch requests
        job_input['mode'] = 'realtime'  # Force realtime mode
        
        async def run_streaming():
            results = []
            async for frame_result in realtime_streaming_handler(job):
                results.append(frame_result)
            return results
        
        # Use nest_asyncio for compatibility
        import asyncio
        import nest_asyncio
        nest_asyncio.apply()
        
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        results = loop.run_until_complete(run_streaming())
        loop.close()
        
        return {"stream_results": results, "mode": "realtime"}


# RunPod serverless handler
runpod.serverless.start({"handler": handler})
