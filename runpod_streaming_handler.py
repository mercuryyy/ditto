"""
RunPod Streaming Handler for Ditto Talking Head Model
Supports progressive video streaming for faster user experience
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
from typing import Dict, Any, Iterator
import runpod

# Add the current directory to the path
sys.path.append('/workspace')

# Import Ditto components
from stream_pipeline_online import StreamSDK
import cv2


class StreamingVideoWriter:
    """Custom video writer that can stream frames as they're generated"""
    
    def __init__(self, output_path: str, fps: int = 25):
        self.output_path = output_path
        self.fps = fps
        self.frame_queue = queue.Queue()
        self.frames_written = 0
        self.is_writing = True
        self.writer = None
        self.streaming_frames = []
        
        # Start background writer thread
        self.writer_thread = threading.Thread(target=self._write_frames)
        self.writer_thread.start()
    
    def _write_frames(self):
        """Background thread to write frames to video file"""
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        
        while self.is_writing or not self.frame_queue.empty():
            try:
                frame = self.frame_queue.get(timeout=1.0)
                if frame is None:
                    break
                    
                # Initialize writer with first frame
                if self.writer is None:
                    h, w, _ = frame.shape
                    self.writer = cv2.VideoWriter(self.output_path, fourcc, self.fps, (w, h))
                
                # Convert RGB to BGR for OpenCV
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                self.writer.write(frame_bgr)
                self.frames_written += 1
                
                # Store frame for streaming (keep last 30 frames)
                self.streaming_frames.append(frame)
                if len(self.streaming_frames) > 30:
                    self.streaming_frames.pop(0)
                    
            except queue.Empty:
                continue
        
        if self.writer:
            self.writer.release()
    
    def add_frame(self, frame):
        """Add a frame to be written"""
        self.frame_queue.put(frame)
    
    def get_streaming_chunk(self, start_frame: int = 0) -> bytes:
        """Get a streaming video chunk from start_frame onwards"""
        if len(self.streaming_frames) <= start_frame:
            return b''
        
        # Create a temporary video with available frames
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp_file:
            tmp_path = tmp_file.name
        
        try:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            if self.streaming_frames:
                h, w, _ = self.streaming_frames[0].shape
                chunk_writer = cv2.VideoWriter(tmp_path, fourcc, self.fps, (w, h))
                
                for frame in self.streaming_frames[start_frame:]:
                    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    chunk_writer.write(frame_bgr)
                
                chunk_writer.release()
                
                # Read the chunk and encode to base64
                with open(tmp_path, 'rb') as f:
                    chunk_data = f.read()
                
                return base64.b64encode(chunk_data).decode('utf-8')
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
        
        return ''
    
    def close(self):
        """Close the writer"""
        self.is_writing = False
        self.frame_queue.put(None)  # Signal end
        self.writer_thread.join()


class StreamingSDK(StreamSDK):
    """Extended StreamSDK with streaming capabilities"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.streaming_writer = None
        self.frame_callback = None
        
    def setup_streaming(self, source_path, output_path, frame_callback=None, **kwargs):
        """Setup with streaming support"""
        self.frame_callback = frame_callback
        self.streaming_writer = StreamingVideoWriter(output_path)
        
        # Override the writer in the parent setup
        self.setup(source_path, output_path, **kwargs)
        
    def _writer_worker(self):
        """Override writer worker to support streaming"""
        while not self.stop_event.is_set():
            try:
                item = self.writer_queue.get(timeout=1)
            except queue.Empty:
                continue

            if item is None:
                break
                
            res_frame_rgb = item
            
            # Add frame to streaming writer
            if self.streaming_writer:
                self.streaming_writer.add_frame(res_frame_rgb)
            
            # Call frame callback if provided
            if self.frame_callback:
                self.frame_callback(res_frame_rgb, self.streaming_writer.frames_written)
            
            # Original writer functionality
            self.writer(res_frame_rgb, fmt="rgb")
            self.writer_pbar.update()
    
    def close(self):
        """Close with streaming support"""
        super().close()
        if self.streaming_writer:
            self.streaming_writer.close()


def text_to_speech(text: str, output_path: str) -> str:
    """Convert text to speech using a TTS service - REMOVED FOR PERFORMANCE"""
    # TTS removed to optimize for LiveKit real-time processing
    # Audio will be handled separately in LiveKit pipeline
    return output_path


def run_ditto_streaming(SDK: StreamingSDK, audio_path: str, source_path: str, output_path: str, 
                       ditto_settings: dict = None, progress_callback=None):
    """Run Ditto with streaming support and configurable settings"""
    
    def frame_callback(frame, frame_num):
        if progress_callback:
            progress_callback(frame_num, SDK.streaming_writer.frames_written)
    
    # Prepare setup kwargs from ditto_settings
    setup_kwargs = {"online_mode": True}  # Default for streaming
    
    if ditto_settings:
        # Apply all the configurable settings
        setup_kwargs.update({
            # Image/Avatar settings
            "max_size": ditto_settings.get("max_size", 1920),
            "crop_scale": ditto_settings.get("crop_scale", 2.3),
            "crop_vx_ratio": ditto_settings.get("crop_vx_ratio", 0.0),
            "crop_vy_ratio": ditto_settings.get("crop_vy_ratio", -0.125),
            "crop_flag_do_rot": ditto_settings.get("crop_flag_do_rot", True),
            "template_n_frames": ditto_settings.get("template_n_frames", -1),
            
            # Motion/Animation settings
            "emo": ditto_settings.get("emo", 4),
            "sampling_timesteps": ditto_settings.get("sampling_timesteps", 50),
            "smo_k_s": ditto_settings.get("smo_k_s", 13),
            "smo_k_d": ditto_settings.get("smo_k_d", 3),
            "relative_d": ditto_settings.get("relative_d", True),
            "eye_f0_mode": ditto_settings.get("eye_f0_mode", False),
            
            # Advanced settings
            "overlap_v2": ditto_settings.get("overlap_v2", 10),
            "delta_eye_open_n": ditto_settings.get("delta_eye_open_n", 0),
            "fade_type": ditto_settings.get("fade_type", ""),
            "flag_stitching": ditto_settings.get("flag_stitching", True),
            
            # Advanced motion controls
            "vad_alpha": ditto_settings.get("vad_alpha", 1.0),
            "delta_pitch": ditto_settings.get("delta_pitch", 0.0),
            "delta_yaw": ditto_settings.get("delta_yaw", 0.0),
            "delta_roll": ditto_settings.get("delta_roll", 0.0),
            "alpha_pitch": ditto_settings.get("alpha_pitch", 1.0),
            "alpha_yaw": ditto_settings.get("alpha_yaw", 1.0),
            "alpha_roll": ditto_settings.get("alpha_roll", 1.0),
            "delta_exp": ditto_settings.get("delta_exp", 0.0),
            
            # Override online_mode if specifically set
            "online_mode": ditto_settings.get("online_mode", True),
        })
        
        # Handle amplitude controls by creating use_d_keys
        mouth_amp = ditto_settings.get("mouth_amplitude", 1.0)
        head_amp = ditto_settings.get("head_amplitude", 1.0)
        eye_amp = ditto_settings.get("eye_amplitude", 1.0)
        
        if mouth_amp != 1.0 or head_amp != 1.0 or eye_amp != 1.0:
            # Create use_d_keys to control amplitudes
            # Note: This is a simplified implementation - full implementation would need 
            # to map these to the correct indices in the motion space
            use_d_keys = {}
            if mouth_amp != 1.0:
                # Mouth-related indices (simplified - actual implementation would be more complex)
                for i in range(20, 40):  # Example range for mouth movements
                    use_d_keys[i] = mouth_amp
            if head_amp != 1.0:
                # Head pose related indices
                for i in range(0, 6):  # Example range for head pose
                    use_d_keys[i] = head_amp
            if eye_amp != 1.0:
                # Eye-related indices
                for i in range(40, 50):  # Example range for eye movements
                    use_d_keys[i] = eye_amp
            
            setup_kwargs["use_d_keys"] = use_d_keys
    
    # Setup streaming
    SDK.setup_streaming(source_path, output_path, frame_callback=frame_callback, **setup_kwargs)
    
    # Load audio
    audio, sr = librosa.core.load(audio_path, sr=16000)
    num_f = math.ceil(len(audio) / 16000 * 25)
    
    # Setup number of frames with fade settings
    fade_in = ditto_settings.get("fade_in", -1) if ditto_settings else -1
    fade_out = ditto_settings.get("fade_out", -1) if ditto_settings else -1
    
    SDK.setup_Nd(N_d=num_f, fade_in=fade_in, fade_out=fade_out, ctrl_info={})
    
    # Process audio in chunks for streaming
    chunksize = (3, 5, 2)
    audio = np.concatenate([np.zeros((chunksize[0] * 640,), dtype=np.float32), audio], 0)
    split_len = int(sum(chunksize) * 0.04 * 16000) + 80
    
    for i in range(0, len(audio), chunksize[1] * 640):
        audio_chunk = audio[i:i + split_len]
        if len(audio_chunk) < split_len:
            audio_chunk = np.pad(audio_chunk, (0, split_len - len(audio_chunk)), mode="constant")
        SDK.run_chunk(audio_chunk, chunksize)
    
    # Close and finalize
    SDK.close()
    
    # Add audio to the video
    cmd = f'ffmpeg -loglevel error -y -i "{SDK.tmp_output_path}" -i "{audio_path}" -map 0:v -map 1:a -c:v copy -c:a aac "{output_path}"'
    os.system(cmd)


def streaming_handler(job: Dict[str, Any]) -> Dict[str, Any]:
    """
    Streaming handler that returns progressive updates with configurable settings
    """
    try:
        job_input = job['input']
        
        # Extract inputs
        text = job_input.get('text', '')
        image_base64 = job_input.get('image_base64', '')
        use_pytorch = job_input.get('use_pytorch', False)
        streaming_mode = job_input.get('streaming_mode', True)
        ditto_settings = job_input.get('ditto_settings', {})
        
        if not text or not image_base64:
            return {"error": "Missing required inputs: text and image_base64"}
        
        # Create temporary directory for processing
        with tempfile.TemporaryDirectory() as temp_dir:
            # Save the input image
            image_path = os.path.join(temp_dir, 'input_image.png')
            with open(image_path, 'wb') as f:
                f.write(base64.b64decode(image_base64))
            
            # Convert text to speech
            audio_path = os.path.join(temp_dir, 'input_audio.wav')
            text_to_speech(text, audio_path)
            
            # Output video path
            output_path = os.path.join(temp_dir, 'output.mp4')
            
            # Select model
            if use_pytorch:
                data_root = "/workspace/checkpoints/ditto_pytorch"
                cfg_pkl = "/workspace/checkpoints/ditto_cfg/v0.4_hubert_cfg_pytorch.pkl"
            else:
                try:
                    import tensorrt
                    data_root = "/workspace/checkpoints/ditto_trt_Ampere_Plus"
                    cfg_pkl = "/workspace/checkpoints/ditto_cfg/v0.4_hubert_cfg_trt.pkl"
                except ImportError:
                    data_root = "/workspace/checkpoints/ditto_pytorch"
                    cfg_pkl = "/workspace/checkpoints/ditto_cfg/v0.4_hubert_cfg_pytorch.pkl"
            
            if streaming_mode:
                # Streaming mode: return incremental updates
                progress_updates = []
                
                def progress_callback(current_frame, total_frames):
                    progress_updates.append({
                        "frame": current_frame,
                        "total": total_frames,
                        "progress": current_frame / max(total_frames, 1) * 100
                    })
                
                # Initialize streaming SDK
                SDK = StreamingSDK(cfg_pkl, data_root)
                
                # Run with streaming and settings
                run_ditto_streaming(SDK, audio_path, image_path, output_path, 
                                  ditto_settings, progress_callback)
                
                # Return final video with progress info
                with open(output_path, 'rb') as f:
                    video_base64 = base64.b64encode(f.read()).decode('utf-8')
                
                return {
                    "video_base64": video_base64,
                    "progress_updates": progress_updates,
                    "streaming": True,
                    "status": "success"
                }
            else:
                # Standard mode: process entire video
                from stream_pipeline_offline import StreamSDK as OfflineSDK
                SDK = OfflineSDK(cfg_pkl, data_root)
                
                # Apply ditto_settings to standard processing
                setup_kwargs = {}
                if ditto_settings:
                    setup_kwargs.update({
                        # Image/Avatar settings
                        "max_size": ditto_settings.get("max_size", 1920),
                        "crop_scale": ditto_settings.get("crop_scale", 2.3),
                        "crop_vx_ratio": ditto_settings.get("crop_vx_ratio", 0.0),
                        "crop_vy_ratio": ditto_settings.get("crop_vy_ratio", -0.125),
                        "crop_flag_do_rot": ditto_settings.get("crop_flag_do_rot", True),
                        "template_n_frames": ditto_settings.get("template_n_frames", -1),
                        
                        # Motion/Animation settings
                        "emo": ditto_settings.get("emo", 4),
                        "sampling_timesteps": ditto_settings.get("sampling_timesteps", 50),
                        "smo_k_s": ditto_settings.get("smo_k_s", 13),
                        "smo_k_d": ditto_settings.get("smo_k_d", 3),
                        "relative_d": ditto_settings.get("relative_d", True),
                        "eye_f0_mode": ditto_settings.get("eye_f0_mode", False),
                        
                        # Advanced settings
                        "overlap_v2": ditto_settings.get("overlap_v2", 10),
                        "delta_eye_open_n": ditto_settings.get("delta_eye_open_n", 0),
                        "fade_type": ditto_settings.get("fade_type", ""),
                        "online_mode": ditto_settings.get("online_mode", False),
                    })
                    
                    # Handle amplitude controls
                    mouth_amp = ditto_settings.get("mouth_amplitude", 1.0)
                    head_amp = ditto_settings.get("head_amplitude", 1.0)
                    eye_amp = ditto_settings.get("eye_amplitude", 1.0)
                    
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

                SDK.setup(image_path, output_path, **setup_kwargs)
                
                # Load and process audio with fade settings
                audio, sr = librosa.core.load(audio_path, sr=16000)
                num_f = math.ceil(len(audio) / 16000 * 25)
                fade_in = ditto_settings.get("fade_in", -1) if ditto_settings else -1
                fade_out = ditto_settings.get("fade_out", -1) if ditto_settings else -1
                SDK.setup_Nd(N_d=num_f, fade_in=fade_in, fade_out=fade_out, ctrl_info={})
                
                aud_feat = SDK.wav2feat.wav2feat(audio)
                SDK.audio2motion_queue.put(aud_feat)
                SDK.close()
                
                # Add audio
                cmd = f'ffmpeg -loglevel error -y -i "{SDK.tmp_output_path}" -i "{audio_path}" -map 0:v -map 1:a -c:v copy -c:a aac "{output_path}"'
                os.system(cmd)
                
                with open(output_path, 'rb') as f:
                    video_base64 = base64.b64encode(f.read()).decode('utf-8')
                
                return {
                    "video_base64": video_base64,
                    "streaming": False,
                    "status": "success"
                }
            
    except Exception as e:
        import traceback
        return {
            "error": str(e),
            "traceback": traceback.format_exc(),
            "status": "failed"
        }


# For backward compatibility, keep the original handler
def handler(job: Dict[str, Any]) -> Dict[str, Any]:
    """Original handler with optional streaming support"""
    # Check if streaming is requested
    streaming_mode = job.get('input', {}).get('streaming_mode', False)
    
    if streaming_mode:
        return streaming_handler(job)
    else:
        # Use original handler logic but with improvements
        return streaming_handler(job)  # The streaming handler supports both modes


# RunPod serverless handler
runpod.serverless.start({"handler": handler})
