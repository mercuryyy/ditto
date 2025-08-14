"""
RunPod Serverless Handler for Ditto Talking Head Model
"""
import os
import sys
import base64
import tempfile
import subprocess
import librosa
import math
import numpy as np
from typing import Dict, Any
import runpod

# Add the current directory to the path
sys.path.append('/workspace')

# Import Ditto components (using the correct imports from inference.py)
from stream_pipeline_offline import StreamSDK


def check_models():
    """Check if models are present in the container"""
    
    # Check if models exist
    if not os.path.exists('/workspace/checkpoints/ditto_cfg/v0.4_hubert_cfg_pytorch.pkl'):
        raise Exception("Models not found in container. Please rebuild the Docker image.")
        
    print("Models verified successfully")
    return


def text_to_speech(text: str, output_path: str) -> str:
    """Convert text to speech using a TTS service"""
    try:
        # Using gTTS as primary option
        from gtts import gTTS
        tts = gTTS(text=text, lang='en')
        tts.save(output_path)
        print(f"Generated TTS audio: {output_path}")
        return output_path
    except Exception as e:
        print(f"gTTS failed: {e}, trying pyttsx3...")
        try:
            # Fallback to pyttsx3 if gTTS fails
            import pyttsx3
            engine = pyttsx3.init()
            engine.save_to_file(text, output_path)
            engine.runAndWait()
            print(f"Generated TTS audio with pyttsx3: {output_path}")
            return output_path
        except Exception as e2:
            print(f"Both TTS methods failed: gTTS={e}, pyttsx3={e2}")
            raise Exception(f"Text-to-speech failed: {e2}")


def run_ditto(SDK: StreamSDK, audio_path: str, source_path: str, output_path: str):
    """
    Run the Ditto talking head generation (based on inference.py)
    """
    # Setup the SDK
    setup_kwargs = {}
    SDK.setup(source_path, output_path, **setup_kwargs)
    
    # Load audio
    audio, sr = librosa.core.load(audio_path, sr=16000)
    num_f = math.ceil(len(audio) / 16000 * 25)
    
    # Setup number of frames
    SDK.setup_Nd(N_d=num_f, fade_in=-1, fade_out=-1, ctrl_info={})
    
    # Process audio to features and run the pipeline
    online_mode = SDK.online_mode
    if online_mode:
        # Online mode processing
        chunksize = (3, 5, 2)
        audio = np.concatenate([np.zeros((chunksize[0] * 640,), dtype=np.float32), audio], 0)
        split_len = int(sum(chunksize) * 0.04 * 16000) + 80  # 6480
        for i in range(0, len(audio), chunksize[1] * 640):
            audio_chunk = audio[i:i + split_len]
            if len(audio_chunk) < split_len:
                audio_chunk = np.pad(audio_chunk, (0, split_len - len(audio_chunk)), mode="constant")
            SDK.run_chunk(audio_chunk, chunksize)
    else:
        # Offline mode processing
        aud_feat = SDK.wav2feat.wav2feat(audio)
        SDK.audio2motion_queue.put(aud_feat)
    
    # Close and finalize
    SDK.close()
    
    # Add audio to the video using ffmpeg
    cmd = f'ffmpeg -loglevel error -y -i "{SDK.tmp_output_path}" -i "{audio_path}" -map 0:v -map 1:a -c:v copy -c:a aac "{output_path}"'
    print(f"Running ffmpeg: {cmd}")
    os.system(cmd)
    
    print(f"Output video generated: {output_path}")


def handler(job: Dict[str, Any]) -> Dict[str, Any]:
    """
    RunPod handler function
    
    Expected input format:
    {
        "input": {
            "text": "Text to synthesize",
            "image_base64": "base64 encoded image",
            "use_pytorch": false  # Optional, defaults to false (use TensorRT)
        }
    }
    """
    try:
        job_input = job['input']
        
        # Check models are present
        check_models()
        
        # Extract inputs
        text = job_input.get('text', '')
        image_base64 = job_input.get('image_base64', '')
        use_pytorch = job_input.get('use_pytorch', False)
        
        if not text or not image_base64:
            return {"error": "Missing required inputs: text and image_base64"}
        
        # Create temporary directory for processing
        with tempfile.TemporaryDirectory() as temp_dir:
            # Save the input image
            image_path = os.path.join(temp_dir, 'input_image.png')
            with open(image_path, 'wb') as f:
                f.write(base64.b64decode(image_base64))
            print(f"Saved input image: {image_path}")
            
            # Convert text to speech
            audio_path = os.path.join(temp_dir, 'input_audio.wav')
            text_to_speech(text, audio_path)
            
            # Output video path
            output_path = os.path.join(temp_dir, 'output.mp4')
            
            # Select the appropriate model and config
            # Check if TensorRT is available
            if not use_pytorch:
                try:
                    import tensorrt
                    # TensorRT is available, use it
                    if os.path.exists("/workspace/checkpoints/ditto_trt_custom"):
                        data_root = "/workspace/checkpoints/ditto_trt_custom"
                    else:
                        data_root = "/workspace/checkpoints/ditto_trt_Ampere_Plus"
                    cfg_pkl = "/workspace/checkpoints/ditto_cfg/v0.4_hubert_cfg_trt.pkl"
                    print("Using TensorRT model")
                except ImportError:
                    # TensorRT not available, fallback to PyTorch
                    print("TensorRT not available, using PyTorch model")
                    use_pytorch = True
            
            if use_pytorch:
                data_root = "/workspace/checkpoints/ditto_pytorch"
                cfg_pkl = "/workspace/checkpoints/ditto_cfg/v0.4_hubert_cfg_pytorch.pkl"
                print("Using PyTorch model")
            
            # Initialize SDK (following inference.py pattern)
            print(f"Initializing SDK with cfg_pkl: {cfg_pkl}, data_root: {data_root}")
            SDK = StreamSDK(cfg_pkl, data_root)
            
            # Run the ditto pipeline
            run_ditto(SDK, audio_path, image_path, output_path)
            
            # Read the output video and encode to base64
            with open(output_path, 'rb') as f:
                video_base64 = base64.b64encode(f.read()).decode('utf-8')
            
            return {
                "video_base64": video_base64,
                "status": "success"
            }
            
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        print(f"Handler error: {error_trace}")
        return {
            "error": str(e),
            "traceback": error_trace,
            "status": "failed"
        }


# RunPod serverless handler
runpod.serverless.start({"handler": handler})
