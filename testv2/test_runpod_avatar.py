#!/usr/bin/env python3
"""
Test script to verify RunPod avatar generation with real image and audio
"""
import os
import asyncio
import aiohttp
import json
import base64
import numpy as np

async def test_runpod_avatar():
    """Test RunPod avatar generation"""
    print("🔍 Testing RunPod Avatar Generation...")
    
    endpoint_id = os.getenv("RUNPOD_ENDPOINT_ID")
    api_key = os.getenv("RUNPOD_API_KEY")
    
    if not endpoint_id or not api_key:
        print("❌ RunPod credentials not found in environment!")
        return False
    
    print(f"📡 Using RunPod endpoint: {endpoint_id}")
    
    # Load a test avatar image - check multiple possible paths
    possible_paths = [
        "avatar_storage/74775662a536662e5da988582094621d.png",  # When run from testv2/
        "testv2/avatar_storage/74775662a536662e5da988582094621d.png",  # When run from project root
        "../example/image.png",  # Fallback to example
        "example/image.png"  # Another fallback
    ]
    
    avatar_path = None
    for path in possible_paths:
        if os.path.exists(path):
            avatar_path = path
            break
    
    # If we still don't have a path, list available avatars and use the first one
    if not avatar_path:
        avatar_dir = "avatar_storage" if os.path.exists("avatar_storage") else "testv2/avatar_storage"
        if os.path.exists(avatar_dir):
            avatars = [f for f in os.listdir(avatar_dir) if f.endswith('.png')]
            if avatars:
                avatar_path = os.path.join(avatar_dir, avatars[0])
                print(f"📂 Found {len(avatars)} avatars in {avatar_dir}, using: {avatars[0]}")
    
    if avatar_path and os.path.exists(avatar_path):
        print(f"🖼️ Loading avatar from: {avatar_path}")
        file_size = os.path.getsize(avatar_path)
        print(f"   File size: {file_size:,} bytes")
        with open(avatar_path, 'rb') as f:
            avatar_b64 = base64.b64encode(f.read()).decode('utf-8')
    else:
        print("❌ No avatar image found! Available files in current directory:")
        for item in os.listdir('.'):
            print(f"   - {item}")
        print("\nPlease ensure you have avatar images in the avatar_storage directory.")
        return False
    
    # Create test audio chunks with actual audio data (sine wave to simulate speech)
    print("🎵 Creating test audio chunks with simulated speech...")
    audio_chunks_b64 = []
    sample_rate = 16000
    chunk_size = 1024  # 64ms at 16kHz
    duration_per_chunk = chunk_size / sample_rate
    
    # Generate 10 chunks (~640ms of audio) with varying frequencies to simulate speech
    for i in range(10):
        # Create a sine wave with varying frequency (simulating voice pitch variations)
        time = np.linspace(i * duration_per_chunk, (i + 1) * duration_per_chunk, chunk_size)
        
        # Mix multiple frequencies to simulate speech harmonics
        frequency1 = 200 + i * 20  # Base frequency varying from 200-380 Hz
        frequency2 = 400 + i * 30  # Second harmonic
        frequency3 = 800 + i * 40  # Third harmonic
        
        # Generate complex waveform
        audio_chunk = (
            0.5 * np.sin(2 * np.pi * frequency1 * time) +  # Fundamental
            0.3 * np.sin(2 * np.pi * frequency2 * time) +  # 2nd harmonic
            0.2 * np.sin(2 * np.pi * frequency3 * time)    # 3rd harmonic
        )
        
        # Add some amplitude variation to simulate speech dynamics
        amplitude_envelope = 0.3 + 0.7 * np.abs(np.sin(np.pi * i / 5))
        audio_chunk = audio_chunk * amplitude_envelope
        
        # Normalize to [-1, 1] range
        audio_chunk = np.clip(audio_chunk, -1.0, 1.0).astype(np.float32)
        
        # Convert to bytes and base64
        chunk_bytes = audio_chunk.tobytes()
        chunk_b64 = base64.b64encode(chunk_bytes).decode('utf-8')
        audio_chunks_b64.append(chunk_b64)
    
    print(f"  Created {len(audio_chunks_b64)} audio chunks with simulated speech patterns")
    
    # Prepare the payload
    payload = {
        "input": {
            "mode": "realtime",
            "image_base64": avatar_b64,
            "audio_chunks_base64": audio_chunks_b64,
            "ditto_settings": {
                "max_size": 512,  # Lower for testing
                "sampling_timesteps": 15,  # Faster for testing
                "mouth_amplitude": 0.8,
                "head_amplitude": 0.6,
                "eye_amplitude": 0.9,
                "emo": 3,
                "smo_k_s": 5,
                "smo_k_d": 2,
                "overlap_v2": 3,
                "crop_scale": 2.3,
                "crop_vx_ratio": 0.0,
                "crop_vy_ratio": -0.125,
                "flag_stitching": True,
                "relative_d": True
            },
            "stream_mode": "chunks"
        }
    }
    
    try:
        print("🚀 Sending avatar generation request to RunPod...")
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"https://api.runpod.ai/v2/{endpoint_id}/runsync",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json"
                },
                json=payload,
                timeout=aiohttp.ClientTimeout(total=60)  # 60 second timeout
            ) as response:
                print(f"📡 Response status: {response.status}")
                
                if response.status == 200:
                    data = await response.json()
                    
                    # Check for successful output
                    if data.get("status") == "COMPLETED":
                        print("✅ Avatar generation completed successfully!")
                        
                        # Check if we got stream results
                        output = data.get("output", {})
                        if "stream_results" in output:
                            results = output["stream_results"]
                            print(f"📊 Received {len(results)} streaming events")
                            
                            # Count frame types and errors
                            frame_count = sum(1 for r in results if r.get("type") == "frame")
                            error_count = sum(1 for r in results if r.get("type") == "error")
                            
                            print(f"🎬 Generated {frame_count} video frames")
                            if error_count > 0:
                                print(f"⚠️ Encountered {error_count} errors")
                            
                            # Show all events with details
                            for i, result in enumerate(results):
                                event_type = result.get("type", "unknown")
                                print(f"  Event {i}: {event_type}")
                                
                                # Show error details if it's an error
                                if event_type == "error":
                                    error_msg = result.get("error", "Unknown error")
                                    print(f"    Error: {error_msg}")
                                    if "traceback" in result:
                                        print(f"    Traceback (last line): {result['traceback'].splitlines()[-1]}")
                            
                            # Check if we actually generated frames
                            if frame_count == 0 and error_count > 0:
                                print("\n❌ No frames generated due to errors!")
                                return False
                        else:
                            print("⚠️ No stream_results in output")
                            print(f"Output keys: {list(output.keys())}")
                    
                    elif data.get("status") == "FAILED":
                        print("❌ Avatar generation failed!")
                        if "error" in data:
                            print(f"Error: {data['error']}")
                    
                    else:
                        print(f"⚠️ Unexpected status: {data.get('status')}")
                        print(f"Full response: {json.dumps(data, indent=2)}")
                    
                    return data.get("status") == "COMPLETED"
                    
                else:
                    text = await response.text()
                    print(f"❌ RunPod API error: {response.status}")
                    print(f"Response: {text[:500]}")  # First 500 chars
                    return False
                    
    except asyncio.TimeoutError:
        print("❌ Request timed out after 60 seconds")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


if __name__ == "__main__":
    # Set environment if not already set
    if not os.getenv("RUNPOD_ENDPOINT_ID"):
        print("Setting environment variables from .env file...")
        from dotenv import load_dotenv
        load_dotenv("testv2/.env")
    
    result = asyncio.run(test_runpod_avatar())
    
    if result:
        print("\n🎉 RunPod avatar generation test PASSED!")
    else:
        print("\n❌ RunPod avatar generation test FAILED!")
        print("\nTroubleshooting:")
        print("1. Check if the RunPod endpoint is active")
        print("2. Verify the Docker image was pushed correctly")
        print("3. Check RunPod logs for detailed errors")
