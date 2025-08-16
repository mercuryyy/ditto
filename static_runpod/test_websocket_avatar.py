#!/usr/bin/env python3
"""
Test script to verify RunPod Pod WebSocket avatar streaming with real image and audio
"""
import os
import asyncio
import websockets
import json
import base64
import numpy as np
from PIL import Image
import io

async def test_websocket_avatar():
    """Test RunPod Pod WebSocket avatar streaming"""
    print("🔍 Testing RunPod Pod WebSocket Avatar Streaming...")
    
    # Get Pod URL from environment or prompt
    pod_url = os.getenv("RUNPOD_POD_URL")
    if not pod_url:
        pod_url = input("Enter your RunPod Pod URL (e.g., https://abcd1234-8888.proxy.runpod.net): ").strip()
        if not pod_url:
            print("❌ Pod URL is required!")
            return False
    
    # Convert HTTP to WebSocket URL
    if pod_url.startswith("http://"):
        ws_url = pod_url.replace("http://", "ws://") + "/ws"
    elif pod_url.startswith("https://"):
        ws_url = pod_url.replace("https://", "wss://") + "/ws"
    else:
        ws_url = f"wss://{pod_url}/ws"
    
    print(f"📡 Connecting to WebSocket: {ws_url}")
    
    # Load a test avatar image - check multiple possible paths
    possible_paths = [
        "/Users/israelcohen/Downloads/israelwhatsappleads.csv-min.png",
        "../testv2/avatar_storage/74775662a536662e5da988582094621d.png",
        "../example/image.png",
        "example/image.png"
    ]
    
    avatar_path = None
    for path in possible_paths:
        if os.path.exists(path):
            avatar_path = path
            break
    
    # If we still don't have a path, look in testv2/avatar_storage
    if not avatar_path:
        avatar_dir = "../testv2/avatar_storage"
        if os.path.exists(avatar_dir):
            avatars = [f for f in os.listdir(avatar_dir) if f.endswith('.png')]
            if avatars:
                avatar_path = os.path.join(avatar_dir, avatars[0])
                print(f"📂 Found {len(avatars)} avatars in {avatar_dir}, using: {avatars[0]}")
    
    if avatar_path and os.path.exists(avatar_path):
        print(f"🖼️ Loading avatar from: {avatar_path}")
        file_size = os.path.getsize(avatar_path)
        print(f"   File size: {file_size:,} bytes")
        
        try:
            # Open and verify the image
            with Image.open(avatar_path) as img:
                print(f"   Image size: {img.size}, mode: {img.mode}")
                # Convert to RGB if needed
                if img.mode == 'RGBA':
                    img = img.convert('RGB')
                elif img.mode != 'RGB':
                    img = img.convert('RGB')
                
                # Save to bytes and encode
                buffer = io.BytesIO()
                img.save(buffer, format='PNG')
                avatar_b64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
                print(f"   Base64 encoded: {len(avatar_b64)} characters")
        except Exception as e:
            print(f"❌ Failed to load/process image: {e}")
            return False
    else:
        print("❌ No avatar image found! Please place avatar images in testv2/avatar_storage/")
        return False
    
    # Create test audio data
    print("🎵 Creating test audio chunks...")
    sample_rate = 16000
    chunk_size = 1024  # 64ms chunks for real-time streaming
    num_chunks = 20    # ~1.3 seconds of audio
    
    audio_chunks = []
    for i in range(num_chunks):
        # Create varied sine wave to simulate speech
        time = np.linspace(i * chunk_size / sample_rate, (i + 1) * chunk_size / sample_rate, chunk_size)
        frequency = 200 + (i % 5) * 50  # Vary frequency
        audio_chunk = 0.5 * np.sin(2 * np.pi * frequency * time)
        audio_chunks.append(audio_chunk.astype(np.float32))
    
    print(f"  Created {len(audio_chunks)} audio chunks")
    
    try:
        # Test basic HTTP endpoint first
        import aiohttp
        async with aiohttp.ClientSession() as session:
            try:
                async with session.get(pod_url.replace("/ws", "/health"), timeout=5) as response:
                    if response.status == 200:
                        health_data = await response.json()
                        print(f"✅ Pod health check passed: {health_data}")
                    else:
                        print(f"⚠️ Pod health check failed: {response.status}")
            except Exception as e:
                print(f"⚠️ Could not reach Pod health endpoint: {e}")
        
        print("🚀 Connecting to WebSocket...")
        
        # Connect to WebSocket
        async with websockets.connect(ws_url, timeout=10) as websocket:
            print("✅ WebSocket connected successfully!")
            
            # Send initialization message
            init_message = {
                "type": "init",
                "avatar_image_base64": avatar_b64,
                "settings": {
                    "max_size": 512,  # Lower for testing
                    "sampling_timesteps": 20,  # Faster for testing
                    "mouth_amplitude": 0.8,
                    "head_amplitude": 0.6,
                    "eye_amplitude": 0.9,
                    "emo": 3,
                    "smo_k_s": 5,
                    "smo_k_d": 2,
                    "overlap_v2": 5,
                    "crop_scale": 2.3,
                    "relative_d": True
                }
            }
            
            print("📤 Sending avatar initialization...")
            await websocket.send(json.dumps(init_message))
            
            # Wait for ready signal
            ready_response = await websocket.recv()
            ready_data = json.loads(ready_response)
            
            if ready_data.get("type") == "ready":
                print("✅ Avatar session initialized and ready!")
                print(f"   Session ID: {ready_data.get('session_id')}")
            elif ready_data.get("type") == "error":
                print(f"❌ Initialization failed: {ready_data.get('message')}")
                return False
            else:
                print(f"⚠️ Unexpected response: {ready_data}")
                return False
            
            # Start streaming audio chunks
            print("🎵 Starting audio streaming...")
            frames_received = 0
            
            # Create task to send audio chunks
            async def send_audio():
                for i, audio_chunk in enumerate(audio_chunks):
                    # Send audio chunk as binary data
                    await websocket.send(audio_chunk.tobytes())
                    print(f"   Sent audio chunk {i+1}/{len(audio_chunks)}")
                    
                    # Real-time delay between chunks
                    await asyncio.sleep(chunk_size / sample_rate)  # 64ms delay
                
                print("🎵 All audio chunks sent")
                
                # Send close signal after a short delay
                await asyncio.sleep(1.0)
                await websocket.send(json.dumps({"type": "close"}))
            
            # Create task to receive frames
            async def receive_frames():
                nonlocal frames_received
                try:
                    while True:
                        response = await websocket.recv()
                        data = json.loads(response)
                        
                        if data.get("type") == "frame":
                            frames_received += 1
                            frame_number = data.get("frame_number", frames_received)
                            data_size = len(data.get("data", ""))
                            print(f"📹 Received frame {frame_number} (size: {data_size} chars)")
                            
                            # Save first frame for verification
                            if frames_received == 1:
                                frame_data = data.get("data")
                                if frame_data:
                                    # Decode and save first frame
                                    frame_bytes = base64.b64decode(frame_data)
                                    with open("first_frame.jpg", "wb") as f:
                                        f.write(frame_bytes)
                                    print("   💾 Saved first frame as 'first_frame.jpg'")
                        
                        elif data.get("type") == "error":
                            print(f"❌ Error: {data.get('message')}")
                            break
                        
                        else:
                            print(f"📨 Received: {data.get('type', 'unknown')}")
                
                except websockets.exceptions.ConnectionClosed:
                    print("🔌 WebSocket connection closed")
                except Exception as e:
                    print(f"❌ Error receiving frames: {e}")
            
            # Run both tasks concurrently
            try:
                await asyncio.gather(
                    send_audio(),
                    receive_frames()
                )
            except Exception as e:
                print(f"❌ Streaming error: {e}")
                return False
            
            print(f"\n🎬 Streaming complete! Received {frames_received} frames")
            
            if frames_received > 0:
                print("✅ WebSocket avatar streaming test PASSED!")
                return True
            else:
                print("❌ No frames received - streaming failed!")
                return False
    
    except websockets.exceptions.WebSocketException as e:
        print(f"❌ WebSocket error: {e}")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


async def test_pod_health(pod_url):
    """Test Pod health endpoint"""
    print("🔍 Testing Pod health endpoint...")
    
    import aiohttp
    async with aiohttp.ClientSession() as session:
        try:
            health_url = pod_url.rstrip('/') + "/health"
            async with session.get(health_url, timeout=10) as response:
                if response.status == 200:
                    data = await response.json()
                    print(f"✅ Pod is healthy: {data}")
                    return True
                else:
                    print(f"❌ Pod health check failed: {response.status}")
                    return False
        except Exception as e:
            print(f"❌ Could not reach Pod: {e}")
            return False


if __name__ == "__main__":
    # Check if Pod URL is provided
    pod_url = os.getenv("RUNPOD_POD_URL")
    if not pod_url:
        print("💡 Set RUNPOD_POD_URL environment variable or you'll be prompted for it")
        print("   Example: export RUNPOD_POD_URL=https://abcd1234-8888.proxy.runpod.net")
        print()
    
    # Run tests
    async def run_tests():
        # Test Pod health first
        if pod_url:
            health_ok = await test_pod_health(pod_url)
            if not health_ok:
                print("\n❌ Pod health check failed - aborting WebSocket test")
                return False
        
        # Test WebSocket avatar streaming
        result = await test_websocket_avatar()
        return result
    
    success = asyncio.run(run_tests())
    
    if success:
        print("\n🎉 All tests PASSED! Your RunPod Pod WebSocket streaming is working! 🚀")
        print("\n📡 Your Pod is ready for LiveKit integration:")
        print(f"   WebSocket URL: {pod_url.replace('http', 'ws') if pod_url else 'wss://your-pod-url'}/ws")
    else:
        print("\n❌ Tests FAILED!")
        print("\n🔧 Troubleshooting:")
        print("1. Ensure your Pod is running and accessible")
        print("2. Check Pod logs for any errors")
        print("3. Verify the correct image was deployed")
        print("4. Test health endpoint manually")
