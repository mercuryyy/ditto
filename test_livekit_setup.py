"""
Test script to verify LiveKit Ditto optimization setup
"""
import requests
import base64
import json
import sys
import os

def test_elevenlabs_tts(api_key, voice_id, text="Hello, this is a test of ElevenLabs TTS integration"):
    """Test ElevenLabs TTS integration"""
    print("🎤 Testing ElevenLabs TTS integration...")
    
    if not api_key or not voice_id:
        print("❌ Please provide ElevenLabs API key and voice ID")
        return False
    
    try:
        url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
        headers = {
            "Accept": "audio/mpeg",
            "Content-Type": "application/json",
            "xi-api-key": api_key
        }
        data = {
            "text": text,
            "model_id": "eleven_monolingual_v1",
            "voice_settings": {
                "stability": 0.5,
                "similarity_boost": 0.5
            }
        }
        
        response = requests.post(url, json=data, headers=headers, timeout=30)
        
        if response.status_code == 200:
            print("✅ ElevenLabs TTS integration working!")
            print(f"   Generated {len(response.content)} bytes of audio")
            return True
        else:
            print(f"❌ ElevenLabs API Error: {response.status_code}")
            print(f"   Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ TTS Test Error: {str(e)}")
        return False

def test_runpod_endpoint(endpoint, api_key, test_image_path="example/image.png"):
    """Test RunPod endpoint with optimized settings"""
    print("🚀 Testing RunPod endpoint with LiveKit optimized settings...")
    
    if not os.path.exists(test_image_path):
        print(f"❌ Test image not found: {test_image_path}")
        return False
    
    try:
        # Read and encode test image
        with open(test_image_path, "rb") as f:
            image_base64 = base64.b64encode(f.read()).decode()
        
        # LiveKit optimized payload
        payload = {
            "input": {
                "text": "This is a test of the LiveKit optimized Ditto avatar system.",
                "image_base64": image_base64,
                "use_pytorch": False,
                "streaming_mode": True,
                "mode": "realtime",  # Use real-time handler
                "ditto_settings": {
                    # RTX 4090 optimized settings
                    "max_size": 1280,
                    "sampling_timesteps": 25,
                    "smo_k_s": 5,
                    "smo_k_d": 2,
                    "overlap_v2": 3,
                    "emo": 3,
                    "mouth_amplitude": 0.8,
                    "head_amplitude": 0.6,
                    "eye_amplitude": 0.9,
                    "online_mode": True,
                    "relative_d": True,
                    "flag_stitching": True
                }
            }
        }
        
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        print("📋 Sending test request to RunPod...")
        response = requests.post(
            f"{endpoint}/run",
            json=payload,
            headers=headers,
            timeout=60
        )
        
        if response.status_code == 200:
            result = response.json()
            if "id" in result:
                print("✅ RunPod endpoint working!")
                print(f"   Job ID: {result['id']}")
                print("   Real-time streaming handler active")
                return True
            elif "stream_results" in result:
                print("✅ RunPod real-time streaming working!")
                print(f"   Mode: {result.get('mode', 'unknown')}")
                return True
            elif "output" in result:
                print("✅ RunPod endpoint working (sync mode)!")
                return True
            else:
                print(f"❌ Unexpected response format: {result}")
                return False
        else:
            print(f"❌ RunPod Error: {response.status_code}")
            print(f"   Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ RunPod Test Error: {str(e)}")
        return False

def verify_files():
    """Verify all required files exist"""
    print("📁 Verifying file setup...")
    
    required_files = [
        "web_ui_streaming_livekit_optimized.py",
        "runpod_realtime_streaming_handler.py", 
        "runpod_streaming_handler.py",
        "requirements_livekit.txt",
        "Dockerfile",
        "LIVEKIT_OPTIMIZATION_SUMMARY.md"
    ]
    
    all_exist = True
    for file in required_files:
        if os.path.exists(file):
            print(f"   ✅ {file}")
        else:
            print(f"   ❌ {file} - MISSING!")
            all_exist = False
    
    return all_exist

def main():
    """Main test function"""
    print("🔧 LiveKit Ditto Setup Verification")
    print("=" * 50)
    
    # Verify files
    files_ok = verify_files()
    print()
    
    # Get environment variables or command line args
    runpod_endpoint = os.getenv("RUNPOD_ENDPOINT")
    runpod_api_key = os.getenv("RUNPOD_API_KEY")
    elevenlabs_api_key = os.getenv("ELEVENLABS_API_KEY")
    elevenlabs_voice_id = os.getenv("ELEVENLABS_VOICE_ID", "21m00Tcm4TlvDq8ikWAM")
    
    if len(sys.argv) >= 3:
        runpod_endpoint = sys.argv[1]
        runpod_api_key = sys.argv[2]
        if len(sys.argv) >= 4:
            elevenlabs_api_key = sys.argv[3]
        if len(sys.argv) >= 5:
            elevenlabs_voice_id = sys.argv[4]
    
    # Test ElevenLabs TTS
    if elevenlabs_api_key:
        tts_ok = test_elevenlabs_tts(elevenlabs_api_key, elevenlabs_voice_id)
        print()
    else:
        print("⚠️  Skipping ElevenLabs test - no API key provided")
        print("   Set ELEVENLABS_API_KEY environment variable or pass as argument")
        tts_ok = None
        print()
    
    # Test RunPod endpoint
    if runpod_endpoint and runpod_api_key:
        runpod_ok = test_runpod_endpoint(runpod_endpoint, runpod_api_key)
        print()
    else:
        print("⚠️  Skipping RunPod test - no endpoint/API key provided")
        print("   Set RUNPOD_ENDPOINT and RUNPOD_API_KEY environment variables")
        runpod_ok = None
        print()
    
    # Summary
    print("📊 VERIFICATION SUMMARY:")
    print("=" * 30)
    print(f"Files Setup: {'✅ PASS' if files_ok else '❌ FAIL'}")
    if tts_ok is not None:
        print(f"ElevenLabs TTS: {'✅ PASS' if tts_ok else '❌ FAIL'}")
    else:
        print("ElevenLabs TTS: ⚠️  SKIPPED")
    if runpod_ok is not None:
        print(f"RunPod Endpoint: {'✅ PASS' if runpod_ok else '❌ FAIL'}")
    else:
        print("RunPod Endpoint: ⚠️  SKIPPED")
    print()
    
    if files_ok and (tts_ok or tts_ok is None) and (runpod_ok or runpod_ok is None):
        print("🎉 Setup verification completed successfully!")
        print("Your LiveKit Ditto optimization is ready to use.")
    else:
        print("❌ Some issues found. Please check the errors above.")

if __name__ == "__main__":
    main()
