#!/usr/bin/env python3
"""
Test script to verify RunPod connection and environment variables
"""
import os
import asyncio
import aiohttp
import json

async def test_runpod():
    """Test RunPod connection"""
    print("🔍 Checking environment variables...")
    
    endpoint_id = os.getenv("RUNPOD_ENDPOINT_ID")
    api_key = os.getenv("RUNPOD_API_KEY")
    
    print(f"RUNPOD_ENDPOINT_ID: {endpoint_id or 'NOT SET ❌'}")
    print(f"RUNPOD_API_KEY: {'SET ✅' if api_key else 'NOT SET ❌'}")
    
    if not endpoint_id or not api_key:
        print("\n❌ RunPod credentials not found in environment!")
        print("\nPlease set them with:")
        print('export RUNPOD_ENDPOINT_ID="your_endpoint_id"')
        print('export RUNPOD_API_KEY="your_api_key"')
        return False
    
    print(f"\n🚀 Testing connection to RunPod endpoint: {endpoint_id}")
    
    # Test with a simple request to the endpoint
    try:
        async with aiohttp.ClientSession() as session:
            # Test with a simple "Hello World" request
            test_payload = {
                "input": {
                    "prompt": "Hello World"
                }
            }
            
            async with session.post(
                f"https://api.runpod.ai/v2/{endpoint_id}/runsync",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json"
                },
                json=test_payload
            ) as response:
                print(f"📡 Test request response: {response.status}")
                if response.status == 200:
                    data = await response.json()
                    print(f"✅ RunPod response: {json.dumps(data, indent=2)}")
                else:
                    text = await response.text()
                    print(f"❌ RunPod API error: {response.status}")
                    print(f"Response: {text}")
    
    except Exception as e:
        print(f"❌ Connection error: {e}")
        return False
    
    return True

if __name__ == "__main__":
    result = asyncio.run(test_runpod())
    if result:
        print("\n✅ RunPod connection successful!")
    else:
        print("\n❌ RunPod connection failed!")
