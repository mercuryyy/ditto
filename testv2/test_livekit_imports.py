"""
Test script to verify LiveKit Agents SDK imports
Run this to check if all required modules are properly installed
"""

def test_imports():
    """Test all required imports for the LiveKit agent"""
    
    print("🧪 Testing LiveKit Agents SDK imports...")
    
    try:
        from livekit.agents import (
            AutoSubscribe,
            JobContext,
            WorkerOptions,
            cli,
        )
        print("✅ livekit.agents core imports successful")
    except ImportError as e:
        print(f"❌ livekit.agents core import failed: {e}")
        return False
    
    try:
        from livekit import rtc
        print("✅ livekit.rtc import successful")
    except ImportError as e:
        print(f"❌ livekit.rtc import failed: {e}")
        return False
    
    try:
        from livekit.plugins import deepgram
        print("✅ livekit.plugins.deepgram import successful")
    except ImportError as e:
        print(f"❌ livekit.plugins.deepgram import failed: {e}")
        print("💡 Try: pip install livekit-plugins-deepgram")
        return False
    
    try:
        from livekit.plugins import elevenlabs
        print("✅ livekit.plugins.elevenlabs import successful")
    except ImportError as e:
        print(f"❌ livekit.plugins.elevenlabs import failed: {e}")
        print("💡 Try: pip install livekit-plugins-elevenlabs")
        return False
    
    try:
        from livekit.plugins import openai
        print("✅ livekit.plugins.openai import successful")
    except ImportError as e:
        print(f"❌ livekit.plugins.openai import failed: {e}")
        print("💡 Try: pip install livekit-plugins-openai")
        return False
    
    print("\n🎉 All imports successful! LiveKit agent should work properly.")
    return True

def test_environment_variables():
    """Test if required environment variables are set"""
    import os
    
    print("\n🔧 Testing environment variables...")
    
    required_vars = [
        "LK_API_KEY",
        "LK_API_SECRET", 
        "ELEVEN_API_KEY",  # Required by LiveKit ElevenLabs plugin
        "OPENAI_API_KEY",
        "DEEPGRAM_API_KEY"
    ]
    
    missing_vars = []
    
    for var in required_vars:
        if os.getenv(var):
            print(f"✅ {var} is set")
        else:
            print(f"❌ {var} is missing")
            missing_vars.append(var)
    
    if missing_vars:
        print(f"\n💡 Missing environment variables: {', '.join(missing_vars)}")
        print("Update your .env file with the required API keys")
        return False
    else:
        print("\n🎉 All environment variables are set!")
        return True

if __name__ == "__main__":
    print("🚀 LiveKit Agent Installation Test\n")
    
    imports_ok = test_imports()
    env_ok = test_environment_variables()
    
    if imports_ok and env_ok:
        print("\n✅ Ready to run LiveKit agent!")
        print("Run: python livekit_agent.py dev")
    else:
        print("\n❌ Setup incomplete. Please fix the issues above.")
