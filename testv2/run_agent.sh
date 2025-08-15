#!/bin/bash

# Load environment variables from .env file
if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
fi

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    source venv/bin/activate
fi

# Run the LiveKit agent
echo "🚀 Starting LiveKit Agent..."
echo "📍 Current directory: $(pwd)"
echo "🔑 Environment variables loaded from .env"

# Verify key environment variables are set
echo "✅ Checking environment variables:"
echo "   OPENAI_API_KEY: $(if [ -n "$OPENAI_API_KEY" ]; then echo 'SET'; else echo 'NOT SET'; fi)"
echo "   DEEPGRAM_API_KEY: $(if [ -n "$DEEPGRAM_API_KEY" ]; then echo 'SET'; else echo 'NOT SET'; fi)"
echo "   ELEVENLABS_API_KEY: $(if [ -n "$ELEVENLABS_API_KEY" ]; then echo 'SET'; else echo 'NOT SET'; fi)"
echo "   LK_API_KEY: $(if [ -n "$LK_API_KEY" ]; then echo 'SET'; else echo 'NOT SET'; fi)"

python livekit_agent.py
