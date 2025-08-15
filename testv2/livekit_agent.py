"""
LiveKit Agent with Deepgram STT + ElevenLabs TTS + OpenAI LLM + RunPod Ditto Avatar
Integrated voice pipeline with real-time avatar synchronization
"""
import asyncio
import logging
import os
import json
import base64
import numpy as np
from typing import Optional, AsyncIterator

from livekit.agents import (
    JobContext,
    WorkerOptions,
    cli,
    Agent,
    AgentSession,
    function_tool,
)
from livekit.plugins import deepgram, elevenlabs, openai
from livekit import rtc

# Configure logging - set to DEBUG for more detailed output
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


class RunPodAvatarSession:
    """Custom avatar session that integrates with RunPod Ditto with direct audio streaming"""
    
    def __init__(self, room: rtc.Room, avatar_image_b64: str):
        self.room = room
        self.avatar_image_b64 = avatar_image_b64
        self.runpod_endpoint_id = os.getenv("RUNPOD_ENDPOINT_ID")
        self.runpod_api_key = os.getenv("RUNPOD_API_KEY")
        self.avatar_participant_identity = "ditto-avatar"
        self.video_source = rtc.VideoSource(1280, 720)  # RTX 4090 optimized resolution
        self.is_active = False
        
        # Audio chunking for optimal lip-sync
        self.audio_buffer = []
        self.chunk_size = 1024  # 64ms at 16kHz
        self.sample_rate = 16000
        self.overlap_size = 256  # 25% overlap for continuity
        
        if not self.runpod_endpoint_id or not self.runpod_api_key:
            logger.error("RunPod credentials not configured")
            
    async def start(self):
        """Start the avatar session and join as a participant"""
        if self.is_active:
            return
            
        # Publish video track for avatar output
        video_track = rtc.LocalVideoTrack.create_video_track(
            "liveportrait", self.video_source
        )
        
        # We'll publish this track when we get frames from RunPod
        # Note: TrackPublishOptions doesn't have a 'name' parameter in newer versions
        await self.room.local_participant.publish_track(video_track, rtc.TrackPublishOptions(
            source=rtc.TrackSource.SOURCE_CAMERA
        ))
        
        self.is_active = True
        logger.info("🎭 RunPod Avatar session started")
    
    async def generate_avatar_from_audio_chunks(self, audio_chunks: list) -> bool:
        """Generate avatar video from direct audio chunks for optimal lip-sync"""
        if not self.is_active:
            logger.warning("Avatar session not active, skipping generation")
            return False
        if not self.runpod_endpoint_id:
            logger.error("RunPod endpoint not configured")
            return False
            
        try:
            logger.info(f"🎬 Generating avatar from {len(audio_chunks)} audio chunks...")
            logger.info(f"🔑 Using RunPod endpoint: {self.runpod_endpoint_id}")
            
            # Convert audio chunks to base64
            audio_chunks_b64 = []
            for chunk in audio_chunks:
                if isinstance(chunk, np.ndarray):
                    chunk_bytes = chunk.astype(np.float32).tobytes()
                    chunk_b64 = base64.b64encode(chunk_bytes).decode('utf-8')
                    audio_chunks_b64.append(chunk_b64)
            
            # Prepare RunPod payload for direct audio streaming
            payload = {
                "input": {
                    "mode": "realtime",
                    "image_base64": self.avatar_image_b64,
                    "audio_chunks_base64": audio_chunks_b64,
                    "ditto_settings": {
                        # Optimized for chunked audio processing with quality lip-sync
                        "max_size": 1280,
                        "sampling_timesteps": 25,
                        "mouth_amplitude": 0.8,
                        "head_amplitude": 0.6,
                        "eye_amplitude": 0.9,
                        "emo": 3,
                        "smo_k_s": 7,  # Higher smoothing for quality
                        "smo_k_d": 3,  # More temporal smoothing  
                        "overlap_v2": 5,  # Higher overlap for continuity
                        "crop_scale": 2.3,
                        "crop_vx_ratio": 0.0,
                        "crop_vy_ratio": -0.125,
                        "flag_stitching": True,
                        "relative_d": True
                    },
                    "stream_mode": "chunks"
                }
            }
            
            # Call RunPod endpoint using /runsync for synchronous execution
            import aiohttp
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"https://api.runpod.ai/v2/{self.runpod_endpoint_id}/runsync",
                    headers={
                        "Authorization": f"Bearer {self.runpod_api_key}",
                        "Content-Type": "application/json"
                    },
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=30)  # 30 second timeout
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        logger.info(f"🎯 RunPod job started: {result.get('id', 'unknown')}")
                        
                        # Process streaming results if available
                        if 'output' in result and 'stream_results' in result['output']:
                            await self._process_avatar_frames(result['output']['stream_results'])
                        
                        return True
                    else:
                        logger.error(f"RunPod API error: {response.status}")
                        return False
                        
        except Exception as e:
            logger.error(f"Avatar generation error: {e}")
            return False
    
    def add_audio_chunk(self, audio_data: np.ndarray):
        """Add audio chunk to buffer for processing"""
        logger.debug(f"📊 Adding audio chunk: {len(audio_data)} samples")
        
        if len(audio_data) != self.chunk_size:
            # Pad or truncate to correct size
            if len(audio_data) < self.chunk_size:
                audio_data = np.pad(audio_data, (0, self.chunk_size - len(audio_data)), mode='constant')
            else:
                audio_data = audio_data[:self.chunk_size]
        
        self.audio_buffer.append(audio_data)
        logger.debug(f"📦 Audio buffer size: {len(self.audio_buffer)} chunks")
        
        # Process buffer when we have enough chunks for overlap
        if len(self.audio_buffer) >= 5:  # Process every 5 chunks (~320ms)
            logger.info(f"🚀 Processing {len(self.audio_buffer)} audio chunks for avatar generation")
            asyncio.create_task(self.generate_avatar_from_audio_chunks(self.audio_buffer[-5:]))
    
    async def process_tts_audio_stream(self, audio_stream: AsyncIterator[rtc.AudioFrame]):
        """Process TTS audio stream in real-time for avatar generation"""
        try:
            async for frame in audio_stream:
                # Convert audio frame to numpy array
                audio_data = np.frombuffer(frame.data, dtype=np.int16).astype(np.float32) / 32768.0
                
                # Resample to 16kHz if needed
                if frame.sample_rate != self.sample_rate:
                    import librosa
                    audio_data = librosa.resample(audio_data, orig_sr=frame.sample_rate, target_sr=self.sample_rate)
                
                # Add to processing buffer
                self.add_audio_chunk(audio_data)
                
        except Exception as e:
            logger.error(f"TTS audio stream processing error: {e}")
    
    async def _process_avatar_frames(self, stream_results: list):
        """Process avatar frames from RunPod and stream to LiveKit"""
        for result in stream_results:
            if result.get("type") == "frame" and "frame_data" in result:
                try:
                    # Decode base64 frame
                    frame_data = base64.b64decode(result["frame_data"])
                    
                    # Convert to video frame and stream to LiveKit
                    # This would typically involve more sophisticated frame handling
                    logger.debug(f"📹 Received avatar frame {result.get('frame_number', 0)}")
                    
                    # TODO: Convert frame data to rtc.VideoFrame and push to video_source
                    # This requires additional image processing
                    
                except Exception as e:
                    logger.error(f"Frame processing error: {e}")


class DittoVoiceAssistant(Agent):
    """LiveKit Agent with integrated RunPod Ditto Avatar"""
    
    def __init__(self, avatar_session: Optional[RunPodAvatarSession] = None):
        super().__init__(
            instructions=(
                "You are a helpful AI assistant with a visual avatar. "
                "Keep responses conversational and engaging, but concise for real-time interaction. "
                "You can interact with users through voice and video in real-time."
            )
        )
        self.avatar_session = avatar_session
        
    async def on_enter(self) -> None:
        """Called when the agent enters the session"""
        logger.info("🎭 Ditto Voice Assistant activated")
        
        # Start avatar session if available
        if self.avatar_session:
            await self.avatar_session.start()
            
            # TEST: Generate some test audio chunks to verify RunPod connection
            logger.info("🧪 Testing avatar generation with dummy audio...")
            test_chunks = []
            for i in range(5):
                # Create a dummy audio chunk (silence)
                test_chunk = np.zeros(1024, dtype=np.float32)
                test_chunks.append(test_chunk)
            
            # Try to generate avatar with test chunks
            await self.avatar_session.generate_avatar_from_audio_chunks(test_chunks)
        
        # Greet the user
        await self.session.generate_reply(
            instructions="Greet the user warmly and let them know you're ready to help."
        )
    
    async def on_tts_audio_generated(self, audio_stream: AsyncIterator[rtc.AudioFrame]) -> None:
        """Called when TTS generates audio - process for avatar generation"""
        if self.avatar_session:
            # Process TTS audio stream for avatar generation
            await self.avatar_session.process_tts_audio_stream(audio_stream)
    
    @function_tool()
    async def get_current_time(self):
        """Get the current time"""
        from datetime import datetime
        current_time = datetime.now().strftime("%H:%M:%S on %B %d, %Y")
        return f"The current time is {current_time}"
    
    @function_tool()
    async def help_with_task(self, task_description: str):
        """Help the user with a specific task
        
        Args:
            task_description: Description of what the user needs help with
        """
        return f"I'd be happy to help you with: {task_description}. What specific aspect would you like me to focus on?"


async def entrypoint(ctx: JobContext):
    """Main entrypoint for the LiveKit agent with avatar integration"""
    
    logger.info("🚀 Starting Ditto Voice Assistant LiveKit Agent...")
    
    # Validate required environment variables
    required_env_vars = ["OPENAI_API_KEY", "DEEPGRAM_API_KEY", "ELEVENLABS_API_KEY"]
    missing_vars = [var for var in required_env_vars if not os.getenv(var)]
    
    if missing_vars:
        logger.error(f"❌ Missing required environment variables: {missing_vars}")
        return
    
    # Initialize avatar_session as None - will be created when metadata is received
    avatar_session = None
    agent = None  # Will be initialized later
    
    # Function to check and initialize avatar from metadata
    async def check_and_init_avatar():
        nonlocal avatar_session, agent
        if ctx.room.metadata and not avatar_session:
            try:
                logger.info(f"📝 Room metadata received: {ctx.room.metadata[:100]}...")
                room_data = json.loads(ctx.room.metadata)
                
                # Check for avatar_id (new file storage approach)
                avatar_id = room_data.get("avatar_id")
                avatar_server = room_data.get("avatar_server", "http://localhost:3000")
                
                if avatar_id:
                    logger.info(f"🔍 Avatar ID found: {avatar_id}, fetching from server...")
                    
                    # Fetch avatar from server
                    import aiohttp
                    try:
                        async with aiohttp.ClientSession() as session:
                            async with session.get(f"{avatar_server}/avatar/{avatar_id}") as response:
                                if response.status == 200:
                                    avatar_data = await response.json()
                                    avatar_image_b64 = avatar_data.get("avatar_b64")
                                    if avatar_image_b64:
                                        logger.info("✅ Avatar fetched successfully from server")
                                else:
                                    logger.error(f"Failed to fetch avatar: HTTP {response.status}")
                                    return False
                    except Exception as e:
                        logger.error(f"Error fetching avatar from server: {e}")
                        return False
                
                # Fallback: check for direct avatar_image_b64 (old approach, kept for compatibility)
                elif room_data.get("avatar_image_b64"):
                    avatar_image_b64 = room_data.get("avatar_image_b64")
                    logger.info("🎭 Avatar image found directly in metadata (legacy mode)")
                else:
                    return False
                
                if avatar_image_b64:
                    logger.info("🎭 Creating RunPod avatar session with fetched image")
                    avatar_session = RunPodAvatarSession(ctx.room, avatar_image_b64)
                    await avatar_session.start()
                    # Update the agent's avatar session if it exists
                    if agent:
                        agent.avatar_session = avatar_session
                        logger.info("✅ Avatar session connected to agent")
                    return True
                    
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse room metadata: {e}")
        return False
    
    # Initial check for metadata
    if not await check_and_init_avatar():
        logger.info("⚠️ No avatar image in initial metadata, will check for updates...")
        
        # Wait a bit and check again (metadata might be set after agent joins)
        await asyncio.sleep(2.0)  # Give time for metadata to be set
        if await check_and_init_avatar():
            logger.info("✅ Avatar session initialized after delay")
        else:
            # Start a background task to periodically check for metadata
            async def metadata_checker():
                for _ in range(10):  # Check up to 10 times
                    await asyncio.sleep(3.0)
                    if await check_and_init_avatar():
                        logger.info("✅ Avatar session initialized from periodic check")
                        break
                else:
                    logger.warning("⚠️ No avatar metadata received after 30 seconds")
            
            asyncio.create_task(metadata_checker())
    
    # Create agent session with proper voice pipeline
    session = AgentSession(
        stt=deepgram.STT(
            model="nova-2",
            language="en-US",
            smart_format=True,
            filler_words=False,
            interim_results=True,
        ),
        llm=openai.LLM(
            model="gpt-4",
            temperature=0.7,
        ),
        tts=elevenlabs.TTS(
            model="eleven_monolingual_v1",
            voice_id=os.getenv("ELEVENLABS_VOICE_ID", "21m00Tcm4TlvDq8ikWAM"),
        ),
    )
    
    # Create the Ditto voice assistant with avatar integration
    agent = DittoVoiceAssistant(avatar_session)
    
    # Handle room events
    @ctx.room.on("participant_connected")
    def on_participant_connected(participant: rtc.RemoteParticipant):
        logger.info(f"Participant joined: {participant.identity}")
    
    @ctx.room.on("participant_disconnected") 
    def on_participant_disconnected(participant: rtc.RemoteParticipant):
        logger.info(f"Participant left: {participant.identity}")
    
    # Listen for room metadata updates
    @ctx.room.on("room_metadata_changed")
    def on_room_metadata_changed(old_metadata: str, new_metadata: str):
        logger.info("📝 Room metadata updated")
        # Check if we can now initialize the avatar
        async def check_metadata():
            if await check_and_init_avatar():
                logger.info("✅ Avatar session initialized from metadata update")
        asyncio.create_task(check_metadata())
    
    # Start the agent session
    await session.start(
        room=ctx.room,
        agent=agent,
    )
    
    logger.info("🎤 Ditto LiveKit Agent session started!")
    logger.info(f"🏠 Connected to room: {ctx.room.name}")
    logger.info("🔊 Deepgram STT + ElevenLabs TTS + OpenAI LLM + RunPod Avatar active")


def prewarm(proc: JobContext):
    """Prewarm function to initialize models"""
    logger.info("Prewarming LiveKit agent...")
    
    try:
        # Validate environment variables during prewarm
        if not os.getenv("OPENAI_API_KEY"):
            logger.warning("⚠️  OPENAI_API_KEY not found in environment")
        
        if not os.getenv("DEEPGRAM_API_KEY"):
            logger.warning("⚠️  DEEPGRAM_API_KEY not found in environment")
            
        if not os.getenv("ELEVENLABS_API_KEY"):
            logger.warning("⚠️  ELEVENLABS_API_KEY not found in environment")
        
        # Prewarm TTS
        proc.userdata["tts"] = elevenlabs.TTS(
            model="eleven_monolingual_v1",
            voice_id=os.getenv("ELEVENLABS_VOICE_ID", "21m00Tcm4TlvDq8ikWAM"),
        )
        
        # Prewarm STT  
        proc.userdata["stt"] = deepgram.STT(
            model="nova-2",
            language="en-US",
        )
        
        # Prewarm LLM
        proc.userdata["llm"] = openai.LLM(
            model="gpt-4",
            temperature=0.7,
        )
        
        logger.info("✅ LiveKit agent prewarmed and ready!")
        
    except Exception as e:
        logger.error(f"Prewarm error: {e}")
        # Don't raise - let it continue without prewarm


if __name__ == "__main__":
    # Run the LiveKit agent
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
            prewarm_fnc=prewarm,
        ),
    )
