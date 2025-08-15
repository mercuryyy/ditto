const express = require('express');
const path = require('path');
const dotenv = require('dotenv');
const { AccessToken } = require('livekit-server-sdk');

// Load environment variables
dotenv.config();

const app = express();
const PORT = process.env.PORT || 3000;

// Middleware
app.use(express.static('public'));
app.use(express.json({ limit: '50mb' })); // For base64 images

// Basic route
app.get('/', (req, res) => {
    res.sendFile(path.join(__dirname, 'public', 'index.html'));
});

// Health check endpoint
app.get('/health', (req, res) => {
    res.json({ 
        status: 'healthy',
        timestamp: new Date().toISOString(),
        environment: process.env.NODE_ENV || 'development'
    });
});

// LiveKit token generation
app.get('/token', (req, res) => {
    try {
        const identity = req.query.identity || `web-${Date.now()}`;
        const room = req.query.room || 'liveportrait-test';
        
        if (!process.env.LK_API_KEY || !process.env.LK_API_SECRET) {
            return res.status(500).json({ 
                error: 'LiveKit credentials not configured',
                details: 'Missing LK_API_KEY or LK_API_SECRET' 
            });
        }

        const token = new AccessToken(
            process.env.LK_API_KEY,
            process.env.LK_API_SECRET,
            {
                identity,
                // Token expires in 1 hour
                ttl: '1h',
            }
        );

        token.addGrant({
            room,
            roomJoin: true,
            canPublish: true,
            canSubscribe: true,
        });

        const jwt = token.toJwt();
        
        res.json({ 
            token: jwt,
            url: process.env.LIVEKIT_URL,
            identity,
            room 
        });

    } catch (error) {
        console.error('Token generation error:', error);
        res.status(500).json({ 
            error: 'Failed to generate token',
            details: error.message 
        });
    }
});

// Start LivePortrait with RunPod integration
app.post('/start-lp', async (req, res) => {
    try {
        const { 
            identity, 
            room, 
            avatar_b64,
            // RTX 4090 Optimization Settings
            max_size = 1280,
            sampling_timesteps = 25,
            // Movement Controls
            mouth_amplitude = 0.8,
            head_amplitude = 0.6,
            eye_amplitude = 0.9,
            // Advanced Settings
            emo = 3,
            overlap_v2 = 3,
            smo_k_s = 5,
            smo_k_d = 2,
            // Expert Controls
            crop_scale = 2.3,
            crop_vx_ratio = 0.0,
            crop_vy_ratio = -0.125,
            vad_alpha = 1.0,
            delta_exp = 0.0,
            flag_stitching = true,
            crop_flag_do_rot = true,
            relative_d = true,
            // Basic Settings
            use_pytorch = false,
            streaming_mode = true
        } = req.body;

        if (!avatar_b64) {
            return res.status(400).json({ error: 'Missing avatar_b64' });
        }

        if (!process.env.LK_API_KEY || !process.env.LK_API_SECRET) {
            return res.status(500).json({ 
                error: 'LiveKit not configured',
                details: 'Missing LK_API_KEY or LK_API_SECRET' 
            });
        }

        // Prepare optimized settings for RunPod
        const dittoSettings = {
            max_size,
            sampling_timesteps,
            mouth_amplitude,
            head_amplitude,
            eye_amplitude,
            emo,
            overlap_v2,
            smo_k_s,
            smo_k_d,
            crop_scale,
            crop_vx_ratio,
            crop_vy_ratio,
            vad_alpha,
            delta_exp,
            flag_stitching,
            crop_flag_do_rot,
            relative_d
        };

        console.log('🚀 Setting up LiveKit room with avatar integration');

        // Create or update room with avatar metadata
        const { RoomServiceClient } = require('livekit-server-sdk');
        const roomService = new RoomServiceClient(process.env.LIVEKIT_URL, process.env.LK_API_KEY, process.env.LK_API_SECRET);
        
        const roomMetadata = JSON.stringify({
            avatar_image_b64: avatar_b64,
            ditto_settings: dittoSettings,
            created_at: new Date().toISOString()
        });

        try {
            // Update room metadata with avatar information
            await roomService.updateRoomMetadata(room, roomMetadata);
            console.log(`📝 Room ${room} metadata updated with avatar data`);
        } catch (roomError) {
            console.warn('Room metadata update failed:', roomError.message);
            // Continue - the agent can still work without metadata
        }

        res.json({
            status: 'room_configured',
            room: room,
            identity: identity,
            optimization_settings: dittoSettings,
            message: 'LiveKit room configured for avatar integration - agent will handle RunPod processing'
        });

    } catch (error) {
        console.error('Start LivePortrait error:', error);
        res.status(500).json({ 
            error: 'Failed to start LivePortrait',
            details: error.message 
        });
    }
});

// Get optimization defaults
app.get('/optimization-defaults', (req, res) => {
    res.json({
        // RTX 4090 Optimized Settings
        max_size: 1280,
        sampling_timesteps: 25,
        
        // Professional Movement Controls
        mouth_amplitude: 0.8,
        head_amplitude: 0.6,
        eye_amplitude: 0.9,
        
        // Low-Latency Settings
        emo: 3,
        overlap_v2: 3,
        smo_k_s: 5,
        smo_k_d: 2,
        
        // Expert Controls
        crop_scale: 2.3,
        crop_vx_ratio: 0.0,
        crop_vy_ratio: -0.125,
        vad_alpha: 1.0,
        delta_exp: 0.0,
        flag_stitching: true,
        crop_flag_do_rot: true,
        relative_d: true,
        
        // Streaming Configuration
        use_pytorch: false,
        streaming_mode: true,
        
        performance_notes: {
            expected_speed: "3-5 seconds per second of video",
            memory_usage: "6-10GB VRAM at 1280px",
            latency_reduction: "60% with optimized settings"
        }
    });
});

// API endpoint to get LiveKit configuration
app.get('/api/config', (req, res) => {
    res.json({
        livekit_url: process.env.LIVEKIT_URL,
        // Don't expose sensitive keys to the frontend
        environment: process.env.NODE_ENV || 'development'
    });
});

// Start the server
app.listen(PORT, () => {
    console.log(`🚀 Server running on port ${PORT}`);
    console.log(`📱 Frontend: http://localhost:${PORT}`);
    console.log(`💚 Health check: http://localhost:${PORT}/health`);
    console.log(`🔧 Environment: ${process.env.NODE_ENV || 'development'}`);
    console.log(`🎯 LiveKit: ${process.env.LIVEKIT_URL ? 'configured' : 'not configured'}`);
    console.log(`⚡ RunPod: ${process.env.RUNPOD_ENDPOINT_ID ? 'configured' : 'not configured'}`);
});
