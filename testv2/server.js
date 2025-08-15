const express = require('express');
const path = require('path');
const dotenv = require('dotenv');
const fs = require('fs').promises;
const crypto = require('crypto');
const { AccessToken } = require('livekit-server-sdk');

// Load environment variables
dotenv.config();

const app = express();
const PORT = process.env.PORT || 3000;

// Avatar storage directory
const AVATAR_DIR = path.join(__dirname, 'avatar_storage');

// Ensure avatar directory exists
async function ensureAvatarDir() {
    try {
        await fs.mkdir(AVATAR_DIR, { recursive: true });
    } catch (error) {
        console.error('Failed to create avatar directory:', error);
    }
}
ensureAvatarDir();

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
app.get('/token', async (req, res) => {
    try {
        const identity = req.query.identity || `web-${Date.now()}`;
        const room = req.query.room || 'liveportrait-test';
        
        // DEBUG: Log token generation details
        console.log('\n🔐 Token Generation Request:');
        console.log(`  Room: ${room}`);
        console.log(`  Identity: ${identity}`);
        console.log(`  Using API Key: ${process.env.LK_API_KEY}`);
        console.log(`  Using API Secret: ${process.env.LK_API_SECRET}`);
        console.log(`  Will connect to: ${process.env.LIVEKIT_URL}\n`);
        
        if (!process.env.LK_API_KEY || !process.env.LK_API_SECRET) {
            return res.status(500).json({ 
                error: 'LiveKit credentials not configured',
                details: 'Missing LK_API_KEY or LK_API_SECRET' 
            });
        }

        console.log('  Creating AccessToken with:');
        console.log('    API Key:', process.env.LK_API_KEY);
        console.log('    API Secret:', process.env.LK_API_SECRET);
        
        const at = new AccessToken(process.env.LK_API_KEY, process.env.LK_API_SECRET, {
            identity: identity,
            ttl: 60 * 60, // 1 hour in seconds
        });

        console.log('  AccessToken created:', at ? 'YES' : 'NO');
        
        at.addGrant({
            roomJoin: true,
            room: room,
            canPublish: true,
            canSubscribe: true,
        });
        
        console.log('  Grant added');
        
        // toJwt() returns a Promise, so we need to await it
        console.log('  Calling toJwt() (async)...');
        const jwt = await at.toJwt();
        
        console.log('  JWT generated successfully');
        console.log('  JWT type:', typeof jwt);
        
        // DEBUG: Log the generated token (safely)
        if (typeof jwt === 'string' && jwt.length > 0) {
            console.log(`  Generated JWT (first 50 chars): ${jwt.substring(0, 50)}...`);
            console.log(`  JWT length: ${jwt.length} characters`);
        } else {
            console.log(`  JWT is not a valid string! Type: ${typeof jwt}, Value:`, jwt);
        }
        
        if (!jwt || jwt === '{}' || typeof jwt !== 'string') {
            console.error('❌ Token generation failed! JWT is not a valid string.');
            console.error('  Received:', jwt);
            throw new Error('Failed to generate valid JWT token');
        }
        
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

// Get avatar endpoint
app.get('/avatar/:id', async (req, res) => {
    try {
        const avatarPath = path.join(AVATAR_DIR, `${req.params.id}.png`);
        
        // Check if file exists
        await fs.access(avatarPath);
        
        // Read and return the base64 avatar
        const avatarData = await fs.readFile(avatarPath, 'utf8');
        res.json({ 
            avatar_id: req.params.id,
            avatar_b64: avatarData 
        });
        
    } catch (error) {
        console.error('Avatar retrieval error:', error);
        res.status(404).json({ 
            error: 'Avatar not found',
            details: `Avatar ID ${req.params.id} does not exist` 
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

        // Generate unique avatar ID
        const avatarId = crypto.randomBytes(16).toString('hex');
        const avatarPath = path.join(AVATAR_DIR, `${avatarId}.png`);
        
        // Save avatar to file system
        await fs.writeFile(avatarPath, avatar_b64, 'utf8');
        console.log(`💾 Avatar saved to file system with ID: ${avatarId}`);
        console.log(`   Size: ${Buffer.from(avatar_b64).length} bytes (base64)`);

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
        
        // Now we only pass the avatar ID in metadata (much smaller!)
        const roomMetadata = JSON.stringify({
            avatar_id: avatarId,
            avatar_server: `http://localhost:${PORT}`,  // So agent knows where to fetch from
            ditto_settings: dittoSettings,
            created_at: new Date().toISOString()
        });

        try {
            // Update room metadata with avatar information
            await roomService.updateRoomMetadata(room, roomMetadata);
            console.log(`📝 Room ${room} metadata updated with avatar ID: ${avatarId}`);
            console.log(`   Metadata size: ${roomMetadata.length} bytes (well under 64KB limit!)`);
        } catch (roomError) {
            console.warn('Room metadata update failed:', roomError.message);
            // Continue - the agent can still work without metadata
        }

        res.json({
            status: 'room_configured',
            room: room,
            identity: identity,
            avatar_id: avatarId,
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
    
    // DEBUG: Display LiveKit credentials for verification
    console.log('\n========================================');
    console.log('🔐 DEBUG - LiveKit Credentials (for meet.livekit.io testing):');
    console.log('========================================');
    console.log(`📍 URL: ${process.env.LIVEKIT_URL || 'NOT SET'}`);
    console.log(`🔑 API Key: ${process.env.LK_API_KEY || 'NOT SET'}`);
    console.log(`🔒 API Secret: ${process.env.LK_API_SECRET || 'NOT SET'}`);
    console.log('========================================');
    console.log('⚠️  SECURITY WARNING: Remove this debug code in production!');
    console.log('========================================\n');
});
