// server.js (CommonJS)
const express = require("express");
const crypto = require("crypto");
require("dotenv").config();

const app = express();
app.use(express.json({ limit: "10mb" }));
app.use(express.static(__dirname + "/public"));
app.enable("trust proxy");

const {
  LIVEKIT_URL,
  LK_API_KEY,
  LK_API_SECRET,
  RUNPOD_ENDPOINT_ID,
  RUNPOD_API_KEY,
  PORT = 3000,
} = process.env;

function signHS256(header, payload, secret) {
  const enc = (obj) => Buffer.from(JSON.stringify(obj)).toString("base64url");
  const h = enc(header);
  const p = enc(payload);
  const data = `${h}.${p}`;
  const sig = crypto.createHmac("sha256", secret).update(data).digest("base64url");
  return `${data}.${sig}`;
}
function createLiveKitToken({ apiKey, apiSecret, identity, room, name }) {
  const now = Math.floor(Date.now() / 1000);
  const payload = {
    iss: apiKey,
    aud: "livekit",          // optional but harmless
    exp: now + 60 * 60,
    nbf: now - 10,
    sub: identity,
    name: name || identity,
    video: {
      room,                  // room name
      roomJoin: true,        // <-- REQUIRED to join
      roomCreate: true,      // allow creating if it doesn't exist
      canPublish: true,
      canSubscribe: true,
      canPublishData: true,
    },
  };
  return signHS256({ alg: "HS256", typ: "JWT" }, payload, apiSecret);
}


app.get("/token", (req, res) => {
  const { identity = `web-${Date.now()}`, room = "liveportrait-test" } = req.query;
  const token = createLiveKitToken({
    apiKey: LK_API_KEY,
    apiSecret: LK_API_SECRET,
    identity,
    room,
    name: identity,
  });
  res.json({ token, url: LIVEKIT_URL });
});

app.post("/start-lp", async (req, res) => {
  try {
    const {
      identity, room, avatar_b64,
      width = 512, height = 512, fps = 24, backend = "torch",
    } = req.body || {};

    const workerToken = createLiveKitToken({
      apiKey: LK_API_KEY,
      apiSecret: LK_API_SECRET,
      identity: `lp-worker-${Date.now()}`,
      room,
      name: "lp-worker",
    });

    const payload = {
      input: {
        livekit_url: LIVEKIT_URL,
        token: workerToken,
        driver_identity: identity,
        track_sid: null,
        source_image_b64: avatar_b64,
        width, height, fps, backend,
      },
    };

    // Node 18+ has global fetch; if yours doesn't, `npm i undici` and:
    // const { fetch } = require('undici');
    const rp = await fetch(`https://api.runpod.ai/v2/${RUNPOD_ENDPOINT_ID}/run`, {
      method: "POST",
      headers: {
        Authorization: `Bearer ${RUNPOD_API_KEY}`,
        "Content-Type": "application/json",
      },
      body: JSON.stringify(payload),
    });

    if (!rp.ok) {
      const txt = await rp.text();
      return res.status(502).json({ error: "runpod_failed", detail: txt });
    }
    const data = await rp.json();
    res.json({ jobId: data.id || data.jobId || data.job_id || null });
  } catch (e) {
    console.error(e);
    res.status(500).json({ error: "server_error", detail: String(e) });
  }
});

app.listen(PORT, () => {
  console.log(`Web demo on http://localhost:${PORT}`);
});

