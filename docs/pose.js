// pose.js
// Robust MediaPipe Pose loader for GitHub Pages
// Fixes: GitHub Pages sometimes returns 0-byte body for .task via fetch()
// Solution: prefer raw.githubusercontent.com for model fetch, keep Pages fallback.

import { PoseLandmarker, FilesetResolver } from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.9";

// ---- Landmark names / indices ----
export const POSE_LANDMARK_NAMES = [
  "nose",
  "left_eye_inner", "left_eye", "left_eye_outer",
  "right_eye_inner", "right_eye", "right_eye_outer",
  "left_ear", "right_ear",
  "mouth_left", "mouth_right",
  "left_shoulder", "right_shoulder",
  "left_elbow", "right_elbow",
  "left_wrist", "right_wrist",
  "left_pinky", "right_pinky",
  "left_index", "right_index",
  "left_thumb", "right_thumb",
  "left_hip", "right_hip",
  "left_knee", "right_knee",
  "left_ankle", "right_ankle",
  "left_heel", "right_heel",
  "left_foot_index", "right_foot_index",
];

export const LANDMARK_INDEX = Object.fromEntries(
  POSE_LANDMARK_NAMES.map((name, idx) => [name, idx])
);

// ---- Simple skeleton connections (name -> name) ----
export const POSE_CONNECTIONS = [
  ["left_shoulder", "right_shoulder"],
  ["left_hip", "right_hip"],
  ["left_shoulder", "left_elbow"],
  ["left_elbow", "left_wrist"],
  ["right_shoulder", "right_elbow"],
  ["right_elbow", "right_wrist"],
  ["left_shoulder", "left_hip"],
  ["right_shoulder", "right_hip"],
  ["left_hip", "left_knee"],
  ["left_knee", "left_ankle"],
  ["right_hip", "right_knee"],
  ["right_knee", "right_ankle"],
  ["left_ankle", "left_heel"],
  ["left_heel", "left_foot_index"],
  ["right_ankle", "right_heel"],
  ["right_heel", "right_foot_index"],
  ["nose", "left_shoulder"],
  ["nose", "right_shoulder"],
];

// ---- Internal singleton ----
let poseLandmarker = null;

// ---- CONFIG: set your RAW model URL here ----
// If your repo name is different, change only the <REPO> part.
const RAW_MODEL_URL =
  "https://raw.githubusercontent.com/giorgosbouh/gait_analysis_mediapipe/main/docs/models/pose_landmarker_lite.task";

// ---- Helpers ----
async function fetchArrayBuffer(url, { useRange = false } = {}) {
  const headers = useRange ? { Range: "bytes=0-" } : undefined;

  const res = await fetch(url, {
    cache: "reload",
    credentials: "omit",
    headers,
  });

  // For range requests, 206 is normal. For normal requests, 200 is normal.
  if (!res.ok) {
    throw new Error(`Fetch failed: ${url} (HTTP ${res.status})`);
  }

  const buf = await res.arrayBuffer();
  return buf;
}

async function fetchModelBufferBestEffort(pagesUrl) {
  // 1) Try GitHub Pages first (fast if it works)
  try {
    const buf = await fetchArrayBuffer(pagesUrl);
    if (buf && buf.byteLength > 1024 * 100) return buf;
    console.warn(`[Pose] Pages model returned ${buf?.byteLength ?? 0} bytes, retrying with Range…`);

    // Some CDNs/servers behave better with Range
    const buf2 = await fetchArrayBuffer(pagesUrl, { useRange: true });
    if (buf2 && buf2.byteLength > 1024 * 100) return buf2;

    console.warn(`[Pose] Pages+Range model returned ${buf2?.byteLength ?? 0} bytes. Falling back to RAW…`);
  } catch (e) {
    console.warn("[Pose] Pages model fetch failed, falling back to RAW…", e);
  }

  // 2) RAW fallback (most reliable for fetch body)
  const rawBuf = await fetchArrayBuffer(RAW_MODEL_URL);
  if (!rawBuf || rawBuf.byteLength < 1024 * 100) {
    throw new Error(`RAW model fetch suspicious size (${rawBuf?.byteLength ?? 0} bytes): ${RAW_MODEL_URL}`);
  }
  return rawBuf;
}

// ---- Public API expected by app.js ----
export async function createPoseLandmarker() {
  if (poseLandmarker) return poseLandmarker;

  const wasmBase = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.9/wasm";

  // Resolve Pages model URL relative to this file
  const pagesModelUrl = new URL("./models/pose_landmarker_lite.task", import.meta.url).toString();

  console.log("[Pose] pose.js URL:", import.meta.url);
  console.log("[Pose] Pages model URL:", pagesModelUrl);
  console.log("[Pose] RAW model URL:", RAW_MODEL_URL);
  console.log("[Pose] wasm base:", wasmBase);

  // Load model into memory (buffer) to avoid WASM file-size / StartGraph issues
  const modelAssetBuffer = await fetchModelBufferBestEffort(pagesModelUrl);

  const vision = await FilesetResolver.forVisionTasks(wasmBase);

  poseLandmarker = await PoseLandmarker.createFromOptions(vision, {
    baseOptions: { modelAssetBuffer },
    runningMode: "VIDEO",
    numPoses: 1,
  });

  console.log("[Pose] PoseLandmarker READY");
  return poseLandmarker;
}
