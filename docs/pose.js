// pose.js
// MediaPipe Pose (Tasks Vision) helper for in-browser gait analysis
// ✅ Uses modelAssetBuffer (fetch -> ArrayBuffer) to avoid WASM "Unable to get file size" failures
// ✅ Resolves model URL relative to this file via import.meta.url
// ✅ Keeps exports expected by app.js: createPoseLandmarker, POSE_LANDMARK_NAMES, POSE_CONNECTIONS, LANDMARK_INDEX

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
// Keep this stable; app.js draws using these names + LANDMARK_INDEX.
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

// ---- Helpers ----
async function fetchAsArrayBuffer(url) {
  const res = await fetch(url, { cache: "no-store" });

  if (!res.ok) {
    throw new Error(`Model fetch failed: ${url} (HTTP ${res.status})`);
  }

  const buf = await res.arrayBuffer();

  // A .task model should be much larger than a tiny HTML error page.
  if (!buf || buf.byteLength < 1024 * 100) {
    throw new Error(`Model fetch suspicious size (${buf?.byteLength} bytes): ${url}`);
  }

  return buf;
}

// ---- Public API expected by app.js ----
export async function createPoseLandmarker() {
  if (poseLandmarker) return poseLandmarker;

  // WASM runtime base (CDN)
  const wasmBase = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.9/wasm";

  // Resolve model URL relative to THIS file (robust on GitHub Pages subpaths)
  // IMPORTANT: if pose.js lives under /assets/, this resolves to /assets/models/...
  // So keep pose.js at same level as models/, OR adjust the relative path below accordingly.
  const modelUrl = new URL("./models/pose_landmarker_lite.task", import.meta.url).toString();

  console.log("[Pose] pose.js URL:", import.meta.url);
  console.log("[Pose] resolved model URL:", modelUrl);
  console.log("[Pose] wasm base:", wasmBase);

  // ✅ Load model into memory to bypass WASM "Unable to get file size" / external_file_handler issues
  const modelAssetBuffer = await fetchAsArrayBuffer(modelUrl);

  const vision = await FilesetResolver.forVisionTasks(wasmBase);

  poseLandmarker = await PoseLandmarker.createFromOptions(vision, {
    baseOptions: {
      modelAssetBuffer, // ✅ key fix
    },
    runningMode: "VIDEO",
    numPoses: 1,
  });

  console.log("[Pose] PoseLandmarker READY");
  return poseLandmarker;
}
