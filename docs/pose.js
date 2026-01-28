// pose.js (RAW-only, no Range, no GitHub Pages fetch)
// Fixes: GitHub Pages fetch returns 0 bytes for .task, Range not supported (HTTP 416)
// Solution: fetch the model from raw.githubusercontent.com and pass as modelAssetBuffer.

import { PoseLandmarker, FilesetResolver } from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.9";

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

let poseLandmarker = null;

// ✅ Your repo is GAIT_ANALYSIS_MEDIAPIPE (case-sensitive)
const RAW_MODEL_URL =
  "https://raw.githubusercontent.com/giorgosbouh/GAIT_ANALYSIS_MEDIAPIPE/main/docs/models/pose_landmarker_lite.task";

async function fetchAsArrayBuffer(url) {
  const res = await fetch(url, { cache: "no-store", credentials: "omit" });
  if (!res.ok) throw new Error(`Model fetch failed: ${url} (HTTP ${res.status})`);

  const buf = await res.arrayBuffer();

  if (!buf || buf.byteLength < 1024 * 100) {
    throw new Error(`Model fetch suspicious size (${buf?.byteLength ?? 0} bytes): ${url}`);
  }

  return buf;
}

export async function createPoseLandmarker() {
  if (poseLandmarker) return poseLandmarker;

  const wasmBase = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.9/wasm";

  console.log("[Pose] pose.js URL:", import.meta.url);
  console.log("[Pose] RAW model URL:", RAW_MODEL_URL);
  console.log("[Pose] wasm base:", wasmBase);

  const modelAssetBuffer = await fetchAsArrayBuffer(RAW_MODEL_URL);

  const vision = await FilesetResolver.forVisionTasks(wasmBase);

  poseLandmarker = await PoseLandmarker.createFromOptions(vision, {
    baseOptions: { modelAssetBuffer },
    runningMode: "VIDEO",
    numPoses: 1,
  });

  console.log("[Pose] PoseLandmarker READY");
  return poseLandmarker;
}
