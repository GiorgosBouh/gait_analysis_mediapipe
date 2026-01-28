// pose.js
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

async function fetchArrayBufferWithFallback(url) {
  // 1) Normal fetch
  let res = await fetch(url, {
    cache: "reload",
    credentials: "omit",
  });

  if (!res.ok) {
    throw new Error(`Model fetch failed: ${url} (HTTP ${res.status})`);
  }

  let buf = await res.arrayBuffer();

  // If GitHub Pages / caching returns empty body, try Range request as fallback
  if (!buf || buf.byteLength === 0) {
    console.warn("[Pose] Model fetch returned 0 bytes. Retrying with Range…");

    res = await fetch(url, {
      cache: "reload",
      credentials: "omit",
      headers: { Range: "bytes=0-" },
    });

    if (!res.ok && res.status !== 206) {
      throw new Error(`Model range fetch failed: ${url} (HTTP ${res.status})`);
    }

    buf = await res.arrayBuffer();
  }

  if (!buf || buf.byteLength < 1024 * 100) {
    throw new Error(`Model fetch suspicious size (${buf?.byteLength ?? 0} bytes): ${url}`);
  }

  return buf;
}

export async function createPoseLandmarker() {
  if (poseLandmarker) return poseLandmarker;

  const wasmBase = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.9/wasm";

  // IMPORTANT: resolve relative to THIS file
  const modelUrl = new URL("./models/pose_landmarker_lite.task", import.meta.url).toString();

  console.log("[Pose] pose.js URL:", import.meta.url);
  console.log("[Pose] resolved model URL:", modelUrl);

  const modelAssetBuffer = await fetchArrayBufferWithFallback(modelUrl);

  const vision = await FilesetResolver.forVisionTasks(wasmBase);

  poseLandmarker = await PoseLandmarker.createFromOptions(vision, {
    baseOptions: { modelAssetBuffer },
    runningMode: "VIDEO",
    numPoses: 1,
  });

  console.log("[Pose] PoseLandmarker READY");
  return poseLandmarker;
}
