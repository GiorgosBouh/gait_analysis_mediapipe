// pose.js — Diagnostics + robust model loading for GitHub Pages
// - Prefer self-hosted model in docs/models
// - Fallback to RAW GitHub URL if Pages returns invalid bytes
// - Avoid Range requests; always fetch full bytes
// - Use Uint8Array buffer for MediaPipe Tasks

import { PoseLandmarker, FilesetResolver } from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.9";

const BUILD_STAMP = "20260201_1035";
console.log(`POSE.JS VERSION: ${BUILD_STAMP}`);

export const POSE_LANDMARK_NAMES = [
  "nose",
  "left_eye_inner","left_eye","left_eye_outer",
  "right_eye_inner","right_eye","right_eye_outer",
  "left_ear","right_ear",
  "mouth_left","mouth_right",
  "left_shoulder","right_shoulder",
  "left_elbow","right_elbow",
  "left_wrist","right_wrist",
  "left_pinky","right_pinky",
  "left_index","right_index",
  "left_thumb","right_thumb",
  "left_hip","right_hip",
  "left_knee","right_knee",
  "left_ankle","right_ankle",
  "left_heel","right_heel",
  "left_foot_index","right_foot_index",
];

export const LANDMARK_INDEX = Object.fromEntries(
  POSE_LANDMARK_NAMES.map((n, i) => [n, i])
);

export const POSE_CONNECTIONS = [
  ["left_shoulder","right_shoulder"],
  ["left_hip","right_hip"],
  ["left_shoulder","left_elbow"],
  ["left_elbow","left_wrist"],
  ["right_shoulder","right_elbow"],
  ["right_elbow","right_wrist"],
  ["left_shoulder","left_hip"],
  ["right_shoulder","right_hip"],
  ["left_hip","left_knee"],
  ["left_knee","left_ankle"],
  ["right_hip","right_knee"],
  ["right_knee","right_ankle"],
  ["left_ankle","left_heel"],
  ["left_heel","left_foot_index"],
  ["right_ankle","right_heel"],
  ["right_heel","right_foot_index"],
  ["nose","left_shoulder"],
  ["nose","right_shoulder"],
];

let poseLandmarker = null;

const MIN_MODEL_BYTES = 200_000;
const WASM_BASE = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.9/wasm";
const MODEL_URL = new URL("./models/pose_landmarker_lite.task", import.meta.url).href;

// ✅ IMPORTANT: repo name is case-sensitive on raw.githubusercontent.com
// If your default branch is "master" instead of "main", change it.
const RAW_MODEL_URL =
  "https://raw.githubusercontent.com/giorgosbouh/gait_analysis_mediapipe/main/docs/models/pose_landmarker_lite.task";

async function fetchModelAsUint8Array(url, label) {
  const response = await fetch(url, { cache: "no-store", credentials: "omit" });
  const contentType = response.headers.get("content-type");
  const contentLength = response.headers.get("content-length");

  if (!response.ok) {
    console.warn(`[Pose] ${label} fetch failed`, {
      url,
      status: response.status,
      contentType,
      contentLength,
    });
    throw new Error(`Model fetch failed: ${url} (HTTP ${response.status})`);
  }

  const arrayBuffer = await response.arrayBuffer();
  const byteLength = arrayBuffer.byteLength;
  console.log(`[Pose] ${label} fetch ok`, {
    url,
    status: response.status,
    contentType,
    contentLength,
    byteLength,
  });

  if (!byteLength || byteLength < MIN_MODEL_BYTES) {
    throw new Error(`Model buffer too small (${byteLength} bytes): ${url}`);
  }

  return new Uint8Array(arrayBuffer);
}

async function loadModelBytes() {
  try {
    return await fetchModelAsUint8Array(MODEL_URL, "Pages");
  } catch (err) {
    console.warn("[Pose] Pages model invalid, falling back to RAW GitHub URL.", err);
  }

  return await fetchModelAsUint8Array(RAW_MODEL_URL, "RAW");
}

export async function createPoseLandmarker() {
  if (poseLandmarker) return poseLandmarker;

  console.log("[Pose] model Pages url:", MODEL_URL);
  console.log("[Pose] model RAW url:", RAW_MODEL_URL);
  console.log("[Pose] wasm base:", WASM_BASE);

  const modelBytes = await loadModelBytes();
  console.log("[Pose] model bytes:", modelBytes.byteLength);

  const vision = await FilesetResolver.forVisionTasks(WASM_BASE);

  let created = null;
  let lastError = null;

  try {
    created = await PoseLandmarker.createFromOptions(vision, {
      baseOptions: { modelAssetBuffer: modelBytes },
      runningMode: "VIDEO",
      numPoses: 1,
    });
  } catch (err) {
    lastError = err;
    console.warn("[Pose] createFromOptions(modelAssetBuffer Uint8Array) failed.", err);
  }

  if (!created) {
    try {
      created = await PoseLandmarker.createFromOptions(vision, {
        baseOptions: { modelAssetBuffer: modelBytes.buffer },
        runningMode: "VIDEO",
        numPoses: 1,
      });
    } catch (err) {
      lastError = err;
      console.warn("[Pose] createFromOptions(modelAssetBuffer ArrayBuffer) failed.", err);
    }
  }

  if (!created && typeof PoseLandmarker.createFromModelBuffer === "function") {
    try {
      created = await PoseLandmarker.createFromModelBuffer(vision, modelBytes);
    } catch (err) {
      lastError = err;
      console.warn("[Pose] createFromModelBuffer(Uint8Array) failed.", err);
    }
  }

  if (!created) {
    throw lastError || new Error("Unable to create PoseLandmarker");
  }

  poseLandmarker = created;
  poseLandmarker.setOptions({ runningMode: "VIDEO", numPoses: 1 });

  console.log("[Pose] PoseLandmarker READY");
  return poseLandmarker;
}
