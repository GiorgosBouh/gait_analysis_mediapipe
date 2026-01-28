// pose.js — FIXED
// Uses RAW GitHub model URL + createFromModelBuffer(Uint8Array)
// Avoids GitHub Pages Range issues (416) and avoids "ExternalFile must specify..." by ensuring Uint8Array.

import { PoseLandmarker, FilesetResolver } from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.9";

console.log("POSE.JS VERSION: BUFFER_V2");

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

// ✅ IMPORTANT: repo name is case-sensitive on raw.githubusercontent.com
// If your default branch is "master" instead of "main", change it.
const RAW_MODEL_URL =
  "https://raw.githubusercontent.com/giorgosbouh/GAIT_ANALYSIS_MEDIAPIPE/main/docs/models/pose_landmarker_lite.task";

async function fetchModelAsUint8Array(url) {
  const res = await fetch(url, { cache: "no-store", credentials: "omit" });
  if (!res.ok) throw new Error(`Model fetch failed: ${url} (HTTP ${res.status})`);

  const ab = await res.arrayBuffer();
  const bytes = new Uint8Array(ab);

  // sanity: model should be MBs, not tiny
  if (!bytes || bytes.byteLength < 200_000) {
    throw new Error(`Model buffer too small (${bytes?.byteLength ?? 0} bytes): ${url}`);
  }
  return bytes;
}

export async function createPoseLandmarker() {
  if (poseLandmarker) return poseLandmarker;

  const wasmBase = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.9/wasm";

  console.log("[Pose] model RAW url:", RAW_MODEL_URL);
  console.log("[Pose] wasm base:", wasmBase);

  const modelBytes = await fetchModelAsUint8Array(RAW_MODEL_URL);
  console.log("[Pose] model bytes:", modelBytes.byteLength);

  const vision = await FilesetResolver.forVisionTasks(wasmBase);

  // ✅ This avoids the ExternalFile empty-input problem
  poseLandmarker = await PoseLandmarker.createFromModelBuffer(vision, modelBytes);

  // Ensure we run in VIDEO mode (detectForVideo needs it)
  poseLandmarker.setOptions({ runningMode: "VIDEO", numPoses: 1 });

  console.log("[Pose] PoseLandmarker READY");
  return poseLandmarker;
}
