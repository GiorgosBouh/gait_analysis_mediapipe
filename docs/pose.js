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
  "left_foot_index", "right_foot_index"
];

export const LANDMARK_INDEX = Object.fromEntries(POSE_LANDMARK_NAMES.map((n, i) => [n, i]));

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

async function preflightFetch(url) {
  const res = await fetch(url, { cache: "no-store" });
  if (!res.ok) {
    throw new Error(`Model fetch failed: ${url} (HTTP ${res.status})`);
  }
  const buf = await res.arrayBuffer();
  if (!buf || buf.byteLength < 1024 * 100) {
    // model should be > ~100KB; if it's tiny it's probably an HTML 404 page
    throw new Error(`Model fetch suspicious size (${buf?.byteLength} bytes): ${url}`);
  }
  return url;
}

export async function createPoseLandmarker() {
  if (poseLandmarker) return poseLandmarker;

  // IMPORTANT: this is resolved relative to pose.js location
  const localModelUrl = new URL("./models/pose_landmarker_lite.task", import.meta.url).toString();

  console.log("[Pose] pose.js loaded from:", import.meta.url);
  console.log("[Pose] resolved model URL:", localModelUrl);

  // 1) Confirm model is fetchable from the exact URL the code will use
  await preflightFetch(localModelUrl);

  // 2) Init WASM runtime
  const wasmBase = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.9/wasm";
  console.log("[Pose] loading WASM from:", wasmBase);

  const vision = await FilesetResolver.forVisionTasks(wasmBase);

  // 3) Create landmarker
  poseLandmarker = await PoseLandmarker.createFromOptions(vision, {
    baseOptions: {
      modelAssetPath: localModelUrl, // explicit absolute URL
    },
    runningMode: "VIDEO",
    numPoses: 1,
  });

  console.log("[Pose] PoseLandmarker READY");
  return poseLandmarker;
}
