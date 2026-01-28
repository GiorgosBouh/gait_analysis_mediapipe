// pose.js
import {
  PoseLandmarker,
  FilesetResolver
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.9";

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

export const LANDMARK_INDEX = Object.fromEntries(
  POSE_LANDMARK_NAMES.map((name, idx) => [name, idx])
);

// Simple skeleton connections (name->name). Keep it minimal + stable.
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

export async function createPoseLandmarker() {
  if (poseLandmarker) return poseLandmarker;

  // WASM runtime (CDN)
  const vision = await FilesetResolver.forVisionTasks(
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.9/wasm"
  );

  // ✅ Self-hosted model (NO google storage 404)
  poseLandmarker = await PoseLandmarker.createFromOptions(vision, {
    baseOptions: {
      modelAssetPath: "./models/pose_landmarker_lite.task"
    },
    runningMode: "VIDEO",
    numPoses: 1
  });

  return poseLandmarker;
}
