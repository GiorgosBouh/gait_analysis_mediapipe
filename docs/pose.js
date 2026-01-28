// pose.js - Stable Version 0.10.0
import { PoseLandmarker, FilesetResolver } from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.0";

export const POSE_LANDMARK_NAMES = [
  "nose", "left_eye_inner", "left_eye", "left_eye_outer",
  "right_eye_inner", "right_eye", "right_eye_outer",
  "left_ear", "right_ear", "mouth_left", "mouth_right",
  "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
  "left_wrist", "right_wrist", "left_pinky", "right_pinky",
  "left_index", "right_index", "left_thumb", "right_thumb",
  "left_hip", "right_hip", "left_knee", "right_knee",
  "left_ankle", "right_ankle", "left_heel", "right_heel",
  "left_foot_index", "right_foot_index",
];

export const LANDMARK_INDEX = Object.fromEntries(
  POSE_LANDMARK_NAMES.map((n, i) => [n, i])
);

export const POSE_CONNECTIONS = PoseLandmarker.POSE_CONNECTIONS;

export async function createPoseLandmarker() {
  try {
    console.log("[Pose] 1. Initializing FilesetResolver (WASM)...");
    const vision = await FilesetResolver.forVisionTasks(
      "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.0/wasm"
    );

    console.log("[Pose] 2. Downloading Model from Google...");
    const poseLandmarker = await PoseLandmarker.createFromOptions(vision, {
      baseOptions: {
        modelAssetPath: `https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task`,
        delegate: "GPU"
      },
      runningMode: "VIDEO",
      numPoses: 1,
      minPoseDetectionConfidence: 0.5,
      minPosePresenceConfidence: 0.5,
      minTrackingConfidence: 0.5,
    });

    console.log("[Pose] 3. Model Loaded Successfully!");
    return poseLandmarker;
  } catch (error) {
    console.error("[Pose] FAILED:", error);
    alert(`CRITICAL ERROR: Failed to load AI model.\n${error.message}`);
    throw error;
  }
}