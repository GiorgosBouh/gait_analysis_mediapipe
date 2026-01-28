// pose.js
import {
  PoseLandmarker,
  FilesetResolver
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.9";

let poseLandmarker = null;

export async function initPoseLandmarker() {
  if (poseLandmarker) return poseLandmarker;

  console.log("[Pose] Initializing MediaPipe Pose…");

  const vision = await FilesetResolver.forVisionTasks(
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.9/wasm"
  );

  poseLandmarker = await PoseLandmarker.createFromOptions(vision, {
    baseOptions: {
      // ✅ SELF-HOSTED MODEL (NO 404, NO FIREWALL ISSUES)
      modelAssetPath: "./models/pose_landmarker_lite.task"
    },
    runningMode: "VIDEO",
    numPoses: 1
  });

  console.log("[Pose] PoseLandmarker READY");
  return poseLandmarker;
}

export function getPoseLandmarker() {
  return poseLandmarker;
}
