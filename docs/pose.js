import { PoseLandmarker, FilesetResolver } from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.9";

// ΣΗΜΑΝΤΙΚΟ: Αυτό βρίσκει το αρχείο .task στον ίδιο φάκελο με το pose.js
// ανεξάρτητα από το αν είσαι σε localhost ή στο github.io/repo/
const modelAssetPath = new URL("./pose_landmarker_lite.task", import.meta.url).toString();

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
    console.log("[Pose] Initializing Vision Tasks...");
    const vision = await FilesetResolver.forVisionTasks(
      "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.9/wasm"
    );

    console.log(`[Pose] Loading model from: ${modelAssetPath}`);
    
    // Φόρτωση του μοντέλου
    const poseLandmarker = await PoseLandmarker.createFromOptions(vision, {
      baseOptions: {
        modelAssetPath: modelAssetPath, // Χρησιμοποιούμε το URL που φτιάξαμε
        delegate: "GPU"
      },
      runningMode: "VIDEO",
      numPoses: 1,
      minPoseDetectionConfidence: 0.5,
      minPosePresenceConfidence: 0.5,
      minTrackingConfidence: 0.5,
    });

    console.log("[Pose] Model loaded successfully!");
    return poseLandmarker;
  } catch (error) {
    console.error("[Pose] Error creating landmarker:", error);
    alert("Error loading AI Model. Please check console.");
    throw error;
  }
}