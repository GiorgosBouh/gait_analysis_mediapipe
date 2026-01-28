// pose.js
import { PoseLandmarker, FilesetResolver } from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.9";

// Σημαντικό: Κατασκευάζουμε το URL του μοντέλου δυναμικά με βάση τη θέση αυτού του αρχείου
const MODEL_FILENAME = "pose_landmarker_lite.task";
const MODEL_URL = new URL(`./${MODEL_FILENAME}`, import.meta.url).toString();

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

// Συνάρτηση για να κατεβάσουμε τα bytes του μοντέλου
async function loadModelBytes() {
  console.log(`[Pose] Loading model from: ${MODEL_URL}`);
  const response = await fetch(MODEL_URL);
  if (!response.ok) {
    throw new Error(`Failed to fetch model: ${response.status} ${response.statusText}`);
  }
  return await response.arrayBuffer();
}

export async function createPoseLandmarker() {
  try {
    const vision = await FilesetResolver.forVisionTasks(
      "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.9/wasm"
    );

    // Φορτώνουμε το μοντέλο χειροκίνητα για να αποφύγουμε προβλήματα με paths στο GitHub Pages
    const modelBuffer = await loadModelBytes();
    const modelBytes = new Uint8Array(modelBuffer);

    const poseLandmarker = await PoseLandmarker.createFromOptions(vision, {
      baseOptions: {
        modelAssetBuffer: modelBytes,
        delegate: "GPU" // Προσπάθεια για GPU, θα γυρίσει σε CPU αν αποτύχει
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
    throw error;
  }
}