import { PoseLandmarker, FilesetResolver } from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.9";

// 1. Το τοπικό αρχείο (αυτό που ανέβασες)
const localModelPath = new URL("./pose_landmarker_lite.task", import.meta.url).toString();

// 2. Το online backup αρχείο (από την Google)
const googleModelPath = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task";

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

// Βοηθητική συνάρτηση για να κατεβάσουμε το αρχείο
async function loadModelBuffer(url) {
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`Failed to fetch ${url} - Status: ${response.status}`);
  }
  return await response.arrayBuffer();
}

export async function createPoseLandmarker() {
  const vision = await FilesetResolver.forVisionTasks(
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.9/wasm"
  );

  let modelBuffer;

  // Προσπάθεια 1: Τοπικό αρχείο
  try {
    console.log(`[Pose] Attempting to load local model: ${localModelPath}`);
    modelBuffer = await loadModelBuffer(localModelPath);
    console.log("[Pose] Local model loaded successfully.");
  } catch (localErr) {
    console.warn("[Pose] Local model failed (404). Switching to Google Backup...", localErr);
    
    // Προσπάθεια 2: Google Backup
    try {
      modelBuffer = await loadModelBuffer(googleModelPath);
      console.log("[Pose] Google Backup model loaded successfully.");
    } catch (googleErr) {
      alert("CRITICAL ERROR: Could not load AI model. Check internet connection.");
      throw googleErr;
    }
  }

  // Δημιουργία του Landmarker με τα bytes που κατεβάσαμε
  try {
    const poseLandmarker = await PoseLandmarker.createFromOptions(vision, {
      baseOptions: {
        modelAssetBuffer: new Uint8Array(modelBuffer), // Στέλνουμε τα bytes απευθείας
        delegate: "GPU"
      },
      runningMode: "VIDEO",
      numPoses: 1,
      minPoseDetectionConfidence: 0.5,
      minPosePresenceConfidence: 0.5,
      minTrackingConfidence: 0.5,
    });

    return poseLandmarker;
  } catch (createError) {
    console.error("[Pose] Error initializing Landmarker:", createError);
    throw createError;
  }
}