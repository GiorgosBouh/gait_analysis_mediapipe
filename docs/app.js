// app.js — FINAL FORCE UNLOCK VERSION
import {
  createPoseLandmarker,
  POSE_LANDMARK_NAMES,
  POSE_CONNECTIONS,
  LANDMARK_INDEX,
} from "./pose.js?v=FORCE_UNLOCK";

import {
  clamp,
  median,
  movingAverage,
  downloadFile,
  formatNumber,
  computeDtStats,
  ensureCanvasSize,
  computePeaks,
} from "./utils.js?v=FORCE_UNLOCK";

import { LineChart } from "./charts.js?v=FORCE_UNLOCK";

console.log("APP STARTED - FORCE UNLOCK VERSION");

// State
const appState = {
  mode: "live",
  running: false,
  poseLandmarker: null,
  lastResult: null,
  lastTimestamp: null,
  data: [],
  dts: [],
  calibration: {
    samples: [],
    totalFrames: 0,
    targetFrames: 45,
    scale: null,
    status: "Not started",
  },
  warnings: [],
  videoSize: { width: 0, height: 0 },
  modelReady: false,
};

// Global vars for elements
let dom = {};
let charts = {};
let animationId = null;
let mediaStream = null;
let uploadUrl = null;

// --- 1. CORE INIT FUNCTION ---
async function initializeSystem() {
  // Grab elements explicitly here to ensure they exist
  dom = {
    appStatus: document.getElementById("appStatus"),
    startLive: document.getElementById("startLive"),
    stopLive: document.getElementById("stopLive"),
    startUpload: document.getElementById("startUpload"),
    stopUpload: document.getElementById("stopUpload"),
    videoFile: document.getElementById("videoFile"),
    video: document.getElementById("video"),
    overlay: document.getElementById("overlay"),
    // Tabs & Charts
    tabs: document.querySelectorAll(".tab"),
    tabContents: {
      live: document.getElementById("tab-live"),
      upload: document.getElementById("tab-upload"),
    },
    // Controls
    heightInput: document.getElementById("heightInput"),
    smoothingWindow: document.getElementById("smoothingWindow"),
    inferenceThrottle: document.getElementById("inferenceThrottle"),
    toggleOverlay: document.getElementById("toggleOverlay"),
    autoCalibrate: document.getElementById("autoCalibrate"),
    calibrateNow: document.getElementById("calibrateNow"),
    exportCsv: document.getElementById("exportCsv"),
    exportMeta: document.getElementById("exportMeta"),
    // Outputs
    calibrationStatus: document.getElementById("calibrationStatus"),
    scaleStatus: document.getElementById("scaleStatus"),
    visibilityStatus: document.getElementById("visibilityStatus"),
    avgStepLength: document.getElementById("avgStepLength"),
    avgStepWidth: document.getElementById("avgStepWidth"),
    avgComSpeed: document.getElementById("avgComSpeed"),
    cadenceValue: document.getElementById("cadenceValue"),
    stepChart: document.getElementById("stepChart"),
    comChart: document.getElementById("comChart"),
  };

  // Init Charts
  if (dom.stepChart) charts.step = new LineChart(dom.stepChart, { lineColor: "#2563eb" });
  if (dom.comChart) charts.com = new LineChart(dom.comChart, { lineColor: "#10b981" });

  // Add listeners
  setupEventListeners();

  // Load Model
  setStatus("Loading AI...", false);
  try {
    appState.poseLandmarker = await createPoseLandmarker();
    appState.modelReady = true;
    
    // SUCCESS!
    setStatus("System Ready", false);
    console.log("Model Ready. Unlocking buttons...");
    
    // FORCE UNLOCK BUTTONS
    forceUnlockButtons();
    
  } catch (error) {
    console.error(error);
    setStatus("Model Error", true);
    alert("Error loading model. Check console.");
  }
}

// --- 2. FORCE UNLOCK LOGIC ---
function forceUnlockButtons() {
  // Ξεκλειδώνουμε το START LIVE άμεσα
  if (dom.startLive) {
    dom.startLive.disabled = false;
    dom.startLive.removeAttribute("disabled");
    dom.startLive.style.cursor = "pointer";
    dom.startLive.style.opacity = "1";
    console.log("Start Live button UNLOCKED");
  }

  // Το Upload θέλει και αρχείο, αλλά ας ενεργοποιήσουμε το input
  if (dom.videoFile) dom.videoFile.disabled = false;
  
  // Update general UI
  updateControls(false);
}

function setStatus(msg, isErr) {
  if (dom.appStatus) {
    dom.appStatus.innerText = msg;
    dom.appStatus.style.background = isErr ? "#ef4444" : (appState.modelReady ? "#10b981" : "#fbbf24");
    dom.appStatus.style.color = isErr ? "#fff" : "#111827";
  }
}

// --- 3. CONTROL LOGIC ---
function updateControls(running) {
  if (!appState.modelReady) return; // Wait for model

  // Live Buttons
  if (dom.startLive) dom.startLive.disabled = running;
  if (dom.stopLive) dom.stopLive.disabled = !running;

  // Upload Buttons
  if (dom.startUpload) {
    const hasFile = dom.videoFile && dom.videoFile.files && dom.videoFile.files.length > 0;
    dom.startUpload.disabled = running || !hasFile;
  }
  if (dom.stopUpload) dom.stopUpload.disabled = !running;
}

// --- 4. APP LOGIC (Simplified) ---

async function startLive() {
  if (!appState.modelReady) return;
  
  appState.mode = "live";
  resetSession();
  setStatus("Starting Cam...");

  try {
    mediaStream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
    dom.video.srcObject = mediaStream;
    await dom.video.play();
    
    await waitForVideoMetadata(dom.video);
    setVideoSize();

    appState.running = true;
    setStatus("Live Running", false);
    updateControls(true);
    runLoop();
  } catch (err) {
    console.error(err);
    alert("Could not access camera. Please allow permissions.");
    setStatus("Cam Error", true);
  }
}

async function startUpload() {
  appState.mode = "upload";
  resetSession();
  
  const file = dom.videoFile.files[0];
  uploadUrl = URL.createObjectURL(file);
  dom.video.srcObject = null;
  dom.video.src = uploadUrl;

  try {
    await dom.video.play();
    await waitForVideoMetadata(dom.video);
    setVideoSize();

    appState.running = true;
    setStatus("Analyzing...", false);
    updateControls(true);
    runLoop();
  } catch (err) {
    console.error(err);
    setStatus("Video Error", true);
  }
}

function stopProcessing() {
  appState.running = false;
  if (animationId) cancelAnimationFrame(animationId);
  if (dom.video) dom.video.pause();
  
  if (mediaStream) {
    mediaStream.getTracks().forEach(t => t.stop());
    mediaStream = null;
  }
  
  setStatus("Stopped", false);
  updateControls(false);
}

function runLoop() {
  if (!appState.running) return;
  const now = (appState.mode === "upload") ? dom.video.currentTime * 1000 : performance.now();
  
  const throttle = dom.inferenceThrottle ? Math.max(1, Number(dom.inferenceThrottle.value)) : 1;
  const idx = appState.calibration.totalFrames + appState.data.length;

  if (idx % throttle === 0 && dom.video.videoWidth > 0) {
    try {
      const res = appState.poseLandmarker.detectForVideo(dom.video, now);
      processResult(res, now);
      appState.lastResult = res;
    } catch(e) { console.warn(e); }
  } else if (appState.lastResult) {
    drawPose(appState.lastResult.landmarks?.[0]);
  }

  if (appState.mode === "upload" && dom.video.ended) {
    stopProcessing();
    setStatus("Done", false);
    return;
  }
  animationId = requestAnimationFrame(runLoop);
}

function processResult(res, time) {
  const lm = res.landmarks?.[0];
  if (!lm) return;
  drawPose(lm);
  
  // Simple Metrics & Calibration logic
  const timeS = time/1000;
  
  if (dom.autoCalibrate.checked && !appState.calibration.scale) {
    const h = Math.abs(lm[LANDMARK_INDEX.nose].y - lm[LANDMARK_INDEX.left_ankle].y) * dom.video.videoHeight;
    appState.calibration.samples.push(h);
    appState.calibration.totalFrames++;
    if (appState.calibration.totalFrames > 30) {
       const med = median(appState.calibration.samples);
       appState.calibration.scale = Number(dom.heightInput.value) / med;
       appState.calibration.status = "Calibrated";
    }
  }

  // Push Data
  const metrics = computeMetrics(lm, appState.calibration.scale);
  appState.data.push({ timestamp_s: timeS, ...metrics });
  
  updateUI();
}

function computeMetrics(lm, scale) {
  if (!scale) return { step_len: 0 };
  const w = dom.video.videoWidth * scale;
  const sl = Math.abs(lm[LANDMARK_INDEX.left_ankle].x - lm[LANDMARK_INDEX.right_ankle].x) * w;
  return { step_length_m2d_apparent: sl };
}

function updateUI() {
  dom.calibrationStatus.innerText = appState.calibration.status;
  dom.scaleStatus.innerText = appState.calibration.scale ? appState.calibration.scale.toFixed(3) : "-";
  
  const len = appState.data.length;
  if (len > 0) {
     const last = appState.data[len-1];
     dom.avgStepLength.innerText = last.step_length_m2d_apparent.toFixed(2);
     dom.exportCsv.disabled = false;
     dom.exportMeta.disabled = false;
  }
}

function drawPose(lm) {
  const ctx = dom.overlay.getContext("2d");
  ctx.clearRect(0,0, dom.overlay.width, dom.overlay.height);
  if (!dom.toggleOverlay.checked || !lm) return;
  
  ctx.strokeStyle = "blue"; 
  ctx.lineWidth = 2;
  POSE_CONNECTIONS.forEach(([s,e]) => {
     const p1 = lm[LANDMARK_INDEX[s]];
     const p2 = lm[LANDMARK_INDEX[e]];
     ctx.beginPath();
     ctx.moveTo(p1.x * dom.overlay.width, p1.y * dom.overlay.height);
     ctx.lineTo(p2.x * dom.overlay.width, p2.y * dom.overlay.height);
     ctx.stroke();
  });
}

function resetSession() {
  appState.data = [];
  appState.calibration = { samples: [], totalFrames: 0, scale: null, status: "Collecting" };
  updateUI();
}

function setVideoSize() {
  if (dom.video.videoWidth) {
    appState.videoSize = { w: dom.video.videoWidth, h: dom.video.videoHeight };
    ensureCanvasSize(dom.overlay, dom.video.videoWidth, dom.video.videoHeight);
  }
}

async function waitForVideoMetadata(v) {
  if(v.videoWidth) return;
  return new Promise(r => v.onloadedmetadata = r);
}

function setupEventListeners() {
  dom.startLive.addEventListener("click", startLive);
  dom.stopLive.addEventListener("click", stopProcessing);
  dom.startUpload.addEventListener("click", startUpload);
  dom.stopUpload.addEventListener("click", stopProcessing);
  dom.videoFile.addEventListener("change", () => updateControls(false));
  
  // Tab switching
  dom.tabs.forEach(tab => {
    tab.addEventListener("click", () => {
       dom.tabs.forEach(t => t.classList.remove("active"));
       tab.classList.add("active");
       dom.tabContents.live.classList.toggle("active", tab.dataset.tab === "live");
       dom.tabContents.upload.classList.toggle("active", tab.dataset.tab === "upload");
    });
  });
  
  // Export Stubs
  dom.exportCsv.addEventListener("click", () => downloadFile("data.csv", JSON.stringify(appState.data)));
}

// --- BOOTSTRAP ---
window.addEventListener("DOMContentLoaded", initializeSystem);