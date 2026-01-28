// app.js
import {
  createPoseLandmarker,
  POSE_LANDMARK_NAMES,
  POSE_CONNECTIONS,
  LANDMARK_INDEX,
} from "./pose.js?v=20260201_1100";
import {
  clamp,
  median,
  movingAverage,
  downloadFile,
  formatNumber,
  computeDtStats,
  ensureCanvasSize,
  computePeaks,
} from "./utils.js?v=20260201_1100";
import { LineChart } from "./charts.js?v=20260201_1100";

const BUILD_STAMP = "20260201_1100";
console.log(`APP.JS VERSION: ${BUILD_STAMP}`);

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
  modelLoading: false,
};

const dom = {
  appStatus: document.getElementById("appStatus"),
  tabs: document.querySelectorAll(".tab"),
  tabContents: {
    live: document.getElementById("tab-live"),
    upload: document.getElementById("tab-upload"),
  },
  video: document.getElementById("video"),
  overlay: document.getElementById("overlay"),
  startLive: document.getElementById("startLive"),
  stopLive: document.getElementById("stopLive"),
  videoFile: document.getElementById("videoFile"),
  startUpload: document.getElementById("startUpload"),
  stopUpload: document.getElementById("stopUpload"),
  heightInput: document.getElementById("heightInput"),
  smoothingWindow: document.getElementById("smoothingWindow"),
  smoothingValue: document.getElementById("smoothingValue"),
  inferenceThrottle: document.getElementById("inferenceThrottle"),
  inferenceValue: document.getElementById("inferenceValue"),
  toggleOverlay: document.getElementById("toggleOverlay"),
  autoCalibrate: document.getElementById("autoCalibrate"),
  calibrateNow: document.getElementById("calibrateNow"),
  exportCsv: document.getElementById("exportCsv"),
  exportMeta: document.getElementById("exportMeta"),
  calibrationStatus: document.getElementById("calibrationStatus"),
  scaleStatus: document.getElementById("scaleStatus"),
  visibilityStatus: document.getElementById("visibilityStatus"),
  warningStatus: document.getElementById("warningStatus"),
  avgStepLength: document.getElementById("avgStepLength"),
  avgStepWidth: document.getElementById("avgStepWidth"),
  avgComSpeed: document.getElementById("avgComSpeed"),
  cadenceValue: document.getElementById("cadenceValue"),
};

const charts = {
  step: new LineChart(document.getElementById("stepChart"), { lineColor: "#2563eb" }),
  com: new LineChart(document.getElementById("comChart"), { lineColor: "#10b981" }),
};

let animationId = null;
let mediaStream = null;
let uploadUrl = null;

function setStatus(message) {
  dom.appStatus.textContent = message;
}

function formatErrorMessage(error) {
  if (!error) return "Unknown error";
  if (typeof error === "string") return error;
  if (error?.message) return error.message;
  return String(error);
}

function reportError(source, error) {
  const msg = formatErrorMessage(error);
  const fullMessage = source ? `${source}: ${msg}` : msg;
  console.error(fullMessage, error);
  setStatus(fullMessage);
  warnOnce(fullMessage);
  updateStatusUI();
}

function setupGlobalErrorHandlers() {
  window.addEventListener("unhandledrejection", (event) => {
    reportError("Unhandled promise rejection", event.reason);
  });
  window.addEventListener("error", (event) => {
    reportError("Unhandled error", event.error || event.message);
  });
}

function warnOnce(message) {
  if (!appState.warnings.includes(message)) appState.warnings.push(message);
}

function resetSession() {
  appState.data = [];
  appState.dts = [];
  appState.lastTimestamp = null;
  appState.lastResult = null;
  appState.warnings = [];
  resetCalibration();
  updateStatusUI();
  dom.visibilityStatus.textContent = "—";
  updateResults();
  updateCharts();
  updateExportButtons();
}

function resetCalibration() {
  appState.calibration = {
    samples: [],
    totalFrames: 0,
    targetFrames: 45,
    scale: null,
    status: "Collecting",
  };
}

function updateStatusUI() {
  dom.calibrationStatus.textContent = appState.calibration.status;
  dom.scaleStatus.textContent = appState.calibration.scale
    ? `${formatNumber(appState.calibration.scale, 5)} m/px`
    : "—";
  dom.warningStatus.textContent = appState.warnings.length ? appState.warnings.join(" | ") : "None";
}

function updateResults() {
  const steps = appState.data
    .map((frame) => frame.step_length_m2d_apparent)
    .filter((v) => Number.isFinite(v));
  const widths = appState.data.map((frame) => frame.step_width_m2d).filter((v) => Number.isFinite(v));
  const speeds = appState.data.map((frame) => frame.com_speed_mps_2d).filter((v) => Number.isFinite(v));

  const avg = (arr) => (arr.length ? arr.reduce((s, v) => s + v, 0) / arr.length : null);

  dom.avgStepLength.textContent = formatNumber(avg(steps));
  dom.avgStepWidth.textContent = formatNumber(avg(widths));
  dom.avgComSpeed.textContent = formatNumber(avg(speeds));

  const cadence = computeCadence();
  dom.cadenceValue.textContent = cadence ? cadence.toFixed(1) : "—";
}

function computeCadence() {
  const trajectory = appState.data.map((frame) => frame.left_ankle_x_m2d).filter((v) => Number.isFinite(v));
  if (trajectory.length < 20) return null;
  const smoothed = movingAverage(trajectory, Math.max(3, Number(dom.smoothingWindow.value)));
  const peaks = computePeaks(smoothed, 6);
  if (peaks.length < 4) return null;
  const duration = appState.data.at(-1)?.timestamp_s - appState.data[0]?.timestamp_s;
  if (!duration || duration <= 0) return null;
  const steps = peaks.length;
  return (steps / duration) * 60;
}

function updateCharts() {
  const stepData = appState.data
    .map((frame) => ({ x: frame.timestamp_s, y: frame.step_length_m2d_apparent }))
    .filter((p) => Number.isFinite(p.y));
  const comData = appState.data
    .map((frame) => ({ x: frame.timestamp_s, y: frame.com_speed_mps_2d }))
    .filter((p) => Number.isFinite(p.y));
  charts.step.draw(stepData);
  charts.com.draw(comData);
}

function updateExportButtons() {
  const enabled = appState.data.length > 0;
  dom.exportCsv.disabled = !enabled;
  dom.exportMeta.disabled = !enabled;
}

function setVisibilityStatus(isFullBody) {
  dom.visibilityStatus.textContent = isFullBody ? "Full body detected" : "Partial body";
  if (!isFullBody) warnOnce("Body not fully visible");
}

function isFullBodyVisible(landmarks) {
  const required = [
    LANDMARK_INDEX.nose,
    LANDMARK_INDEX.left_ankle,
    LANDMARK_INDEX.right_ankle,
    LANDMARK_INDEX.left_hip,
    LANDMARK_INDEX.right_hip,
  ];
  return required.every((idx) => landmarks?.[idx] && (landmarks[idx].visibility ?? 1) > 0.5);
}

function estimatePixelHeight(landmarks, videoHeight) {
  const topIndices = [LANDMARK_INDEX.nose, LANDMARK_INDEX.left_ear, LANDMARK_INDEX.right_ear];
  const bottomIndices = [LANDMARK_INDEX.left_ankle, LANDMARK_INDEX.right_ankle];

  const topYs = topIndices
    .map((idx) => landmarks?.[idx])
    .filter((lm) => lm && (lm.visibility ?? 1) > 0.5)
    .map((lm) => lm.y);

  const bottomYs = bottomIndices
    .map((idx) => landmarks?.[idx])
    .filter((lm) => lm && (lm.visibility ?? 1) > 0.5)
    .map((lm) => lm.y);

  if (!topYs.length || !bottomYs.length) return null;
  const top = Math.min(...topYs);
  const bottom = Math.max(...bottomYs);

  // landmarks are normalized y; convert to px height
  const pixelHeight = Math.abs(bottom - top) * videoHeight;
  return pixelHeight;
}

function updateCalibration(landmarks) {
  if (!dom.autoCalibrate.checked || appState.calibration.scale) return;
  if (!appState.videoSize.height) return;

  const pixelHeight = estimatePixelHeight(landmarks, appState.videoSize.height);
  appState.calibration.totalFrames += 1;
  if (pixelHeight) appState.calibration.samples.push(pixelHeight);

  if (appState.calibration.totalFrames >= appState.calibration.targetFrames) {
    const validCount = appState.calibration.samples.length;
    const required = Math.floor(appState.calibration.targetFrames / 2);
    const med = median(appState.calibration.samples);

    if (validCount >= required && med && med > 0.25 * appState.videoSize.height) {
      const heightMeters = clamp(Number(dom.heightInput.value), 1.0, 2.3);
      appState.calibration.scale = heightMeters / med;
      appState.calibration.status = "Calibrated";
    } else {
      appState.calibration.status = "Calibration failed";
      warnOnce("Calibration failed; 2D meters disabled");
    }
  }
}

function updateManualCalibration() {
  resetCalibration();
  appState.warnings = appState.warnings.filter((w) => !w.startsWith("Calibration failed"));
  updateStatusUI();
}

function drawPose(landmarks) {
  if (!dom.toggleOverlay.checked) return;
  const canvas = dom.overlay;
  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  if (!landmarks) return;

  ctx.strokeStyle = "rgba(37, 99, 235, 0.9)";
  ctx.lineWidth = 2;

  POSE_CONNECTIONS.forEach(([startName, endName]) => {
    const start = landmarks[LANDMARK_INDEX[startName]];
    const end = landmarks[LANDMARK_INDEX[endName]];
    if (!start || !end) return;

    ctx.beginPath();
    ctx.moveTo(start.x * canvas.width, start.y * canvas.height);
    ctx.lineTo(end.x * canvas.width, end.y * canvas.height);
    ctx.stroke();
  });

  ctx.fillStyle = "rgba(16, 185, 129, 0.9)";
  landmarks.forEach((lm) => {
    ctx.beginPath();
    ctx.arc(lm.x * canvas.width, lm.y * canvas.height, 3, 0, Math.PI * 2);
    ctx.fill();
  });
}

function processResult(result, timestampMs) {
  const landmarks = result?.landmarks?.[0];
  const world = result?.worldLandmarks?.[0];
  if (!landmarks) return;

  ensureCanvasSize(dom.overlay, appState.videoSize.width, appState.videoSize.height);
  drawPose(landmarks);

  const timestamp_s = timestampMs / 1000;
  let dt = null;
  if (appState.lastTimestamp !== null) {
    dt = (timestampMs - appState.lastTimestamp) / 1000;
    if (dt > 0) appState.dts.push(dt);
  }
  appState.lastTimestamp = timestampMs;

  updateCalibration(landmarks);
  setVisibilityStatus(isFullBodyVisible(landmarks));

  const scale = appState.calibration.scale;
  const com = computeCom(landmarks, scale);
  const comSpeed = computeComSpeed(com, dt);
  const metrics = computeStepMetrics(landmarks, scale);

  const frameEntry = {
    timestamp_s,
    dt_s: dt,
    landmarks,
    world,
    scale_m_per_px: scale,

    com_x_m2d: com?.x ?? NaN,
    com_y_m2d: com?.y ?? NaN,
    com_speed_mps_2d: comSpeed,

    step_width_m2d: metrics.stepWidth,
    step_length_m2d_apparent: metrics.stepLength,
    stride_length_m2d_proxy: metrics.strideLength,
    left_ankle_x_m2d: metrics.leftAnkleX,
    right_ankle_x_m2d: metrics.rightAnkleX,
  };

  appState.data.push(frameEntry);
  updateStatusUI();
  updateResults();
  updateCharts();
  updateExportButtons();
}

function computeCom(landmarks, scale) {
  if (!scale) return null;
  const left = landmarks?.[LANDMARK_INDEX.left_hip];
  const right = landmarks?.[LANDMARK_INDEX.right_hip];
  if (!left || !right) return null;

  const xPx = ((left.x + right.x) / 2) * appState.videoSize.width;
  const yPx = ((left.y + right.y) / 2) * appState.videoSize.height;
  return { x: xPx * scale, y: yPx * scale };
}

function computeComSpeed(currentCom, dt) {
  if (!currentCom || !dt || appState.data.length === 0) return NaN;

  const windowSize = Number(dom.smoothingWindow.value);
  const comXs = appState.data.map((frame) => frame.com_x_m2d).filter((v) => Number.isFinite(v));
  const comYs = appState.data.map((frame) => frame.com_y_m2d).filter((v) => Number.isFinite(v));

  const smoothX = movingAverage([...comXs, currentCom.x], windowSize);
  const smoothY = movingAverage([...comYs, currentCom.y], windowSize);
  if (smoothX.length < 2 || smoothY.length < 2) return NaN;

  const dx = smoothX[smoothX.length - 1] - smoothX[smoothX.length - 2];
  const dy = smoothY[smoothY.length - 1] - smoothY[smoothY.length - 2];
  return Math.hypot(dx, dy) / dt;
}

function computeStepMetrics(landmarks, scale) {
  const left = landmarks?.[LANDMARK_INDEX.left_ankle];
  const right = landmarks?.[LANDMARK_INDEX.right_ankle];

  if (!left || !right || !scale) {
    return {
      stepWidth: NaN,
      stepLength: NaN,
      strideLength: NaN,
      leftAnkleX: NaN,
      rightAnkleX: NaN,
    };
  }

  const leftX = left.x * appState.videoSize.width * scale;
  const rightX = right.x * appState.videoSize.width * scale;
  const leftY = left.y * appState.videoSize.height * scale;
  const rightY = right.y * appState.videoSize.height * scale;

  const stepWidth = Math.abs(leftY - rightY);
  const stepLength = Math.abs(leftX - rightX);

  return {
    stepWidth,
    stepLength,
    strideLength: stepLength * 2,
    leftAnkleX: leftX,
    rightAnkleX: rightX,
  };
}

function getPixelLandmarks(landmarks) {
  return landmarks.map((lm) => ({
    x: lm.x * appState.videoSize.width,
    y: lm.y * appState.videoSize.height,
    z: lm.z,
  }));
}

function getMetricLandmarks(pixelLandmarks, scale) {
  if (!scale) return pixelLandmarks.map(() => ({ x: NaN, y: NaN }));
  return pixelLandmarks.map((lm) => ({ x: lm.x * scale, y: lm.y * scale }));
}

function exportCsv() {
  const header = ["timestamp_s"];

  POSE_LANDMARK_NAMES.forEach((name) => header.push(`${name}_x`, `${name}_y`, `${name}_z`));
  POSE_LANDMARK_NAMES.forEach((name) => header.push(`${name}_x_px`, `${name}_y_px`));
  POSE_LANDMARK_NAMES.forEach((name) => header.push(`${name}_x_m2d`, `${name}_y_m2d`));
  POSE_LANDMARK_NAMES.forEach((name) => header.push(`${name}_wx`, `${name}_wy`, `${name}_wz`));

  header.push(
    "step_width_m2d",
    "step_length_m2d_apparent",
    "stride_length_m2d_proxy",
    "com_x_m2d",
    "com_y_m2d",
    "com_speed_mps_2d"
  );

  const rows = appState.data.map((frame) => {
    const row = [Number.isFinite(frame.timestamp_s) ? frame.timestamp_s.toFixed(4) : "NaN"];

    const landmarks = frame.landmarks || [];
    const world = frame.world || [];
    const pixel = getPixelLandmarks(landmarks);
    const metric = getMetricLandmarks(pixel, frame.scale_m_per_px);

    POSE_LANDMARK_NAMES.forEach((_, idx) => {
      const lm = landmarks[idx] || {};
      row.push(lm.x ?? NaN, lm.y ?? NaN, lm.z ?? NaN);
    });

    pixel.forEach((lm) => row.push(lm.x ?? NaN, lm.y ?? NaN));
    metric.forEach((lm) => row.push(lm.x ?? NaN, lm.y ?? NaN));

    POSE_LANDMARK_NAMES.forEach((_, idx) => {
      const lm = world[idx] || {};
      row.push(lm.x ?? NaN, lm.y ?? NaN, lm.z ?? NaN);
    });

    row.push(
      frame.step_width_m2d ?? NaN,
      frame.step_length_m2d_apparent ?? NaN,
      frame.stride_length_m2d_proxy ?? NaN,
      frame.com_x_m2d ?? NaN,
      frame.com_y_m2d ?? NaN,
      frame.com_speed_mps_2d ?? NaN
    );

    return row.join(",");
  });

  const content = [header.join(","), ...rows].join("\n");
  downloadFile("gait_landmarks.csv", content, "text/csv");
}

function exportMeta() {
  const heightMeters = clamp(Number(dom.heightInput.value), 1.0, 2.3);
  const dtStats = computeDtStats(appState.dts);
  const payload = {
    height_m: heightMeters,
    scale_m_per_px: appState.calibration.scale,
    calibration_frames_count: appState.calibration.samples.length,
    video_width: appState.videoSize.width,
    video_height: appState.videoSize.height,
    mode: appState.mode,
    dt_stats: dtStats,
    smoothing_window: Number(dom.smoothingWindow.value),
    inference_throttle: Number(dom.inferenceThrottle.value),
    warnings: appState.warnings,
    app_version: "1.0.0",
  };
  downloadFile("gait_meta.json", JSON.stringify(payload, null, 2), "application/json");
}

async function initPose() {
  if (appState.modelReady) return true;
  if (appState.modelLoading) return false;

  appState.modelLoading = true;

  setStatus("Loading model…");

  try {
    appState.poseLandmarker = await createPoseLandmarker();
    appState.modelReady = true;
    setStatus("Ready");
    return true;
  } catch (err) {
    reportError("Model load failed", err);

    appState.poseLandmarker = null;
    appState.modelReady = false;
    return false;
  } finally {
    appState.modelLoading = false;
    updateControls(appState.running);
  }
}

function setVideoSize() {
  const width = dom.video.videoWidth;
  const height = dom.video.videoHeight;
  if (width && height && (width !== appState.videoSize.width || height !== appState.videoSize.height)) {
    appState.videoSize = { width, height };
    ensureCanvasSize(dom.overlay, width, height);
  }
}

function updateControls(running) {
  dom.startLive.disabled = running;
  dom.stopLive.disabled = !running;
  dom.startUpload.disabled = running || !dom.videoFile.files.length;
  dom.stopUpload.disabled = !running;
}

function setMode(mode) {
  appState.mode = mode;
  dom.tabs.forEach((tab) => tab.classList.toggle("active", tab.dataset.tab === mode));
  dom.tabContents.live.classList.toggle("active", mode === "live");
  dom.tabContents.upload.classList.toggle("active", mode === "upload");
}

function stopProcessing() {
  appState.running = false;
  if (animationId) cancelAnimationFrame(animationId);
  animationId = null;

  try {
    dom.video.pause();
  } catch (_) {}

  if (mediaStream) {
    mediaStream.getTracks().forEach((track) => track.stop());
    mediaStream = null;
  }
  if (uploadUrl) {
    URL.revokeObjectURL(uploadUrl);
    uploadUrl = null;
  }

  updateControls(false);
  setStatus("Stopped");
}

async function startLive() {
  if (!(await initPose())) return;

  appState.mode = "live";
  resetSession();
  setStatus("Starting camera…");

  try {
    mediaStream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
  } catch (err) {
    console.error(err);
    setStatus("Camera permission denied");
    warnOnce("Camera permission denied");
    updateStatusUI();
    return;
  }

  dom.video.srcObject = mediaStream;

  try {
    await dom.video.play();
  } catch (err) {
    console.error(err);
    setStatus("Unable to start video playback");
    warnOnce("Unable to start video playback");
    updateStatusUI();
    return;
  }

  // wait for dimensions
  await waitForVideoMetadata(dom.video);
  setVideoSize();

  appState.running = true;
  setStatus("Live capture running");
  updateControls(true);
  runLoop();
}

async function startUpload() {
  if (!(await initPose())) return;

  appState.mode = "upload";
  if (!dom.videoFile.files.length) return;

  resetSession();
  const file = dom.videoFile.files[0];

  uploadUrl = URL.createObjectURL(file);
  dom.video.srcObject = null;
  dom.video.src = uploadUrl;

  try {
    await dom.video.play();
  } catch (err) {
    console.error(err);
    setStatus("Unable to play uploaded video");
    warnOnce("Unable to play uploaded video");
    updateStatusUI();
    return;
  }

  await waitForVideoMetadata(dom.video);
  setVideoSize();

  appState.running = true;
  setStatus("Upload running");
  updateControls(true);
  runLoop();
}

function runLoop() {
  if (!appState.running) return;

  // Safety: never call detect when model isn't ready
  if (!appState.poseLandmarker) {
    animationId = requestAnimationFrame(runLoop);
    return;
  }

  setVideoSize();

  // If video has no dimensions yet, wait.
  if (!appState.videoSize.width || !appState.videoSize.height) {
    animationId = requestAnimationFrame(runLoop);
    return;
  }

  const now = appState.mode === "upload" ? dom.video.currentTime * 1000 : performance.now();
  const throttle = Math.max(1, Number(dom.inferenceThrottle.value));

  // Use frame index for throttle (NOT data length because data only grows on detect frames)
  const frameIndex = appState.calibration.totalFrames + appState.data.length;

  if (frameIndex % throttle === 0) {
    let result = null;
    try {
      result = appState.poseLandmarker.detectForVideo(dom.video, now);
    } catch (err) {
      reportError("Pose detection failed", err);
      animationId = requestAnimationFrame(runLoop);
      return;
    }
    appState.lastResult = result;
    processResult(result, now);
  } else if (appState.lastResult) {
    drawPose(appState.lastResult.landmarks?.[0]);
  }

  if (appState.mode === "upload" && dom.video.ended) {
    stopProcessing();
    setStatus("Upload finished");
    return;
  }

  animationId = requestAnimationFrame(runLoop);
}

function validateHeight() {
  const heightMeters = clamp(Number(dom.heightInput.value), 1.0, 2.3);
  dom.heightInput.value = heightMeters.toFixed(2);
  if (appState.calibration.scale) {
    const latestMedian = median(appState.calibration.samples);
    if (latestMedian) appState.calibration.scale = heightMeters / latestMedian;
  }
  updateStatusUI();
}

function updateInferenceLabel() {
  dom.inferenceValue.textContent = dom.inferenceThrottle.value;
}

function updateSmoothingLabel() {
  dom.smoothingValue.textContent = dom.smoothingWindow.value;
}

function setupEventListeners() {
  dom.tabs.forEach((tab) => {
    tab.addEventListener("click", () => setMode(tab.dataset.tab));
  });

  dom.startLive.addEventListener("click", startLive);
  dom.stopLive.addEventListener("click", stopProcessing);

  dom.videoFile.addEventListener("change", () => {
    dom.startUpload.disabled = !dom.videoFile.files.length || appState.running;
  });
  dom.startUpload.addEventListener("click", startUpload);
  dom.stopUpload.addEventListener("click", stopProcessing);

  dom.exportCsv.addEventListener("click", exportCsv);
  dom.exportMeta.addEventListener("click", exportMeta);

  dom.heightInput.addEventListener("change", validateHeight);

  dom.smoothingWindow.addEventListener("input", () => {
    updateSmoothingLabel();
    // Note: smoothing affects only future computed speeds/cadence, not retroactive
  });

  dom.inferenceThrottle.addEventListener("input", updateInferenceLabel);

  dom.calibrateNow.addEventListener("click", updateManualCalibration);
}

async function waitForVideoMetadata(videoEl) {
  if (videoEl.videoWidth && videoEl.videoHeight) return;
  await new Promise((resolve) => {
    const onLoaded = () => {
      videoEl.removeEventListener("loadedmetadata", onLoaded);
      resolve();
    };
    videoEl.addEventListener("loadedmetadata", onLoaded, { once: true });
  });
}

async function init() {
  setMode("live");
  updateSmoothingLabel();
  updateInferenceLabel();
  updateControls(false);
  updateStatusUI();
  setupEventListeners();
  setupGlobalErrorHandlers();

  // Do not preload the model; allow users to click Start immediately.
  setStatus("Ready (click Start)");
  updateControls(false);
}

init();
