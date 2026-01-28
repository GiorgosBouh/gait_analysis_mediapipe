// app.js
import {
  createPoseLandmarker,
  POSE_LANDMARK_NAMES,
  POSE_CONNECTIONS,
  LANDMARK_INDEX,
} from "./pose.js?v=20260201_1205";
import {
  clamp,
  median,
  movingAverage,
  downloadFile,
  formatNumber,
  computeDtStats,
  ensureCanvasSize,
  computePeaks,
} from "./utils.js?v=20260201_1205";
import { LineChart } from "./charts.js?v=20260201_1205";

const BUILD_STAMP = "20260201_1205";
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

function getEl(id) {
  const el = document.getElementById(id);
  if (!el) {
    console.warn(`[App] Missing element #${id}`);
  }
  return el;
}

const dom = {
  appStatus: getEl("appStatus"),
  tabs: document.querySelectorAll(".tab"),
  tabContents: {
    live: getEl("tab-live"),
    upload: getEl("tab-upload"),
  },
  video: getEl("video"),
  overlay: getEl("overlay"),
  startLive: getEl("startLive"),
  stopLive: getEl("stopLive"),
  videoFile: getEl("videoFile"),
  startUpload: getEl("startUpload"),
  stopUpload: getEl("stopUpload"),
  heightInput: getEl("heightInput"),
  smoothingWindow: getEl("smoothingWindow"),
  smoothingValue: getEl("smoothingValue"),
  inferenceThrottle: getEl("inferenceThrottle"),
  inferenceValue: getEl("inferenceValue"),
  toggleOverlay: getEl("toggleOverlay"),
  autoCalibrate: getEl("autoCalibrate"),
  calibrateNow: getEl("calibrateNow"),
  exportCsv: getEl("exportCsv"),
  exportMeta: getEl("exportMeta"),
  calibrationStatus: getEl("calibrationStatus"),
  scaleStatus: getEl("scaleStatus"),
  visibilityStatus: getEl("visibilityStatus"),
  warningStatus: getEl("warningStatus"),
  avgStepLength: getEl("avgStepLength"),
  avgStepWidth: getEl("avgStepWidth"),
  avgComSpeed: getEl("avgComSpeed"),
  cadenceValue: getEl("cadenceValue"),
  stepChart: getEl("stepChart"),
  comChart: getEl("comChart"),
};

const charts = {
  step: dom.stepChart ? new LineChart(dom.stepChart, { lineColor: "#2563eb" }) : null,
  com: dom.comChart ? new LineChart(dom.comChart, { lineColor: "#10b981" }) : null,
};

let animationId = null;
let mediaStream = null;
let uploadUrl = null;

function setStatus(message) {
  if (dom.appStatus) {
    dom.appStatus.textContent = message;
  }
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
  if (dom.visibilityStatus) {
    dom.visibilityStatus.textContent = "—";
  }
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
  if (dom.calibrationStatus) {
    dom.calibrationStatus.textContent = appState.calibration.status;
  }
  if (dom.scaleStatus) {
    dom.scaleStatus.textContent = appState.calibration.scale
      ? `${formatNumber(appState.calibration.scale, 5)} m/px`
      : "—";
  }
  if (dom.warningStatus) {
    dom.warningStatus.textContent = appState.warnings.length ? appState.warnings.join(" | ") : "None";
  }
}

function updateResults() {
  const steps = appState.data
    .map((frame) => frame.step_length_m2d_apparent)
    .filter((v) => Number.isFinite(v));
  const widths = appState.data.map((frame) => frame.step_width_m2d).filter((v) => Number.isFinite(v));
  const speeds = appState.data.map((frame) => frame.com_speed_mps_2d).filter((v) => Number.isFinite(v));

  const avg = (arr) => (arr.length ? arr.reduce((s, v) => s + v, 0) / arr.length : null);

  if (dom.avgStepLength) dom.avgStepLength.textContent = formatNumber(avg(steps));
  if (dom.avgStepWidth) dom.avgStepWidth.textContent = formatNumber(avg(widths));
  if (dom.avgComSpeed) dom.avgComSpeed.textContent = formatNumber(avg(speeds));

  const cadence = computeCadence();
  if (dom.cadenceValue) dom.cadenceValue.textContent = cadence ? cadence.toFixed(1) : "—";
}

function computeCadence() {
  const trajectory = appState.data.map((frame) => frame.left_ankle_x_m2d).filter((v) => Number.isFinite(v));
  if (trajectory.length < 20) return null;
  const smoothingWindow = dom.smoothingWindow ? Number(dom.smoothingWindow.value) : 5;
  const smoothed = movingAverage(trajectory, Math.max(3, smoothingWindow));
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
  if (charts.step) charts.step.draw(stepData);
  if (charts.com) charts.com.draw(comData);
}

function updateExportButtons() {
  const enabled = appState.data.length > 0;
  if (dom.exportCsv) dom.exportCsv.disabled = !enabled;
  if (dom.exportMeta) dom.exportMeta.disabled = !enabled;
}

function setVisibilityStatus(isFullBody) {
  if (dom.visibilityStatus) {
    dom.visibilityStatus.textContent = isFullBody ? "Full body detected" : "Partial body";
  }
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
  if (!dom.autoCalibrate || !dom.autoCalibrate.checked || appState.calibration.scale) return;
  if (!appState.videoSize.height) return;

  const pixelHeight = estimatePixelHeight(landmarks, appState.videoSize.height);
  appState.calibration.totalFrames += 1;
  if (pixelHeight) appState.calibration.samples.push(pixelHeight);

  if (appState.calibration.totalFrames >= appState.calibration.targetFrames) {
    const validCount = appState.calibration.samples.length;
    const required = Math.floor(appState.calibration.targetFrames / 2);
    const med = median(appState.calibration.samples);

    if (validCount >= required && med && med > 0.25 * appState.videoSize.height) {
      const heightMeters = clamp(Number(dom.heightInput?.value ?? 1.75), 1.0, 2.3);
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
  if (!dom.toggleOverlay || !dom.toggleOverlay.checked) return;
  const canvas = dom.overlay;
  if (!canvas) return;
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

  if (dom.overlay) {
    ensureCanvasSize(dom.overlay, appState.videoSize.width, appState.videoSize.height);
  }
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

  const windowSize = dom.smoothingWindow ? Number(dom.smoothingWindow.value) : 5;
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
  const heightMeters = clamp(Number(dom.heightInput?.value ?? 1.75), 1.0, 2.3);
  const dtStats = computeDtStats(appState.dts);
  const payload = {
    height_m: heightMeters,
    scale_m_per_px: appState.calibration.scale,
    calibration_frames_count: appState.calibration.samples.length,
    video_width: appState.videoSize.width,
    video_height: appState.videoSize.height,
    mode: appState.mode,
    dt_stats: dtStats,
    smoothing_window: dom.smoothingWindow ? Number(dom.smoothingWindow.value) : null,
    inference_throttle: dom.inferenceThrottle ? Number(dom.inferenceThrottle.value) : null,
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
  if (!dom.video || !dom.overlay) return;
  const width = dom.video.videoWidth;
  const height = dom.video.videoHeight;
  if (width && height && (width !== appState.videoSize.width || height !== appState.videoSize.height)) {
    appState.videoSize = { width, height };
    ensureCanvasSize(dom.overlay, width, height);
  }
}

function updateControls(running) {
  if (dom.startLive) dom.startLive.disabled = running;
  if (dom.stopLive) dom.stopLive.disabled = !running;
  if (dom.startUpload) {
    dom.startUpload.disabled = running || !dom.videoFile || !dom.videoFile.files.length;
  }
  if (dom.stopUpload) dom.stopUpload.disabled = !running;
}

function setMode(mode) {
  appState.mode = mode;
  dom.tabs.forEach((tab) => tab.classList.toggle("active", tab.dataset.tab === mode));
  if (dom.tabContents.live) {
    dom.tabContents.live.classList.toggle("active", mode === "live");
  }
  if (dom.tabContents.upload) {
    dom.tabContents.upload.classList.toggle("active", mode === "upload");
  }
}

function stopProcessing() {
  appState.running = false;
  if (animationId) cancelAnimationFrame(animationId);
  animationId = null;

  try {
    if (dom.video) dom.video.pause();
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
  setStatus("Loading model…");
  console.log("[App] Start Live clicked");
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

  if (dom.video) dom.video.srcObject = mediaStream;

  try {
    if (dom.video) await dom.video.play();
  } catch (err) {
    console.error(err);
    setStatus("Unable to start video playback");
    warnOnce("Unable to start video playback");
    updateStatusUI();
    return;
  }

  // wait for dimensions
  if (dom.video) {
    await waitForVideoMetadata(dom.video);
  }
  setVideoSize();

  appState.running = true;
  setStatus("Live capture running");
  updateControls(true);
  runLoop();
}

async function startUpload() {
  setStatus("Loading model…");
  console.log("[App] Start Upload clicked");
  if (!(await initPose())) return;

  appState.mode = "upload";
  if (!dom.videoFile || !dom.videoFile.files.length) return;

  resetSession();
  const file = dom.videoFile.files[0];

  uploadUrl = URL.createObjectURL(file);
  if (dom.video) {
    dom.video.srcObject = null;
    dom.video.src = uploadUrl;
  }

  try {
    if (dom.video) await dom.video.play();
  } catch (err) {
    console.error(err);
    setStatus("Unable to play uploaded video");
    warnOnce("Unable to play uploaded video");
    updateStatusUI();
    return;
  }

  if (dom.video) {
    await waitForVideoMetadata(dom.video);
  }
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

  const now = appState.mode === "upload" && dom.video
    ? dom.video.currentTime * 1000
    : performance.now();
  const throttle = dom.inferenceThrottle ? Math.max(1, Number(dom.inferenceThrottle.value)) : 1;

  // Use frame index for throttle (NOT data length because data only grows on detect frames)
  const frameIndex = appState.calibration.totalFrames + appState.data.length;

  if (frameIndex % throttle === 0) {
    let result = null;
    try {
      if (!dom.video) return;
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

  if (appState.mode === "upload" && dom.video && dom.video.ended) {
    stopProcessing();
    setStatus("Upload finished");
    return;
  }

  animationId = requestAnimationFrame(runLoop);
}

function validateHeight() {
  const heightMeters = clamp(Number(dom.heightInput?.value ?? 1.75), 1.0, 2.3);
  if (dom.heightInput) dom.heightInput.value = heightMeters.toFixed(2);
  if (appState.calibration.scale) {
    const latestMedian = median(appState.calibration.samples);
    if (latestMedian) appState.calibration.scale = heightMeters / latestMedian;
  }
  updateStatusUI();
}

function updateInferenceLabel() {
  if (dom.inferenceValue && dom.inferenceThrottle) {
    dom.inferenceValue.textContent = dom.inferenceThrottle.value;
  }
}

function updateSmoothingLabel() {
  if (dom.smoothingValue && dom.smoothingWindow) {
    dom.smoothingValue.textContent = dom.smoothingWindow.value;
  }
}

function setupEventListeners() {
  dom.tabs.forEach((tab) => {
    tab.addEventListener("click", () => setMode(tab.dataset.tab));
  });

  if (dom.startLive) dom.startLive.addEventListener("click", startLive);
  if (dom.stopLive) dom.stopLive.addEventListener("click", stopProcessing);

  if (dom.videoFile) {
    dom.videoFile.addEventListener("change", () => {
      if (dom.startUpload) {
        dom.startUpload.disabled = !dom.videoFile.files.length || appState.running;
      }
    });
  }
  if (dom.startUpload) dom.startUpload.addEventListener("click", startUpload);
  if (dom.stopUpload) dom.stopUpload.addEventListener("click", stopProcessing);

  if (dom.exportCsv) dom.exportCsv.addEventListener("click", exportCsv);
  if (dom.exportMeta) dom.exportMeta.addEventListener("click", exportMeta);

  if (dom.heightInput) dom.heightInput.addEventListener("change", validateHeight);

  if (dom.smoothingWindow) {
    dom.smoothingWindow.addEventListener("input", () => {
      updateSmoothingLabel();
      // Note: smoothing affects only future computed speeds/cadence, not retroactive
    });
  }

  if (dom.inferenceThrottle) {
    dom.inferenceThrottle.addEventListener("input", updateInferenceLabel);
  }

  if (dom.calibrateNow) dom.calibrateNow.addEventListener("click", updateManualCalibration);
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
