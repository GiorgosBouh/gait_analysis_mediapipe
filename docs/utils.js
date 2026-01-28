export function clamp(value, min, max) {
  return Math.min(Math.max(value, min), max);
}

export function median(values) {
  if (!values.length) return null;
  const sorted = [...values].sort((a, b) => a - b);
  const mid = Math.floor(sorted.length / 2);
  return sorted.length % 2 === 0
    ? (sorted[mid - 1] + sorted[mid]) / 2
    : sorted[mid];
}

export function movingAverage(values, windowSize) {
  if (windowSize <= 1) return values.slice();
  const result = [];
  for (let i = 0; i < values.length; i += 1) {
    const start = Math.max(0, i - windowSize + 1);
    const slice = values.slice(start, i + 1);
    const avg = slice.reduce((sum, v) => sum + v, 0) / slice.length;
    result.push(avg);
  }
  return result;
}

export function downloadFile(filename, content, type = "text/plain") {
  const blob = new Blob([content], { type });
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = filename;
  anchor.click();
  URL.revokeObjectURL(url);
}

export function formatNumber(value, digits = 3) {
  if (value === null || Number.isNaN(value) || value === undefined) return "—";
  return Number(value).toFixed(digits);
}

export function computeDtStats(dts) {
  if (!dts.length) {
    return { min: null, max: null, mean: null };
  }
  const sum = dts.reduce((acc, v) => acc + v, 0);
  return {
    min: Math.min(...dts),
    max: Math.max(...dts),
    mean: sum / dts.length,
  };
}

export function ensureCanvasSize(canvas, width, height) {
  if (canvas.width !== width || canvas.height !== height) {
    canvas.width = width;
    canvas.height = height;
  }
}

export function computePeaks(signal, minDistance = 6) {
  const peaks = [];
  for (let i = 1; i < signal.length - 1; i += 1) {
    if (signal[i] > signal[i - 1] && signal[i] > signal[i + 1]) {
      if (!peaks.length || i - peaks[peaks.length - 1] >= minDistance) {
        peaks.push(i);
      }
    }
  }
  return peaks;
}
