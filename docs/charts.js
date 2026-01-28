export class LineChart {
  constructor(canvas, options = {}) {
    this.canvas = canvas;
    this.ctx = canvas.getContext("2d");
    this.options = {
      lineColor: options.lineColor || "#2563eb",
      axisColor: options.axisColor || "#9ca3af",
      gridColor: options.gridColor || "#e5e7eb",
      background: options.background || "#ffffff",
      padding: 32,
    };
  }

  draw(data) {
    const ctx = this.ctx;
    const { width, height } = this.canvas;
    const { padding } = this.options;

    ctx.clearRect(0, 0, width, height);
    ctx.fillStyle = this.options.background;
    ctx.fillRect(0, 0, width, height);

    if (!data.length) {
      ctx.fillStyle = "#9ca3af";
      ctx.fillText("No data yet", padding, height / 2);
      return;
    }

    const xs = data.map((p) => p.x);
    const ys = data.map((p) => p.y);
    const minX = Math.min(...xs);
    const maxX = Math.max(...xs);
    const minY = Math.min(...ys);
    const maxY = Math.max(...ys);

    const xRange = maxX - minX || 1;
    const yRange = maxY - minY || 1;

    const chartWidth = width - padding * 2;
    const chartHeight = height - padding * 2;

    ctx.strokeStyle = this.options.gridColor;
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(padding, padding);
    ctx.lineTo(padding, height - padding);
    ctx.lineTo(width - padding, height - padding);
    ctx.stroke();

    ctx.strokeStyle = this.options.axisColor;
    ctx.beginPath();
    ctx.moveTo(padding, padding);
    ctx.lineTo(padding, height - padding);
    ctx.lineTo(width - padding, height - padding);
    ctx.stroke();

    ctx.strokeStyle = this.options.lineColor;
    ctx.lineWidth = 2;
    ctx.beginPath();
    data.forEach((point, index) => {
      const x = padding + ((point.x - minX) / xRange) * chartWidth;
      const y = height - padding - ((point.y - minY) / yRange) * chartHeight;
      if (index === 0) {
        ctx.moveTo(x, y);
      } else {
        ctx.lineTo(x, y);
      }
    });
    ctx.stroke();

    ctx.fillStyle = "#6b7280";
    ctx.font = "12px sans-serif";
    ctx.fillText(minY.toFixed(2), 6, height - padding + 4);
    ctx.fillText(maxY.toFixed(2), 6, padding + 4);
  }
}
