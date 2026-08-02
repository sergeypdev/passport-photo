import { mmToPx, TOP_MARGIN_PX, PHOTO_W_PX } from './constants.js';

/**
 * Draw spec-guide overlay onto an already-sized 413×531 passport canvas context.
 * Call after drawing the composed photo — guides are not saved to the downloaded PNG.
 *
 * @param {CanvasRenderingContext2D} ctx
 */
export function drawGuides(ctx) {
  ctx.save();

  const W = PHOTO_W_PX; // 413
  const crownY = TOP_MARGIN_PX; // ~47 px  (4 mm from top)
  const chinMinY = TOP_MARGIN_PX + mmToPx(33); // 33 mm chin-to-crown lower bound
  const chinMaxY = TOP_MARGIN_PX + mmToPx(35); // 35 mm chin-to-crown upper bound
  const centerX = W / 2;

  const TEAL = 'rgba(0, 180, 180, 0.75)';
  const TEAL_FILL = 'rgba(0, 200, 200, 0.13)';
  const LABEL_BG = 'rgba(0, 0, 0, 0.45)';
  const LABEL_FG = '#ffffff';
  const FONT = '11px system-ui, -apple-system, sans-serif';

  // ── Helper: draw a small label with dark pill background ──────────────────
  function label(text, x, y, align = 'left') {
    ctx.save();
    ctx.font = FONT;
    ctx.textBaseline = 'middle';
    ctx.textAlign = align;
    const pad = 4;
    const metrics = ctx.measureText(text);
    const tw = metrics.width;
    const th = 13;
    let rx = x;
    if (align === 'center') rx = x - tw / 2;
    if (align === 'right') rx = x - tw;
    ctx.fillStyle = LABEL_BG;
    ctx.beginPath();
    ctx.roundRect(rx - pad, y - th / 2 - 1, tw + pad * 2, th + 2, 3);
    ctx.fill();
    ctx.fillStyle = LABEL_FG;
    ctx.fillText(text, x, y);
    ctx.restore();
  }

  // ── 1. Inner border frame (1 px, full photo boundary) ────────────────────
  ctx.strokeStyle = TEAL;
  ctx.lineWidth = 1;
  ctx.setLineDash([]);
  ctx.strokeRect(0.5, 0.5, W - 1, ctx.canvas.height - 1);

  // ── 2. Crown line ─────────────────────────────────────────────────────────
  ctx.strokeStyle = TEAL;
  ctx.lineWidth = 1;
  ctx.setLineDash([4, 3]);
  ctx.beginPath();
  ctx.moveTo(0, crownY);
  ctx.lineTo(W, crownY);
  ctx.stroke();
  label('макушка (4 мм)', 6, crownY - 8, 'left');

  // ── 3. Chin acceptance band (33–35 mm below crown) ───────────────────────
  ctx.setLineDash([]);
  ctx.fillStyle = TEAL_FILL;
  ctx.fillRect(0, chinMinY, W, chinMaxY - chinMinY);

  ctx.strokeStyle = TEAL;
  ctx.lineWidth = 1;

  // Top edge of band (33 mm)
  ctx.setLineDash([4, 3]);
  ctx.beginPath();
  ctx.moveTo(0, chinMinY);
  ctx.lineTo(W, chinMinY);
  ctx.stroke();

  // Bottom edge of band (35 mm)
  ctx.beginPath();
  ctx.moveTo(0, chinMaxY);
  ctx.lineTo(W, chinMaxY);
  ctx.stroke();

  label('линия подбородка (33–35 мм)', 6, chinMaxY + 8, 'left');

  // ── 4. Vertical center line (dashed) ─────────────────────────────────────
  ctx.strokeStyle = TEAL;
  ctx.lineWidth = 1;
  ctx.setLineDash([5, 4]);
  ctx.beginPath();
  ctx.moveTo(centerX, 0);
  ctx.lineTo(centerX, ctx.canvas.height);
  ctx.stroke();

  // Reset line dash to solid before restore
  ctx.setLineDash([]);

  ctx.restore();
}
