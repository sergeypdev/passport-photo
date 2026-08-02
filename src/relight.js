/**
 * Cheap portrait shadow-lift: evens out one-sided lighting by lifting shadowed
 * skin toward the lit-side tone. Three choices keep it looking natural:
 *
 *  1. Lift only — gain is clamped to >= 1, so we never darken. (Darkening the
 *     bright side is what previously turned hair near the face muddy.)
 *  2. Multiplicative — newC = C * gain scales all channels together, preserving
 *     hue and saturation, so a shadowed cheek keeps its skin color instead of
 *     going flat gray (which is what equal additive lifting did).
 *  3. Skin-gated — the correction is weighted by how close each pixel is to the
 *     sampled skin tone, so hair, brows, lips and eyes are left alone.
 *
 * The shading is estimated from a large blur of luminance, so fine detail is kept.
 * A learned relight model (e.g. DPR with a flat spherical-harmonic target, or
 * IC-Light) would do better still, at a much larger download.
 */

// How far to lift shadows toward the lit tone. 1 = fully match; lower keeps some
// natural modeling so the face doesn't look flat/pasted.
const STRENGTH = 0.7;

// Never brighten a shadow by more than this — caps noise blow-up in deep shadow.
const MAX_GAIN = 1.6;

// Chroma distance at which skin weight falls to ~0.6 (larger = looser skin gate).
const SKIN_SIGMA = 0.06;

/**
 * Even out lighting on the face. Modifies the canvas in place.
 *
 * @param {HTMLCanvasElement} canvas  foreground cutout (source-sized, RGBA)
 * @param {object} geometry  from analyzeFace: crown, chin, leftEye, rightEye
 * @returns {HTMLCanvasElement} same canvas
 */
export function flattenLighting(canvas, geometry) {
  const ctx = canvas.getContext('2d');
  const { width: w, height: h } = canvas;
  const imageData = ctx.getImageData(0, 0, w, h);
  const data = imageData.data;

  if (!geometry) return canvas; // no face → nothing safe to relight

  const ellipse = faceEllipse(geometry);
  const skin = sampleSkin(data, w, h, ellipse);
  if (!skin) return canvas;

  // Per-pixel weight = inside-face ellipse × skin-tone similarity. Only opaque
  // skin pixels get corrected; hair/brows/lips/background stay at weight ~0.
  const weight = new Float32Array(w * h);
  const lum = new Float32Array(w * h);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const i = y * w + x;
      const r = data[i * 4];
      const g = data[i * 4 + 1];
      const b = data[i * 4 + 2];
      lum[i] = 0.299 * r + 0.587 * g + 0.114 * b;
      if (data[i * 4 + 3] === 0) continue;
      const e = ellipseWeight(x, y, ellipse);
      if (e <= 0) continue;
      weight[i] = e * skinWeight(r, g, b, skin);
    }
  }

  // Low-frequency shading of the skin (blurred through the skin weight so dark
  // hair at the hairline doesn't bleed into the estimate).
  const radius = Math.max(1, Math.floor(Math.min(w, h) / 6));
  const shading = boxBlurLuminance(lum, weight, w, h, radius);

  // Lift each skin pixel's low-frequency level partway toward the lit-side tone.
  for (let i = 0; i < w * h; i++) {
    const wgt = weight[i];
    if (wgt <= 0) continue;
    const s = shading[i];
    if (s <= 1 || s >= skin.targetL) continue; // already at/above lit tone
    const lifted = s + (skin.targetL - s) * STRENGTH;
    let gain = lifted / s;
    if (gain > MAX_GAIN) gain = MAX_GAIN;
    const eff = 1 + (gain - 1) * wgt; // fade the gain out with the weight
    data[i * 4] = clamp8(data[i * 4] * eff);
    data[i * 4 + 1] = clamp8(data[i * 4 + 1] * eff);
    data[i * 4 + 2] = clamp8(data[i * 4 + 2] * eff);
    // alpha untouched
  }

  ctx.putImageData(imageData, 0, 0);
  return canvas;
}

/** Face ellipse (center + radii) in source pixel coords, from landmarks. */
function faceEllipse(geometry) {
  const { crown, chin, leftEye, rightEye } = geometry;
  const cx = (leftEye.x + rightEye.x) / 2;
  const cy = (crown.y + chin.y) / 2;
  const eyeDist = Math.hypot(rightEye.x - leftEye.x, rightEye.y - leftEye.y);
  const rx = Math.max(1, eyeDist * 1.15);
  const ry = Math.max(1, ((chin.y - crown.y) / 2) * 1.05);
  return { cx, cy, rx, ry };
}

/** Feathered membership (0..1) of (x,y) in the face ellipse. */
function ellipseWeight(x, y, { cx, cy, rx, ry }) {
  const dx = (x - cx) / rx;
  const dy = (y - cy) / ry;
  const d = Math.sqrt(dx * dx + dy * dy);
  const INNER = 0.55;
  if (d <= INNER) return 1;
  if (d >= 1) return 0;
  const t = (1 - d) / (1 - INNER);
  return t * t * (3 - 2 * t); // smoothstep
}

/**
 * Sample skin from forehead / cheeks / nose patches. Returns the reference
 * chromaticity plus targetL = the brightest patch (the lit side to lift toward).
 */
function sampleSkin(data, w, h, { cx, cy, rx, ry }) {
  const points = [
    [cx, cy - 0.55 * ry], // forehead
    [cx - 0.55 * rx, cy + 0.15 * ry], // left cheek
    [cx + 0.55 * rx, cy + 0.15 * ry], // right cheek
    [cx, cy], // nose bridge
  ];

  let sumR = 0;
  let sumG = 0;
  let sumB = 0;
  let count = 0;
  let targetL = 0;

  for (const [px, py] of points) {
    let pr = 0;
    let pg = 0;
    let pb = 0;
    let pc = 0;
    for (let dy = -3; dy <= 3; dy++) {
      for (let dx = -3; dx <= 3; dx++) {
        const x = Math.round(px) + dx;
        const y = Math.round(py) + dy;
        if (x < 0 || y < 0 || x >= w || y >= h) continue;
        const i = y * w + x;
        if (data[i * 4 + 3] === 0) continue;
        pr += data[i * 4];
        pg += data[i * 4 + 1];
        pb += data[i * 4 + 2];
        pc++;
      }
    }
    if (pc === 0) continue;
    pr /= pc;
    pg /= pc;
    pb /= pc;
    sumR += pr;
    sumG += pg;
    sumB += pb;
    count++;
    const l = 0.299 * pr + 0.587 * pg + 0.114 * pb;
    if (l > targetL) targetL = l;
  }

  if (count === 0) return null;

  const r = sumR / count;
  const g = sumG / count;
  const b = sumB / count;
  const s = r + g + b || 1;
  return { chromaR: r / s, chromaG: g / s, targetL };
}

/** How skin-like a pixel is (1 = matches sampled skin chromaticity, → 0 otherwise). */
function skinWeight(r, g, b, skin) {
  const s = r + g + b || 1;
  const dr = r / s - skin.chromaR;
  const dg = g / s - skin.chromaG;
  const d2 = dr * dr + dg * dg;
  return Math.exp(-d2 / (2 * SKIN_SIGMA * SKIN_SIGMA));
}

/**
 * Box-blur a single-channel Float32Array (separable). Zero-weight pixels are
 * excluded from blending so transparent / non-skin areas don't pollute the mean.
 */
function boxBlurLuminance(lum, weight, w, h, radius) {
  const tmp = new Float32Array(w * h);
  const out = new Float32Array(w * h);

  for (let y = 0; y < h; y++) {
    let sumL = 0;
    let sumW = 0;
    for (let x = 0; x < w; x++) {
      sumL += lum[y * w + x] * weight[y * w + x];
      sumW += weight[y * w + x];
      if (x - radius - 1 >= 0) {
        const idx = y * w + (x - radius - 1);
        sumL -= lum[idx] * weight[idx];
        sumW -= weight[idx];
      }
      tmp[y * w + x] = sumW > 0 ? sumL / sumW : 0;
    }
  }

  for (let x = 0; x < w; x++) {
    let sumL = 0;
    let sumW = 0;
    for (let y = 0; y < h; y++) {
      sumL += tmp[y * w + x] * weight[y * w + x];
      sumW += weight[y * w + x];
      if (y - radius - 1 >= 0) {
        const oy = y - radius - 1;
        sumL -= tmp[oy * w + x] * weight[oy * w + x];
        sumW -= weight[oy * w + x];
      }
      out[y * w + x] = sumW > 0 ? sumL / sumW : 0;
    }
  }

  return out;
}

function clamp8(v) {
  return v < 0 ? 0 : v > 255 ? 255 : Math.round(v);
}
