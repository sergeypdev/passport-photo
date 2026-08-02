import * as ort from 'onnxruntime-web';

// onnxruntime-web fetches its wasm from a path; point it at the matching CDN build.
ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.27.0/dist/';
// Single-threaded wasm avoids needing SharedArrayBuffer / COOP+COEP headers.
ort.env.wasm.numThreads = 1;

const MODEL_URL = '/dpr_512.onnx';
const SIZE = 512; // DPR operates on 512x512 Lab-L

// DPR order-2 SH target. Coeff order is [DC, y, z, x, ...]. Verified axis convention:
//   -y = toward camera (frontal key)   +z = up (top light)   +x = right (side light)
// Derived from an HDRI env map by projecting its luminance onto the SH basis — see
// tools/envmap_to_sh.py, from studio_small_02_2k.exr, then rotated -14° about the
// vertical axis (the orientation that looked best) and baked in. Centered frontal key,
// mild modeling. Only coefficient ratios matter (exposure locked after).
const TARGET_SH = new Float32Array([0.75, -0.2002, 0.3458, -0.0499, 0.1473, 0.4666, -0.2568, -0.1049, -0.2122]);

// How fully to apply the relighting (1 = full DPR result, lower keeps more original).
const STRENGTH = 0.7;
// Low-pass amount for the gain map, as a PERCENT of the 512px working size.
const LOWPASS_PCT = 100;
// Clamp the per-pixel LINEAR-luminance gain (deep shadow lift / highlight cut).
const MIN_GAIN = 0.5;
const MAX_GAIN = 3.0;
const INV_GAMMA = 1 / 2.2; // linear-gain → sRGB-multiplier conversion

// Absolute exposure: lock the face's mean luma to this (0..255), so a dark input
// still comes out appropriately bright against the white background.
const TARGET_FACE_LUMA = 130;
const EXPOSURE_MIN_SCALE = 0.5;
const EXPOSURE_MAX_SCALE = 2.6;

let _sessionPromise = null;
function getSession() {
  if (!_sessionPromise) {
    _sessionPromise = ort.InferenceSession.create(MODEL_URL, {
      executionProviders: ['wasm'],
      graphOptimizationLevel: 'all',
    });
  }
  return _sessionPromise;
}

/**
 * Relight the face toward soft studio lighting and lock it to a target exposure.
 * Modifies the source-sized foreground canvas in place.
 *
 * @param {HTMLCanvasElement} canvas  foreground cutout (source-sized, RGBA)
 * @param {object} [geometry]  analyzeFace result (crown/chin/eyes) for face metering
 * @returns {Promise<HTMLCanvasElement>} the same canvas
 */
export async function relightDPR(canvas, geometry) {
  const session = await getSession();
  const { width: w, height: h } = canvas;

  const lowpass = Math.max(2, Math.min(SIZE, Math.round((LOWPASS_PCT / 100) * SIZE)));
  const sh = TARGET_SH;

  // Pick the DPR input crop. DPR expects a face-FILLING frame (see its obama.jpg:
  // hairline near the top edge, chin ~3/4 down, cheeks to the sides). Feeding the
  // whole cutout squashed to 512 makes the face small & stretched and degrades the
  // estimate, so crop a square around the face from the landmarks.
  let cropX = 0;
  let cropY = 0;
  let cropW = w;
  let cropH = h;
  if (geometry) {
    const cx = (geometry.leftEye.x + geometry.rightEye.x) / 2;
    const faceH = Math.max(1, geometry.chin.y - geometry.crown.y);
    const s = faceH / 0.7; // face spans ~70% of the crop, matching obama's framing
    cropX = cx - s / 2;
    cropY = geometry.crown.y - 0.08 * s; // ~8% headroom above the crown
    cropW = s;
    cropH = s;
  }

  // 1. Draw that crop into 512x512 and read pixels.
  const small = document.createElement('canvas');
  small.width = SIZE;
  small.height = SIZE;
  const sctx = small.getContext('2d');
  sctx.drawImage(canvas, cropX, cropY, cropW, cropH, 0, 0, SIZE, SIZE);
  const smallData = sctx.getImageData(0, 0, SIZE, SIZE).data;

  // 2. Lab L (0..1) for the model input.
  const inputL = new Float32Array(SIZE * SIZE);
  for (let i = 0; i < SIZE * SIZE; i++) {
    inputL[i] = labL(smallData[i * 4], smallData[i * 4 + 1], smallData[i * 4 + 2]);
  }

  // 3. Run DPR → relit L in [0,1] under the studio target.
  const out = await session.run({
    l: new ort.Tensor('float32', inputL, [1, 1, SIZE, SIZE]),
    sh: new ort.Tensor('float32', sh, [1, 9, 1, 1]),
  });
  const relit = out.out.data;

  // 4. Convert the L* change into a LINEAR-luminance ratio (the physically correct
  //    relight gain), softened by STRENGTH.
  const ratio = new Float32Array(SIZE * SIZE);
  for (let i = 0; i < SIZE * SIZE; i++) {
    const r = labLinv(relit[i]) / Math.max(labLinv(inputL[i]), 1e-4);
    ratio[i] = 1 + (r - 1) * STRENGTH;
  }

  // 5. Encode the clamped gain into a canvas so we can bilinear-upsample it.
  const gainImg = new ImageData(SIZE, SIZE);
  for (let i = 0; i < SIZE * SIZE; i++) {
    let gain = ratio[i];
    if (gain < MIN_GAIN) gain = MIN_GAIN;
    else if (gain > MAX_GAIN) gain = MAX_GAIN;
    const enc = ((gain - MIN_GAIN) / (MAX_GAIN - MIN_GAIN)) * 255;
    gainImg.data[i * 4] = gainImg.data[i * 4 + 1] = gainImg.data[i * 4 + 2] = enc;
    gainImg.data[i * 4 + 3] = 255;
  }
  const gainCanvas = document.createElement('canvas');
  gainCanvas.width = SIZE;
  gainCanvas.height = SIZE;
  gainCanvas.getContext('2d').putImageData(gainImg, 0, 0);

  // 6. Low-pass the gain (512 → LOWPASS) and decode to a small linear-gain grid we
  //    sample manually — the broad lighting gradient survives, and manual sampling
  //    lets us clamp at the crop edges so regions outside the crop keep the nearest
  //    gain (no seam) instead of an abrupt cutoff.
  const lowCanvas = document.createElement('canvas');
  lowCanvas.width = lowpass;
  lowCanvas.height = lowpass;
  const lctx = lowCanvas.getContext('2d');
  lctx.imageSmoothingEnabled = true;
  lctx.drawImage(gainCanvas, 0, 0, lowpass, lowpass);
  const lowData = lctx.getImageData(0, 0, lowpass, lowpass).data;
  const low = new Float32Array(lowpass * lowpass);
  for (let i = 0; i < lowpass * lowpass; i++) {
    low[i] = MIN_GAIN + (lowData[i * 4] / 255) * (MAX_GAIN - MIN_GAIN);
  }

  // 7. Apply the gain in FLOAT (no clamp yet), mapping each full-res pixel through the
  //    crop UV (clamped). Scaling linear luminance by g ≈ multiplying the sRGB value by
  //    g^(1/2.2); same factor on R/G/B preserves hue. Deferring the clamp stops a bright
  //    frontal key from clipping skin to white before the exposure lock pulls it back.
  const gctx = canvas.getContext('2d');
  const fullImg = gctx.getImageData(0, 0, w, h);
  const fdata = fullImg.data;
  const buf = new Float32Array(w * h * 3);

  const ellipse = geometry ? faceEllipse(geometry) : null;
  let faceLuma = 0;
  let faceN = 0;
  for (let i = 0; i < w * h; i++) {
    if (fdata[i * 4 + 3] === 0) continue;
    const x = i % w;
    const y = (i / w) | 0;
    const lin = sampleGain(low, lowpass, (x - cropX) / cropW, (y - cropY) / cropH);
    const g = Math.pow(lin, INV_GAMMA);
    const nr = fdata[i * 4] * g;
    const ng = fdata[i * 4 + 1] * g;
    const nb = fdata[i * 4 + 2] * g;
    buf[i * 3] = nr;
    buf[i * 3 + 1] = ng;
    buf[i * 3 + 2] = nb;
    // Meter the face region (fall back to whole subject if no geometry).
    if (!ellipse || inEllipse(x, y, ellipse)) {
      faceLuma += 0.299 * nr + 0.587 * ng + 0.114 * nb;
      faceN++;
    }
  }

  // 8. Absolute exposure lock: scale so the face's mean luma hits the target.
  let scale = 1;
  if (faceN > 0 && faceLuma > 1e-3) {
    scale = TARGET_FACE_LUMA / (faceLuma / faceN);
    if (scale < EXPOSURE_MIN_SCALE) scale = EXPOSURE_MIN_SCALE;
    else if (scale > EXPOSURE_MAX_SCALE) scale = EXPOSURE_MAX_SCALE;
  }

  // 9. Apply the exposure scale, then a SINGLE clamp. (User brightness/contrast/
  //    saturation are applied separately as fast GPU display filters, not here.)
  for (let i = 0; i < w * h; i++) {
    if (fdata[i * 4 + 3] === 0) continue;
    fdata[i * 4] = clamp8(buf[i * 3] * scale);
    fdata[i * 4 + 1] = clamp8(buf[i * 3 + 1] * scale);
    fdata[i * 4 + 2] = clamp8(buf[i * 3 + 2] * scale);
  }

  gctx.putImageData(fullImg, 0, 0);
  return canvas;
}

/** Bilinear-sample the low-res gain grid at normalized (u,v), clamped to the edges. */
function sampleGain(low, n, u, v) {
  u = (u < 0 ? 0 : u > 1 ? 1 : u) * (n - 1);
  v = (v < 0 ? 0 : v > 1 ? 1 : v) * (n - 1);
  const x0 = Math.floor(u);
  const y0 = Math.floor(v);
  const x1 = Math.min(x0 + 1, n - 1);
  const y1 = Math.min(y0 + 1, n - 1);
  const fx = u - x0;
  const fy = v - y0;
  const top = low[y0 * n + x0] * (1 - fx) + low[y0 * n + x1] * fx;
  const bot = low[y1 * n + x0] * (1 - fx) + low[y1 * n + x1] * fx;
  return top * (1 - fy) + bot * fy;
}

/** Face ellipse (center + radii) in source pixel coords, from landmarks. */
function faceEllipse(g) {
  const cx = (g.leftEye.x + g.rightEye.x) / 2;
  const cy = (g.crown.y + g.chin.y) / 2;
  const eyeDist = Math.hypot(g.rightEye.x - g.leftEye.x, g.rightEye.y - g.leftEye.y);
  const rx = Math.max(1, eyeDist * 1.3);
  const ry = Math.max(1, ((g.chin.y - g.crown.y) / 2) * 1.1);
  return { cx, cy, rx, ry };
}

function inEllipse(x, y, e) {
  const dx = (x - e.cx) / e.rx;
  const dy = (y - e.cy) / e.ry;
  return dx * dx + dy * dy <= 1;
}

/** CIELab lightness L* in [0,1] from sRGB bytes (matches the training color space). */
function labL(r, g, b) {
  const y = 0.2126 * srgb2lin(r) + 0.7152 * srgb2lin(g) + 0.0722 * srgb2lin(b);
  const f = y > 0.008856 ? Math.cbrt(y) : 7.787 * y + 16 / 116;
  return (116 * f - 16) / 100;
}

/** Inverse of labL: L* in [0,1] → relative linear luminance Y in [0,1]. */
function labLinv(lStored) {
  const l100 = lStored * 100;
  const f = (l100 + 16) / 116;
  return l100 > 8 ? f * f * f : l100 / 903.3;
}

function srgb2lin(c) {
  c /= 255;
  return c <= 0.04045 ? c / 12.92 : Math.pow((c + 0.055) / 1.055, 2.4);
}

function clamp8(v) {
  return v < 0 ? 0 : v > 255 ? 255 : Math.round(v);
}
