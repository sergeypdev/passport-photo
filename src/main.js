import { analyzeFace } from './analyze.js';
import { computeMatte } from './matting.js';
import { composePassport } from './framing.js';
import { flattenLighting } from './relight.js';
import { relightDPR } from './relight-dpr.js';
import { drawGuides } from './guides.js';

const fileInput = document.getElementById('fileInput');
const processBtn = document.getElementById('processBtn');
const relightToggle = document.getElementById('relightToggle');
const guidesToggle = document.getElementById('guidesToggle');
const statusEl = document.getElementById('status');
const outputCanvas = document.getElementById('output');
const outputEmpty = document.getElementById('outputEmpty');
const inputCanvas = document.getElementById('inputCanvas');
const inputEmpty = document.getElementById('inputEmpty');
const downloadBtn = document.getElementById('downloadBtn');

// User adjustment sliders (applied as fast GPU CSS filters, no reprocessing).
const brightness = document.getElementById('brightness');
const brightnessVal = document.getElementById('brightnessVal');
const contrast = document.getElementById('contrast');
const contrastVal = document.getElementById('contrastVal');
const saturation = document.getElementById('saturation');
const saturationVal = document.getElementById('saturationVal');

/** Swap a pane between its canvas and its "empty" placeholder. */
function showPane(canvas, placeholder, showCanvas) {
  canvas.classList.toggle('hidden', !showCanvas);
  placeholder.classList.toggle('hidden', showCanvas);
}

let currentBitmap = null;
let bitmapWidth = 0;
let bitmapHeight = 0;

/** The last clean composed passport canvas (guides + adjustments never baked in). */
let lastComposed = null;

// Adjustment state → a CSS filter string. brightness/contrast/saturate are all 1 = neutral.
const adjust = { brightness: 1, contrast: 1, saturation: 1 };

function filterString() {
  return `brightness(${adjust.brightness}) contrast(${adjust.contrast}) saturate(${adjust.saturation})`;
}

/** GPU-composited live adjustment — no canvas redraw needed. */
function applyDisplayFilter() {
  outputCanvas.style.filter = filterString();
}

/**
 * Render the output canvas from lastComposed, optionally overlaying guides.
 * Safe to call when lastComposed is null (no-op).
 */
function renderOutput() {
  if (!lastComposed) return;

  outputCanvas.width = lastComposed.width;
  outputCanvas.height = lastComposed.height;

  const ctx = outputCanvas.getContext('2d');
  ctx.drawImage(lastComposed, 0, 0);

  if (guidesToggle.checked) {
    drawGuides(ctx);
  }

  applyDisplayFilter();
  showPane(outputCanvas, outputEmpty, true);
}

// Load image file into an ImageBitmap when a file is selected
fileInput.addEventListener('change', async () => {
  const file = fileInput.files?.[0];
  if (!file) return;

  try {
    if (currentBitmap) {
      currentBitmap.close();
    }
    currentBitmap = await createImageBitmap(file);
    bitmapWidth = currentBitmap.width;
    bitmapHeight = currentBitmap.height;

    // Show the chosen photo in the Input pane; reset the Result pane.
    inputCanvas.width = bitmapWidth;
    inputCanvas.height = bitmapHeight;
    inputCanvas.getContext('2d').drawImage(currentBitmap, 0, 0);
    showPane(inputCanvas, inputEmpty, true);

    lastComposed = null;
    showPane(outputCanvas, outputEmpty, false);

    statusEl.textContent = 'Фото загружено. Нажмите «Обработать».';
  } catch (err) {
    statusEl.textContent = `Не удалось загрузить изображение: ${err.message}`;
  }
});

// Process the loaded image into a passport photo
processBtn.addEventListener('click', async () => {
  if (!currentBitmap) {
    statusEl.textContent = 'Сначала выберите изображение.';
    return;
  }

  processBtn.disabled = true;
  const resultPane = document.getElementById('resultPane');
  resultPane.classList.add('is-processing');
  try {
    statusEl.textContent = 'Удаляем фон… (при первом запуске загружается модель)';
    const { foregroundCanvas: fg, alpha } = await computeMatte(
      currentBitmap,
      bitmapWidth,
      bitmapHeight
    );

    statusEl.textContent = 'Определяем лицо…';
    const geometry = await analyzeFace(currentBitmap, bitmapWidth, bitmapHeight, alpha);

    if (relightToggle.checked) {
      statusEl.textContent = 'Выравниваем освещение…';
      try {
        await relightDPR(fg, geometry);
      } catch (err) {
        // Fall back to the cheap heuristic if the model can't load/run.
        console.warn('DPR relight failed, using heuristic:', err);
        flattenLighting(fg, geometry);
      }
    }

    statusEl.textContent = 'Кадрируем по стандарту…';
    lastComposed = composePassport(fg, geometry);

    renderOutput();

    resultPane.classList.add('just-captured');
    setTimeout(() => resultPane.classList.remove('just-captured'), 700);

    statusEl.textContent = 'Готово — можно сохранять.';
  } catch (err) {
    statusEl.textContent = `Ошибка: ${err.message}`;
    console.error(err);
  } finally {
    processBtn.disabled = false;
    resultPane.classList.remove('is-processing');
  }
});

// Toggle guides overlay without reprocessing
guidesToggle.addEventListener('change', () => {
  renderOutput();
});

// Live adjustments — just update the CSS filter (GPU), never reprocess.
brightness.addEventListener('input', () => {
  adjust.brightness = Number(brightness.value);
  brightnessVal.textContent = adjust.brightness.toFixed(2);
  applyDisplayFilter();
});
contrast.addEventListener('input', () => {
  adjust.contrast = Number(contrast.value);
  contrastVal.textContent = adjust.contrast.toFixed(2);
  applyDisplayFilter();
});
saturation.addEventListener('input', () => {
  adjust.saturation = Number(saturation.value);
  saturationVal.textContent = adjust.saturation.toFixed(2);
  applyDisplayFilter();
});

// Download a CLEAN copy (no guides), with the adjustments baked in via ctx.filter.
downloadBtn.addEventListener('click', () => {
  if (!lastComposed) {
    statusEl.textContent = 'Нечего сохранять — сначала обработайте фото.';
    return;
  }

  const exp = document.createElement('canvas');
  exp.width = lastComposed.width;
  exp.height = lastComposed.height;
  const ectx = exp.getContext('2d');
  ectx.filter = filterString();
  ectx.drawImage(lastComposed, 0, 0);

  const anchor = document.createElement('a');
  anchor.href = exp.toDataURL('image/png');
  anchor.download = 'passport.png';
  anchor.click();
});
