import { AutoModel, AutoProcessor, RawImage, env } from '@huggingface/transformers';

// Cache ONNX weights in the browser between sessions.
env.allowLocalModels = false;

const MODEL_ID = 'briaai/RMBG-1.4';

let _modelPromise = null;
let _processorPromise = null;

function pickDevice() {
  return typeof navigator !== 'undefined' && navigator.gpu ? 'webgpu' : 'wasm';
}

function getModel() {
  if (!_modelPromise) {
    _modelPromise = AutoModel.from_pretrained(MODEL_ID, {
      device: pickDevice(),
      dtype: 'fp32',
    });
  }
  return _modelPromise;
}

function getProcessor() {
  if (!_processorPromise) {
    _processorPromise = AutoProcessor.from_pretrained(MODEL_ID);
  }
  return _processorPromise;
}

/**
 * Run RMBG-1.4 alpha matting on the source image.
 *
 * Produces a proper soft alpha (great hair edges), unlike the coarse MediaPipe
 * class mask. The returned alpha is also used downstream for crown detection.
 *
 * @param {ImageBitmap|HTMLCanvasElement} source
 * @param {number} width  source pixel width
 * @param {number} height source pixel height
 * @returns {Promise<{foregroundCanvas: HTMLCanvasElement, alpha: Uint8Array, width: number, height: number}>}
 */
export async function computeMatte(source, width, height) {
  const [model, processor] = await Promise.all([getModel(), getProcessor()]);

  // Draw source into a canvas we can both read pixels from and later re-use.
  const canvas = document.createElement('canvas');
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext('2d');
  ctx.drawImage(source, 0, 0, width, height);
  const imageData = ctx.getImageData(0, 0, width, height);

  // RMBG expects RGB; strip the alpha channel for the model input.
  const rawImage = new RawImage(imageData.data, width, height, 4).rgb();

  const { pixel_values } = await processor(rawImage);
  const { output } = await model({ input: pixel_values });

  // output[0] is a [1,H,W] matte in [0,1]; scale to bytes and resize to source.
  const maskImage = await RawImage.fromTensor(output[0].mul(255).to('uint8')).resize(
    width,
    height
  );
  const alpha = maskImage.data; // Uint8Array, length width*height (1 channel)

  // Apply the matte as the alpha channel of the source pixels.
  const pixels = imageData.data;
  for (let i = 0; i < alpha.length; i++) {
    pixels[i * 4 + 3] = alpha[i];
  }
  ctx.putImageData(imageData, 0, 0);

  return { foregroundCanvas: canvas, alpha, width, height };
}
