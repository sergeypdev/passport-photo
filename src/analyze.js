import { getFaceLandmarker } from './mediapipe.js';

/**
 * Analyze face landmarks + matte alpha to extract geometry in source pixel coords.
 *
 * @param {ImageBitmap|HTMLCanvasElement} source
 * @param {number} width  source pixel width
 * @param {number} height source pixel height
 * @param {Uint8Array} alpha  matte alpha (length width*height), used to find the crown
 * @returns {Promise<object>} face geometry object
 */
export async function analyzeFace(source, width, height, alpha) {
  const faceLandmarker = await getFaceLandmarker();

  const faceResult = faceLandmarker.detect(source);
  const landmarks = faceResult.faceLandmarks?.[0];
  if (!landmarks) {
    throw new Error('No face detected');
  }

  // Convert normalized landmarks to pixel coords
  const lm = (index) => ({
    x: landmarks[index].x * width,
    y: landmarks[index].y * height,
  });

  const chin = lm(152);
  const leftEye = lm(33);
  const rightEye = lm(263);
  const noseTip = lm(1);

  // Roll angle: positive when the right eye sits lower than the left in the image
  const roll = Math.atan2(rightEye.y - leftEye.y, rightEye.x - leftEye.x);

  const crown = findCrown(alpha, width, height, noseTip.x, lm(10));

  // Face height: chin-to-crown distance (hypot keeps it robust to head roll)
  const faceHeightPx = Math.hypot(chin.x - crown.x, chin.y - crown.y);

  return { chin, leftEye, rightEye, noseTip, crown, roll, faceHeightPx, width, height };
}

/**
 * Crown = topmost opaque pixel of the matte within a column window around the face.
 * Falls back to the forehead landmark if the matte is empty there.
 */
function findCrown(alpha, width, height, faceCenterX, forehead) {
  const ALPHA_THRESHOLD = 128;
  const halfWindow = Math.floor(width * 0.15);
  const colMin = Math.max(0, Math.floor(faceCenterX - halfWindow));
  const colMax = Math.min(width - 1, Math.floor(faceCenterX + halfWindow));

  for (let row = 0; row < height; row++) {
    const rowOffset = row * width;
    for (let col = colMin; col <= colMax; col++) {
      if (alpha[rowOffset + col] >= ALPHA_THRESHOLD) {
        return { x: col, y: row };
      }
    }
  }

  return { x: forehead.x, y: forehead.y };
}
