import { PHOTO_W_PX, PHOTO_H_PX, TOP_MARGIN_PX, FACE_HEIGHT_PX } from './constants.js';

/**
 * Compose the foreground canvas into a passport-sized canvas.
 * The crown is placed at the horizontal center and TOP_MARGIN_PX from the top.
 * The image is rotated to eliminate roll and scaled so the face height matches FACE_HEIGHT_PX.
 *
 * @param {HTMLCanvasElement} foregroundCanvas - segmented foreground (transparent bg)
 * @param {object} geometry - result from analyzeFace
 * @returns {HTMLCanvasElement} passport-sized canvas
 */
export function composePassport(foregroundCanvas, geometry) {
  const canvas = document.createElement('canvas');
  canvas.width = PHOTO_W_PX;
  canvas.height = PHOTO_H_PX;

  const ctx = canvas.getContext('2d');

  // White background
  ctx.fillStyle = '#ffffff';
  ctx.fillRect(0, 0, PHOTO_W_PX, PHOTO_H_PX);

  const scale = FACE_HEIGHT_PX / geometry.faceHeightPx;
  const { crown } = geometry;

  ctx.save();

  // Place crown at (center, TOP_MARGIN_PX)
  ctx.translate(PHOTO_W_PX / 2, TOP_MARGIN_PX);

  // Counter-rotate to straighten the face
  ctx.rotate(-geometry.roll);

  // Scale so face height matches passport spec
  ctx.scale(scale, scale);

  // Move the crown of the source to the origin
  ctx.translate(-crown.x, -crown.y);

  ctx.drawImage(foregroundCanvas, 0, 0);

  ctx.restore();

  return canvas;
}
