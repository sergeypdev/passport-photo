const DPI = 300;

const PHOTO_W_MM = 35;
const PHOTO_H_MM = 45;
const TOP_MARGIN_MM = 4;
const FACE_HEIGHT_MM = 34; // target chin-to-crown

export function mmToPx(mm) {
  return Math.round((mm / 25.4) * DPI);
}

export const PHOTO_W_PX = mmToPx(PHOTO_W_MM);
export const PHOTO_H_PX = mmToPx(PHOTO_H_MM);
export const TOP_MARGIN_PX = mmToPx(TOP_MARGIN_MM);
export const FACE_HEIGHT_PX = mmToPx(FACE_HEIGHT_MM);
