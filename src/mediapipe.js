import { FilesetResolver, FaceLandmarker } from '@mediapipe/tasks-vision';

const WASM_URL = 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.18/wasm';

let _fileset = null;
async function getFileset() {
  if (!_fileset) {
    _fileset = await FilesetResolver.forVisionTasks(WASM_URL);
  }
  return _fileset;
}

let _faceLandmarker = null;
export async function getFaceLandmarker() {
  if (!_faceLandmarker) {
    const fileset = await getFileset();
    _faceLandmarker = await FaceLandmarker.createFromOptions(fileset, {
      baseOptions: {
        modelAssetPath:
          'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task',
      },
      runningMode: 'IMAGE',
      numFaces: 1,
    });
  }
  return _faceLandmarker;
}
