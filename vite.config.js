import { defineConfig } from 'vite';
import fs from 'fs';
import path from 'path';

// Strip ORT wasm files that Vite copies into dist/ but are never fetched
// (both onnxruntime-web and Transformers.js load wasm from CDN at runtime).
const stripOrtWasm = {
  name: 'strip-ort-wasm',
  closeBundle() {
    const dir = path.resolve('dist/assets');
    if (!fs.existsSync(dir)) return;
    for (const f of fs.readdirSync(dir)) {
      if (f.startsWith('ort-') && f.endsWith('.wasm')) {
        fs.rmSync(path.join(dir, f));
        console.log(`[strip-ort-wasm] removed ${f}`);
      }
    }
  },
};

export default defineConfig({
  server: { port: 5173, open: true },
  // MediaPipe ships large wasm/model assets; don't inline them.
  build: { assetsInlineLimit: 0 },
  // Transformers.js bundles onnxruntime-web; let it resolve its own wasm at runtime.
  optimizeDeps: { exclude: ['@huggingface/transformers'] },
  plugins: [stripOrtWasm],
});
