import { defineConfig } from 'vite';

export default defineConfig({
  server: { port: 5173, open: true },
  // MediaPipe ships large wasm/model assets; don't inline them.
  build: { assetsInlineLimit: 0 },
  // Transformers.js bundles onnxruntime-web; let it resolve its own wasm at runtime.
  optimizeDeps: { exclude: ['@huggingface/transformers'] },
});
