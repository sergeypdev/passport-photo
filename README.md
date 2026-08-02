# Passport Photo

A fully client-side web app that turns any portrait into a passport-spec photo
(35×45 mm) — entirely in your browser, no server, no upload.

## What it does

1. **Background removal** — RMBG-1.4 alpha matting (via Transformers.js) for clean hair edges.
2. **Face analysis** — MediaPipe FaceLandmarker for chin/eyes/pose; the crown (top of head)
   is found from the matte.
3. **Framing** — scales and straightens so the face lands at spec: crown 4 mm from the top,
   ~34 mm chin-to-crown, centered, on a white background (300 DPI → 413×531 px).
4. **Relighting** — DPR (Deep Single-Image Portrait Relighting) exported to ONNX and run with
   onnxruntime-web, retargeting the face to a soft studio light derived from an HDRI.
5. **Guides + adjustments** — a toggleable spec overlay (crown line, 33–35 mm chin band) and
   live Brightness / Contrast / Saturation (GPU CSS filters, baked into the download).

## Develop

```bash
npm install
npm run dev      # http://localhost:5173
npm run build
```

Models are fetched from CDNs on first run and cached by the browser; the DPR model
(`public/dpr_512.onnx`, ~2.9 MB) ships with the app.

## tools/

Offline utilities for **re-deriving the studio lighting** — not needed to run the app.
They use an isolated [uv](https://github.com/astral-sh/uv) env:

```bash
cd tools
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python torch numpy onnx onnxruntime onnxscript opencv-python-headless
git clone --depth 1 https://github.com/zhhoper/DPR DPR

.venv/bin/python export_dpr.py                       # DPR weights → public/dpr_512.onnx
.venv/bin/python envmap_to_sh.py path/to/env.exr     # HDRI → DPR SH coefficients
```

`envmap_to_sh.py` projects an equirectangular HDRI onto DPR's order-2 SH basis (verified
axis convention), auto-orients the key to the front, and prints diagnostics. Paste the
resulting coefficients into `TARGET_SH` in `src/relight-dpr.js`.
