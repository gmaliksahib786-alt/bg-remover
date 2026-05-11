// ═══════════════════════════════════════════════════════════════
//  Background Removal Web Worker
//  Uses: @huggingface/transformers v3 (ESM)
//  Model: briaai/RMBG-1.4 (quantized ~45MB)
//  Device: WebGPU → WASM fallback (automatic)
// ═══════════════════════════════════════════════════════════════

import {
  env,
  AutoModel,
  AutoProcessor,
  RawImage,
} from 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.5.2/dist/transformers.min.js';

// Use jsdelivr CDN — CORS allowed
env.allowLocalModels  = false;
env.allowRemoteModels = true;
env.useBrowserCache   = true;

// Disable WASM proxy in worker context (already off-main-thread)
env.backends.onnx.wasm.proxy = false;

const MODEL_ID = 'briaai/RMBG-1.4';

let model     = null;
let processor = null;
let device    = 'wasm'; // will update after load

// ── MODEL LOAD ────────────────────────────────────────────────
async function loadModel() {
  self.postMessage({ type: 'status', status: 'loading', message: 'Checking WebGPU support...' });

  // Detect WebGPU
  let useWebGPU = false;
  try {
    if (typeof navigator !== 'undefined' && navigator.gpu) {
      const adapter = await navigator.gpu.requestAdapter();
      if (adapter) { useWebGPU = true; device = 'webgpu'; }
    }
  } catch {}

  self.postMessage({
    type: 'status',
    status: 'loading',
    message: useWebGPU
      ? '⚡ WebGPU detected! Loading model (~45MB)...'
      : '🔄 Loading model in WASM mode (~45MB)...',
    device: device
  });

  try {
    model = await AutoModel.from_pretrained(MODEL_ID, {
      config:    { model_type: 'custom' },
      device:    useWebGPU ? 'webgpu' : 'wasm',
      dtype:     useWebGPU ? 'fp32'   : 'q8',
      progress_callback: (p) => {
        if (p.status === 'progress' && p.total) {
          const pct  = Math.round((p.loaded / p.total) * 85) + 5;
          const mb   = (p.loaded / 1024 / 1024).toFixed(1);
          const tot  = (p.total  / 1024 / 1024).toFixed(1);
          self.postMessage({
            type: 'progress',
            pct,
            message: `Downloading model... ${mb}MB / ${tot}MB`,
          });
        }
      },
    });

    self.postMessage({ type: 'progress', pct: 92, message: 'Loading processor...' });

    processor = await AutoProcessor.from_pretrained(MODEL_ID, {
      config: {
        do_normalize: true, do_pad: false, do_rescale: true, do_resize: true,
        image_mean: [0.5, 0.5, 0.5], image_std: [1, 1, 1],
        feature_extractor_type: 'ImageFeatureExtractor',
        resample: 2,
        rescale_factor: 0.00392156862745098,
        size: { width: 1024, height: 1024 },
      },
    });

    self.postMessage({
      type:    'ready',
      device:  device,
      message: useWebGPU ? '⚡ WebGPU ready — ultra fast!' : '✓ WASM ready',
    });

  } catch (err) {
    self.postMessage({ type: 'error', message: 'Model load failed: ' + err.message });
  }
}

// ── PROCESS IMAGE ─────────────────────────────────────────────
async function processImage({ imageData, width, height, id }) {
  try {
    self.postMessage({ type: 'processing', id, pct: 10, message: 'Preprocessing image...' });

    // Create RawImage from ImageData pixels (RGB)
    const rgbData = new Uint8ClampedArray(width * height * 3);
    for (let i = 0; i < width * height; i++) {
      rgbData[i * 3]     = imageData[i * 4];
      rgbData[i * 3 + 1] = imageData[i * 4 + 1];
      rgbData[i * 3 + 2] = imageData[i * 4 + 2];
    }
    const rawImage = new RawImage(rgbData, width, height, 3);

    self.postMessage({ type: 'processing', id, pct: 25, message: 'Running AI model...' });

    const inputs = await processor(rawImage);

    self.postMessage({ type: 'processing', id, pct: 50, message: device === 'webgpu' ? '⚡ WebGPU inference...' : 'WASM inference...' });

    const { output } = await model(inputs);

    self.postMessage({ type: 'processing', id, pct: 80, message: 'Applying mask...' });

    // Resize mask to original dimensions
    const mask = await RawImage.fromTensor(output[0].mul(255).to('uint8'))
      .resize(width, height);

    // Apply mask as alpha channel to original imageData
    const out = new Uint8ClampedArray(imageData.length);
    for (let i = 0; i < width * height; i++) {
      out[i * 4]     = imageData[i * 4];
      out[i * 4 + 1] = imageData[i * 4 + 1];
      out[i * 4 + 2] = imageData[i * 4 + 2];
      out[i * 4 + 3] = mask.data[i];
    }

    self.postMessage({ type: 'processing', id, pct: 95, message: 'Finalizing...' });

    // Transfer buffer back to main thread (zero-copy)
    self.postMessage(
      { type: 'done', id, imageData: out.buffer, width, height },
      [out.buffer]
    );

  } catch (err) {
    self.postMessage({ type: 'error', id, message: 'Processing failed: ' + err.message });
  }
}

// ── MESSAGE HANDLER ───────────────────────────────────────────
self.onmessage = async ({ data }) => {
  if (data.type === 'load')    loadModel();
  if (data.type === 'process') processImage(data);
};

// Auto-start loading
loadModel();
