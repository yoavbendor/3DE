// Runs the ONNX model off the main thread so the page stays responsive.
// Messages in:  {id, type:'load', url} | {id, type:'run', input, V}
// Messages out: {id, type:'progress', done, total} | {id, type:'done', ...} | {id, type:'error', message}

importScripts("https://cdn.jsdelivr.net/npm/onnxruntime-web@1.22.0/dist/ort.min.js");

// GitHub Pages cannot send the COOP/COEP headers that threads need.
ort.env.wasm.numThreads = 1;
ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.22.0/dist/";

const RES = 504;
let session = null;
let loading = null;

async function openCache() {
  try { return await caches.open("da3-model-v1"); } catch { return null; }
}

async function fetchModel(url, id) {
  const cache = await openCache();
  const cached = cache && (await cache.match(url).catch(() => null));
  if (cached) return new Uint8Array(await cached.arrayBuffer());

  const resp = await fetch(url);
  if (!resp.ok) throw new Error(`Model download failed (HTTP ${resp.status})`);
  const total = Number(resp.headers.get("content-length")) || 0;
  const reader = resp.body.getReader();
  const chunks = [];
  let received = 0;
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    chunks.push(value);
    received += value.length;
    postMessage({ id, type: "progress", done: received, total });
  }
  const bytes = new Uint8Array(received);
  let offset = 0;
  for (const c of chunks) { bytes.set(c, offset); offset += c.length; }
  if (cache) cache.put(url, new Response(bytes.slice())).catch(() => {});
  return bytes;
}

async function load(id, url) {
  const bytes = await fetchModel(url, id);
  const providers = "gpu" in navigator ? ["webgpu", "wasm"] : ["wasm"];
  for (const ep of providers) {
    try {
      session = await ort.InferenceSession.create(bytes, {
        executionProviders: [ep],
        graphOptimizationLevel: "all",
      });
      return ep;
    } catch (err) {
      console.warn(`${ep} backend unavailable`, err);
    }
  }
  throw new Error("Could not start the model on this device");
}

self.onmessage = async (e) => {
  const m = e.data;
  try {
    if (m.type === "load") {
      if (!loading) loading = load(m.id, m.url).catch((err) => { loading = null; throw err; });
      const backend = await loading;
      postMessage({ id: m.id, type: "done", backend });
    } else if (m.type === "run") {
      if (!loading) throw new Error("Model not loaded");
      await loading;
      const feeds = { images: new ort.Tensor("float32", m.input, [1, m.V, 3, RES, RES]) };
      const out = await session.run(feeds);
      const result = {
        depth: out.depth.data,
        depth_conf: out.depth_conf.data,
        extrinsics: out.extrinsics.data,
        intrinsics: out.intrinsics.data,
      };
      const transfer = Object.values(result).map((a) => a.buffer);
      postMessage({ id: m.id, type: "done", result }, transfer);
    }
  } catch (err) {
    postMessage({ id: m.id, type: "error", message: err.message || String(err) });
  }
};
