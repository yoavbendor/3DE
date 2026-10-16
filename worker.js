// Runs the ONNX model off the main thread so the page stays responsive.
// In:  {id, type:'load', prefer} | {id, type:'run', input, V, prefer}
//      res: 504 | 336 (picks the model file), prefer: "auto" (WebGPU, then CPU) | "webgpu" | "cpu"
// Out: {type:'log', level, msg, data}                      (diagnostics, any time)
//      {id, type:'progress', done, total} | {id, type:'progress', phase:'infer'}
//      {id, type:'done', backend, result?, stats?, timings?}
//      {id, type:'error', message}

importScripts("https://cdn.jsdelivr.net/npm/onnxruntime-web@1.22.0/dist/ort.min.js");

// GitHub Pages cannot send the COOP/COEP headers that threads need.
ort.env.wasm.numThreads = 1;
ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.22.0/dist/";

const WORKER_BUILD = "debug-3";
// Model file per input size. Two models so phones can run 3+ views at 336.
const MODELS = { 504: "model/da3-small-fp16.onnx", 336: "model/da3-small-336-fp16.onnx" };
const OUTPUTS = ["depth", "depth_conf", "extrinsics", "intrinsics"];

const modelBytesByRes = {};
let session = null;
let sessionRes = null;
let sessionKind = null; // backend actually in use: "webgpu" | "wasm"
let sessionPref = null; // what the page asked for

const now = () => performance.now();
const round = (x) => Math.round(x * 1000) / 1000;
const log = (level, msg, data) => postMessage({ type: "log", level, msg, data });

async function openCache() {
  try { return await caches.open("da3-model-v1"); } catch { return null; }
}

async function getModelBytes(res, id) {
  const MODEL_URL = MODELS[res];
  if (!MODEL_URL) throw new Error(`no model for ${res} px input`);
  if (modelBytesByRes[res]) return modelBytesByRes[res];
  const t = now();
  const cache = await openCache();
  const cached = cache && (await cache.match(MODEL_URL).catch(() => null));
  if (cached) {
    const bytes = new Uint8Array(await cached.arrayBuffer());
    modelBytesByRes[res] = bytes;
    log("info", "model read from browser cache", { res, bytes: bytes.length, ms: round(now() - t) });
    return bytes;
  }

  const resp = await fetch(MODEL_URL);
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
  if (cache) cache.put(MODEL_URL, new Response(bytes.slice())).catch(() => {});
  modelBytesByRes[res] = bytes;
  log("info", "model downloaded", { res, bytes: received, ms: round(now() - t) });
  return bytes;
}

async function ensureSession(pref, res, id) {
  if (session && sessionPref === pref && sessionRes === res) return;
  const bytes = await getModelBytes(res, id);
  if (session) {
    try { await session.release(); } catch { /* ignore */ }
    session = null;
  }
  const hasGpu = "gpu" in navigator;
  const eps = pref === "cpu" ? ["wasm"]
    : pref === "webgpu" ? ["webgpu"]
    : hasGpu ? ["webgpu", "wasm"] : ["wasm"];
  log("info", "creating session", { pref, navigatorGpu: hasGpu, tryOrder: eps });

  for (const ep of eps) {
    const t = now();
    try {
      session = await ort.InferenceSession.create(bytes, {
        executionProviders: [ep],
        graphOptimizationLevel: "all",
      });
      sessionKind = ep;
      sessionPref = pref;
      sessionRes = res;
      log("info", `session ready on ${ep}`, { ms: round(now() - t) });
      return;
    } catch (err) {
      log("warn", `session failed on ${ep}`, { ms: round(now() - t), error: String((err && err.message) || err) });
    }
  }
  throw new Error("Could not start the model on this device (no WebGPU or WebAssembly backend worked)");
}

// Count NaN/Infinity and the finite range, so a broken output is visible in the log.
function statsOf(a) {
  let nonFinite = 0, min = Infinity, max = -Infinity;
  for (let i = 0; i < a.length; i++) {
    const x = a[i];
    if (!Number.isFinite(x)) { nonFinite++; continue; }
    if (x < min) min = x;
    if (x > max) max = x;
  }
  return { n: a.length, nonFinite, min: round(min), max: round(max) };
}

async function infer(input, V, res, id) {
  const feeds = { images: new ort.Tensor("float32", input, [1, V, 3, res, res]) };
  postMessage({ id, type: "progress", phase: "infer" });
  const t = now();
  const out = await session.run(feeds);
  const ms = round(now() - t);
  const result = {};
  for (const k of OUTPUTS) result[k] = out[k].data.slice();
  return { result, ms };
}

async function run(m) {
  const timings = {};
  let t = now();
  await ensureSession(m.prefer, m.res, m.id);
  timings.sessionMs = round(now() - t);

  let fallback = null;
  let attempt;
  try {
    attempt = await infer(m.input, m.V, m.res, m.id);
  } catch (err) {
    if (sessionKind !== "webgpu") throw err;
    log("error", "WebGPU inference threw, retrying on CPU", { error: String((err && err.message) || err) });
    fallback = "webgpu-error";
    await ensureSession("cpu", m.res, m.id);
    attempt = await infer(m.input, m.V, m.res, m.id);
  }

  let stats = {};
  for (const k of OUTPUTS) stats[k] = statsOf(attempt.result[k]);
  const broken = OUTPUTS.some((k) => stats[k].nonFinite > 0);
  if (broken && sessionKind === "webgpu") {
    log("warn", "WebGPU produced NaN/Infinity, retrying on CPU", stats);
    fallback = "non-finite-output";
    await ensureSession("cpu", m.res, m.id);
    attempt = await infer(m.input, m.V, m.res, m.id);
    stats = {};
    for (const k of OUTPUTS) stats[k] = statsOf(attempt.result[k]);
  }

  timings.inferMs = attempt.ms;
  timings.backend = sessionKind;
  timings.fallback = fallback;
  log("info", "inference finished", { backend: sessionKind, V: m.V, res: m.res, inferMs: attempt.ms, fallback, stats });
  return { result: attempt.result, stats, timings };
}

self.onmessage = async (e) => {
  const m = e.data;
  try {
    if (m.type === "load") {
      await ensureSession(m.prefer, m.res, m.id);
      postMessage({ id: m.id, type: "done", backend: sessionKind });
    } else if (m.type === "run") {
      const r = await run(m);
      const transfer = OUTPUTS.map((k) => r.result[k].buffer);
      postMessage({ id: m.id, type: "done", backend: sessionKind, ...r }, transfer);
    }
  } catch (err) {
    const message = String((err && err.message) || err);
    log("error", "worker step failed", { type: m.type, message, stack: err && err.stack ? String(err.stack).slice(0, 800) : undefined });
    postMessage({ id: m.id, type: "error", message });
  }
};

log("info", "worker started", { build: WORKER_BUILD, userAgent: navigator.userAgent });
