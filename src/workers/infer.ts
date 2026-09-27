/// <reference lib="webworker" />
// Inference worker: downloads MoGe-2 once, runs it (WebGPU first, WASM
// fallback), recovers metric geometry and fits the light. Keeps the UI thread
// free for rendering.

import * as ort from "onnxruntime-web/webgpu";
import ortGlue from "onnxruntime-web/ort-wasm-simd-threaded.asyncify.mjs?url";
import { MODEL_URL, MODEL_VERSION, NUM_TOKENS, NUM_TOKENS_WASM, STATS_CROP } from "../config";
import { inputSize, pointsFromDepth, postprocess, type Geometry } from "../geometry/moge-post";
import { getCached, hashBytes, putCached } from "../lib/cache";
import { measure } from "../lib/stats";
import { fitLight } from "../light/fit";
import { transferables, type Analysis, type FromWorker, type ToWorker } from "./protocol";

declare const self: DedicatedWorkerGlobalScope;

// The ORT runtime is over Cloudflare's 25 MiB asset limit, so it is served
// next to the weights from R2 with its version in the name.
ort.env.wasm.wasmPaths = {
  mjs: new URL(ortGlue, self.location.href).href,
  wasm: new URL(`/models/ort-wasm-simd-threaded.asyncify-${ort.env.versions.web}.wasm`, self.location.href).href,
};
ort.env.wasm.numThreads = self.crossOriginIsolated ? Math.min(8, navigator.hardwareConcurrency || 4) : 1;

const post = (msg: FromWorker, transfer: Transferable[] = []) => self.postMessage(msg, transfer);

let modelBytes: Promise<Uint8Array> | null = null;
let prefer: "webgpu" | "wasm" = "webgpu";
let tokenOverride: number | undefined;
let session: Promise<{ session: ort.InferenceSession; backend: string }> | null = null;

async function download(url: string): Promise<Uint8Array> {
  const cache = "caches" in self ? await caches.open("depth-light-models") : null;
  const hit = cache ? await cache.match(url) : undefined;
  if (hit) {
    post({ type: "progress", phase: "download", fraction: 1 });
    return new Uint8Array(await hit.arrayBuffer());
  }
  const res = await fetch(url);
  if (!res.ok || !res.body) throw new Error(`model download failed: ${res.status}`);
  const total = Number(res.headers.get("content-length")) || 0;
  const reader = res.body.getReader();
  const chunks: Uint8Array[] = [];
  let loaded = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    loaded += value.length;
    if (total) post({ type: "progress", phase: "download", fraction: loaded / total });
  }
  const bytes = new Uint8Array(loaded);
  let o = 0;
  for (const c of chunks) {
    bytes.set(c, o);
    o += c.length;
  }
  if (cache) {
    await cache.put(url, new Response(bytes, { headers: { "content-type": "application/octet-stream" } }));
    // Drop weights from earlier versions.
    for (const req of await cache.keys()) if (!req.url.endsWith(url)) await cache.delete(req);
  }
  post({ type: "progress", phase: "download", fraction: 1 });
  return bytes;
}

async function createSession(prefer: "webgpu" | "wasm"): Promise<{ session: ort.InferenceSession; backend: string }> {
  modelBytes ??= download(MODEL_URL);
  const bytes = await modelBytes;
  const order: ("webgpu" | "wasm")[] = prefer === "webgpu" && "gpu" in navigator ? ["webgpu", "wasm"] : ["wasm"];
  let last: unknown;
  for (const ep of order) {
    try {
      const s = await ort.InferenceSession.create(bytes, { executionProviders: [ep], graphOptimizationLevel: "all" });
      return { session: s, backend: ep };
    } catch (e) {
      last = e;
    }
  }
  throw last instanceof Error ? last : new Error(String(last));
}

function getSession(): Promise<{ session: ort.InferenceSession; backend: string }> {
  session ??= createSession(prefer).catch((e) => {
    // Let the next image retry (a flaky connection, say) instead of failing forever.
    session = null;
    modelBytes = null;
    throw e;
  });
  return session;
}

async function runModel(rgba: Uint8ClampedArray, width: number, height: number, tokens: number, s: ort.InferenceSession): Promise<Geometry> {
  const plane = width * height;
  const data = new Float32Array(3 * plane);
  for (let p = 0; p < plane; p++) {
    data[p] = rgba[p * 4] / 255;
    data[plane + p] = rgba[p * 4 + 1] / 255;
    data[2 * plane + p] = rgba[p * 4 + 2] / 255;
  }
  const feeds = {
    image: new ort.Tensor("float32", data, [1, 3, height, width]),
    num_tokens: new ort.Tensor("int64", BigInt64Array.of(BigInt(tokens)), []),
  };
  const out = await s.run(feeds);
  const geometry = postprocess({
    width,
    height,
    points: out.points.data as Float32Array,
    normal: out.normal.data as Float32Array,
    mask: out.mask.data as Float32Array,
    metricScale: (out.metric_scale.data as Float32Array)[0],
  });
  for (const t of Object.values(out)) t.dispose();
  return geometry;
}

function pixels(bitmap: ImageBitmap, sx: number, sy: number, sw: number, sh: number, w: number, h: number): Uint8ClampedArray {
  const canvas = new OffscreenCanvas(w, h);
  const ctx = canvas.getContext("2d", { willReadFrequently: true })!;
  ctx.imageSmoothingEnabled = true;
  ctx.imageSmoothingQuality = "high";
  ctx.drawImage(bitmap, sx, sy, sw, sh, 0, 0, w, h);
  return ctx.getImageData(0, 0, w, h).data;
}

async function analyze(id: number, bytes: ArrayBuffer, mime: string): Promise<void> {
  const t0 = performance.now();
  const bitmap = await createImageBitmap(new Blob([bytes], { type: mime }), { imageOrientation: "from-image" });
  const { width: W, height: H } = bitmap;

  const cw = Math.min(W, STATS_CROP), ch = Math.min(H, STATS_CROP);
  const stats = measure(pixels(bitmap, (W - cw) >> 1, (H - ch) >> 1, cw, ch, cw, ch), cw, ch);

  const s = await getSession();
  const tokens = tokenOverride ?? (s.backend === "webgpu" ? NUM_TOKENS : NUM_TOKENS_WASM);
  const size = inputSize(tokens, W, H);
  const rgba = pixels(bitmap, 0, 0, W, H, size.width, size.height);
  bitmap.close();

  const key = `${MODEL_VERSION}:${tokens}:${await hashBytes(bytes)}`;
  const cached = await getCached(key);
  let geometry: Geometry;
  let backend = s.backend;
  post({ type: "progress", phase: "infer", fraction: 0 });
  if (cached && cached.width === size.width && cached.height === size.height) {
    geometry = { ...cached, points: pointsFromDepth(cached.depth, cached.width, cached.height, cached.fx, cached.fy), fovY: (2 * Math.atan(0.5 / cached.fy) * 180) / Math.PI };
    backend = "cache";
  } else {
    try {
      geometry = await runModel(rgba, size.width, size.height, tokens, s.session);
    } catch (e) {
      if (s.backend !== "webgpu") throw e;
      // An op the WebGPU backend cannot run: fall back for good.
      session = createSession("wasm");
      const fallback = await session;
      backend = fallback.backend;
      geometry = await runModel(rgba, size.width, size.height, tokens, fallback.session);
    }
    await putCached(key, {
      width: geometry.width,
      height: geometry.height,
      depth: geometry.depth,
      normals: geometry.normals,
      mask: geometry.mask,
      fx: geometry.fx,
      fy: geometry.fy,
      metricScale: geometry.metricScale,
    });
  }
  post({ type: "progress", phase: "fit", fraction: 0 });
  const fit = fitLight(rgba, geometry);
  const analysis: Analysis = { geometry, fit, stats, rgba, backend, ms: performance.now() - t0 };
  post({ type: "result", id, analysis }, transferables(analysis));
}

self.onmessage = async (ev: MessageEvent<ToWorker>) => {
  const msg = ev.data;
  try {
    if (msg.type === "load") {
      prefer = msg.prefer;
      tokenOverride = msg.tokens;
      const s = await getSession();
      post({ type: "ready", backend: s.backend });
    } else if (msg.type === "analyze") {
      await analyze(msg.id, msg.bytes, msg.mime);
    }
  } catch (e) {
    post({ type: "error", id: msg.type === "analyze" ? msg.id : undefined, message: e instanceof Error ? e.message : String(e) });
  }
};
