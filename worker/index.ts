// Serves /models/* (model weights and the ONNX Runtime binary). Everything
// else is a static asset and never reaches this Worker (see run_worker_first
// in wrangler.jsonc).
//
// Both files exceed Cloudflare's 25 MiB per-asset limit. R2 is tried first;
// if the bucket does not hold the file, it is streamed back together from
// the parts in /weights/ (made by scripts/chunk-models.mjs), so the site
// works with no bucket uploads at all.

// Minimal slice of the Workers runtime types, so the app does not need
// @cloudflare/workers-types just for this file.
interface R2ObjectBody {
  body: ReadableStream;
  size: number;
  httpEtag: string;
}
interface R2Bucket {
  get(key: string): Promise<R2ObjectBody | null>;
}
interface Fetcher {
  fetch(input: Request | string | URL, init?: RequestInit): Promise<Response>;
}
interface ExecutionContext {
  waitUntil(promise: Promise<unknown>): void;
}
declare class FixedLengthStream extends TransformStream<Uint8Array, Uint8Array> {
  constructor(length: number);
}

interface Env {
  MODELS: R2Bucket;
  ASSETS: Fetcher;
}

interface Manifest {
  [key: string]: { size: number; sha256: string; parts: string[] };
}

function headers(key: string, size: number, etag: string): HeadersInit {
  // _headers only applies to static assets, so the Worker sets its own.
  return {
    "content-type": key.endsWith(".wasm") ? "application/wasm" : "application/octet-stream",
    "content-length": String(size),
    "cache-control": "public, max-age=31536000, immutable",
    etag,
    "cross-origin-resource-policy": "same-origin",
    "x-content-type-options": "nosniff",
  };
}

async function fromParts(req: Request, env: Env, ctx: ExecutionContext, key: string): Promise<Response | null> {
  const res = await env.ASSETS.fetch(new URL("/weights/manifest.json", req.url));
  if (!res.ok) return null;
  const entry = ((await res.json()) as Manifest)[key];
  if (!entry) return null;
  const h = headers(key, entry.size, `"${entry.sha256}"`);
  if (req.method === "HEAD") return new Response(null, { headers: h });
  const { readable, writable } = new FixedLengthStream(entry.size);
  const pump = async () => {
    for (const part of entry.parts) {
      const r = await env.ASSETS.fetch(new URL(`/weights/${part}`, req.url));
      if (!r.ok || !r.body) throw new Error(`missing part ${part}`);
      await r.body.pipeTo(writable, { preventClose: true });
    }
    await writable.close();
  };
  ctx.waitUntil(pump().catch((e) => writable.abort(e)));
  return new Response(readable, { headers: h });
}

export default {
  async fetch(req: Request, env: Env, ctx: ExecutionContext): Promise<Response> {
    const path = new URL(req.url).pathname;
    if (!path.startsWith("/models/")) return new Response(null, { status: 404 });
    if (req.method !== "GET" && req.method !== "HEAD") return new Response(null, { status: 405 });
    const key = decodeURIComponent(path.slice("/models/".length));

    const obj = await env.MODELS.get(key);
    if (obj) return new Response(req.method === "HEAD" ? null : obj.body, { headers: headers(key, obj.size, obj.httpEtag) });

    const parts = await fromParts(req, env, ctx, key);
    if (parts) return parts;

    // Name the key, so a missing or misnamed file is easy to spot.
    return new Response(`not found: ${key}\n`, { status: 404, headers: { "x-depth-light-missing": key } });
  },
};
