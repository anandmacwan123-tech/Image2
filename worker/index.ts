// Serves /models/* from R2. Everything else is a static asset and never
// reaches this Worker (see run_worker_first in wrangler.jsonc).

// Minimal slice of the Workers R2 types, so the app does not need
// @cloudflare/workers-types just for this file.
interface R2ObjectBody {
  body: ReadableStream;
  size: number;
  httpEtag: string;
}
interface R2Bucket {
  get(key: string): Promise<R2ObjectBody | null>;
}

export default {
  async fetch(req: Request, env: { MODELS: R2Bucket }): Promise<Response> {
    const path = new URL(req.url).pathname;
    if (!path.startsWith("/models/")) return new Response(null, { status: 404 });
    if (req.method !== "GET" && req.method !== "HEAD") return new Response(null, { status: 405 });
    const key = decodeURIComponent(path.slice("/models/".length));
    const obj = await env.MODELS.get(key);
    // Name the key, so a missing or misnamed upload is easy to spot.
    if (!obj) return new Response(`not in R2: ${key}\n`, { status: 404, headers: { "x-depth-light-missing": key } });
    // _headers only applies to static assets, so the Worker sets its own.
    return new Response(req.method === "HEAD" ? null : obj.body, {
      headers: {
        "content-type": key.endsWith(".wasm") ? "application/wasm" : "application/octet-stream",
        "content-length": String(obj.size),
        "cache-control": "public, max-age=31536000, immutable",
        etag: obj.httpEtag,
        "cross-origin-resource-policy": "same-origin",
        "x-content-type-options": "nosniff",
      },
    });
  },
};
