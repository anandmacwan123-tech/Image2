import { createHash } from "node:crypto";
import { createReadStream, existsSync, readFileSync, statSync } from "node:fs";
import { resolve } from "node:path";
import { defaultClientConditions, type Plugin } from "vite";
import { defineConfig } from "vitest/config";

// Cross-origin isolation lets the WASM fallback use threads. Production gets
// the same headers from public/_headers and worker/index.ts.
const isolation = {
  "Cross-Origin-Opener-Policy": "same-origin",
  "Cross-Origin-Embedder-Policy": "require-corp",
};

// In production the Cloudflare Worker streams /models/* from R2. In dev and
// preview this serves the same paths from ./models, falling back to the ORT
// runtime files in node_modules so a fresh checkout only needs the weights.
function localModels(): Plugin {
  const dirs = [resolve("models"), resolve("node_modules/onnxruntime-web/dist")];
  const handler = (req: { url?: string }, res: import("node:http").ServerResponse, next: () => void) => {
    const url = req.url ?? "";
    if (!url.startsWith("/models/")) return next();
    const name = decodeURIComponent(url.slice("/models/".length).split("?")[0]);
    if (name.includes("..") || name.includes("/")) return next();
    const candidates = dirs.map((d) => resolve(d, name));
    // ORT ships its runtime unversioned; we publish it with the version in the name.
    const ort = name.match(/^(ort-wasm-simd-threaded\.[a-z]+)-[\d.]+\.wasm$/);
    if (ort) candidates.push(resolve(dirs[1], `${ort[1]}.wasm`));
    const file = candidates.find((f) => existsSync(f) && statSync(f).isFile());
    if (!file) {
      res.statusCode = 404;
      res.end();
      return;
    }
    res.setHeader("content-type", name.endsWith(".wasm") ? "application/wasm" : "application/octet-stream");
    res.setHeader("content-length", String(statSync(file).size));
    res.setHeader("cache-control", "no-cache");
    for (const [k, v] of Object.entries(isolation)) res.setHeader(k, v);
    createReadStream(file).pipe(res);
  };
  return {
    name: "local-models",
    configureServer(server) {
      server.middlewares.use(handler);
    },
    configurePreviewServer(server) {
      server.middlewares.use(handler);
    },
  };
}

// Offline after first load: at build time, emit a service worker that
// precaches the app shell. Its version comes from the bundle's file names, so
// every deploy installs a fresh one and drops the old shell.
function serviceWorker(): Plugin {
  return {
    name: "service-worker",
    apply: "build",
    generateBundle(_, bundle) {
      const files = Object.keys(bundle)
        .filter((f) => !f.endsWith(".map") && f !== "index.html")
        .map((f) => `/${f}`);
      const version = createHash("sha256").update(files.sort().join("\n")).digest("hex").slice(0, 12);
      const template = readFileSync(resolve("src/sw.js"), "utf8");
      this.emitFile({
        type: "asset",
        fileName: "sw.js",
        source: template.replace("__VERSION__", version).replace("__SHELL__", JSON.stringify(["/", ...files])),
      });
    },
  };
}

export default defineConfig({
  plugins: [localModels(), serviceWorker()],
  // ORT's default entry bundles its 26 MB runtime as an asset, over the
  // Cloudflare per-file limit. The extern-wasm entry loads it from the URL
  // set in the worker instead (served from R2 under /models/).
  resolve: { conditions: ["onnxruntime-web-use-extern-wasm", ...defaultClientConditions] },
  server: { headers: isolation },
  preview: { headers: isolation },
  worker: { format: "es" },
  optimizeDeps: { exclude: ["onnxruntime-web"] },
  build: { target: "es2022", assetsInlineLimit: 0 },
  test: { include: ["test/**/*.test.ts"], testTimeout: 60000 },
});
