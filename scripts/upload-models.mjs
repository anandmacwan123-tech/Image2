// Upload the model weights and the ONNX Runtime WebAssembly binary to R2.
// Both exceed Cloudflare's 25 MiB static asset limit, so the Worker serves
// them from the bucket under /models/.
//
//   npm run models:upload            uploads to the real bucket (--remote)
//   npm run models:upload -- --local writes to Wrangler's local simulator

import { spawnSync } from "node:child_process";
import { existsSync, readFileSync } from "node:fs";

const BUCKET = "depth-light-models";
const MODEL_VERSION = readFileSync("src/config.ts", "utf8").match(/MODEL_VERSION = "([^"]+)"/)[1];
const ORT_VERSION = JSON.parse(readFileSync("node_modules/onnxruntime-web/package.json", "utf8")).version;

const files = [
  { key: `moge-2-vits-normal-${MODEL_VERSION}.onnx`, file: `models/moge-2-vits-normal-${MODEL_VERSION}.onnx`, type: "application/octet-stream" },
  {
    key: `ort-wasm-simd-threaded.asyncify-${ORT_VERSION}.wasm`,
    file: "node_modules/onnxruntime-web/dist/ort-wasm-simd-threaded.asyncify.wasm",
    type: "application/wasm",
  },
];

const local = process.argv.includes("--local");
for (const { key, file, type } of files) {
  if (!existsSync(file)) {
    console.error(`missing ${file}${file.startsWith("models/") ? " (run: python scripts/export-moge.py)" : " (run: npm install)"}`);
    process.exit(1);
  }
  console.log(`${file} -> r2://${BUCKET}/${key}`);
  const args = ["wrangler", "r2", "object", "put", `${BUCKET}/${key}`, `--file=${file}`, `--content-type=${type}`, local ? "--local" : "--remote"];
  const r = spawnSync("npx", args, { stdio: "inherit" });
  if (r.status !== 0) process.exit(r.status ?? 1);
}
