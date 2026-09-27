// Split the model weights and the ORT runtime into static assets under the
// 25 MiB Cloudflare limit. The Worker streams them back together at
// /models/<name> whenever R2 does not hold the file, so the site works with
// no bucket uploads. Rerun after a new export or an onnxruntime-web upgrade.
//
//   node scripts/chunk-models.mjs

import { createHash } from "node:crypto";
import { mkdirSync, readFileSync, readdirSync, rmSync, writeFileSync } from "node:fs";

const PART = 20 * 1024 * 1024;
const OUT = "public/weights";
const MODEL_VERSION = readFileSync("src/config.ts", "utf8").match(/MODEL_VERSION = "([^"]+)"/)[1];
const ORT_VERSION = JSON.parse(readFileSync("node_modules/onnxruntime-web/package.json", "utf8")).version;

const files = [
  { key: `moge-2-vits-normal-${MODEL_VERSION}.onnx`, file: `models/moge-2-vits-normal-${MODEL_VERSION}.onnx` },
  { key: `ort-wasm-simd-threaded.asyncify-${ORT_VERSION}.wasm`, file: "node_modules/onnxruntime-web/dist/ort-wasm-simd-threaded.asyncify.wasm" },
];

rmSync(OUT, { recursive: true, force: true });
mkdirSync(OUT, { recursive: true });
const manifest = {};
for (const { key, file } of files) {
  const bytes = readFileSync(file);
  const sha = createHash("sha256").update(bytes).digest("hex");
  const parts = [];
  for (let o = 0, i = 0; o < bytes.length; o += PART, i++) {
    const name = `${key}.${sha.slice(0, 12)}.part${i}`;
    writeFileSync(`${OUT}/${name}`, bytes.subarray(o, o + PART));
    parts.push(name);
  }
  manifest[key] = { size: bytes.length, sha256: sha, parts };
  console.log(`${key}: ${parts.length} parts, ${bytes.length} bytes`);
}
writeFileSync(`${OUT}/manifest.json`, JSON.stringify(manifest, null, 2) + "\n");
console.log(readdirSync(OUT).join("\n"));
