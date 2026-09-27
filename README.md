# depth-light

Drop in a photo and get its depth map and light map, then place 3D objects that the photo's light hits correctly as you move them. Everything runs in the browser: the photo never leaves the device.

The heavy work runs once per image. MoGe-2 estimates depth, normals and the lens in one pass, and an in-house fit recovers the light. After that it's all real-time rendering, so moving an object never reruns a model.

## Run it locally

```sh
npm install

# One-time: export the model (CPU is enough, about 2 minutes).
python -m venv .venv && . .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install "git+https://github.com/microsoft/MoGe.git" onnx onnxruntime onnxslim pillow
python scripts/export-moge.py --verify-image some-photo.jpg

npm run dev          # http://localhost:5173
npm test             # light-fit accuracy on the synthetic room, focal/shift port
npm run build        # typecheck and build into dist/
```

The export writes `models/moge-2-vits-normal-v1.onnx` (67 MiB). In dev and preview, Vite serves `/models/*` from `models/`, and takes the ONNX Runtime WebAssembly binary from `node_modules`, so no Cloudflare account is needed to work on the app.

URL switches, mainly for debugging:

| Parameter | Does |
| --- | --- |
| `?backend=wasm` | Skip WebGPU, for machines whose GPU driver misbehaves |
| `?tokens=1200` … `3600` | ViT tokens for MoGe-2. Default is 1800 on WebGPU and 1200 on WASM |
| `?debug` | Exposes the renderer as `window.stage` |

## Deploy

The site is static files plus one Worker for `/models/*`. The weights (67 MiB) and the ONNX Runtime WebGPU binary (25.5 MiB) are both over Cloudflare's 25 MiB per-asset limit, so they ship as 20 MiB parts in `public/weights/`. The Worker streams the parts back together at `/models/<name>`. R2 is checked first: if the `depth-light-models` bucket holds a file, that copy wins, otherwise the parts are used. No uploads are needed.

1. Create the bucket once: `npx wrangler r2 bucket create depth-light-models`. The binding needs it to exist, even empty.
2. In the Cloudflare dashboard, connect this repo in Workers Builds. Build command `npm run build`, deploy command `npx wrangler deploy`.

After a new model export (bump `MODEL_VERSION` in `src/config.ts` first) or an `onnxruntime-web` upgrade, run `node scripts/chunk-models.mjs` and commit `public/weights/`. To serve from R2 instead and keep the parts out of git, run `npm run models:upload` and delete `public/weights/`.

## Using it

| Control | Does |
| --- | --- |
| Open | Pick an image. Dropping one anywhere works too |
| Image / Depth / Light | Switch view. Keys 1, 2, 3 |
| Mono / Colour | Switch output. Key M |
| Export | Swaps the right of the top bar for Image, Depth, Light, HDRI, Data. Esc closes |
| Sphere / Cube / Chrome | Add an object at the next click |
| Key / Fill / Soft | Key strength, fill strength, shadow softness. Double-click resets |
| Probe ball | Drag to move the key light, past the rim to go behind; double-click resets |
| Object | Drag slides it along surfaces, Option-drag rotates, scroll or pinch scales, Backspace deletes |

Drop a `.glb` on the canvas to place it where it lands. Glb files keep their real size. Primitives start at about an eighth of the view height where they land, so they're usable on a tabletop and in a valley alike.

Exports render at the photo's full resolution:

- **Image**: the composite as PNG.
- **Depth**: 16-bit grey PNG, white is near: value = (far − z) / (far − near), with near and far in the Data file. Sky is 0.
- **Light**: the light map as PNG, meaning the scene as if every surface were white paint.
- **HDRI**: the fitted light as a 1024×512 equirectangular `.hdr`, levelled so +y is up, in three.js's equirect convention.
- **Data**: JSON with the SH coefficients (shading and three.js LightProbe form), the key, sources, field of view and depth range.

## How it works

```
once per image (worker)                       every frame (main thread)
MoGe-2 → focal + shift → metric geometry      local refit around each moved object
light fit: regions → SH ambient → key → sources   shadow map, reflections, contact shadow
                                              background, catcher, objects, grade
```

**Geometry.** The ONNX graph is MoGe-2's raw forward pass. `src/geometry/moge-post.ts` ports the Python `infer()` post-processing: a 1-D Levenberg–Marquardt solve for the z shift (with focal in closed form), then metric depth, points and intrinsics. `test/geometry.test.ts` checks it against the Python output on a real image. The input is sized on the model's token grid, because the exported graph resizes without antialiasing.

**Light.** A fit, not a model (`src/light/`):

1. Prepare: linear light; drop invalid, clipped, near-black and depth-edge pixels.
2. Group: k-means on chromaticity. Upward-facing pixels always form their own region. Each chroma region is then split into connected surfaces, and surfaces whose median albedo differs by over 15% under a first key fit become separate regions. That split is what separates a grey object from a white wall.
3. Key: for a fixed direction, the region scales and the key/ambient ratio have a closed form, so scoring a direction is one pass over the samples. The search covers 1,200 directions, polishes the best with Nelder–Mead, then reweights robustly (Cauchy on the key model's residuals) and polishes again.
4. Ambient: 9 SH coefficients plus region scales, alternating, using the key fit's robust weights.
5. Sources: clipped regions projected to 3D. One within 25° of the key becomes the key, placed at its real position; the others get strengths from a non-negative fit with a capped inverse-square falloff.
6. Scale: the largest near-neutral region that isn't the floor is treated as white paint (albedo 0.8), with grey world as the colour fallback.

The ambient is the SH fit minus the key's and sources' SH projections, so the key isn't counted twice.

**Rendering** (`src/render/stage.ts`). The scene mesh built from the point map is used three ways. It's a depth-only occluder in the object pass. It's a shadow catcher in the background pass, multiplying the photo by the fraction of light that survives: the key where an object's shadow falls, the fill where it touches the surface. And it's a photo-textured room for each object's reflection cube map. Each object gets its own ambient and key visibility from the local refit (`src/light/local.ts`). A grade pass then matches the photo's measured edge softness, black level and noise, and applies mono.

## Results so far

- **Synthetic room** (dark floor, white walls, 12 objects, three light directions): key error of 0.11–0.23° in every case, with dark and white floors, and with and without cast shadows. Without regions and the floor split it's off by more than 5°. The tests enforce under 1°.
- **Export**: the shipped file stores fp16 weights and computes in fp32. Against PyTorch, median depth error is 1e-4 and normals and masks are identical.
- **Speed** (headless Chromium, WASM, 4 threads, 1200 tokens): about 12 s per photo including the light fit, and about 3 s to reopen a photo from the geometry cache.
- **WebGPU vs WASM**: the same photo gives the same field of view (to 1e-5°), the same mask and the same key direction to five decimals. That was checked in headless Chromium, whose software GPU is far too slow to time.
- **Real photos**: sunlit outdoor scenes and tabletops give plausible keys. Flat-lit interiors mostly come out as ambient, and scenes with mixed lamps and windows can land on the wrong side. The probe ball and sliders are the fix.

## Differences from the plan

- **fp16**: `onnxconverter-common` and ORT's float16 converter both produced invalid graphs on this model (duplicate cast nodes on shared constants). The export instead stores weights in fp16 and casts them to fp32 on load. The download is the same size, accuracy is fp32, and one file serves WebGPU and WASM.
- **ORT runtime in R2**: its WebGPU build exceeds the asset limit, so it's served next to the weights under `/models/`.
- **Ambient** reaches objects as per-object SH uniforms rather than one scene `LightProbe`. The maths is the same, but three.js probes are per scene, and the local refit gives every object its own.
- **Sources** are drawn in the object shader with the fit's own falloff (capped at 4×), not with three.js `PointLight`. Unbounded inverse-square light blew out any object placed near a lamp.
- **Key refinement** uses the closed-form inner solve and a dense search described above, rather than a general nonlinear solve. Regions gained the albedo split, which was needed to stay under 1° once the synthetic room had grey objects.
- **Soft shadows** use 32 PCF taps in the catcher, because three.js's 5 taps dither without temporal AA. Objects also darken where they meet their surface.
- Labels are sentence case.

## Layout

```
public/_headers            cross-origin isolation, immutable assets
src/main.ts                app state and wiring
src/ui/                    labels, sliders, progress line, probe ball
src/geometry/              moge-post.ts (focal + shift), mesh.ts
src/light/                 sh, regions, ambient, refine, sources, local, fit
src/render/                stage (scene and passes), objects, shaders, export
src/workers/infer.ts       onnxruntime-web session, MoGe-2, light fit
src/lib/                   png16, hdr, cache, stats, linalg
src/sw.js                  offline service worker template
worker/index.ts            serves /models/* from R2
scripts/export-moge.py     ONNX export, fp16 weights, verification
scripts/upload-models.mjs  weights and ORT runtime to R2
test/                      synthetic room, light and geometry tests
```
