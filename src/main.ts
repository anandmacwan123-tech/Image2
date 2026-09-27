// App state and wiring: the worker does the heavy work once per image, the
// Stage renders, and every interaction after that is real time.

import "@fontsource/ibm-plex-mono/400.css";
import "./style.css";
import { MAX_PHOTO_SIDE } from "./config";
import { dataJSON, depthPNG, download, hdri, pngFromRGBA } from "./render/export";
import { loadGLB, type Kind, type Placed, type View } from "./render/objects";
import { Stage } from "./render/stage";
import { ProgressLine, press, slider } from "./ui/controls";
import { Probe } from "./ui/probe";
import type { FromWorker, ToWorker } from "./workers/protocol";

const $ = <T extends Element = HTMLElement>(sel: string) => document.querySelector<T>(sel)!;
const $$ = <T extends Element = HTMLElement>(sel: string) => Array.from(document.querySelectorAll<T>(sel));

const canvas = $<HTMLCanvasElement>("#view");
const probeCanvas = $<HTMLCanvasElement>("#probe");
const stageEl = $("#stage");
const dropLabel = $("#drop");
const fileInput = $<HTMLInputElement>("#file");
const progress = new ProgressLine($("#progress"));

const stage = new Stage(canvas);
if (new URLSearchParams(location.search).has("debug")) Object.assign(window, { stage });
const worker = new Worker(new URL("./workers/infer.ts", import.meta.url), { type: "module" });
const send = (msg: ToWorker, transfer: Transferable[] = []) => worker.postMessage(msg, transfer);

const state = {
  request: 0,
  pending: null as { id: number; bitmap: ImageBitmap; name: string } | null,
  name: "image",
  adding: null as Kind | null,
  selected: null as Placed | null,
  drag: null as { o: Placed; rotate: boolean; x: number; du: number; dv: number } | null,
  busy: false,
};

// ---- Worker ---------------------------------------------------------------

let creep = 0;
function startCreep(from: number, to: number): void {
  clearInterval(creep);
  let f = from;
  progress.set(f);
  creep = window.setInterval(() => {
    f += (to - f) * 0.04;
    progress.set(f);
  }, 100);
}

worker.onmessage = (ev: MessageEvent<FromWorker>) => {
  const msg = ev.data;
  if (msg.type === "progress") {
    if (msg.phase === "download") progress.set(msg.fraction * (state.pending ? 0.6 : 1));
    else if (msg.phase === "infer") startCreep(0.65, 0.95);
    else startCreep(0.95, 0.99);
  } else if (msg.type === "ready") {
    if (!state.pending) progress.done();
  } else if (msg.type === "result") {
    if (!state.pending || msg.id !== state.pending.id) return;
    clearInterval(creep);
    const { bitmap, name } = state.pending;
    state.pending = null;
    state.name = name;
    state.selected = null;
    stage.load(bitmap, msg.analysis);
    showApp();
    layout();
    drawProbe();
    progress.done();
    console.info(`depth-light: ${msg.analysis.backend}, ${Math.round(msg.analysis.ms)} ms, fov ${msg.analysis.geometry.fovY.toFixed(1)}°`, msg.analysis.fit.info);
  } else if (msg.type === "error") {
    clearInterval(creep);
    progress.done();
    if (msg.id !== undefined && state.pending?.id === msg.id) {
      state.pending.bitmap.close();
      state.pending = null;
    }
    console.error("depth-light:", msg.message);
    if (!stage.photo) dropLabel.textContent = /download|fetch|404/i.test(msg.message) ? "Model unavailable" : "Could not read image";
  }
};
// ?backend=wasm skips WebGPU, for machines whose GPU driver misbehaves;
// ?tokens=1200…3600 trades detail for speed.
const params = new URLSearchParams(location.search);
const tokens = Number(params.get("tokens"));
send({
  type: "load",
  prefer: params.get("backend") === "wasm" ? "wasm" : "webgpu",
  tokens: tokens >= 1200 && tokens <= 3600 ? Math.round(tokens) : undefined,
});

if (import.meta.env.PROD && "serviceWorker" in navigator) {
  navigator.serviceWorker.register("/sw.js").catch(() => {
    // Offline support is a bonus; the app works without it.
  });
}

// ---- Opening files ----------------------------------------------------------

function isModel(f: File): boolean {
  return /\.glb$/i.test(f.name) || f.type === "model/gltf-binary";
}

async function openImage(file: File): Promise<void> {
  const id = ++state.request;
  const bytes = await file.arrayBuffer();
  let bitmap = await createImageBitmap(file, { imageOrientation: "from-image" });
  const gl = stage.renderer.getContext();
  const limit = Math.min(MAX_PHOTO_SIDE, gl.getParameter(gl.MAX_TEXTURE_SIZE));
  if (Math.max(bitmap.width, bitmap.height) > limit) {
    const s = limit / Math.max(bitmap.width, bitmap.height);
    const w = Math.round(bitmap.width * s), h = Math.round(bitmap.height * s);
    bitmap.close();
    bitmap = await createImageBitmap(file, { imageOrientation: "from-image", resizeWidth: w, resizeHeight: h, resizeQuality: "high" });
  }
  state.pending?.bitmap.close();
  state.pending = { id, bitmap, name: file.name.replace(/\.[^.]+$/, "") || "image" };
  dropLabel.textContent = "Drop image";
  progress.set(0.02);
  send({ type: "analyze", id, bytes, mime: file.type || "image/jpeg" }, [bytes]);
}

async function addModel(file: File, u = 0.5, v = 0.6): Promise<void> {
  if (!stage.photo) return;
  const model = await loadGLB(await file.arrayBuffer());
  const o = stage.addObject("model", u, v, model) ?? stage.addObject("model", 0.5, 0.75, model);
  if (o) state.selected = o;
}

function openFile(file: File | undefined, u?: number, v?: number): void {
  if (!file) return;
  if (isModel(file)) void addModel(file, u, v);
  else if (file.type.startsWith("image/") || /\.(jpe?g|png|webp|avif|heic|gif|bmp)$/i.test(file.name)) void openImage(file);
}

fileInput.addEventListener("change", () => {
  openFile(fileInput.files?.[0]);
  fileInput.value = "";
});
dropLabel.addEventListener("click", () => fileInput.click());

window.addEventListener("dragover", (ev) => {
  ev.preventDefault();
  document.body.classList.add("dragging");
});
window.addEventListener("dragleave", (ev) => {
  if (!ev.relatedTarget) document.body.classList.remove("dragging");
});
window.addEventListener("drop", (ev) => {
  ev.preventDefault();
  document.body.classList.remove("dragging");
  const file = ev.dataTransfer?.files[0];
  const r = canvas.getBoundingClientRect();
  const u = (ev.clientX - r.left) / r.width, v = (ev.clientY - r.top) / r.height;
  const inside = u >= 0 && u <= 1 && v >= 0 && v <= 1;
  openFile(file, inside ? u : undefined, inside ? v : undefined);
});

// ---- Layout and frame loop --------------------------------------------------

function showApp(): void {
  $("#top").hidden = false;
  $("#bottom").hidden = false;
  probeCanvas.hidden = false;
  dropLabel.hidden = true;
}

function layout(): void {
  const photo = stage.photo;
  if (!photo) return;
  const r = stageEl.getBoundingClientRect();
  const aspect = photo.width / photo.height;
  let w = r.width, h = w / aspect;
  if (h > r.height) {
    h = r.height;
    w = h * aspect;
  }
  canvas.style.width = `${Math.floor(w)}px`;
  canvas.style.height = `${Math.floor(h)}px`;
  const dpr = Math.min(2, window.devicePixelRatio || 1);
  stage.resize(Math.max(1, Math.round(Math.floor(w) * dpr)), Math.max(1, Math.round(Math.floor(h) * dpr)));
  // The probe sits over the canvas's bottom-right corner, not the letterbox.
  const c = canvas.getBoundingClientRect();
  probeCanvas.style.right = `${r.right - c.right + 16}px`;
  probeCanvas.style.bottom = `${r.bottom - c.bottom + 16}px`;
}
window.addEventListener("resize", layout);

function tick(): void {
  stage.frame();
  requestAnimationFrame(tick);
}
requestAnimationFrame(tick);

// ---- Controls -----------------------------------------------------------------

function setView(view: View): void {
  stage.setView(view);
  press($$("[data-view]"), (b) => b.dataset.view === view);
}
for (const b of $$("[data-view]")) b.addEventListener("click", () => setView(b.dataset.view as View));

function setMono(mono: boolean): void {
  stage.setMono(mono);
  press($$("[data-mode]"), (b) => (b.dataset.mode === "mono") === mono);
  drawProbe();
}
for (const b of $$("[data-mode]")) b.addEventListener("click", () => setMono(b.dataset.mode === "mono"));

$("[data-action=open]").addEventListener("click", () => fileInput.click());

function setAdding(kind: Kind | null): void {
  state.adding = kind;
  press($$("[data-add]"), (b) => b.dataset.add === kind);
  canvas.classList.toggle("placing", !!kind);
}
for (const b of $$("[data-add]")) b.addEventListener("click", () => setAdding(state.adding === b.dataset.add ? null : (b.dataset.add as Kind)));

function setExportOpen(open: boolean): void {
  $("[data-group=settings]").hidden = open;
  $("[data-group=export]").hidden = !open;
}
$("[data-action=export]").addEventListener("click", (ev) => {
  ev.stopPropagation();
  setExportOpen(true);
});
document.addEventListener("click", (ev) => {
  if (!(ev.target as HTMLElement).closest("[data-group=export]")) setExportOpen(false);
});
for (const b of $$("[data-export]")) {
  b.addEventListener("click", async () => {
    if (state.busy) return;
    state.busy = true;
    b.setAttribute("aria-pressed", "true");
    startCreep(0.1, 0.9);
    try {
      await exportAs(b.dataset.export!);
    } catch (e) {
      console.error("depth-light: export failed", e);
    } finally {
      clearInterval(creep);
      progress.done();
      b.setAttribute("aria-pressed", "false");
      state.busy = false;
      setExportOpen(false);
    }
  });
}

const sliders = {
  key: slider($("[data-slider=key]"), 0, 2, 1, (v) => onSlider({ key: v })),
  fill: slider($("[data-slider=fill]"), 0, 2, 1, (v) => onSlider({ fill: v })),
  soft: slider($("[data-slider=soft]"), 0, 1, 0.35, (v) => onSlider({ soft: v })),
};
void sliders;
function onSlider(c: { key?: number; fill?: number; soft?: number }): void {
  stage.setControls(c);
  drawProbe();
}

const probe = new Probe(
  probeCanvas,
  (dir) => {
    stage.setControls({ dir });
    drawProbe();
  },
  () => {
    stage.setControls({ dir: null });
    drawProbe();
  },
);

function drawProbe(): void {
  const m = stage.model;
  if (!m || !stage.fit) return;
  const d = stage.keyDirection();
  probe.draw({
    ambient: m.ambient,
    key: m.key.strength.map((k) => k * stage.controls.key),
    fill: stage.controls.fill,
    dir: [d.x, d.y, d.z],
    white: stage.fit.white,
  });
}

// ---- Placing and moving objects -------------------------------------------------

function uvOf(ev: { clientX: number; clientY: number }): { u: number; v: number } {
  const r = canvas.getBoundingClientRect();
  return { u: (ev.clientX - r.left) / r.width, v: (ev.clientY - r.top) / r.height };
}

// Active touches, for pinch-to-scale on phones (the equivalent of scrolling).
const touches = new Map<number, { x: number; y: number }>();
let pinch: { o: Placed; distance: number } | null = null;
const spread = () => {
  const [a, b] = [...touches.values()];
  return Math.hypot(a.x - b.x, a.y - b.y);
};

canvas.addEventListener("pointerdown", (ev) => {
  touches.set(ev.pointerId, { x: ev.clientX, y: ev.clientY });
  if (touches.size === 2 && state.selected) {
    state.drag = null;
    pinch = { o: state.selected, distance: spread() };
    return;
  }
  const { u, v } = uvOf(ev);
  if (state.adding) {
    const o = stage.addObject(state.adding, u, v);
    if (o) {
      state.selected = o;
      setAdding(null);
    }
    return;
  }
  const o = stage.pick(u, v);
  state.selected = o;
  if (!o) return;
  const at = stage.project(o);
  state.drag = { o, rotate: ev.altKey, x: ev.clientX, du: at.x - u, dv: at.y - v };
  canvas.setPointerCapture(ev.pointerId);
  canvas.style.cursor = "grabbing";
});

canvas.addEventListener("pointermove", (ev) => {
  if (touches.has(ev.pointerId)) touches.set(ev.pointerId, { x: ev.clientX, y: ev.clientY });
  if (pinch && touches.size === 2) {
    const d = spread();
    stage.scaleObject(pinch.o, d / Math.max(pinch.distance, 1));
    pinch.distance = d;
    return;
  }
  const d = state.drag;
  if (!d) return;
  if (d.rotate) {
    stage.rotateObject(d.o, (ev.clientX - d.x) * 0.01);
    d.x = ev.clientX;
  } else {
    const { u, v } = uvOf(ev);
    stage.moveObject(d.o, u + d.du, v + d.dv);
  }
});

const endDrag = (ev: PointerEvent) => {
  touches.delete(ev.pointerId);
  if (touches.size < 2) pinch = null;
  state.drag = null;
  canvas.style.cursor = "";
};
canvas.addEventListener("pointerup", endDrag);
canvas.addEventListener("pointercancel", endDrag);

canvas.addEventListener(
  "wheel",
  (ev) => {
    const { u, v } = uvOf(ev);
    const o = stage.pick(u, v) ?? state.selected;
    if (!o) return;
    ev.preventDefault();
    state.selected = o;
    stage.scaleObject(o, Math.exp(-ev.deltaY * 0.0015));
  },
  { passive: false },
);

window.addEventListener("keydown", (ev) => {
  if (ev.metaKey || ev.ctrlKey || !stage.photo) return;
  if (ev.key === "1") setView("image");
  else if (ev.key === "2") setView("depth");
  else if (ev.key === "3") setView("light");
  else if (ev.key === "m" || ev.key === "M") setMono(!stage.mono);
  else if ((ev.key === "Backspace" || ev.key === "Delete") && state.selected) {
    stage.removeObject(state.selected);
    state.selected = null;
  } else if (ev.key === "Escape") {
    setAdding(null);
    setExportOpen(false);
  } else return;
  ev.preventDefault();
});

// ---- Export ---------------------------------------------------------------------------

async function exportAs(kind: string): Promise<void> {
  const photo = stage.photo, fit = stage.fit, g = stage.geometry, m = stage.model;
  if (!photo || !fit || !g || !m) return;
  // Let the progress line paint before the heavy work.
  await new Promise((r) => requestAnimationFrame(() => setTimeout(r, 0)));
  const s = Math.min(1, stage.maxExportSide() / Math.max(photo.width, photo.height));
  const w = Math.round(photo.width * s), h = Math.round(photo.height * s);
  const base = state.name;
  const mode = stage.mono ? "mono" : "colour";
  const dir = stage.keyDirection();
  const settings = { key: stage.controls.key, fill: stage.controls.fill, keyDir: [dir.x, dir.y, dir.z] as [number, number, number] };
  if (kind === "image") {
    download(await pngFromRGBA(stage.renderPixels(w, h), w, h), `${base}-image.png`);
  } else if (kind === "light") {
    download(await pngFromRGBA(stage.renderPixels(w, h, { view: "light", objects: false }), w, h), `${base}-light.png`);
  } else if (kind === "depth") {
    const { near, far } = stage.depthRange;
    download(new Blob([depthPNG(g, w, h, near, far)], { type: "image/png" }), `${base}-depth.png`);
  } else if (kind === "hdri") {
    download(new Blob([hdri(fit, m, settings)], { type: "image/vnd.radiance" }), `${base}-light.hdr`);
  } else if (kind === "data") {
    const json = dataJSON(fit, m, g, { width: photo.width, height: photo.height }, stage.depthRange, { ...settings, soft: stage.controls.soft, overridden: !!stage.controls.dir }, mode);
    download(new Blob([json], { type: "application/json" }), `${base}-light.json`);
  }
}
