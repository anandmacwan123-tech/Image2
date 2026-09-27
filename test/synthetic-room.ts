// A synthetic room for the light-fit tests: dark floor, white walls and
// ceiling, 12 coloured objects, one distant light. Rendered by ray casting
// with exact normals and matte surfaces, then quantised to an 8-bit sRGB
// photo, which is the best case the plan describes.

import type { Geometry } from "../src/geometry/moge-post";

type V3 = [number, number, number];

export interface RoomOptions {
  width?: number;
  height?: number;
  /** Unit vector toward the light, three.js camera space. */
  light: V3;
  floorAlbedo?: V3;
  ambient?: number;
  key?: number;
  /** Tint of the light. */
  lightColour?: V3;
  /** Cast shadows from the objects onto the room. */
  shadows?: boolean;
}

export interface Room {
  rgba: Uint8ClampedArray;
  geometry: Geometry;
  /** Linear albedo per pixel, for inspecting the fit. */
  albedo: Float32Array;
  /** Scale from scene radiance to the 8-bit photo's linear values. */
  exposure: number;
}

interface Hit {
  t: number;
  n: V3;
  albedo: V3;
}

type Shape =
  | { kind: "plane"; n: V3; d: number; albedo: V3 }
  | { kind: "sphere"; c: V3; r: number; albedo: V3 }
  | { kind: "box"; min: V3; max: V3; albedo: V3 };

const FLOOR_Y = -1.4;

export function objects(): Shape[] {
  const onFloor = (x: number, z: number, r: number): V3 => [x, FLOOR_Y + r, z];
  return [
    { kind: "sphere", c: onFloor(-1.2, -3.2, 0.35), r: 0.35, albedo: [0.7, 0.1, 0.1] },
    { kind: "sphere", c: onFloor(0.2, -4.2, 0.45), r: 0.45, albedo: [0.1, 0.6, 0.15] },
    { kind: "sphere", c: onFloor(1.4, -3.0, 0.3), r: 0.3, albedo: [0.1, 0.2, 0.7] },
    { kind: "sphere", c: onFloor(-0.4, -2.6, 0.2), r: 0.2, albedo: [0.75, 0.7, 0.1] },
    { kind: "sphere", c: onFloor(1.9, -4.8, 0.4), r: 0.4, albedo: [0.6, 0.6, 0.6] },
    { kind: "sphere", c: [-1.8, 0.2, -4.5], r: 0.3, albedo: [0.7, 0.4, 0.1] },
    { kind: "box", min: [-2.3, FLOOR_Y, -5.5], max: [-1.5, FLOOR_Y + 0.9, -4.7], albedo: [0.15, 0.5, 0.6] },
    { kind: "box", min: [0.8, FLOOR_Y, -5.6], max: [1.6, FLOOR_Y + 0.5, -5.0], albedo: [0.55, 0.2, 0.55] },
    { kind: "box", min: [-0.9, FLOOR_Y, -5.0], max: [-0.3, FLOOR_Y + 1.2, -4.4], albedo: [0.8, 0.8, 0.78] },
    { kind: "box", min: [0.5, FLOOR_Y, -2.6], max: [0.9, FLOOR_Y + 0.4, -2.2], albedo: [0.35, 0.25, 0.15] },
    { kind: "box", min: [-2.5, 0.4, -3.4], max: [-2.2, 0.8, -2.8], albedo: [0.2, 0.2, 0.25] },
    { kind: "sphere", c: [1.2, 0.9, -5.4], r: 0.25, albedo: [0.9, 0.5, 0.6] },
  ];
}

function room(floor: V3): Shape[] {
  const white: V3 = [0.8, 0.8, 0.8];
  return [
    { kind: "plane", n: [0, 1, 0], d: -FLOOR_Y, albedo: floor },
    { kind: "plane", n: [0, -1, 0], d: 1.8, albedo: white },
    { kind: "plane", n: [1, 0, 0], d: 2.5, albedo: white },
    { kind: "plane", n: [-1, 0, 0], d: 2.5, albedo: white },
    { kind: "plane", n: [0, 0, 1], d: 6, albedo: white },
  ];
}

function intersect(shape: Shape, o: V3, d: V3): Hit | null {
  if (shape.kind === "plane") {
    // n·p + d = 0
    const den = shape.n[0] * d[0] + shape.n[1] * d[1] + shape.n[2] * d[2];
    if (den >= -1e-9) return null;
    const t = -(shape.n[0] * o[0] + shape.n[1] * o[1] + shape.n[2] * o[2] + shape.d) / den;
    return t > 1e-6 ? { t, n: shape.n, albedo: shape.albedo } : null;
  }
  if (shape.kind === "sphere") {
    const oc: V3 = [o[0] - shape.c[0], o[1] - shape.c[1], o[2] - shape.c[2]];
    const b = oc[0] * d[0] + oc[1] * d[1] + oc[2] * d[2];
    const c = oc[0] * oc[0] + oc[1] * oc[1] + oc[2] * oc[2] - shape.r * shape.r;
    const disc = b * b - c;
    if (disc < 0) return null;
    const t = -b - Math.sqrt(disc);
    if (t <= 1e-6) return null;
    const p = [o[0] + t * d[0], o[1] + t * d[1], o[2] + t * d[2]];
    return { t, n: [(p[0] - shape.c[0]) / shape.r, (p[1] - shape.c[1]) / shape.r, (p[2] - shape.c[2]) / shape.r], albedo: shape.albedo };
  }
  let tmin = -Infinity, tmax = Infinity, axis = 0, sign = 1;
  for (let k = 0; k < 3; k++) {
    if (Math.abs(d[k]) < 1e-12) {
      if (o[k] < shape.min[k] || o[k] > shape.max[k]) return null;
      continue;
    }
    let t1 = (shape.min[k] - o[k]) / d[k], t2 = (shape.max[k] - o[k]) / d[k];
    let s = -1;
    if (t1 > t2) { [t1, t2] = [t2, t1]; s = 1; }
    if (t1 > tmin) { tmin = t1; axis = k; sign = s; }
    tmax = Math.min(tmax, t2);
  }
  if (tmin > tmax || tmin <= 1e-6) return null;
  const n: V3 = [0, 0, 0];
  n[axis] = sign;
  return { t: tmin, n, albedo: shape.albedo };
}

function trace(shapes: Shape[], o: V3, d: V3): Hit | null {
  let best: Hit | null = null;
  for (const s of shapes) {
    const h = intersect(s, o, d);
    if (h && (!best || h.t < best.t)) best = h;
  }
  return best;
}

export function renderRoom(opts: RoomOptions): Room {
  const W = opts.width ?? 320, H = opts.height ?? 240;
  const fovY = (60 * Math.PI) / 180;
  const fy = 0.5 / Math.tan(fovY / 2);
  const fx = (fy * H) / W;
  const floor = opts.floorAlbedo ?? [0.08, 0.08, 0.08];
  const obj = objects();
  const shapes = [...room(floor), ...obj];
  const a = opts.ambient ?? 0.25, k = opts.key ?? 0.9;
  const tint = opts.lightColour ?? [1, 1, 1];
  const L = opts.light;

  const depth = new Float32Array(W * H), points = new Float32Array(W * H * 3), normals = new Float32Array(W * H * 3);
  const mask = new Uint8Array(W * H), albedo = new Float32Array(W * H * 3), lin = new Float32Array(W * H * 3);
  let max = 0;
  for (let j = 0; j < H; j++) {
    for (let i = 0; i < W; i++) {
      const p = j * W + i;
      const dir: V3 = [((i + 0.5) / W - 0.5) / fx, -((j + 0.5) / H - 0.5) / fy, -1];
      const len = Math.hypot(...dir);
      const d: V3 = [dir[0] / len, dir[1] / len, dir[2] / len];
      const hit = trace(shapes, [0, 0, 0], d)!;
      const pos: V3 = [d[0] * hit.t, d[1] * hit.t, d[2] * hit.t];
      mask[p] = 1;
      depth[p] = -pos[2];
      points.set(pos, p * 3);
      normals.set(hit.n, p * 3);
      albedo.set(hit.albedo, p * 3);
      let cos = Math.max(0, hit.n[0] * L[0] + hit.n[1] * L[1] + hit.n[2] * L[2]);
      if (opts.shadows && cos > 0) {
        const o: V3 = [pos[0] + hit.n[0] * 1e-4, pos[1] + hit.n[1] * 1e-4, pos[2] + hit.n[2] * 1e-4];
        if (obj.some((s) => intersect(s, o, L))) cos = 0;
      }
      for (let c = 0; c < 3; c++) {
        const v = hit.albedo[c] * tint[c] * (a + k * cos);
        lin[p * 3 + c] = v;
        max = Math.max(max, v);
      }
    }
  }
  const exposure = 0.92 / max;
  const rgba = new Uint8ClampedArray(W * H * 4);
  for (let p = 0; p < W * H; p++) {
    for (let c = 0; c < 3; c++) {
      const v = lin[p * 3 + c] * exposure;
      const s = v <= 0.0031308 ? v * 12.92 : 1.055 * Math.pow(v, 1 / 2.4) - 0.055;
      rgba[p * 4 + c] = Math.round(s * 255);
    }
    rgba[p * 4 + 3] = 255;
  }
  const geometry: Geometry = {
    width: W,
    height: H,
    depth,
    points,
    normals,
    mask,
    fx,
    fy,
    fovY: (fovY * 180) / Math.PI,
    metricScale: 1,
  };
  return { rgba, geometry, albedo, exposure };
}

export function normalize(v: V3): V3 {
  const l = Math.hypot(...v);
  return [v[0] / l, v[1] / l, v[2] / l];
}
