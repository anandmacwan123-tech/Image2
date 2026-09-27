// Find visible light sources: clipped regions such as windows and lamps.
// Each becomes a 3D position through the point map, or, where MoGe masks the
// pixels out (sky, anything infinitely far), a direction only.

import type { Geometry } from "../geometry/moge-post";
import type { Vec3 } from "./refine";

export interface Source {
  /** Metric position, three.js camera space; null for sky and far sources. */
  position: Vec3 | null;
  /** Unit direction of the source's centre from the camera. */
  direction: Vec3;
  /** Size in working-resolution pixels. */
  pixels: number;
  /** Approximate emitting area in m² (0 for sky sources). */
  area: number;
  /** Colour from the unclipped rim around the source, max channel 1. */
  rgb: Vec3;
}

export function findSources(clipped: Uint8Array, geometry: Geometry, linear: Float32Array, maxSources = 4): Source[] {
  const { width: W, height: H, mask, points, depth, fx, fy } = geometry;
  const minPixels = Math.max(12, Math.round(W * H * 0.0002));
  const label = new Int32Array(W * H).fill(-1);
  const stack: number[] = [];
  const comps: number[][] = [];
  for (let start = 0; start < W * H; start++) {
    if (!clipped[start] || label[start] >= 0) continue;
    const pix: number[] = [];
    label[start] = comps.length;
    stack.push(start);
    while (stack.length) {
      const p = stack.pop()!;
      pix.push(p);
      const x = p % W, y = (p / W) | 0;
      if (x > 0) visit(p - 1);
      if (x < W - 1) visit(p + 1);
      if (y > 0) visit(p - W);
      if (y < H - 1) visit(p + W);
    }
    comps.push(pix);
  }
  function visit(q: number) {
    if (clipped[q] && label[q] < 0) {
      label[q] = comps.length;
      stack.push(q);
    }
  }

  const sources: Source[] = [];
  for (const pix of comps.sort((a, b) => b.length - a.length).slice(0, maxSources)) {
    if (pix.length < minPixels) break;
    let cx = 0, cy = 0;
    const xs: number[] = [], ys: number[] = [], zs: number[] = [];
    let area = 0;
    for (const p of pix) {
      cx += p % W;
      cy += (p / W) | 0;
      if (mask[p]) {
        xs.push(points[p * 3]);
        ys.push(points[p * 3 + 1]);
        zs.push(points[p * 3 + 2]);
        // A pixel covers (z / focal_px)² m² facing the camera.
        area += (depth[p] / (fx * W)) * (depth[p] / (fy * H));
      }
    }
    cx /= pix.length;
    cy /= pix.length;
    const ray: Vec3 = [((cx + 0.5) / W - 0.5) / fx, -((cy + 0.5) / H - 0.5) / fy, -1];
    const len = Math.hypot(...ray);
    const direction: Vec3 = [ray[0] / len, ray[1] / len, ray[2] / len];
    const positional = xs.length >= pix.length * 0.5;
    const med = (a: number[]) => a.sort((u, v) => u - v)[a.length >> 1];
    sources.push({
      position: positional ? [med(xs), med(ys), med(zs)] : null,
      direction,
      pixels: pix.length,
      area: positional ? area : 0,
      rgb: rimColour(pix, clipped, linear, W, H),
    });
  }
  return sources;
}

/** Mean colour of unclipped pixels a few pixels outside the component. */
function rimColour(pix: number[], clipped: Uint8Array, linear: Float32Array, W: number, H: number): Vec3 {
  let r = 0, g = 0, b = 0, n = 0;
  const step = 3;
  for (const p of pix) {
    const x = p % W, y = (p / W) | 0;
    for (const [dx, dy] of [[step, 0], [-step, 0], [0, step], [0, -step]]) {
      const qx = x + dx, qy = y + dy;
      if (qx < 0 || qy < 0 || qx >= W || qy >= H) continue;
      const q = qy * W + qx;
      if (clipped[q]) continue;
      r += linear[q * 3];
      g += linear[q * 3 + 1];
      b += linear[q * 3 + 2];
      n++;
    }
  }
  const m = Math.max(r, g, b);
  return n > 0 && m > 0 ? [r / m, g / m, b / m] : [1, 1, 1];
}
