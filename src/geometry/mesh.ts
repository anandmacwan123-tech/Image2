// Triangle mesh from the point map: one vertex per grid cell, triangles
// dropped where they would bridge a depth jump (their surface would run
// almost parallel to the view ray) or touch an invalid pixel.

import type { Geometry } from "./moge-post";

export interface SceneMesh {
  positions: Float32Array;
  normals: Float32Array;
  /** Texture coordinates with v = 0 at the top row. */
  uvs: Float32Array;
  index: Uint32Array;
  cols: number;
  rows: number;
}

export function buildSceneMesh(g: Geometry, longSide = 320): SceneMesh {
  const step = Math.max(1, Math.round(Math.max(g.width, g.height) / longSide));
  const cols = Math.floor((g.width - 1) / step) + 1;
  const rows = Math.floor((g.height - 1) / step) + 1;
  const positions = new Float32Array(cols * rows * 3);
  const normals = new Float32Array(cols * rows * 3);
  const uvs = new Float32Array(cols * rows * 2);
  const valid = new Uint8Array(cols * rows);
  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      const v = r * cols + c;
      const p = r * step * g.width + c * step;
      valid[v] = g.mask[p];
      for (let k = 0; k < 3; k++) {
        positions[v * 3 + k] = g.points[p * 3 + k];
        normals[v * 3 + k] = g.normals[p * 3 + k];
      }
      uvs[v * 2] = (c * step + 0.5) / g.width;
      uvs[v * 2 + 1] = (r * step + 0.5) / g.height;
    }
  }

  const index: number[] = [];
  const tri = (a: number, b: number, c: number) => {
    if (!valid[a] || !valid[b] || !valid[c]) return;
    const pa = a * 3, pb = b * 3, pc = c * 3;
    const za = -positions[pa + 2], zb = -positions[pb + 2], zc = -positions[pc + 2];
    const zmin = Math.min(za, zb, zc), zmax = Math.max(za, zb, zc);
    const jump = zmax / zmin - 1;
    if (jump > 0.4) return;
    if (jump > 0.03) {
      // Face normal against the ray to the centroid: near-perpendicular means
      // the triangle spans a silhouette rather than a surface.
      const ux = positions[pb] - positions[pa], uy = positions[pb + 1] - positions[pa + 1], uz = positions[pb + 2] - positions[pa + 2];
      const vx = positions[pc] - positions[pa], vy = positions[pc + 1] - positions[pa + 1], vz = positions[pc + 2] - positions[pa + 2];
      const nx = uy * vz - uz * vy, ny = uz * vx - ux * vz, nz = ux * vy - uy * vx;
      const cx = positions[pa] + positions[pb] + positions[pc];
      const cy = positions[pa + 1] + positions[pb + 1] + positions[pc + 1];
      const cz = positions[pa + 2] + positions[pb + 2] + positions[pc + 2];
      const cos = Math.abs(nx * cx + ny * cy + nz * cz) / (Math.hypot(nx, ny, nz) * Math.hypot(cx, cy, cz) || 1);
      if (cos < 0.12) return;
    }
    index.push(a, b, c);
  };
  for (let r = 0; r < rows - 1; r++) {
    for (let c = 0; c < cols - 1; c++) {
      const a = r * cols + c, b = a + 1, d = a + cols, e = d + 1;
      // Counter-clockwise seen from the camera (v grows downward).
      tri(a, d, b);
      tri(b, d, e);
    }
  }
  return { positions, normals, uvs, index: Uint32Array.from(index), cols, rows };
}

/** Surface point and normal under a pixel, from the working-resolution maps. */
export function surfaceAt(g: Geometry, u: number, v: number): { point: [number, number, number]; normal: [number, number, number] } | null {
  const x = Math.min(g.width - 1, Math.max(0, Math.floor(u * g.width)));
  const y = Math.min(g.height - 1, Math.max(0, Math.floor(v * g.height)));
  const p = y * g.width + x;
  if (!g.mask[p]) return null;
  // Average normals over a small window of pixels on the same surface.
  let nx = 0, ny = 0, nz = 0;
  const r = 3;
  const z0 = g.depth[p];
  for (let dy = -r; dy <= r; dy++) {
    for (let dx = -r; dx <= r; dx++) {
      const xx = x + dx, yy = y + dy;
      if (xx < 0 || yy < 0 || xx >= g.width || yy >= g.height) continue;
      const q = yy * g.width + xx;
      if (!g.mask[q] || Math.abs(g.depth[q] - z0) > 0.05 * z0) continue;
      nx += g.normals[q * 3];
      ny += g.normals[q * 3 + 1];
      nz += g.normals[q * 3 + 2];
    }
  }
  const len = Math.hypot(nx, ny, nz) || 1;
  return {
    point: [g.points[p * 3], g.points[p * 3 + 1], g.points[p * 3 + 2]],
    normal: [nx / len, ny / len, nz / len],
  };
}
