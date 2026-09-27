// Port of the post-processing in MoGe's Python `infer()`: recover focal length
// and z shift from the raw affine point map, then rebuild metric depth, points
// and intrinsics. Output is converted from OpenCV camera space (x right, y
// down, z forward) to three.js camera space (x right, y up, z backward).

export interface MogeRaw {
  width: number;
  height: number;
  /** (H, W, 3) affine point map as returned by forward(). */
  points: Float32Array;
  /** (H, W, 3) unit normals, OpenCV camera space. */
  normal: Float32Array;
  /** (H, W) soft mask in [0, 1]. */
  mask: Float32Array;
  metricScale: number;
}

export interface Geometry {
  width: number;
  height: number;
  /** Metric depth along the view axis, 0 where invalid. */
  depth: Float32Array;
  /** (H, W, 3) metric points, three.js camera space. */
  points: Float32Array;
  /** (H, W, 3) unit normals, three.js camera space, facing the camera. */
  normals: Float32Array;
  /** 1 where the model trusts the geometry (not sky, not infinitely far). */
  mask: Uint8Array;
  /** Focal lengths normalised by image width and height. */
  fx: number;
  fy: number;
  /** Vertical field of view in degrees. */
  fovY: number;
  metricScale: number;
}

/** Token grid the model uses for a given aspect ratio, as in forward(). */
export function tokenGrid(numTokens: number, aspect: number): { rows: number; cols: number } {
  return { rows: Math.round(Math.sqrt(numTokens / aspect)), cols: Math.round(Math.sqrt(numTokens * aspect)) };
}

/**
 * Input size for the model: height on the token grid (so the model's own
 * resize, which has no antialiasing in the ONNX graph, is a near no-op) and
 * width keeping the photo's aspect ratio exactly.
 */
export function inputSize(numTokens: number, width: number, height: number): { width: number; height: number } {
  const aspect = width / height;
  const h = tokenGrid(numTokens, aspect).rows * 14;
  return { width: Math.max(14, Math.round(h * aspect)), height: h };
}

/** Pixel-centre view-plane coordinates in units of the half diagonal. */
function viewPlaneUV(i: number, j: number, width: number, height: number): [number, number] {
  const aspect = width / height;
  const d = Math.sqrt(1 + aspect * aspect);
  const spanX = aspect / d;
  const spanY = 1 / d;
  return [spanX * ((2 * i + 1 - width) / width), spanY * ((2 * j + 1 - height) / height)];
}

/**
 * Solve min over (focal, shift) of |focal · xy / (z + shift) − uv|², on a
 * 64×64 nearest-neighbour subsample like the reference implementation. The
 * focal has a closed form for any shift, so this is a 1-D Levenberg–Marquardt
 * solve over the shift. Returns focal relative to the half diagonal.
 */
export function recoverFocalShift(
  points: Float32Array,
  mask: Float32Array | Uint8Array,
  width: number,
  height: number,
  size = 64,
): { focal: number; shift: number } {
  const xs: number[] = [], ys: number[] = [], zs: number[] = [], us: number[] = [], vs: number[] = [];
  for (let r = 0; r < size; r++) {
    const j = Math.floor((r * height) / size);
    for (let c = 0; c < size; c++) {
      const i = Math.floor((c * width) / size);
      const p = j * width + i;
      if (!(mask[p] > 0.5)) continue;
      const [u, v] = viewPlaneUV(i, j, width, height);
      xs.push(points[p * 3]);
      ys.push(points[p * 3 + 1]);
      zs.push(points[p * 3 + 2]);
      us.push(u);
      vs.push(v);
    }
  }
  const n = xs.length;
  if (n < 2) return { focal: 1, shift: 0 };

  let zMin = Infinity;
  for (const z of zs) zMin = Math.min(zMin, z);

  const focalFor = (s: number): number => {
    let num = 0, den = 0;
    for (let k = 0; k < n; k++) {
      const w = 1 / (zs[k] + s);
      const px = xs[k] * w, py = ys[k] * w;
      num += px * us[k] + py * vs[k];
      den += px * px + py * py;
    }
    return den > 0 ? num / den : 1;
  };
  const residuals = (s: number, out: Float64Array): number => {
    const f = focalFor(s);
    let cost = 0;
    for (let k = 0; k < n; k++) {
      const w = f / (zs[k] + s);
      const rx = xs[k] * w - us[k];
      const ry = ys[k] * w - vs[k];
      out[2 * k] = rx;
      out[2 * k + 1] = ry;
      cost += rx * rx + ry * ry;
    }
    return cost;
  };

  // Keep every z + shift positive; the optimum always satisfies this.
  const floor = -zMin + 1e-6 * (Math.abs(zMin) + 1);
  const r0 = new Float64Array(2 * n), r1 = new Float64Array(2 * n), r2 = new Float64Array(2 * n);
  let s = Math.max(0, floor + 1e-6);
  let cost = residuals(s, r0);
  let lambda = 1e-3;
  for (let it = 0; it < 100; it++) {
    const h = 1e-6 * (1 + Math.abs(s));
    residuals(s + h, r1);
    residuals(s - h, r2);
    let jtj = 0, jtr = 0;
    for (let k = 0; k < 2 * n; k++) {
      const jk = (r1[k] - r2[k]) / (2 * h);
      jtj += jk * jk;
      jtr += jk * r0[k];
    }
    if (jtj === 0) break;
    let improved = false;
    for (let tries = 0; tries < 20; tries++) {
      let next = s - jtr / (jtj * (1 + lambda));
      if (next <= floor) next = 0.5 * (s + floor);
      const c = residuals(next, r1);
      if (c < cost) {
        const rel = (cost - c) / Math.max(cost, 1e-30);
        s = next;
        cost = c;
        r0.set(r1);
        lambda = Math.max(lambda / 10, 1e-12);
        improved = true;
        if (rel < 1e-12) it = 100;
        break;
      }
      lambda *= 10;
    }
    if (!improved) break;
  }
  return { focal: focalFor(s), shift: s };
}

export function postprocess(raw: MogeRaw): Geometry {
  const { width: W, height: H, points: P, normal: N, mask: M, metricScale } = raw;
  const count = W * H;
  const { focal, shift } = recoverFocalShift(P, M, W, H);
  const aspect = W / H;
  const diag = Math.sqrt(1 + aspect * aspect);
  const fx = (focal / 2) * (diag / aspect);
  const fy = (focal / 2) * diag;

  const depth = new Float32Array(count);
  const points = new Float32Array(count * 3);
  const normals = new Float32Array(count * 3);
  const mask = new Uint8Array(count);
  for (let j = 0; j < H; j++) {
    const v = ((j + 0.5) / H - 0.5) / fy;
    for (let i = 0; i < W; i++) {
      const p = j * W + i;
      const z = (P[p * 3 + 2] + shift) * metricScale;
      const ok = M[p] > 0.5 && z > 0 && Number.isFinite(z);
      // Normals are kept everywhere (the sky still has a direction to face);
      // depth and points only where the mask trusts them.
      normals[p * 3] = N[p * 3];
      normals[p * 3 + 1] = -N[p * 3 + 1];
      normals[p * 3 + 2] = -N[p * 3 + 2];
      if (!ok) continue;
      mask[p] = 1;
      depth[p] = z;
      const u = ((i + 0.5) / W - 0.5) / fx;
      points[p * 3] = u * z;
      points[p * 3 + 1] = -v * z;
      points[p * 3 + 2] = -z;
    }
  }
  const fovY = (2 * Math.atan(0.5 / fy) * 180) / Math.PI;
  return { width: W, height: H, depth, points, normals, mask, fx, fy, fovY, metricScale };
}

/** Rebuild points from depth and intrinsics (used when loading from cache). */
export function pointsFromDepth(depth: Float32Array, width: number, height: number, fx: number, fy: number): Float32Array {
  const points = new Float32Array(width * height * 3);
  for (let j = 0; j < height; j++) {
    const v = ((j + 0.5) / height - 0.5) / fy;
    for (let i = 0; i < width; i++) {
      const p = j * width + i;
      const z = depth[p];
      if (!(z > 0)) continue;
      points[p * 3] = (((i + 0.5) / width - 0.5) / fx) * z;
      points[p * 3 + 1] = -v * z;
      points[p * 3 + 2] = -z;
    }
  }
  return points;
}
