// Group pixels into regions that share an albedo, so only brightness changes
// within a region count as light. Regions come from chromaticity (colour with
// brightness removed), except upward-facing pixels, which always form their
// own region: a dark floor otherwise reads as a shadowed floor.

import { rng } from "../lib/linalg";

export interface Samples {
  count: number;
  /** Linear RGB, 3 per sample. */
  rgb: Float32Array;
  /** Unit normals, three.js camera space, 3 per sample. */
  normal: Float32Array;
  /** Metric positions, three.js camera space, 3 per sample. */
  position: Float32Array;
  /** Pixel index in the working-resolution image. */
  pixel: Uint32Array;
}

export interface Regions {
  /** Region id per sample. */
  id: Uint16Array;
  count: number;
  /** Id of the upward-facing region, or -1 if there is none. */
  floor: number;
  /** Mean linear RGB per region. */
  meanRgb: Float32Array;
  /** Sample count per region. */
  size: Uint32Array;
  /** Chroma region (before any albedo split) for any linear colour and normal. */
  classify: (r: number, g: number, b: number, nx: number, ny: number, nz: number) => number;
}

/** cos 25°: pixels whose normal is within 25° of up count as floor. */
export const FLOOR_COS = Math.cos((25 * Math.PI) / 180);

/**
 * Estimate "up" from the normals: the mean of the normals that point roughly
 * along camera +y, refined once. Falls back to camera +y.
 */
export function estimateUp(normal: Float32Array, count: number): [number, number, number] {
  let up: [number, number, number] = [0, 1, 0];
  for (const threshold of [0.8, 0.94]) {
    let x = 0, y = 0, z = 0, n = 0;
    for (let i = 0; i < count; i++) {
      const nx = normal[i * 3], ny = normal[i * 3 + 1], nz = normal[i * 3 + 2];
      if (nx * up[0] + ny * up[1] + nz * up[2] > threshold) {
        x += nx; y += ny; z += nz; n++;
      }
    }
    if (n < Math.max(50, count * 0.01)) break;
    const len = Math.hypot(x, y, z);
    up = [x / len, y / len, z / len];
  }
  return up;
}

function chroma(rgb: Float32Array, i: number): [number, number] {
  const r = rgb[i * 3], g = rgb[i * 3 + 1], b = rgb[i * 3 + 2];
  const s = r + g + b + 1e-6;
  return [r / s, g / s];
}

/**
 * k-means on (r, g) chromaticity with k-means++ seeding, then small clusters
 * merge into their nearest neighbour. Deterministic for a given input.
 */
export function groupRegions(samples: Samples, up: [number, number, number], k = 12, minFraction = 0.002): Regions {
  const { count, rgb, normal } = samples;
  const isFloor = new Uint8Array(count);
  const rest: number[] = [];
  for (let i = 0; i < count; i++) {
    const d = normal[i * 3] * up[0] + normal[i * 3 + 1] * up[1] + normal[i * 3 + 2] * up[2];
    if (d > FLOOR_COS) isFloor[i] = 1;
    else rest.push(i);
  }

  const rand = rng(1234);
  const cr = new Float32Array(count * 2);
  for (let i = 0; i < count; i++) {
    const [a, b] = chroma(rgb, i);
    cr[i * 2] = a;
    cr[i * 2 + 1] = b;
  }

  // Train on a subsample, assign everything.
  const train = rest.length > 20000 ? Array.from({ length: 20000 }, () => rest[Math.floor(rand() * rest.length)]) : rest;
  const kk = Math.max(1, Math.min(k, train.length));
  const centers = new Float64Array(kk * 2);
  if (train.length > 0) {
    const first = train[Math.floor(rand() * train.length)];
    centers[0] = cr[first * 2];
    centers[1] = cr[first * 2 + 1];
    const d2 = new Float64Array(train.length).fill(Infinity);
    for (let c = 1; c < kk; c++) {
      let total = 0;
      for (let t = 0; t < train.length; t++) {
        const i = train[t];
        const dx = cr[i * 2] - centers[(c - 1) * 2], dy = cr[i * 2 + 1] - centers[(c - 1) * 2 + 1];
        d2[t] = Math.min(d2[t], dx * dx + dy * dy);
        total += d2[t];
      }
      let pick = rand() * total, chosen = train[train.length - 1];
      for (let t = 0; t < train.length; t++) {
        pick -= d2[t];
        if (pick <= 0) { chosen = train[t]; break; }
      }
      centers[c * 2] = cr[chosen * 2];
      centers[c * 2 + 1] = cr[chosen * 2 + 1];
    }
    const sums = new Float64Array(kk * 3);
    for (let it = 0; it < 25; it++) {
      sums.fill(0);
      for (const i of train) {
        const c = nearest(centers, kk, cr[i * 2], cr[i * 2 + 1]);
        sums[c * 3] += cr[i * 2];
        sums[c * 3 + 1] += cr[i * 2 + 1];
        sums[c * 3 + 2]++;
      }
      for (let c = 0; c < kk; c++) {
        if (sums[c * 3 + 2] > 0) {
          centers[c * 2] = sums[c * 3] / sums[c * 3 + 2];
          centers[c * 2 + 1] = sums[c * 3 + 1] / sums[c * 3 + 2];
        }
      }
    }
  }

  // Assign, then fold clusters below minFraction into their nearest neighbour.
  const cluster = new Int32Array(count).fill(-1);
  const sizes = new Float64Array(kk);
  for (const i of rest) {
    const c = nearest(centers, kk, cr[i * 2], cr[i * 2 + 1]);
    cluster[i] = c;
    sizes[c]++;
  }
  const alive = new Uint8Array(kk);
  for (let c = 0; c < kk; c++) alive[c] = sizes[c] >= Math.max(1, minFraction * rest.length) ? 1 : 0;
  if (!alive.some((a) => a) && kk > 0) alive[sizes.indexOf(Math.max(...sizes))] = 1;
  const remap = new Int32Array(kk).fill(-1);
  let next = 0;
  for (let c = 0; c < kk; c++) if (alive[c]) remap[c] = next++;
  for (let c = 0; c < kk; c++) {
    if (alive[c]) continue;
    let best = -1, bestD = Infinity;
    for (let o = 0; o < kk; o++) {
      if (!alive[o]) continue;
      const dx = centers[c * 2] - centers[o * 2], dy = centers[c * 2 + 1] - centers[o * 2 + 1];
      const d = dx * dx + dy * dy;
      if (d < bestD) { bestD = d; best = o; }
    }
    remap[c] = remap[best];
  }

  const hasFloor = count - rest.length > 0;
  const floor = hasFloor ? next : -1;
  const regionCount = next + (hasFloor ? 1 : 0);
  const id = new Uint16Array(count);
  for (let i = 0; i < count; i++) id[i] = isFloor[i] ? floor : remap[cluster[i]];

  const classify = (r: number, g: number, b: number, nx: number, ny: number, nz: number): number => {
    if (hasFloor && nx * up[0] + ny * up[1] + nz * up[2] > FLOOR_COS) return floor;
    const s = r + g + b + 1e-6;
    return remap[nearest(centers, kk, r / s, g / s)];
  };
  const { meanRgb, size } = regionStats(id, regionCount, rgb);
  return { id, count: regionCount, floor, meanRgb, size, classify };
}

function regionStats(id: Uint16Array, regionCount: number, rgb: Float32Array): { meanRgb: Float32Array; size: Uint32Array } {
  const meanRgb = new Float32Array(regionCount * 3);
  const size = new Uint32Array(regionCount);
  for (let i = 0; i < id.length; i++) {
    const r = id[i];
    size[r]++;
    meanRgb[r * 3] += rgb[i * 3];
    meanRgb[r * 3 + 1] += rgb[i * 3 + 1];
    meanRgb[r * 3 + 2] += rgb[i * 3 + 2];
  }
  for (let r = 0; r < regionCount; r++) {
    if (size[r] === 0) continue;
    meanRgb[r * 3] /= size[r];
    meanRgb[r * 3 + 1] /= size[r];
    meanRgb[r * 3 + 2] /= size[r];
  }
  return { meanRgb, size };
}

/** 4-connected components of a label image; -1 marks excluded pixels. */
export function components(label: Int32Array, width: number, height: number): Int32Array {
  const comp = new Int32Array(label.length).fill(-1);
  const stack: number[] = [];
  let next = 0;
  for (let start = 0; start < label.length; start++) {
    if (label[start] < 0 || comp[start] >= 0) continue;
    const l = label[start];
    comp[start] = next;
    stack.push(start);
    while (stack.length) {
      const p = stack.pop()!;
      const x = p % width;
      if (x > 0 && label[p - 1] === l && comp[p - 1] < 0) { comp[p - 1] = next; stack.push(p - 1); }
      if (x < width - 1 && label[p + 1] === l && comp[p + 1] < 0) { comp[p + 1] = next; stack.push(p + 1); }
      if (p >= width && label[p - width] === l && comp[p - width] < 0) { comp[p - width] = next; stack.push(p - width); }
      if (p + width < width * height && label[p + width] === l && comp[p + width] < 0) { comp[p + width] = next; stack.push(p + width); }
    }
    next++;
  }
  return comp;
}

/**
 * Chromaticity cannot tell white paint from grey paint. Split each region
 * into its connected surfaces, estimate each surface's albedo under a first
 * light fit, and separate surfaces whose median albedo differs by more than
 * `ratio`. Surfaces too small to judge join the nearest group.
 */
export function splitByAlbedo(
  regions: Regions,
  rgb: Float32Array,
  component: Int32Array,
  albedo: Float32Array,
  ratio = 1.15,
  minSamples = 15,
): Regions {
  const gap = Math.log(ratio);
  const id = Uint16Array.from(regions.id);
  let count = regions.count;
  const byRegion = new Map<number, Map<number, number[]>>();
  for (let i = 0; i < id.length; i++) {
    const r = id[i];
    if (r === regions.floor || !(albedo[i] > 0)) continue;
    let comps = byRegion.get(r);
    if (!comps) byRegion.set(r, (comps = new Map()));
    let list = comps.get(component[i]);
    if (!list) comps.set(component[i], (list = []));
    list.push(i);
  }
  const med = (idx: number[]) => {
    const v = idx.map((i) => Math.log(albedo[i])).sort((a, b) => a - b);
    return v[v.length >> 1];
  };
  for (const [r, comps] of byRegion) {
    const big = [...comps.values()].filter((l) => l.length >= minSamples).map((l) => ({ l, m: med(l) }));
    if (big.length < 2) continue;
    big.sort((a, b) => a.m - b.m);
    const groups: { m: number[]; members: number[][]; n: number }[] = [];
    for (const c of big) {
      const last = groups[groups.length - 1];
      if (last && c.m - last.m[last.m.length - 1] <= gap) {
        last.m.push(c.m);
        last.members.push(c.l);
        last.n += c.l.length;
      } else groups.push({ m: [c.m], members: [c.l], n: c.l.length });
    }
    if (groups.length < 2) continue;
    const keep = groups.reduce((a, b, k) => (b.n > groups[a].n ? k : a), 0);
    const ids = groups.map((_, k) => (k === keep ? r : count++));
    const centre = groups.map((g) => g.m[g.m.length >> 1]);
    groups.forEach((g, k) => g.members.forEach((l) => l.forEach((i) => (id[i] = ids[k]))));
    for (const l of comps.values()) {
      if (l.length >= minSamples) continue;
      const m = med(l);
      let best = 0;
      for (let k = 1; k < groups.length; k++) if (Math.abs(centre[k] - m) < Math.abs(centre[best] - m)) best = k;
      for (const i of l) id[i] = ids[best];
    }
  }
  if (count === regions.count) return regions;
  const { meanRgb, size } = regionStats(id, count, rgb);
  return { ...regions, id, count, meanRgb, size };
}

function nearest(centers: Float64Array, k: number, a: number, b: number): number {
  let best = 0, bestD = Infinity;
  for (let c = 0; c < k; c++) {
    const dx = a - centers[c * 2], dy = b - centers[c * 2 + 1];
    const d = dx * dx + dy * dy;
    if (d < bestD) { bestD = d; best = c; }
  }
  return best;
}

/** Saturation of a region's mean colour: 0 for grey, toward 1 for pure hues. */
export function saturation(meanRgb: Float32Array, r: number): number {
  const a = meanRgb[r * 3], b = meanRgb[r * 3 + 1], c = meanRgb[r * 3 + 2];
  const mx = Math.max(a, b, c), mn = Math.min(a, b, c);
  return mx > 0 ? (mx - mn) / mx : 0;
}
