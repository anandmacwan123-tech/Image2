// The light fit, run once per image in the worker:
//
// 1. Prepare: linear light, drop invalid, clipped, near-black and depth-edge pixels.
// 2. Group: chromaticity regions, with the floor always split off.
// 3. Ambient: 9 SH coefficients and region scales, robustly reweighted.
// 4. Key: dense direction search with a closed-form inner solve.
// 5. Sources: clipped regions projected to 3D; one near the key becomes the key.
// 6. Output: models for mono and colour, plus the data the local refit needs.
//
// The fit has a scale ambiguity (bright light on dark paint looks like dim
// light on white paint). It is fixed by treating the largest near-neutral
// region that is not the floor, usually a wall, as white paint.

import type { Geometry } from "../geometry/moge-post";
import { nnlsNormal, rng } from "../lib/linalg";
import { basisFor, fitAmbientSH } from "./ambient";
import { makeLocalData, type LocalData } from "./local";
import { angleBetween, directionalTerm, fitKeyDirection, keyWeights, positionalTerm, solveRatio, type Vec3 } from "./refine";
import { components, estimateUp, groupRegions, saturation, splitByAlbedo, type Regions, type Samples } from "./regions";
import { SH_COUNT, shDirectional, shDominantDirection, shMinimum } from "./sh";
import { findSources, type Source } from "./sources";

export type { Vec3 } from "./refine";

export interface PointSource {
  position: Vec3;
  /** Shading added at distance d0, per channel. */
  strength: number[];
  /** Distance from the scene's reference point, where strength applies. */
  d0: number;
}

export interface LightModel {
  channels: 1 | 3;
  /** Everything the fit explains, as shading SH per channel. */
  total: number[][];
  /** Total minus key and sources: the fill light. */
  ambient: number[][];
  key: {
    dir: Vec3;
    strength: number[];
    /** Set when a visible source is the key: the light sits here. */
    source: Vec3 | null;
  };
  sources: PointSource[];
}

export interface LightFit {
  mono: LightModel;
  colour: LightModel;
  up: Vec3;
  /** Robust-weighted centroid of the fitted pixels. */
  refPoint: Vec3;
  detected: Source[];
  local: LocalData;
  /** Albedo assumed for the reference region. */
  white: number;
  info: { samples: number; regions: number; keyCost: number; neutralReference: boolean };
}

export const WHITE = 0.8;
const MAX_ALBEDO = 0.95;
const LUMA = [0.2126, 0.7152, 0.0722];

const SRGB_TO_LINEAR = new Float32Array(256);
for (let i = 0; i < 256; i++) {
  const c = i / 255;
  SRGB_TO_LINEAR[i] = c <= 0.04045 ? c / 12.92 : Math.pow((c + 0.055) / 1.055, 2.4);
}
export function srgbToLinear(v: number): number {
  return SRGB_TO_LINEAR[v];
}

export interface Prepared {
  samples: Samples;
  clipped: Uint8Array;
  linear: Float32Array;
  /** 1 for pixels usable in the fit. */
  eligible: Uint8Array;
}

/** Linearise, mask and subsample. rgba must match the geometry's size. */
export function prepare(rgba: Uint8ClampedArray | Uint8Array, geometry: Geometry, maxSamples = 60000): Prepared {
  const { width: W, height: H, mask, depth, normals, points } = geometry;
  const count = W * H;
  const linear = new Float32Array(count * 3);
  const clipped = new Uint8Array(count);
  for (let p = 0; p < count; p++) {
    const r = rgba[p * 4], g = rgba[p * 4 + 1], b = rgba[p * 4 + 2];
    linear[p * 3] = SRGB_TO_LINEAR[r];
    linear[p * 3 + 1] = SRGB_TO_LINEAR[g];
    linear[p * 3 + 2] = SRGB_TO_LINEAR[b];
    if (Math.max(r, g, b) >= 250) clipped[p] = 1;
  }

  // Depth edges and creases, where normals are unreliable, plus a margin
  // around clipped pixels, where bloom is not shading.
  const bad = new Uint8Array(count);
  for (let y = 0; y < H; y++) {
    for (let x = 0; x < W; x++) {
      const p = y * W + x;
      if (!mask[p]) continue;
      for (const q of [x + 1 < W ? p + 1 : -1, y + 1 < H ? p + W : -1]) {
        if (q < 0 || !mask[q]) continue;
        const rel = Math.abs(depth[p] - depth[q]) / Math.min(depth[p], depth[q]);
        const dot = normals[p * 3] * normals[q * 3] + normals[p * 3 + 1] * normals[q * 3 + 1] + normals[p * 3 + 2] * normals[q * 3 + 2];
        if (rel > 0.03 || dot < 0.8) bad[p] = bad[q] = 1;
      }
      if (clipped[p]) bad[p] = 2;
    }
  }
  const excluded = new Uint8Array(count);
  for (let y = 0; y < H; y++) {
    for (let x = 0; x < W; x++) {
      const p = y * W + x;
      if (!bad[p]) continue;
      const r = bad[p] === 2 ? 3 : 1;
      for (let dy = -r; dy <= r; dy++) {
        const yy = y + dy;
        if (yy < 0 || yy >= H) continue;
        for (let dx = -r; dx <= r; dx++) {
          const xx = x + dx;
          if (xx >= 0 && xx < W) excluded[yy * W + xx] = 1;
        }
      }
    }
  }

  const eligible: number[] = [];
  const usable = new Uint8Array(count);
  for (let p = 0; p < count; p++) {
    if (!mask[p] || excluded[p]) continue;
    const lum = LUMA[0] * linear[p * 3] + LUMA[1] * linear[p * 3 + 1] + LUMA[2] * linear[p * 3 + 2];
    if (lum < 0.002) continue;
    eligible.push(p);
    usable[p] = 1;
  }
  const rand = rng(42);
  const take = Math.min(maxSamples, eligible.length);
  for (let i = 0; i < take; i++) {
    const j = i + Math.floor(rand() * (eligible.length - i));
    const t = eligible[i];
    eligible[i] = eligible[j];
    eligible[j] = t;
  }
  const chosen = eligible.slice(0, take).sort((a, b) => a - b);

  const samples: Samples = {
    count: take,
    rgb: new Float32Array(take * 3),
    normal: new Float32Array(take * 3),
    position: new Float32Array(take * 3),
    pixel: Uint32Array.from(chosen),
  };
  for (let i = 0; i < take; i++) {
    const p = chosen[i];
    for (let k = 0; k < 3; k++) {
      samples.rgb[i * 3 + k] = linear[p * 3 + k];
      samples.normal[i * 3 + k] = normals[p * 3 + k];
      samples.position[i * 3 + k] = points[p * 3 + k];
    }
  }
  return { samples, clipped, linear, eligible: usable };
}

/**
 * I ≈ s_r · Σ_j coef_j · term_j with coef ≥ 0, alternating between the
 * coefficients (NNLS) and the region scales (closed form).
 */
export function fitLinearTerms(
  I: Float32Array,
  w: Float32Array,
  id: Uint16Array,
  regions: number,
  terms: Float32Array[],
  sInit: Float64Array,
  iters = 10,
): { coef: Float64Array; s: Float64Array } {
  const n = I.length, J = terms.length;
  const s = Float64Array.from(sInit);
  let coef: Float64Array = new Float64Array(J);
  const A = new Float64Array(J * J), b = new Float64Array(J);
  const num = new Float64Array(regions), den = new Float64Array(regions), cnt = new Float64Array(regions);
  for (let i = 0; i < n; i++) cnt[id[i]]++;
  for (let it = 0; it < iters; it++) {
    A.fill(0);
    b.fill(0);
    for (let i = 0; i < n; i++) {
      const si = s[id[i]], wi = w[i];
      for (let p = 0; p < J; p++) {
        const tp = terms[p][i] * si * wi;
        b[p] += tp * I[i];
        for (let q = p; q < J; q++) A[p * J + q] += tp * terms[q][i] * si;
      }
    }
    for (let p = 0; p < J; p++) for (let q = 0; q < p; q++) A[p * J + q] = A[q * J + p];
    coef = nnlsNormal(A, b, J);
    num.fill(0);
    den.fill(0);
    for (let i = 0; i < n; i++) {
      let v = 0;
      for (let p = 0; p < J; p++) v += coef[p] * terms[p][i];
      num[id[i]] += w[i] * I[i] * v;
      den[id[i]] += w[i] * v * v;
    }
    for (let r = 0; r < regions; r++) if (den[r] > 1e-12) s[r] = Math.max(num[r] / den[r], 1e-6);
    let m = 0, tot = 0;
    for (let r = 0; r < regions; r++) { m += s[r] * cnt[r]; tot += cnt[r]; }
    m = tot > 0 ? m / tot : 1;
    for (let r = 0; r < regions; r++) s[r] /= m;
    for (let p = 0; p < J; p++) coef[p] *= m;
  }
  return { coef, s };
}

/** The region treated as white paint: largest near-neutral non-floor region. */
export function referenceRegion(regions: Regions): { index: number; neutral: boolean } {
  let best = -1, fallback = -1;
  for (let r = 0; r < regions.count; r++) {
    if (r === regions.floor) continue;
    if (fallback < 0 || regions.size[r] > regions.size[fallback]) fallback = r;
    if (saturation(regions.meanRgb, r) < 0.2 && (best < 0 || regions.size[r] > regions.size[best])) best = r;
  }
  if (best >= 0) return { index: best, neutral: true };
  return { index: fallback >= 0 ? fallback : 0, neutral: false };
}

/**
 * Per-channel gauge g: albedo = s · g. The reference region gets WHITE (and
 * is neutral in colour); if it is not near-neutral, colour falls back to a
 * grey-world balance. No sizeable region may exceed MAX_ALBEDO.
 */
export function gauges(scales: Float64Array[], regions: Regions, ref: { index: number; neutral: boolean }, monoScales: Float64Array): number[] {
  const R = regions.count;
  const big = (r: number) => regions.size[r] >= 0.03 * regions.id.length;
  const weightedMean = (s: Float64Array) => {
    let m = 0, t = 0;
    for (let r = 0; r < R; r++) { m += s[r] * regions.size[r]; t += regions.size[r]; }
    return t > 0 ? m / t : 1;
  };
  const gMono = WHITE / Math.max(monoScales[ref.index], 1e-9);
  let g: number[];
  if (scales.length === 1 || ref.neutral) {
    g = scales.map((s) => WHITE / Math.max(s[ref.index], 1e-9));
  } else {
    const mMono = weightedMean(monoScales);
    g = scales.map((s) => (gMono * mMono) / Math.max(weightedMean(s), 1e-9));
  }
  let max = 0;
  for (let r = 0; r < R; r++) {
    if (!big(r)) continue;
    for (let c = 0; c < scales.length; c++) max = Math.max(max, scales[c][r] * g[c]);
  }
  if (max > MAX_ALBEDO) g = g.map((v) => (v * MAX_ALBEDO) / max);
  return g;
}

export function fitLight(rgba: Uint8ClampedArray | Uint8Array, geometry: Geometry): LightFit {
  const { samples, clipped, linear, eligible } = prepare(rgba, geometry);
  const n = samples.count;
  const up = estimateUp(samples.normal, n);
  const chroma = groupRegions(samples, up);

  const channel = (k: number) => {
    const out = new Float32Array(n);
    for (let i = 0; i < n; i++) out[i] = samples.rgb[i * 3 + k];
    return out;
  };
  const rgbI = [channel(0), channel(1), channel(2)];
  const lumI = new Float32Array(n);
  for (let i = 0; i < n; i++) lumI[i] = LUMA[0] * rgbI[0][i] + LUMA[1] * rgbI[1][i] + LUMA[2] * rgbI[2][i];

  // 3. A first ambient fit only seeds the key search.
  const Y = basisFor(samples.normal, n);
  const ones = new Float32Array(n).fill(1);
  const seed = fitAmbientSH(lumI, Y, chroma.id, chroma.count, null, false, 10);

  // 4. Key direction on a subsample, luminance only, with its own robust
  // weights. A first pass on chroma regions gives each connected surface an
  // albedo; regions whose surfaces disagree are split, and the key refit.
  const stride = Math.max(1, Math.floor(n / 10000));
  const sub = Math.ceil(n / stride);
  const subNormal = new Float32Array(sub * 3), subI = new Float32Array(sub);
  for (let j = 0, i = 0; i < n; i += stride, j++) {
    subNormal.set(samples.normal.subarray(i * 3, i * 3 + 3), j * 3);
    subI[j] = lumI[i];
  }
  const subIds = (ids: Uint16Array) => {
    const out = new Uint16Array(sub);
    for (let j = 0, i = 0; i < n; i += stride, j++) out[j] = ids[i];
    return out;
  };
  const subOnes = new Float32Array(sub).fill(1);
  const first = fitKeyDirection(subNormal, subI, subOnes, subIds(chroma.id), chroma.count, shDominantDirection(seed.c));

  const { width: W, height: H, normals } = geometry;
  const label = new Int32Array(W * H).fill(-1);
  for (let p = 0; p < W * H; p++) {
    if (!eligible[p]) continue;
    label[p] = chroma.classify(linear[p * 3], linear[p * 3 + 1], linear[p * 3 + 2], normals[p * 3], normals[p * 3 + 1], normals[p * 3 + 2]);
  }
  const comp = components(label, W, H);
  const sampleComp = new Int32Array(n);
  for (let i = 0; i < n; i++) sampleComp[i] = comp[samples.pixel[i]];
  const t0 = directionalTerm(samples.normal, n, first.dir, new Float32Array(n));
  const relAlbedo = new Float32Array(n);
  for (let i = 0; i < n; i++) relAlbedo[i] = lumI[i] / (1 + first.rho * t0[i]);
  const regions = splitByAlbedo(chroma, samples.rgb, sampleComp, relAlbedo);
  const R = regions.count;
  const { id } = regions;
  const keyFit = regions === chroma ? first : fitKeyDirection(subNormal, subI, subOnes, subIds(id), R, first.dir);
  let keyDir = keyFit.dir;

  // 5. Sources.
  let wsum = 0;
  const refPoint: Vec3 = [0, 0, 0];
  for (let i = 0; i < n; i++) {
    for (let k = 0; k < 3; k++) refPoint[k] += samples.position[i * 3 + k];
    wsum++;
  }
  for (let k = 0; k < 3; k++) refPoint[k] /= Math.max(wsum, 1e-9);
  const detected = findSources(clipped, geometry, linear);
  const toward = (p: Vec3): Vec3 => {
    const d: Vec3 = [p[0] - refPoint[0], p[1] - refPoint[1], p[2] - refPoint[2]];
    const l = Math.hypot(...d) || 1;
    return [d[0] / l, d[1] / l, d[2] / l];
  };
  let keySource: Vec3 | null = null;
  let bestAngle = (25 * Math.PI) / 180;
  for (const s of detected) {
    if (s.position) {
      const a = angleBetween(toward(s.position), keyDir);
      if (a < bestAngle) { bestAngle = a; keySource = s.position; }
    } else if (s.pixels < 0.01 * geometry.width * geometry.height && angleBetween(s.direction, keyDir) < (15 * Math.PI) / 180) {
      keyDir = s.direction; // a small clipped patch in the sky: the sun
    }
  }
  if (keySource) keyDir = toward(keySource);
  const others = detected.filter((s) => s.position && s.position !== keySource);

  const keyTerm = keySource
    ? positionalTerm(samples.normal, samples.position, n, keySource, new Float32Array(n))
    : directionalTerm(samples.normal, n, keyDir, new Float32Array(n));
  const srcTerms = others.map((s) => {
    const p = s.position!;
    const d0 = Math.hypot(p[0] - refPoint[0], p[1] - refPoint[1], p[2] - refPoint[2]);
    const t = positionalTerm(samples.normal, samples.position, n, p, new Float32Array(n));
    for (let i = 0; i < n; i++) {
      const d = Math.hypot(p[0] - samples.position[i * 3], p[1] - samples.position[i * 3 + 1], p[2] - samples.position[i * 3 + 2]);
      t[i] *= Math.min(4, (d0 * d0) / Math.max(d * d, 1e-6));
    }
    return { position: p, d0, t };
  });
  // Robust weights for everything that follows come from the key model, which
  // (unlike band-limited SH) can represent a sharp terminator.
  let w = keyWeights(lumI, ones, id, keyTerm, solveRatio(lumI, ones, id, R, keyTerm));
  w = keyWeights(lumI, ones, id, keyTerm, solveRatio(lumI, w, id, R, keyTerm));
  const shMono = fitAmbientSH(lumI, Y, id, R, w, false);
  const shRgb = rgbI.map((I) => fitAmbientSH(I, Y, id, R, w, false));

  const terms = [ones, keyTerm, ...srcTerms.map((s) => s.t)];
  const linMono = fitLinearTerms(lumI, w, id, R, terms, shMono.s);
  const linRgb = rgbI.map((I, c) => fitLinearTerms(I, w, id, R, terms, shRgb[c].s));

  // 6. Gauge and assemble.
  const ref = referenceRegion(regions);
  const gShMono = gauges([shMono.s], regions, ref, shMono.s)[0];
  const gShRgb = gauges(shRgb.map((f) => f.s), regions, ref, shMono.s);
  const gLinMono = gauges([linMono.s], regions, ref, linMono.s)[0];
  const gLinRgb = gauges(linRgb.map((f) => f.s), regions, ref, linMono.s);

  const build = (sh: { c: Float64Array }[], lin: { coef: Float64Array }[], gSh: number[], gLin: number[]): LightModel => {
    const channels = sh.length as 1 | 3;
    const total = sh.map((f, c) => Array.from(f.c, (v) => v / gSh[c]));
    const keyStrength = lin.map((f, c) => f.coef[1] / gLin[c]);
    const sources: PointSource[] = srcTerms.map((s, j) => ({
      position: s.position,
      d0: s.d0,
      strength: lin.map((f, c) => f.coef[2 + j] / gLin[c]),
    }));
    const ambient = total.map((tc, c) => {
      const a = Float64Array.from(tc);
      const k = shDirectional(keyStrength[c], ...(keySource ? toward(keySource) : keyDir));
      for (let i = 0; i < SH_COUNT; i++) a[i] -= k[i];
      for (const s of sources) {
        const d = shDirectional(s.strength[c], ...toward(s.position));
        for (let i = 0; i < SH_COUNT; i++) a[i] -= d[i];
      }
      const min = shMinimum(a);
      if (min < 0) a[0] -= min / 0.282095;
      return Array.from(a);
    });
    return { channels, total, ambient, key: { dir: keyDir, strength: keyStrength, source: keySource }, sources };
  };
  const mono = build([shMono], [linMono], [gShMono], [gLinMono]);
  const colour = build(shRgb, linRgb, gShRgb, gLinRgb);

  // Local refit data: a subsample carrying everything fixed by the global fit.
  const lstride = Math.max(1, Math.floor(n / 16000));
  const ln = Math.ceil(n / lstride);
  const lpos = new Float32Array(ln * 3), lnorm = new Float32Array(ln * 3);
  const lval = new Float32Array(ln * 4), lalb = new Float32Array(ln * 4), lsrc = new Float32Array(ln * 4);
  const lkey = new Float32Array(ln), lw = new Float32Array(ln);
  for (let j = 0, i = 0; i < n; i += lstride, j++) {
    lpos.set(samples.position.subarray(i * 3, i * 3 + 3), j * 3);
    lnorm.set(samples.normal.subarray(i * 3, i * 3 + 3), j * 3);
    lval[j * 4] = lumI[i];
    lalb[j * 4] = shMono.s[id[i]] * gShMono;
    let srcMono = 0;
    for (let s = 0; s < srcTerms.length; s++) srcMono += mono.sources[s].strength[0] * srcTerms[s].t[i];
    lsrc[j * 4] = srcMono;
    for (let c = 0; c < 3; c++) {
      lval[j * 4 + 1 + c] = rgbI[c][i];
      lalb[j * 4 + 1 + c] = shRgb[c].s[id[i]] * gShRgb[c];
      let v = 0;
      for (let s = 0; s < srcTerms.length; s++) v += colour.sources[s].strength[c] * srcTerms[s].t[i];
      lsrc[j * 4 + 1 + c] = v;
    }
    lkey[j] = keyTerm[i];
    lw[j] = w[i];
  }
  const local = makeLocalData(ln, lpos, lnorm, lval, lalb, lkey, lsrc, lw);

  return {
    mono,
    colour,
    up,
    refPoint,
    detected,
    local,
    white: WHITE,
    info: { samples: n, regions: R, keyCost: keyFit.cost, neutralReference: ref.neutral },
  };
}
