// Fit the ambient: I(x) ≈ s_r(x) · Σ c_i Y_i(n_x), alternating between the
// region scales (closed form each) and the 9 harmonic coefficients (one 9×9
// solve), with robust reweighting so cast shadows and highlights drop out as
// outliers instead of bending the light.

import { solveSPD, median } from "../lib/linalg";
import { SH_COUNT, shBasis } from "./sh";

export interface ShFit {
  /** Shading coefficients, in the fit's own gauge (see fit.ts for albedo). */
  c: Float64Array;
  /** Per-region scale. */
  s: Float64Array;
  /** Final per-sample weights (robust weights times the input weights). */
  w: Float32Array;
}

/** Precomputed basis values, 9 per sample. */
export function basisFor(normal: Float32Array, count: number): Float32Array {
  const Y = new Float32Array(count * SH_COUNT);
  const tmp = new Float64Array(SH_COUNT);
  for (let i = 0; i < count; i++) {
    shBasis(normal[i * 3], normal[i * 3 + 1], normal[i * 3 + 2], tmp);
    Y.set(tmp, i * SH_COUNT);
  }
  return Y;
}

export function fitAmbientSH(
  I: Float32Array,
  Y: Float32Array,
  id: Uint16Array,
  regions: number,
  weights: Float32Array | null,
  robust: boolean,
  iters = 16,
): ShFit {
  const K = 9; // SH_COUNT, local so hot loops never touch a module binding
  const n = I.length;
  const base = weights ?? new Float32Array(n).fill(1);
  const w = Float32Array.from(base);
  const s = new Float64Array(regions);
  const cnt = new Float64Array(regions);
  for (let i = 0; i < n; i++) {
    s[id[i]] += I[i];
    cnt[id[i]]++;
  }
  for (let r = 0; r < regions; r++) s[r] = cnt[r] > 0 ? Math.max(s[r] / cnt[r], 1e-6) : 1;

  let c: Float64Array = new Float64Array(K);
  const A = new Float64Array(K * K);
  const b = new Float64Array(K);
  const num = new Float64Array(regions);
  const den = new Float64Array(regions);
  const S = new Float32Array(n);
  const reg = [0, 1e-4, 1e-4, 1e-4, 1e-3, 1e-3, 1e-3, 1e-3, 1e-3];

  for (let it = 0; it < iters; it++) {
    // Light, given the scales.
    A.fill(0);
    b.fill(0);
    for (let i = 0; i < n; i++) {
      const wi = w[i];
      if (wi === 0) continue;
      const si = s[id[i]];
      const ws2 = wi * si * si;
      const wsI = wi * si * I[i];
      const o = i * K;
      for (let p = 0; p < K; p++) {
        const yp = Y[o + p];
        b[p] += wsI * yp;
        const t = ws2 * yp;
        for (let q = p; q < K; q++) A[p * K + q] += t * Y[o + q];
      }
    }
    let trace = 0;
    for (let p = 0; p < K; p++) {
      for (let q = 0; q < p; q++) A[p * K + q] = A[q * K + p];
      trace += A[p * K + p];
    }
    for (let p = 0; p < K; p++) A[p * K + p] += (reg[p] * trace) / K;
    c = solveSPD(A, b, K);

    // Shading per sample, then scales given the light.
    num.fill(0);
    den.fill(0);
    for (let i = 0; i < n; i++) {
      const o = i * K;
      let v = 0;
      for (let p = 0; p < K; p++) v += c[p] * Y[o + p];
      S[i] = v;
      const wi = w[i];
      num[id[i]] += wi * I[i] * v;
      den[id[i]] += wi * v * v;
    }
    for (let r = 0; r < regions; r++) if (den[r] > 1e-12) s[r] = Math.max(num[r] / den[r], 1e-6);

    // Fix the scale ambiguity: size-weighted mean scale of 1.
    let m = 0, tot = 0;
    for (let r = 0; r < regions; r++) { m += s[r] * cnt[r]; tot += cnt[r]; }
    m = tot > 0 ? m / tot : 1;
    if (m > 0) {
      for (let r = 0; r < regions; r++) s[r] /= m;
      for (let p = 0; p < K; p++) c[p] *= m;
      for (let i = 0; i < n; i++) S[i] *= m;
    }

    if (robust && it >= 2) {
      // Cauchy weights on relative residuals, scaled by their MAD.
      const e = new Float32Array(n);
      let meanI = 0;
      for (let i = 0; i < n; i++) meanI += I[i];
      meanI /= Math.max(n, 1);
      for (let i = 0; i < n; i++) {
        const pred = s[id[i]] * S[i];
        e[i] = (I[i] - pred) / Math.max(pred, 0.02 * meanI);
      }
      const abs = e.map(Math.abs);
      const sigma = Math.max(1.4826 * median(abs), 0.01);
      const k = 2.5 * sigma;
      for (let i = 0; i < n; i++) {
        const q = e[i] / k;
        w[i] = base[i] / (1 + q * q);
      }
    }
  }
  return { c, s, w };
}
