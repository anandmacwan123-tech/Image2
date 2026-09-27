// Local refit: when an object moves, pixels near it in 3D get more weight in
// a refit of the ambient and of how much key light reaches that spot. With few
// nearby pixels the prior pulls the result back to the global fit.
//
//   I(x) ≈ ρ_r(x) · ( Σ c_i Y_i(n_x) + m · k · t(x) + src(x) )
//
// ρ is the fitted albedo, t the key's cosine term and src the fitted point
// sources, all fixed from the global fit. The unknowns are the 9 ambient
// coefficients c and the key visibility m (0 in a real cast shadow, 1 in the
// open), solved as one 10×10 system per channel.

import { solveSPD } from "../lib/linalg";
import { SH_COUNT, shBasis } from "./sh";

export interface LocalData {
  count: number;
  position: Float32Array;
  /** Basis values, 9 per sample. */
  basis: Float32Array;
  /** Observed linear value per channel: [luminance, r, g, b] per sample. */
  value: Float32Array;
  /** Albedo per sample: [mono, r, g, b]. */
  albedo: Float32Array;
  /** Key cosine term per sample (positional if the key is a visible source). */
  keyTerm: Float32Array;
  /** Fitted point-source shading per sample: [mono, r, g, b]. */
  sourceTerm: Float32Array;
  /** Robust weight from the global fit. */
  weight: Float32Array;
}

export interface LocalLight {
  /** Ambient shading coefficients per channel. */
  ambient: Float64Array[];
  /** Key visibility, 1 = as fitted globally. */
  keyScale: number;
  /** Effective number of nearby pixels that informed this fit. */
  support: number;
}

export interface LocalPrior {
  channels: 1 | 3;
  ambient: ArrayLike<number>[];
  keyStrength: number[];
}

const LUMA = [0.2126, 0.7152, 0.0722];

/** Prior strength, in pixels' worth of evidence. */
const PRIOR_PIXELS = 300;

export function makeLocalData(
  count: number,
  position: Float32Array,
  normal: Float32Array,
  value: Float32Array,
  albedo: Float32Array,
  keyTerm: Float32Array,
  sourceTerm: Float32Array,
  weight: Float32Array,
): LocalData {
  const basis = new Float32Array(count * SH_COUNT);
  const tmp = new Float64Array(SH_COUNT);
  for (let i = 0; i < count; i++) {
    shBasis(normal[i * 3], normal[i * 3 + 1], normal[i * 3 + 2], tmp);
    basis.set(tmp, i * SH_COUNT);
  }
  return { count, position, basis, value, albedo, keyTerm, sourceTerm, weight };
}

/**
 * Refit around a point. `radius` is the object's size; the neighbourhood is a
 * Gaussian falloff a few times that, never tighter than 0.4 m.
 */
export function localRefit(data: LocalData, prior: LocalPrior, center: ArrayLike<number>, radius: number): LocalLight {
  const n = data.count;
  const sigma = Math.max(0.4, 2 * radius);
  const inv2s2 = 1 / (2 * sigma * sigma);
  const cutoff = 9 * sigma * sigma;
  const channels = prior.channels;
  const K = 9; // SH_COUNT, local so hot loops never touch a module binding
  const N = K + 1;

  const A = Array.from({ length: channels }, () => new Float64Array(N * N));
  const b = Array.from({ length: channels }, () => new Float64Array(N));
  // Diagonal scale of the data per unknown, for a prior in "pixels" units.
  const diag = Array.from({ length: channels }, () => new Float64Array(N));
  let support = 0, totalW = 0;
  const row = new Float64Array(N);

  for (let i = 0; i < n; i++) {
    const dx = data.position[i * 3] - center[0];
    const dy = data.position[i * 3 + 1] - center[1];
    const dz = data.position[i * 3 + 2] - center[2];
    const d2 = dx * dx + dy * dy + dz * dz;
    const wr = data.weight[i];
    for (let c = 0; c < channels; c++) {
      const ch = channels === 1 ? 0 : c + 1;
      const rho = data.albedo[i * 4 + ch];
      const k = prior.keyStrength[c];
      for (let p = 0; p < K; p++) row[p] = rho * data.basis[i * K + p];
      row[K] = rho * k * data.keyTerm[i];
      for (let p = 0; p < N; p++) diag[c][p] += wr * row[p] * row[p];
    }
    totalW += wr;
    if (d2 > cutoff) continue;
    const w = wr * Math.exp(-d2 * inv2s2);
    if (w < 1e-6) continue;
    support += w;
    for (let c = 0; c < channels; c++) {
      const ch = channels === 1 ? 0 : c + 1;
      const rho = data.albedo[i * 4 + ch];
      const k = prior.keyStrength[c];
      for (let p = 0; p < K; p++) row[p] = rho * data.basis[i * K + p];
      row[K] = rho * k * data.keyTerm[i];
      const y = data.value[i * 4 + ch] - rho * data.sourceTerm[i * 4 + ch];
      const Ac = A[c], bc = b[c];
      for (let p = 0; p < N; p++) {
        const wp = w * row[p];
        bc[p] += wp * y;
        for (let q = p; q < N; q++) Ac[p * N + q] += wp * row[q];
      }
    }
  }

  const ambient: Float64Array[] = [];
  const scales: number[] = [];
  for (let c = 0; c < channels; c++) {
    const Ac = A[c], bc = b[c];
    for (let p = 0; p < N; p++) for (let q = 0; q < p; q++) Ac[p * N + q] = Ac[q * N + p];
    for (let p = 0; p < N; p++) {
      const lambda = (PRIOR_PIXELS * diag[c][p]) / Math.max(totalW, 1e-9) + 1e-12;
      const target = p < K ? prior.ambient[c][p] : 1;
      Ac[p * N + p] += lambda;
      bc[p] += lambda * target;
    }
    const x = solveSPD(Ac, bc, N);
    ambient.push(x.subarray(0, K).slice());
    scales.push(x[K]);
  }
  const keyScale = channels === 1 ? scales[0] : scales.reduce((acc, v, c) => acc + v * LUMA[c], 0);
  return { ambient, keyScale: Math.min(1.25, Math.max(0, keyScale)), support };
}
