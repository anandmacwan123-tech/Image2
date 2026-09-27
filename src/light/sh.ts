// Real spherical harmonics up to band 2, in three.js order:
// [ Y00, Y1-1 (y), Y10 (z), Y11 (x), Y2-2 (xy), Y2-1 (yz), Y20, Y21 (xz), Y22 ].
//
// The fit works in "shading" units: a surface with albedo ρ and normal n shows
// up in the photo as ρ · S(n), where S(n) = Σ c_i Y_i(n). three.js renders a
// Lambertian surface as ρ/π · E(n) with E(n) = Σ Â_l L_i Y_i(n), so matching
// the two gives L_i = π c_i / Â_l (see toThreeProbe).

export const SH_COUNT = 9;

// Clamped-cosine convolution weights per band: Â_0 = π, Â_1 = 2π/3, Â_2 = π/4.
export const A_HAT = [Math.PI, (2 * Math.PI) / 3, Math.PI / 4];
export const BAND = [0, 1, 1, 1, 2, 2, 2, 2, 2];

export function shBasis(x: number, y: number, z: number, out: Float64Array | number[], offset = 0): void {
  out[offset] = 0.282095;
  out[offset + 1] = 0.488603 * y;
  out[offset + 2] = 0.488603 * z;
  out[offset + 3] = 0.488603 * x;
  out[offset + 4] = 1.092548 * x * y;
  out[offset + 5] = 1.092548 * y * z;
  out[offset + 6] = 0.315392 * (3 * z * z - 1);
  out[offset + 7] = 1.092548 * x * z;
  out[offset + 8] = 0.546274 * (x * x - y * y);
}

export function shEval(c: ArrayLike<number>, x: number, y: number, z: number): number {
  return (
    c[0] * 0.282095 +
    c[1] * 0.488603 * y +
    c[2] * 0.488603 * z +
    c[3] * 0.488603 * x +
    c[4] * 1.092548 * x * y +
    c[5] * 1.092548 * y * z +
    c[6] * 0.315392 * (3 * z * z - 1) +
    c[7] * 1.092548 * x * z +
    c[8] * 0.546274 * (x * x - y * y)
  );
}

/** Shading coefficients of a constant a: S(n) = a. */
export function shConstant(a: number): Float64Array {
  const c = new Float64Array(SH_COUNT);
  c[0] = a / 0.282095;
  return c;
}

/**
 * Shading coefficients of a directional light, k · max(0, n·ℓ), band-limited
 * to l ≤ 2. For a delta of radiance at ℓ, irradiance is Σ Â_l Y_i(ℓ) Y_i(n).
 */
export function shDirectional(k: number, lx: number, ly: number, lz: number): Float64Array {
  const y = new Float64Array(SH_COUNT);
  shBasis(lx, ly, lz, y);
  for (let i = 0; i < SH_COUNT; i++) y[i] *= k * A_HAT[BAND[i]];
  return y;
}

/**
 * Dominant direction of a shading function: the band-1 vector points toward
 * the brightest side. Returns a unit vector, or +y if band 1 is empty.
 */
export function shDominantDirection(c: ArrayLike<number>): [number, number, number] {
  const x = c[3], y = c[1], z = c[2];
  const n = Math.hypot(x, y, z);
  return n > 1e-12 ? [x / n, y / n, z / n] : [0, 1, 0];
}

/** Shading SH → three.js LightProbe radiance coefficients (per channel). */
export function toThreeProbe(c: ArrayLike<number>): Float64Array {
  const out = new Float64Array(SH_COUNT);
  for (let i = 0; i < SH_COUNT; i++) out[i] = (Math.PI * c[i]) / A_HAT[BAND[i]];
  return out;
}

/**
 * Radiance SH evaluated as an environment (for HDRI export and the reflection
 * backdrop). Band 2 is windowed to tame ringing, and negatives are clamped.
 */
export function radianceAt(L: ArrayLike<number>, x: number, y: number, z: number): number {
  const w2 = 0.75;
  const v =
    L[0] * 0.282095 +
    L[1] * 0.488603 * y +
    L[2] * 0.488603 * z +
    L[3] * 0.488603 * x +
    w2 * (L[4] * 1.092548 * x * y +
      L[5] * 1.092548 * y * z +
      L[6] * 0.315392 * (3 * z * z - 1) +
      L[7] * 1.092548 * x * z +
      L[8] * 0.546274 * (x * x - y * y));
  return v > 0 ? v : 0;
}

/**
 * Lowest value of S(n) over the sphere, sampled on a Fibonacci set. Used to
 * keep the ambient term non-negative after the key light is removed.
 */
export function shMinimum(c: ArrayLike<number>, samples = 256): number {
  let min = Infinity;
  const golden = Math.PI * (3 - Math.sqrt(5));
  for (let i = 0; i < samples; i++) {
    const y = 1 - (2 * (i + 0.5)) / samples;
    const r = Math.sqrt(1 - y * y);
    const t = golden * i;
    const v = shEval(c, Math.cos(t) * r, y, Math.sin(t) * r);
    if (v < min) min = v;
  }
  return min;
}

/** Mean of S(n) over the sphere, which is just the band-0 term. */
export function shMean(c: ArrayLike<number>): number {
  return c[0] * 0.282095;
}
