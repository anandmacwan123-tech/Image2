// Measure the photo so the rendered object can match it: sensor noise, lens
// softness and black and white levels. Measured once per image at full
// resolution on a centre crop, so the numbers are in full-resolution pixels.

export interface PhotoStats {
  /** Noise standard deviation in sRGB units (0–1), full-resolution pixels. */
  noise: number;
  /** Gaussian blur sigma of edges, in full-resolution pixels. */
  blur: number;
  /** Per-channel black level in linear light (0.5th percentile). */
  black: [number, number, number];
  /** Per-channel white level in linear light (99.5th percentile). */
  white: [number, number, number];
}

const LUMA = [0.2126, 0.7152, 0.0722];

/** rgba: sRGB 8-bit pixels of a crop taken at full resolution. */
export function measure(rgba: Uint8ClampedArray, width: number, height: number): PhotoStats {
  const n = width * height;
  const L = new Float32Array(n);
  for (let p = 0; p < n; p++) L[p] = (LUMA[0] * rgba[p * 4] + LUMA[1] * rgba[p * 4 + 1] + LUMA[2] * rgba[p * 4 + 2]) / 255;
  return { noise: noiseSigma(L, width, height), blur: blurSigma(L, width, height), ...levels(rgba, n) };
}

/**
 * Immerkær's fast noise estimate: the Laplacian-difference kernel cancels
 * smooth image content, and its mean absolute response scales with noise.
 * Pixels on strong edges are skipped so texture does not read as noise.
 */
function noiseSigma(L: Float32Array, W: number, H: number): number {
  let sum = 0, count = 0;
  for (let y = 1; y < H - 1; y++) {
    for (let x = 1; x < W - 1; x++) {
      const p = y * W + x;
      const gx = L[p + 1] - L[p - 1], gy = L[p + W] - L[p - W];
      if (gx * gx + gy * gy > 0.01) continue;
      const v =
        L[p - W - 1] - 2 * L[p - W] + L[p - W + 1] -
        2 * L[p - 1] + 4 * L[p] - 2 * L[p + 1] +
        L[p + W - 1] - 2 * L[p + W] + L[p + W + 1];
      sum += Math.abs(v);
      count++;
    }
  }
  return count > 0 ? (Math.sqrt(Math.PI / 2) * sum) / (6 * count) : 0;
}

/**
 * Edge softness by re-blurring: at a step edge blurred by σ, the gradient
 * peak scales as 1/σ; blurring again by σ0 makes it 1/√(σ² + σ0²). The ratio
 * of the two peaks gives σ. Median over the strongest edges.
 */
function blurSigma(L: Float32Array, W: number, H: number): number {
  const s0 = 1;
  const B = gaussian(L, W, H, s0);
  const grad = (img: Float32Array, p: number) => {
    const gx = img[p + 1] - img[p - 1], gy = img[p + W] - img[p - W];
    return Math.sqrt(gx * gx + gy * gy);
  };
  const mags: { p: number; g: number }[] = [];
  for (let y = 4; y < H - 4; y += 1) {
    for (let x = 4; x < W - 4; x += 1) {
      const p = y * W + x;
      const g = grad(L, p);
      if (g > 0.08) mags.push({ p, g });
    }
  }
  if (mags.length < 50) return 0.8;
  mags.sort((a, b) => b.g - a.g);
  const top = mags.slice(0, Math.max(50, Math.floor(mags.length * 0.1)));
  const sig: number[] = [];
  for (const { p, g } of top) {
    // Only local maxima along x or y, i.e. edge centres.
    if (g < grad(L, p - 1) || g < grad(L, p + 1)) {
      if (g < grad(L, p - W) || g < grad(L, p + W)) continue;
    }
    const r = g / Math.max(grad(B, p), 1e-6);
    if (r > 1.01) sig.push(s0 / Math.sqrt(r * r - 1));
  }
  if (sig.length < 20) return 0.8;
  sig.sort((a, b) => a - b);
  return Math.min(8, Math.max(0.3, sig[sig.length >> 1]));
}

function gaussian(src: Float32Array, W: number, H: number, sigma: number): Float32Array {
  const r = Math.ceil(sigma * 3);
  const k: number[] = [];
  let ks = 0;
  for (let i = -r; i <= r; i++) {
    const v = Math.exp(-(i * i) / (2 * sigma * sigma));
    k.push(v);
    ks += v;
  }
  for (let i = 0; i < k.length; i++) k[i] /= ks;
  const tmp = new Float32Array(W * H), out = new Float32Array(W * H);
  for (let y = 0; y < H; y++) {
    for (let x = 0; x < W; x++) {
      let v = 0;
      for (let i = -r; i <= r; i++) v += k[i + r] * src[y * W + Math.min(W - 1, Math.max(0, x + i))];
      tmp[y * W + x] = v;
    }
  }
  for (let y = 0; y < H; y++) {
    for (let x = 0; x < W; x++) {
      let v = 0;
      for (let i = -r; i <= r; i++) v += k[i + r] * tmp[Math.min(H - 1, Math.max(0, y + i)) * W + x];
      out[y * W + x] = v;
    }
  }
  return out;
}

function levels(rgba: Uint8ClampedArray, n: number): { black: [number, number, number]; white: [number, number, number] } {
  const black: [number, number, number] = [0, 0, 0];
  const white: [number, number, number] = [1, 1, 1];
  for (let c = 0; c < 3; c++) {
    const hist = new Uint32Array(256);
    for (let p = 0; p < n; p++) hist[rgba[p * 4 + c]]++;
    let acc = 0, lo = -1, hi = 255;
    for (let v = 0; v < 256; v++) {
      acc += hist[v];
      if (lo < 0 && acc >= n * 0.005) lo = v;
      if (acc >= n * 0.995) { hi = v; break; }
    }
    black[c] = toLinear(Math.max(lo, 0) / 255);
    white[c] = toLinear(hi / 255);
  }
  return { black, white };
}

function toLinear(c: number): number {
  return c <= 0.04045 ? c / 12.92 : Math.pow((c + 0.055) / 1.055, 2.4);
}
