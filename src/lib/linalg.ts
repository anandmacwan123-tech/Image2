// Small dense solvers for the light fits. Matrices are row-major Float64Array.

/**
 * Solve A x = b for symmetric positive (semi)definite A by Cholesky, adding a
 * small diagonal jitter if A is near singular. A and b are left untouched.
 */
export function solveSPD(A: Float64Array, b: Float64Array, n: number): Float64Array {
  const L = new Float64Array(n * n);
  let trace = 0;
  for (let i = 0; i < n; i++) trace += A[i * n + i];
  let jitter = 0;
  for (let attempt = 0; attempt < 6; attempt++) {
    if (cholesky(A, L, n, jitter)) return cholSolve(L, b, n);
    jitter = jitter === 0 ? 1e-10 * (trace / n + 1e-30) : jitter * 100;
  }
  return new Float64Array(n);
}

function cholesky(A: Float64Array, L: Float64Array, n: number, jitter: number): boolean {
  L.fill(0);
  for (let i = 0; i < n; i++) {
    for (let j = 0; j <= i; j++) {
      let s = A[i * n + j] + (i === j ? jitter : 0);
      for (let k = 0; k < j; k++) s -= L[i * n + k] * L[j * n + k];
      if (i === j) {
        if (!(s > 0)) return false;
        L[i * n + i] = Math.sqrt(s);
      } else {
        L[i * n + j] = s / L[j * n + j];
      }
    }
  }
  return true;
}

function cholSolve(L: Float64Array, b: Float64Array, n: number): Float64Array {
  const y = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    let s = b[i];
    for (let k = 0; k < i; k++) s -= L[i * n + k] * y[k];
    y[i] = s / L[i * n + i];
  }
  const x = new Float64Array(n);
  for (let i = n - 1; i >= 0; i--) {
    let s = y[i];
    for (let k = i + 1; k < n; k++) s -= L[k * n + i] * x[k];
    x[i] = s / L[i * n + i];
  }
  return x;
}

/**
 * Non-negative least squares on normal equations (A = MᵀM, b = Mᵀy) by
 * projected coordinate descent. n is tiny here (a handful of light terms).
 */
export function nnlsNormal(A: Float64Array, b: Float64Array, n: number, iters = 200): Float64Array {
  const x = new Float64Array(n);
  for (let it = 0; it < iters; it++) {
    let change = 0;
    for (let i = 0; i < n; i++) {
      const d = A[i * n + i];
      if (d <= 0) continue;
      let g = b[i];
      for (let j = 0; j < n; j++) if (j !== i) g -= A[i * n + j] * x[j];
      const v = Math.max(0, g / d);
      change = Math.max(change, Math.abs(v - x[i]));
      x[i] = v;
    }
    if (change < 1e-12) break;
  }
  return x;
}

/** Median of a numeric array (copies). */
export function median(values: ArrayLike<number>): number {
  const a = Float64Array.from(values as ArrayLike<number>).sort();
  if (a.length === 0) return 0;
  const m = a.length >> 1;
  return a.length % 2 ? a[m] : 0.5 * (a[m - 1] + a[m]);
}

/** Value at quantile q ∈ [0, 1] (copies). */
export function quantile(values: ArrayLike<number>, q: number): number {
  const a = Float64Array.from(values as ArrayLike<number>).sort();
  if (a.length === 0) return 0;
  const i = Math.min(a.length - 1, Math.max(0, Math.round(q * (a.length - 1))));
  return a[i];
}

/** Deterministic PRNG (mulberry32) so fits are reproducible. */
export function rng(seed: number): () => number {
  let t = seed >>> 0;
  return () => {
    t = (t + 0x6d2b79f5) >>> 0;
    let r = Math.imul(t ^ (t >>> 15), 1 | t);
    r = (r + Math.imul(r ^ (r >>> 7), 61 | r)) ^ r;
    return ((r ^ (r >>> 14)) >>> 0) / 4294967296;
  };
}
