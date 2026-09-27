// Refine the key light: fit I(x) ≈ s_r(x) · (a + k · max(0, n_x · ℓ)).
//
// For a fixed direction ℓ the rest has a closed form. Write u_r = s_r·a and
// ρ = k/a; then I ≈ u_r · (1 + ρ t) with t = max(0, n·ℓ), each u_r is a
// one-line least-squares solve, and the cost left over is a rational function
// of ρ built from six sums per region. So scoring a direction is one pass over
// the samples, cheap enough to search the whole sphere densely and then
// polish the best candidates with Nelder–Mead.

export type Vec3 = [number, number, number];

export interface RatioFit {
  /** Normalised residual: 0 is a perfect fit, 1 explains nothing. */
  cost: number;
  /** Key-to-ambient ratio k/a. */
  rho: number;
  /** Per-region u_r = s_r · a. */
  u: Float64Array;
}

/** Sample terms t = max(0, n · ℓ) for a distant light. */
export function directionalTerm(normal: Float32Array, count: number, dir: Vec3, out: Float32Array): Float32Array {
  const [lx, ly, lz] = dir;
  for (let i = 0; i < count; i++) {
    const d = normal[i * 3] * lx + normal[i * 3 + 1] * ly + normal[i * 3 + 2] * lz;
    out[i] = d > 0 ? d : 0;
  }
  return out;
}

/** Sample terms for a light at a point: t = max(0, n · normalize(p_light − x)). */
export function positionalTerm(normal: Float32Array, position: Float32Array, count: number, light: Vec3, out: Float32Array): Float32Array {
  for (let i = 0; i < count; i++) {
    const dx = light[0] - position[i * 3], dy = light[1] - position[i * 3 + 1], dz = light[2] - position[i * 3 + 2];
    const len = Math.hypot(dx, dy, dz) || 1;
    const d = (normal[i * 3] * dx + normal[i * 3 + 1] * dy + normal[i * 3 + 2] * dz) / len;
    out[i] = d > 0 ? d : 0;
  }
  return out;
}

/** Best ρ ≥ 0 and region scales for fixed light terms t. */
export function solveRatio(I: Float32Array, w: Float32Array, id: Uint16Array, regions: number, t: Float32Array): RatioFit {
  const S = new Float64Array(regions * 6);
  const n = I.length;
  for (let i = 0; i < n; i++) {
    const wi = w[i];
    if (wi === 0) continue;
    const r = id[i] * 6, ti = t[i], Ii = I[i];
    const wt = wi * ti, wI = wi * Ii;
    S[r] += wi;
    S[r + 1] += wt;
    S[r + 2] += wt * ti;
    S[r + 3] += wI;
    S[r + 4] += wI * ti;
    S[r + 5] += wI * Ii;
  }
  let total = 0;
  for (let r = 0; r < regions; r++) total += S[r * 6 + 5];
  const cost = (rho: number): number => {
    let c = 0;
    for (let r = 0; r < regions; r++) {
      const o = r * 6;
      const den = S[o] + 2 * rho * S[o + 1] + rho * rho * S[o + 2];
      const num = S[o + 3] + rho * S[o + 4];
      c += den > 0 ? S[o + 5] - (num * num) / den : S[o + 5];
    }
    return c;
  };

  // ρ = tan θ. Coarse scan over θ, then golden section in the best bracket.
  const steps = 32, top = Math.PI / 2 - 1e-3;
  let bestK = 0, bestC = Infinity;
  for (let k = 0; k <= steps; k++) {
    const c = cost(Math.tan((k / steps) * top));
    if (c < bestC) { bestC = c; bestK = k; }
  }
  let lo = (Math.max(0, bestK - 1) / steps) * top;
  let hi = (Math.min(steps, bestK + 1) / steps) * top;
  const g = (Math.sqrt(5) - 1) / 2;
  let x1 = hi - g * (hi - lo), x2 = lo + g * (hi - lo);
  let f1 = cost(Math.tan(x1)), f2 = cost(Math.tan(x2));
  for (let it = 0; it < 48; it++) {
    if (f1 < f2) { hi = x2; x2 = x1; f2 = f1; x1 = hi - g * (hi - lo); f1 = cost(Math.tan(x1)); }
    else { lo = x1; x1 = x2; f1 = f2; x2 = lo + g * (hi - lo); f2 = cost(Math.tan(x2)); }
  }
  const theta = f1 < f2 ? x1 : x2;
  const rho = Math.tan(theta);
  const u = new Float64Array(regions);
  for (let r = 0; r < regions; r++) {
    const o = r * 6;
    const den = S[o] + 2 * rho * S[o + 1] + rho * rho * S[o + 2];
    u[r] = den > 0 ? (S[o + 3] + rho * S[o + 4]) / den : 0;
  }
  return { cost: total > 0 ? cost(rho) / total : 1, rho, u };
}

/** Evenly spread unit vectors (Fibonacci sphere). */
export function fibonacciSphere(n: number): Vec3[] {
  const out: Vec3[] = [];
  const golden = Math.PI * (3 - Math.sqrt(5));
  for (let i = 0; i < n; i++) {
    const y = 1 - (2 * (i + 0.5)) / n;
    const r = Math.sqrt(1 - y * y);
    out.push([Math.cos(golden * i) * r, y, Math.sin(golden * i) * r]);
  }
  return out;
}

function normalize(v: Vec3): Vec3 {
  const l = Math.hypot(v[0], v[1], v[2]) || 1;
  return [v[0] / l, v[1] / l, v[2] / l];
}

function tangentBasis(d: Vec3): [Vec3, Vec3] {
  const a: Vec3 = Math.abs(d[1]) < 0.9 ? [0, 1, 0] : [1, 0, 0];
  const e1 = normalize([a[1] * d[2] - a[2] * d[1], a[2] * d[0] - a[0] * d[2], a[0] * d[1] - a[1] * d[0]]);
  const e2: Vec3 = [d[1] * e1[2] - d[2] * e1[1], d[2] * e1[0] - d[0] * e1[2], d[0] * e1[1] - d[1] * e1[0]];
  return [e1, e2];
}

export function angleBetween(a: Vec3, b: Vec3): number {
  const d = (a[0] * b[0] + a[1] * b[1] + a[2] * b[2]) / (Math.hypot(...a) * Math.hypot(...b) || 1);
  return Math.acos(Math.max(-1, Math.min(1, d)));
}

export interface KeyFit extends RatioFit {
  dir: Vec3;
  /** Robust weights under the key model, per input sample. */
  w: Float32Array;
}

/**
 * Cauchy weights on the relative residuals of I ≈ u_r (1 + ρ t), scaled by
 * their median absolute deviation. Cast shadows, highlights and pixels whose
 * albedo differs from the rest of their region fall away as outliers.
 */
export function keyWeights(I: Float32Array, base: Float32Array, id: Uint16Array, t: Float32Array, fit: RatioFit, out?: Float32Array): Float32Array {
  const n = I.length;
  const e = new Float32Array(n);
  let mean = 0;
  for (let i = 0; i < n; i++) mean += I[i];
  mean /= Math.max(n, 1);
  const abs = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    const pred = fit.u[id[i]] * (1 + fit.rho * t[i]);
    e[i] = (I[i] - pred) / Math.max(pred, 0.02 * mean);
    abs[i] = Math.abs(e[i]);
  }
  abs.sort();
  const sigma = Math.max(1.4826 * abs[n >> 1], 0.01);
  const k = 2.5 * sigma;
  const w = out ?? new Float32Array(n);
  for (let i = 0; i < n; i++) {
    const q = e[i] / k;
    w[i] = base[i] / (1 + q * q);
  }
  return w;
}

/**
 * Find the key direction. Scores a dense set of directions plus the seed from
 * the ambient fit, polishes the best few with Nelder–Mead on the tangent plane
 * of each candidate, then reweights robustly and polishes again.
 */
export function fitKeyDirection(
  normal: Float32Array,
  I: Float32Array,
  base: Float32Array,
  id: Uint16Array,
  regions: number,
  seed: Vec3,
  candidates = 1200,
  rounds = 3,
): KeyFit {
  const n = I.length;
  const t = new Float32Array(n);
  let w = base;
  const score = (d: Vec3): RatioFit => solveRatio(I, w, id, regions, directionalTerm(normal, n, d, t));
  const polish = (start: Vec3, step: number): { dir: Vec3; fit: RatioFit } => {
    const [e1, e2] = tangentBasis(start);
    const at = (p: [number, number]): Vec3 =>
      normalize([
        start[0] + p[0] * e1[0] + p[1] * e2[0],
        start[1] + p[0] * e1[1] + p[1] * e2[1],
        start[2] + p[0] * e1[2] + p[1] * e2[2],
      ]);
    const dir = at(nelderMead2((p) => score(at(p)).cost, step, 120, 1e-6));
    return { dir, fit: score(dir) };
  };

  const scored: { dir: Vec3; cost: number }[] = [];
  for (const d of [normalize(seed), ...fibonacciSphere(candidates)]) scored.push({ dir: d, cost: score(d).cost });
  scored.sort((a, b) => a.cost - b.cost);
  const starts: Vec3[] = [];
  for (const s of scored) {
    if (starts.every((d) => angleBetween(d, s.dir) > 0.25)) starts.push(s.dir);
    if (starts.length === 4) break;
  }
  let best = { dir: starts[0], fit: score(starts[0]) };
  for (const start of starts) {
    const r = polish(start, 0.06);
    if (r.fit.cost < best.fit.cost) best = r;
  }

  // Reweight against the key model itself and polish from where we are.
  for (let round = 0; round < rounds; round++) {
    w = keyWeights(I, base, id, directionalTerm(normal, n, best.dir, t), best.fit);
    const r = polish(best.dir, 0.03);
    best = r.fit.cost <= score(best.dir).cost ? r : { dir: best.dir, fit: score(best.dir) };
  }
  return { dir: best.dir, ...best.fit, w };
}

/** Minimise a 2-D function from the origin with Nelder–Mead. */
export function nelderMead2(f: (p: [number, number]) => number, step: number, iters: number, tol: number): [number, number] {
  let s: { p: [number, number]; v: number }[] = [
    { p: [0, 0], v: 0 },
    { p: [step, 0], v: 0 },
    { p: [0, step], v: 0 },
  ];
  for (const x of s) x.v = f(x.p);
  for (let it = 0; it < iters; it++) {
    s.sort((a, b) => a.v - b.v);
    const size = Math.max(Math.hypot(s[1].p[0] - s[0].p[0], s[1].p[1] - s[0].p[1]), Math.hypot(s[2].p[0] - s[0].p[0], s[2].p[1] - s[0].p[1]));
    if (size < tol) break;
    const c: [number, number] = [(s[0].p[0] + s[1].p[0]) / 2, (s[0].p[1] + s[1].p[1]) / 2];
    const w = s[2];
    const lerp = (k: number): [number, number] => [c[0] + k * (w.p[0] - c[0]), c[1] + k * (w.p[1] - c[1])];
    const r = lerp(-1), fr = f(r);
    if (fr < s[0].v) {
      const e = lerp(-2), fe = f(e);
      s[2] = fe < fr ? { p: e, v: fe } : { p: r, v: fr };
    } else if (fr < s[1].v) {
      s[2] = { p: r, v: fr };
    } else {
      const k = fr < w.v ? -0.5 : 0.5;
      const q = lerp(k), fq = f(q);
      if (fq < Math.min(fr, w.v)) s[2] = { p: q, v: fq };
      else {
        s = s.map((x, i) => (i === 0 ? x : { p: [(x.p[0] + s[0].p[0]) / 2, (x.p[1] + s[0].p[1]) / 2] as [number, number], v: 0 }));
        s[1].v = f(s[1].p);
        s[2].v = f(s[2].p);
      }
    }
  }
  s.sort((a, b) => a.v - b.v);
  return s[0].p;
}
