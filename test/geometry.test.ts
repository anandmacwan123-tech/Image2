import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import { inputSize, postprocess, recoverFocalShift, tokenGrid } from "../src/geometry/moge-post";

describe("focal and shift recovery", () => {
  it("recovers a known focal and shift from a synthetic point map", () => {
    const W = 96, H = 72, focal = 1.3, shift = 0.7;
    const aspect = W / H, diag = Math.sqrt(1 + aspect * aspect);
    const points = new Float32Array(W * H * 3);
    const mask = new Float32Array(W * H).fill(1);
    for (let j = 0; j < H; j++) {
      for (let i = 0; i < W; i++) {
        const u = (aspect / diag) * ((2 * i + 1 - W) / W);
        const v = (1 / diag) * ((2 * j + 1 - H) / H);
        const z = 2 + Math.sin(i * 0.1) + 0.5 * Math.cos(j * 0.13);
        const p = (j * W + i) * 3;
        points[p] = (u * z) / focal;
        points[p + 1] = (v * z) / focal;
        points[p + 2] = z - shift;
      }
    }
    const r = recoverFocalShift(points, mask, W, H);
    expect(r.focal).toBeCloseTo(focal, 4);
    expect(r.shift).toBeCloseTo(shift, 4);
  });

  it("matches MoGe's Python infer() on a real output", () => {
    const ref = JSON.parse(readFileSync(new URL("./fixtures/moge-raw.json", import.meta.url), "utf8"));
    const buf = readFileSync(new URL("./fixtures/moge-raw.f32", import.meta.url));
    const all = new Float32Array(buf.buffer, buf.byteOffset, buf.byteLength / 4);
    const { width: W, height: H } = ref;
    const points = all.slice(0, W * H * 3);
    const mask = all.slice(W * H * 3, W * H * 4);
    const normal = new Float32Array(W * H * 3);
    for (let p = 0; p < W * H; p++) normal[p * 3 + 2] = -1;

    const r = recoverFocalShift(points, mask, W, H);
    // scipy stops at ftol=1e-3, so allow a little slack against it.
    expect(Math.abs(r.focal - ref.focal) / ref.focal).toBeLessThan(5e-3);
    expect(Math.abs(r.shift - ref.shift)).toBeLessThan(5e-3);

    const g = postprocess({ width: W, height: H, points, normal, mask, metricScale: ref.metricScale });
    expect(Math.abs(g.fx - ref.fx) / ref.fx).toBeLessThan(5e-3);
    expect(Math.abs(g.fy - ref.fy) / ref.fy).toBeLessThan(5e-3);
    const depths = Array.from(g.depth).filter((d) => d > 0).sort((a, b) => a - b);
    expect(depths.length).toBe(ref.validCount);
    const med = depths.length % 2 ? depths[depths.length >> 1] : 0.5 * (depths[depths.length / 2 - 1] + depths[depths.length / 2]);
    expect(Math.abs(med - ref.depthMedian) / ref.depthMedian).toBeLessThan(5e-3);
    // Converted to three.js space: the camera looks down -z.
    const i = g.depth.findIndex((d) => d > 0);
    expect(g.points[i * 3 + 2]).toBeCloseTo(-g.depth[i], 5);
  });

  it("sizes the model input on the token grid", () => {
    expect(tokenGrid(1800, 4 / 3)).toEqual({ rows: 37, cols: 49 });
    const s = inputSize(1800, 4032, 3024);
    expect(s.height).toBe(37 * 14);
    expect(Math.abs(s.width / s.height - 4 / 3)).toBeLessThan(0.005);
  });
});
