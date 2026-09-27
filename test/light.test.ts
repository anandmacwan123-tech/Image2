import { describe, expect, it } from "vitest";
import { fitLight, prepare } from "../src/light/fit";
import { fitAmbientSH, basisFor } from "../src/light/ambient";
import { fitKeyDirection, angleBetween, type Vec3 } from "../src/light/refine";
import { localRefit } from "../src/light/local";
import { shDirectional, shEval, shConstant, toThreeProbe } from "../src/light/sh";
import { normalize, renderRoom } from "./synthetic-room";

const deg = (rad: number) => (rad * 180) / Math.PI;

// Three light directions (toward the light, three.js camera space).
const LIGHTS: Record<string, Vec3> = {
  "upper right": normalize([0.55, 0.75, 0.35]),
  "upper left, behind": normalize([-0.6, 0.7, -0.4]),
  "from the camera": normalize([0.15, 0.45, 0.88]),
};

describe("spherical harmonics", () => {
  it("projects a directional light to within band-limit error", () => {
    const l = normalize([0.3, 0.8, 0.5]);
    const c = shDirectional(1, ...l);
    // Mean of max(0, n·l) over the sphere is 1/4; SH keeps it exactly.
    expect(c[0] * 0.282095).toBeCloseTo(0.25, 5);
    // Peak of the band-limited clamped cosine is close to 1 (≈ 1.02 at l ≤ 2).
    expect(shEval(c, ...l)).toBeGreaterThan(0.95);
    expect(shEval(c, ...l)).toBeLessThan(1.1);
  });

  it("maps shading to three.js probe coefficients", () => {
    // A constant shading of 1 must render a surface of albedo ρ as ρ:
    // three.js irradiance is 0.886227·L00, and ρ/π·E = ρ needs E = π.
    const L = toThreeProbe(shConstant(1));
    expect(L[0] * 0.886227).toBeCloseTo(Math.PI, 4);
  });
});

describe("light fit on a synthetic room", () => {
  for (const [name, light] of Object.entries(LIGHTS)) {
    it(`finds the key within 1° (${name}, dark floor)`, () => {
      const room = renderRoom({ light });
      const fit = fitLight(room.rgba, room.geometry);
      expect(deg(angleBetween(fit.mono.key.dir, light))).toBeLessThan(1);
      expect(deg(angleBetween(fit.colour.key.dir, light))).toBeLessThan(1);
    });

    it(`finds the key within 1° (${name}, white floor)`, () => {
      const room = renderRoom({ light, floorAlbedo: [0.8, 0.8, 0.8] });
      const fit = fitLight(room.rgba, room.geometry);
      expect(deg(angleBetween(fit.mono.key.dir, light))).toBeLessThan(1);
    });
  }

  it("does much worse without regions or a floor split (the failure the plan measured)", () => {
    const light = LIGHTS["upper right"];
    const room = renderRoom({ light });
    const { samples } = prepare(room.rgba, room.geometry);
    const n = samples.count;
    const I = new Float32Array(n);
    for (let i = 0; i < n; i++) I[i] = 0.2126 * samples.rgb[i * 3] + 0.7152 * samples.rgb[i * 3 + 1] + 0.0722 * samples.rgb[i * 3 + 2];
    const one = new Uint16Array(n);
    const w = new Float32Array(n).fill(1);
    const plain = fitKeyDirection(samples.normal, I, w, one, 1, [0, 1, 0]);
    expect(deg(angleBetween(plain.dir, light))).toBeGreaterThan(5);
  });

  it("keeps the key close with cast shadows present", () => {
    const light = LIGHTS["upper right"];
    const room = renderRoom({ light, shadows: true });
    const fit = fitLight(room.rgba, room.geometry);
    expect(deg(angleBetween(fit.mono.key.dir, light))).toBeLessThan(5);
  });

  it("treats the white walls as white and recovers the light's strength", () => {
    const light = LIGHTS["upper right"];
    const room = renderRoom({ light, ambient: 0.3, key: 0.9 });
    const fit = fitLight(room.rgba, room.geometry);
    // Walls are albedo 0.8 = WHITE, so shading is exactly exposure × light.
    expect(fit.info.neutralReference).toBe(true);
    expect(fit.mono.key.strength[0] / (room.exposure * 0.9)).toBeCloseTo(1, 1);
  });

  it("white-balances a tinted light in colour mode", () => {
    const light = LIGHTS["upper right"];
    const room = renderRoom({ light, lightColour: [1, 0.8, 0.55] });
    const fit = fitLight(room.rgba, room.geometry);
    const k = fit.colour.key.strength;
    // The walls are white, so the light itself carries the tint.
    expect(k[1] / k[0]).toBeCloseTo(0.8, 1);
    expect(k[2] / k[0]).toBeCloseTo(0.55, 1);
  });
});

describe("ambient SH fit", () => {
  it("reproduces shading at the visible normals when albedo is known", () => {
    const light = LIGHTS["upper right"];
    const room = renderRoom({ light, floorAlbedo: [0.8, 0.8, 0.8] });
    const { samples } = prepare(room.rgba, room.geometry);
    const n = samples.count;
    const I = new Float32Array(n);
    for (let i = 0; i < n; i++) I[i] = samples.rgb[i * 3 + 1] / room.albedo[samples.pixel[i] * 3 + 1];
    const fit = fitAmbientSH(I, basisFor(samples.normal, n), new Uint16Array(n), 1, null, false);
    // Band-limited SH cannot draw a sharp terminator, and with only
    // camera-facing normals its band-1 direction drifts (about 20° here),
    // which is why the key is refined separately. The shading itself holds.
    let err = 0, tot = 0;
    for (let i = 0; i < n; i++) {
      const pred = fit.s[0] * shEval(fit.c, samples.normal[i * 3], samples.normal[i * 3 + 1], samples.normal[i * 3 + 2]);
      err += (pred - I[i]) ** 2;
      tot += I[i] ** 2;
    }
    expect(Math.sqrt(err / tot)).toBeLessThan(0.1);
  });
});

describe("local refit", () => {
  it("matches the global fit with no nearby data and follows local shadow", () => {
    const light = LIGHTS["upper right"];
    const room = renderRoom({ light, shadows: true });
    const fit = fitLight(room.rgba, room.geometry);
    const prior = { channels: 1 as const, ambient: fit.mono.ambient, keyStrength: fit.mono.key.strength };
    const far = localRefit(fit.local, prior, [100, 100, 100], 0.2);
    expect(far.keyScale).toBeCloseTo(1, 3);
    for (let i = 0; i < 9; i++) expect(far.ambient[0][i]).toBeCloseTo(fit.mono.ambient[0][i], 3);
    const open = localRefit(fit.local, prior, [-1.5, -1.3, -2.0], 0.2);
    expect(open.keyScale).toBeGreaterThan(0.7);
  });

  it("runs fast enough for every frame", () => {
    const room = renderRoom({ light: LIGHTS["upper right"] });
    const fit = fitLight(room.rgba, room.geometry);
    const prior = { channels: 3 as const, ambient: fit.colour.ambient, keyStrength: fit.colour.key.strength };
    const t0 = performance.now();
    for (let i = 0; i < 20; i++) localRefit(fit.local, prior, [0, -1, -3 - i * 0.05], 0.3);
    const ms = (performance.now() - t0) / 20;
    expect(ms).toBeLessThan(16);
  });
});
