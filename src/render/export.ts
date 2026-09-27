// Exports at the photo's full resolution. The composite and light map are
// rendered by the Stage; depth, HDRI and data are computed here.

import type { Geometry } from "../geometry/moge-post";
import type { LightFit, LightModel, Vec3 } from "../light/fit";
import { radianceAt, toThreeProbe } from "../light/sh";
import { encodeGrey16 } from "../lib/png16";
import { encodeHDR } from "../lib/hdr";

export function download(data: Blob, name: string): void {
  const url = URL.createObjectURL(data);
  const a = document.createElement("a");
  a.href = url;
  a.download = name;
  document.body.append(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 10000);
}

export async function pngFromRGBA(rgba: Uint8Array<ArrayBuffer>, width: number, height: number): Promise<Blob> {
  const canvas = new OffscreenCanvas(width, height);
  const ctx = canvas.getContext("2d")!;
  ctx.putImageData(new ImageData(new Uint8ClampedArray(rgba.buffer, rgba.byteOffset, rgba.byteLength), width, height), 0, 0);
  return canvas.convertToBlob({ type: "image/png" });
}

/**
 * Depth at full resolution as a 16-bit PNG: white is near, black is far,
 * value = (far − z) / (far − near). Sky and invalid pixels are 0. Upsampled
 * bilinearly, never blending across invalid pixels.
 */
export function depthPNG(g: Geometry, width: number, height: number, near: number, far: number): Uint8Array<ArrayBuffer> {
  const out = new Uint16Array(width * height);
  const range = far - near;
  for (let y = 0; y < height; y++) {
    const sy = Math.min(g.height - 1, Math.max(0, ((y + 0.5) / height) * g.height - 0.5));
    const y0 = Math.floor(sy), y1 = Math.min(g.height - 1, y0 + 1), fy = sy - y0;
    for (let x = 0; x < width; x++) {
      const sx = Math.min(g.width - 1, Math.max(0, ((x + 0.5) / width) * g.width - 0.5));
      const x0 = Math.floor(sx), x1 = Math.min(g.width - 1, x0 + 1), fx = sx - x0;
      const taps = [
        [y0 * g.width + x0, (1 - fx) * (1 - fy)],
        [y0 * g.width + x1, fx * (1 - fy)],
        [y1 * g.width + x0, (1 - fx) * fy],
        [y1 * g.width + x1, fx * fy],
      ];
      let z = 0, w = 0;
      for (const [p, wt] of taps) {
        if (g.mask[p] && wt > 0) {
          z += g.depth[p] * wt;
          w += wt;
        }
      }
      // Nearest decides validity, so silhouettes stay sharp.
      const nearest = (fy < 0.5 ? y0 : y1) * g.width + (fx < 0.5 ? x0 : x1);
      if (!g.mask[nearest] || w === 0) continue;
      const v = Math.min(1, Math.max(0, (far - z / w) / range));
      out[y * width + x] = Math.round(v * 65535);
    }
  }
  return encodeGrey16(out, width, height);
}

/** Rotation taking camera-space "up" to +y, as a 3×3 row-major matrix. */
export function levelRotation(up: Vec3): number[] {
  const [x, y, z] = up;
  // Rodrigues from up to (0, 1, 0), about up × (0, 1, 0).
  const axis: Vec3 = [-z, 0, x];
  const s = Math.hypot(...axis), c = y;
  if (s < 1e-9) return [1, 0, 0, 0, 1, 0, 0, 0, 1];
  const k: Vec3 = [axis[0] / s, axis[1] / s, axis[2] / s];
  const K = [0, -k[2], k[1], k[2], 0, -k[0], -k[1], k[0], 0];
  const K2 = [0, 0, 0, 0, 0, 0, 0, 0, 0];
  for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) for (let m = 0; m < 3; m++) K2[i * 3 + j] += K[i * 3 + m] * K[m * 3 + j];
  const R = [1, 0, 0, 0, 1, 0, 0, 0, 1];
  for (let i = 0; i < 9; i++) R[i] += s * K[i] + (1 - c) * K2[i];
  return R;
}

function mul(R: number[], v: Vec3): Vec3 {
  return [R[0] * v[0] + R[1] * v[1] + R[2] * v[2], R[3] * v[0] + R[4] * v[1] + R[5] * v[2], R[6] * v[0] + R[7] * v[1] + R[8] * v[2]];
}

export interface LightSettings {
  key: number;
  fill: number;
  keyDir: Vec3;
}

/**
 * The fitted light as an equirectangular .hdr (three.js convention: +y up,
 * u = 0.5 looking along +x, u = 0.25 along −z), levelled so gravity points
 * down. Ambient from the SH, the key and each source as small discs whose
 * radiance gives the fitted irradiance.
 */
export function hdri(fit: LightFit, model: LightModel, settings: LightSettings, width = 1024): Uint8Array<ArrayBuffer> {
  const height = width / 2;
  const R = levelRotation(fit.up);
  const Rt = [R[0], R[3], R[6], R[1], R[4], R[7], R[2], R[5], R[8]];
  const ch = model.channels;
  const L = model.ambient.map((c) => toThreeProbe(c));
  const rgb = new Float32Array(width * height * 3);
  const dirs = new Float32Array(width * height * 3);
  const solid = new Float32Array(width * height);
  for (let j = 0; j < height; j++) {
    const lat = (0.5 - (j + 0.5) / height) * Math.PI;
    const dOmega = ((2 * Math.PI) / width) * (Math.PI / height) * Math.cos(lat);
    for (let i = 0; i < width; i++) {
      const phi = ((i + 0.5) / width - 0.5) * 2 * Math.PI;
      const world: Vec3 = [Math.cos(phi) * Math.cos(lat), Math.sin(lat), Math.sin(phi) * Math.cos(lat)];
      const d = mul(Rt, world);
      const p = j * width + i;
      dirs.set(d, p * 3);
      solid[p] = dOmega;
      for (let c = 0; c < 3; c++) rgb[p * 3 + c] = radianceAt(L[ch === 1 ? 0 : c], d[0], d[1], d[2]) * settings.fill;
    }
  }
  const disc = (dir: Vec3, radiusDeg: number, strength: number[]) => {
    const cosR = Math.cos((radiusDeg * Math.PI) / 180);
    let omega = 0;
    const inside: number[] = [];
    for (let p = 0; p < width * height; p++) {
      if (dirs[p * 3] * dir[0] + dirs[p * 3 + 1] * dir[1] + dirs[p * 3 + 2] * dir[2] >= cosR) {
        inside.push(p);
        omega += solid[p];
      }
    }
    if (omega === 0) return;
    for (const p of inside) for (let c = 0; c < 3; c++) rgb[p * 3 + c] += (Math.PI * strength[ch === 1 ? 0 : c]) / omega;
  };
  disc(settings.keyDir, 2, model.key.strength.map((k) => k * settings.key));
  for (const s of model.sources) {
    const v: Vec3 = [s.position[0] - fit.refPoint[0], s.position[1] - fit.refPoint[1], s.position[2] - fit.refPoint[2]];
    const l = Math.hypot(...v) || 1;
    disc([v[0] / l, v[1] / l, v[2] / l], 3, s.strength);
  }
  return encodeHDR(rgb, width, height);
}

export function dataJSON(
  fit: LightFit,
  model: LightModel,
  g: Geometry,
  photo: { width: number; height: number },
  depth: { near: number; far: number },
  settings: LightSettings & { soft: number; overridden: boolean },
  mode: "mono" | "colour",
): string {
  const round = (v: number) => Math.round(v * 1e6) / 1e6;
  const r = (a: number[]) => a.map(round);
  const fovX = (2 * Math.atan(0.5 / g.fx) * 180) / Math.PI;
  return JSON.stringify(
    {
      app: "depth-light",
      version: 1,
      image: photo,
      frame: "three.js camera space: +x right, +y up, -z forward, metres",
      camera: { fovY: round(g.fovY), fovX: round(fovX), fx: round(g.fx), fy: round(g.fy), metricScale: round(g.metricScale) },
      depth: { near: round(depth.near), far: round(depth.far), png: "16-bit grey, value = (far - z) / (far - near), 0 = sky or invalid" },
      up: r(fit.up),
      mode,
      light: {
        units: "shading S: a surface of albedo ρ appears as ρ·S in linear sRGB; the reference wall is albedo " + fit.white,
        basis: "real spherical harmonics l ≤ 2, three.js order: Y00, Y1-1, Y10, Y11, Y2-2, Y2-1, Y20, Y21, Y22",
        total: model.total.map(r),
        ambient: model.ambient.map(r),
        ambientThreeLightProbe: model.ambient.map((c) => r(Array.from(toThreeProbe(c)))),
        key: {
          direction: r(settings.keyDir),
          strength: r(model.key.strength),
          position: model.key.source ? r(model.key.source) : null,
          threeDirectionalLightIntensity: r(model.key.strength.map((k) => Math.PI * k * settings.key)),
        },
        sources: model.sources.map((s) => ({ position: r(s.position), strength: r(s.strength), d0: round(s.d0) })),
        controls: { key: round(settings.key), fill: round(settings.fill), soft: round(settings.soft), directionOverridden: settings.overridden },
      },
      hdri: { projection: "equirectangular, three.js convention, levelled so +y is up", rotationCameraToWorld: r(levelRotation(fit.up)) },
    },
    null,
    2,
  );
}
