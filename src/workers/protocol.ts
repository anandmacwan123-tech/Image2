// Messages between the main thread and the inference worker. Large arrays
// travel as transferables, so maps move without copying.

import type { Geometry } from "../geometry/moge-post";
import type { LightFit } from "../light/fit";
import type { PhotoStats } from "../lib/stats";

export type ToWorker =
  | { type: "load"; prefer: "webgpu" | "wasm"; tokens?: number }
  | { type: "analyze"; id: number; bytes: ArrayBuffer; mime: string };

export interface Analysis {
  geometry: Geometry;
  fit: LightFit;
  stats: PhotoStats;
  /** Working-resolution sRGB pixels, the size of the geometry. */
  rgba: Uint8ClampedArray;
  /** Execution provider that ran the model, or "cache". */
  backend: string;
  ms: number;
}

export type FromWorker =
  | { type: "progress"; phase: "download" | "infer" | "fit"; fraction: number }
  | { type: "ready"; backend: string }
  | { type: "result"; id: number; analysis: Analysis }
  | { type: "error"; id?: number; message: string };

/** Every transferable buffer inside an analysis. */
export function transferables(a: Analysis): Transferable[] {
  const g = a.geometry, l = a.fit.local;
  const list = [g.depth, g.points, g.normals, g.mask, a.rgba, l.position, l.basis, l.value, l.albedo, l.keyTerm, l.sourceTerm, l.weight];
  return [...new Set(list.map((x) => x.buffer))] as Transferable[];
}
