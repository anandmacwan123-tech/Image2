// The probe ball: a small white sphere lit by the current light. Dragging it
// moves the key; the point under the pointer becomes the light direction, and
// dragging past the rim swings the light round behind the ball. A double
// click returns to the estimate.

import type { Vec3 } from "../light/fit";
import { shEval } from "../light/sh";

export interface ProbeLight {
  /** Ambient shading SH per channel (1 or 3). */
  ambient: number[][];
  /** Key shading per channel, sliders applied. */
  key: number[];
  fill: number;
  dir: Vec3;
  white: number;
}

export class Probe {
  private ctx: CanvasRenderingContext2D;
  private dragging = false;

  constructor(
    private canvas: HTMLCanvasElement,
    onDrag: (dir: Vec3) => void,
    onReset: () => void,
  ) {
    this.ctx = canvas.getContext("2d")!;
    const toDir = (ev: PointerEvent): Vec3 => {
      const r = canvas.getBoundingClientRect();
      const x = ((ev.clientX - r.left) / r.width) * 2 - 1;
      const y = 1 - ((ev.clientY - r.top) / r.height) * 2;
      // Inside the disc: the visible hemisphere. Beyond it: continue over the
      // rim, reaching straight behind at twice the radius.
      const d = Math.min(Math.hypot(x, y), 2);
      const theta = (d * Math.PI) / 2;
      const phi = Math.atan2(y, x);
      return [Math.sin(theta) * Math.cos(phi), Math.sin(theta) * Math.sin(phi), Math.cos(theta)];
    };
    canvas.addEventListener("pointerdown", (ev) => {
      this.dragging = true;
      canvas.setPointerCapture(ev.pointerId);
      canvas.style.cursor = "grabbing";
      onDrag(toDir(ev));
    });
    canvas.addEventListener("pointermove", (ev) => {
      if (this.dragging) onDrag(toDir(ev));
    });
    const end = () => {
      this.dragging = false;
      canvas.style.cursor = "";
    };
    canvas.addEventListener("pointerup", end);
    canvas.addEventListener("pointercancel", end);
    canvas.addEventListener("dblclick", onReset);
  }

  draw(light: ProbeLight): void {
    const dpr = Math.min(2, window.devicePixelRatio || 1);
    const size = Math.round(this.canvas.clientWidth * dpr) || 128;
    if (this.canvas.width !== size) {
      this.canvas.width = size;
      this.canvas.height = size;
    }
    const img = this.ctx.createImageData(size, size);
    const ch = light.ambient.length;
    const r = size / 2 - 1;
    for (let j = 0; j < size; j++) {
      for (let i = 0; i < size; i++) {
        const x = (i + 0.5 - size / 2) / r, y = (size / 2 - j - 0.5) / r;
        const rr = x * x + y * y;
        const o = (j * size + i) * 4;
        if (rr > 1) continue;
        const z = Math.sqrt(1 - rr);
        const cos = Math.max(0, x * light.dir[0] + y * light.dir[1] + z * light.dir[2]);
        for (let c = 0; c < 3; c++) {
          const k = ch === 1 ? 0 : c;
          const s = Math.max(0, shEval(light.ambient[k], x, y, z)) * light.fill + light.key[k] * cos;
          img.data[o + c] = Math.round(255 * encode(Math.min(1, light.white * s)));
        }
        // Soft edge.
        img.data[o + 3] = Math.round(255 * Math.min(1, (1 - Math.sqrt(rr)) * r));
      }
    }
    this.ctx.putImageData(img, 0, 0);
  }
}

function encode(v: number): number {
  return v <= 0.0031308 ? v * 12.92 : 1.055 * Math.pow(v, 1 / 2.4) - 0.055;
}
