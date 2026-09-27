// Radiance .hdr (RGBE) writer with run-length encoded scanlines.

export function encodeHDR(rgb: Float32Array, width: number, height: number): Uint8Array<ArrayBuffer> {
  const header = `#?RADIANCE\n# depth-light fitted environment\nFORMAT=32-bit_rle_rgbe\nEXPOSURE=1.0\n\n-Y ${height} +X ${width}\n`;
  const head = new TextEncoder().encode(header);
  const chunks: Uint8Array[] = [head];
  const rgbe = new Uint8Array(width * 4);
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const i = (y * width + x) * 3;
      const [r, g, b] = toRGBE(rgb[i], rgb[i + 1], rgb[i + 2]);
      rgbe[x * 4] = r;
      rgbe[x * 4 + 1] = g;
      rgbe[x * 4 + 2] = b;
      rgbe[x * 4 + 3] = rgbeExp(rgb[i], rgb[i + 1], rgb[i + 2]);
    }
    chunks.push(scanline(rgbe, width));
  }
  const total = chunks.reduce((n, c) => n + c.length, 0);
  const out = new Uint8Array(total);
  let o = 0;
  for (const c of chunks) {
    out.set(c, o);
    o += c.length;
  }
  return out;
}

function rgbeExp(r: number, g: number, b: number): number {
  const m = Math.max(r, g, b);
  if (m < 1e-32) return 0;
  return Math.floor(Math.log2(m)) + 1 + 128;
}

function toRGBE(r: number, g: number, b: number): [number, number, number] {
  const m = Math.max(r, g, b);
  if (m < 1e-32) return [0, 0, 0];
  const e = Math.floor(Math.log2(m)) + 1;
  const scale = 256 / Math.pow(2, e);
  return [Math.min(255, Math.floor(r * scale)), Math.min(255, Math.floor(g * scale)), Math.min(255, Math.floor(b * scale))];
}

/** New-style RLE: 2 2 hi lo, then each of the four channels run-length encoded. */
function scanline(rgbe: Uint8Array, width: number): Uint8Array {
  if (width < 8 || width > 0x7fff) return rgbe.slice();
  const out: number[] = [2, 2, (width >> 8) & 0xff, width & 0xff];
  for (let c = 0; c < 4; c++) {
    let x = 0;
    while (x < width) {
      // Look for a run of at least 3.
      let run = 1;
      while (x + run < width && run < 127 && rgbe[(x + run) * 4 + c] === rgbe[x * 4 + c]) run++;
      if (run >= 3) {
        out.push(128 + run, rgbe[x * 4 + c]);
        x += run;
        continue;
      }
      // Literal span until the next run of 3 or 128 bytes.
      let len = 0;
      while (x + len < width && len < 128) {
        const v = rgbe[(x + len) * 4 + c];
        if (x + len + 2 < width && rgbe[(x + len + 1) * 4 + c] === v && rgbe[(x + len + 2) * 4 + c] === v) break;
        len++;
      }
      if (len === 0) len = 1;
      out.push(len);
      for (let k = 0; k < len; k++) out.push(rgbe[(x + k) * 4 + c]);
      x += len;
    }
  }
  return Uint8Array.from(out);
}
