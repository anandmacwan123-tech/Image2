// 16-bit greyscale PNG, for depth maps that Blender and TouchDesigner can read
// without banding.

import { encode } from "fast-png";

export function encodeGrey16(data: Uint16Array, width: number, height: number): Uint8Array<ArrayBuffer> {
  const png = encode({ width, height, data, depth: 16, channels: 1 });
  return png.buffer instanceof ArrayBuffer ? (png as Uint8Array<ArrayBuffer>) : new Uint8Array(png);
}
