// Model files carry a version in their name so a new export never collides
// with a cached one. Bump MODEL_VERSION when the weights change.
export const MODEL_VERSION = "v1";
export const MODEL_URL = `/models/moge-2-vits-normal-${MODEL_VERSION}.onnx`;

/**
 * ViT tokens for MoGe-2 (the model accepts 1200–3600). 1800 puts the input
 * at 518 px tall, which runs in about a second on WebGPU and keeps the WASM
 * fallback usable.
 */
export const NUM_TOKENS = 1800;
export const NUM_TOKENS_WASM = 1200;

/** Longest side of the photo kept for display and export. */
export const MAX_PHOTO_SIDE = 8192;

/** Side of the centre crop used to measure noise and blur. */
export const STATS_CROP = 768;
