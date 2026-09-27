// GLSL shared by the background, the shadow catcher and the reflection
// backdrop. Textures from the model use v = 0 for the top row.

export const MAX_SOURCES = 4;
export const MAX_CONTACTS = 16;

export const colourGLSL = /* glsl */ `
vec3 srgbToLinearV(vec3 c) {
  return mix(c / 12.92, pow((c + 0.055) / 1.055, vec3(2.4)), step(vec3(0.04045), c));
}
vec3 linearToSrgbV(vec3 c) {
  return mix(c * 12.92, 1.055 * pow(c, vec3(1.0 / 2.4)) - 0.055, step(vec3(0.0031308), c));
}
float luma(vec3 c) { return dot(c, vec3(0.2126, 0.7152, 0.0722)); }
`;

/** Shading model of the fit: ambient SH + key + point sources. */
export const lightGLSL = /* glsl */ `
uniform vec3 uAmb[9];
uniform vec3 uKey;
uniform vec3 uKeyFit;
uniform vec3 uKeyDir;
uniform vec3 uKeyPos;
uniform float uKeyPositional;
uniform float uFill;
uniform int uSrcCount;
uniform vec3 uSrcPos[${MAX_SOURCES}];
uniform vec3 uSrcK[${MAX_SOURCES}];
uniform float uSrcD0[${MAX_SOURCES}];

vec3 shShade(vec3 c[9], vec3 n) {
  return c[0] * 0.282095
    + c[1] * 0.488603 * n.y + c[2] * 0.488603 * n.z + c[3] * 0.488603 * n.x
    + c[4] * 1.092548 * n.x * n.y + c[5] * 1.092548 * n.y * n.z
    + c[6] * 0.315392 * (3.0 * n.z * n.z - 1.0)
    + c[7] * 1.092548 * n.x * n.z + c[8] * 0.546274 * (n.x * n.x - n.y * n.y);
}
vec3 keyDirAt(vec3 p) {
  return uKeyPositional > 0.5 ? normalize(uKeyPos - p) : uKeyDir;
}
float keyCos(vec3 n, vec3 p) { return max(dot(n, keyDirAt(p)), 0.0); }
vec3 sourceShade(vec3 n, vec3 p) {
  vec3 s = vec3(0.0);
  for (int i = 0; i < ${MAX_SOURCES}; i++) {
    if (i >= uSrcCount) break;
    vec3 d = uSrcPos[i] - p;
    float l2 = max(dot(d, d), 1e-6);
    float f = min(4.0, uSrcD0[i] * uSrcD0[i] / l2);
    s += uSrcK[i] * max(dot(n, d * inversesqrt(l2)), 0.0) * f;
  }
  return s;
}
vec3 ambientShade(vec3 n) { return max(shShade(uAmb, n), vec3(0.0)); }
`;

export const fullscreenVert = /* glsl */ `
varying vec2 vUv;
void main() {
  vUv = uv;
  gl_Position = vec4(position.xy, 0.0, 1.0);
}
`;

/** Image, depth or light map behind everything. */
export const backgroundFrag = /* glsl */ `
precision highp float;
uniform sampler2D uPhoto;
uniform sampler2D uNormal;
uniform sampler2D uPosition;
uniform sampler2D uDepth;
uniform int uView;
uniform float uNear;
uniform float uFar;
uniform float uWhite;
varying vec2 vUv;
${colourGLSL}
${lightGLSL}
void main() {
  vec2 uv = vec2(vUv.x, 1.0 - vUv.y);
  vec3 photo = texture2D(uPhoto, uv).rgb;
  vec3 col;
  if (uView == 1) {
    float z = texture2D(uDepth, uv).r;
    float g = z > 0.0 ? clamp((uFar - z) / (uFar - uNear), 0.0, 1.0) : 0.0;
    col = srgbToLinearV(vec3(g));
  } else if (uView == 2) {
    vec4 pos = texture2D(uPosition, uv);
    vec3 n = normalize(texture2D(uNormal, uv).xyz);
    // The sky is the light: show it as it is.
    col = pos.w > 0.5 ? uWhite * (ambientShade(n) * uFill + uKey * keyCos(n, pos.xyz) + sourceShade(n, pos.xyz)) : photo;
  } else {
    col = photo;
  }
  gl_FragColor = vec4(col, 1.0);
}
`;

/** Composite, then match the photo: blur, levels, mono and grain. */
export const gradeFrag = /* glsl */ `
precision highp float;
uniform sampler2D tBg;
uniform sampler2D tFg;
uniform vec2 uTexel;
uniform float uBlur;
uniform float uNoise;
uniform vec3 uBlack;
uniform float uMono;
uniform float uMatch;
varying vec2 vUv;
${colourGLSL}

// Hash-based Gaussian noise, fixed to the pixel grid so grain does not crawl.
float hash(vec2 p) {
  p = fract(p * vec2(443.897, 441.423));
  p += dot(p, p.yx + 19.19);
  return fract((p.x + p.y) * p.x);
}
float gauss(vec2 p) {
  float u1 = max(hash(p), 1e-6), u2 = hash(p + 17.3);
  return sqrt(-2.0 * log(u1)) * cos(6.2831853 * u2);
}

vec4 foreground(vec2 uv) {
  if (uBlur < 0.35) return texture2D(tFg, uv);
  vec4 acc = vec4(0.0);
  float wsum = 0.0;
  float r = uBlur * 0.9;
  for (int y = -1; y <= 1; y++) {
    for (int x = -1; x <= 1; x++) {
      vec2 o = vec2(float(x), float(y));
      float w = exp(-dot(o, o) * 0.5 * 0.81);
      acc += texture2D(tFg, uv + o * r * uTexel) * w;
      wsum += w;
    }
  }
  return acc / wsum;
}

void main() {
  vec3 bg = texture2D(tBg, vUv).rgb;
  vec4 fg = foreground(vUv);
  vec3 fgc = fg.a > 1e-4 ? fg.rgb / fg.a : vec3(0.0);
  fgc = mix(fgc, uBlack + (1.0 - uBlack) * fgc, uMatch);
  vec3 col = fgc * fg.a + bg * (1.0 - fg.a);
  if (uMono > 0.5) col = vec3(luma(col));
  vec3 enc = linearToSrgbV(clamp(col, 0.0, 1.0));
  enc += gauss(gl_FragCoord.xy) * uNoise * fg.a * uMatch;
  gl_FragColor = vec4(clamp(enc, 0.0, 1.0), 1.0);
}
`;

/** Photo-textured room for the reflection cube camera. */
export const envMeshVert = /* glsl */ `
varying vec2 vUv;
void main() {
  vUv = uv;
  gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
}
`;

export const envMeshFrag = /* glsl */ `
precision highp float;
uniform sampler2D uPhoto;
uniform sampler2D uClipped;
uniform float uBoost;
uniform float uMono;
varying vec2 vUv;
${colourGLSL}
void main() {
  vec3 c = texture2D(uPhoto, vUv).rgb;
  c *= mix(1.0, uBoost, texture2D(uClipped, vUv).r);
  if (uMono > 0.5) c = vec3(luma(c));
  gl_FragColor = vec4(c, 1.0);
}
`;

/** Backdrop: photo sky inside the frame, the ambient fit everywhere else. */
export const backdropVert = /* glsl */ `
varying vec3 vDir;
void main() {
  vDir = position;
  gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
}
`;

export const backdropFrag = /* glsl */ `
precision highp float;
uniform sampler2D uPhoto;
uniform sampler2D uPosition;
uniform sampler2D uClipped;
uniform vec3 uRad[9];
uniform float uFx;
uniform float uFy;
uniform float uBoost;
uniform float uMono;
varying vec3 vDir;
${colourGLSL}
vec3 radiance(vec3 d) {
  vec3 v = uRad[0] * 0.282095
    + uRad[1] * 0.488603 * d.y + uRad[2] * 0.488603 * d.z + uRad[3] * 0.488603 * d.x
    + 0.75 * (uRad[4] * 1.092548 * d.x * d.y + uRad[5] * 1.092548 * d.y * d.z
    + uRad[6] * 0.315392 * (3.0 * d.z * d.z - 1.0)
    + uRad[7] * 1.092548 * d.x * d.z + uRad[8] * 0.546274 * (d.x * d.x - d.y * d.y));
  return max(v, vec3(0.0));
}
void main() {
  vec3 d = normalize(vDir);
  vec3 c = radiance(d);
  if (d.z < -1e-3) {
    vec2 uv = vec2(0.5 + uFx * d.x / -d.z, 0.5 - uFy * d.y / -d.z);
    if (all(greaterThanEqual(uv, vec2(0.0))) && all(lessThanEqual(uv, vec2(1.0))) && texture2D(uPosition, uv).w < 0.5) {
      c = texture2D(uPhoto, uv).rgb * mix(1.0, uBoost, texture2D(uClipped, uv).r);
    }
  }
  if (uMono > 0.5) c = vec3(luma(c));
  gl_FragColor = vec4(c, 1.0);
}
`;

/** Depth view for objects, on the same scale as the depth map. */
export const depthObjVert = /* glsl */ `
varying float vZ;
void main() {
  vec4 mv = modelViewMatrix * vec4(position, 1.0);
  vZ = -mv.z;
  gl_Position = projectionMatrix * mv;
}
`;

export const depthObjFrag = /* glsl */ `
precision highp float;
uniform float uNear;
uniform float uFar;
varying float vZ;
${colourGLSL}
void main() {
  float g = clamp((uFar - vZ) / (uFar - uNear), 0.0, 1.0);
  gl_FragColor = vec4(srgbToLinearV(vec3(g)), 1.0);
}
`;

/** Contact map: darkness falls off with height above the contact plane. */
export const contactVert = /* glsl */ `
varying float vDepth;
void main() {
  gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
  vDepth = gl_Position.z * 0.5 + 0.5;
}
`;

export const contactFrag = /* glsl */ `
precision highp float;
varying float vDepth;
void main() {
  float a = 1.0 - clamp(vDepth, 0.0, 1.0);
  gl_FragColor = vec4(0.0, 0.0, 0.0, a * a);
}
`;

export const blurFrag = /* glsl */ `
precision highp float;
uniform sampler2D tMap;
uniform vec2 uStep;
varying vec2 vUv;
void main() {
  float w[5];
  w[0] = 0.2270270270; w[1] = 0.1945945946; w[2] = 0.1216216216; w[3] = 0.0540540541; w[4] = 0.0162162162;
  vec4 c = texture2D(tMap, vUv) * w[0];
  for (int i = 1; i < 5; i++) {
    c += texture2D(tMap, vUv + uStep * float(i)) * w[i];
    c += texture2D(tMap, vUv - uStep * float(i)) * w[i];
  }
  gl_FragColor = c;
}
`;
