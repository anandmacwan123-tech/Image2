// The renderer. Three.js draws objects over the photo; an invisible mesh built
// from the point map stands in for the room: it hides objects behind nearer
// things, catches their shadows and supplies their reflections.
//
// Each frame:
//   1. background pass: photo (or depth or light map), then the shadow
//      catcher multiplies in cast and contact shadows;
//   2. object pass: occluder depth, then objects, into a premultiplied layer;
//   3. grade pass: composite and match the photo's blur, levels and grain,
//      then mono.

import * as THREE from "three";
import { FullScreenQuad } from "three/addons/postprocessing/Pass.js";
import { buildSceneMesh, surfaceAt } from "../geometry/mesh";
import type { Geometry } from "../geometry/moge-post";
import type { LightFit, LightModel, Vec3 } from "../light/fit";
import { localRefit } from "../light/local";
import { toThreeProbe } from "../light/sh";
import type { PhotoStats } from "../lib/stats";
import type { Analysis } from "../workers/protocol";
import {
  MAX_OBJECT_SOURCES,
  PRIMITIVE_SIZE,
  center,
  createObject,
  dispose,
  place,
  radius,
  setView,
  sourceUniforms,
  updateTransform,
  type Kind,
  type Placed,
  type View,
} from "./objects";
import {
  MAX_CONTACTS,
  MAX_SOURCES,
  backdropFrag,
  backdropVert,
  backgroundFrag,
  blurFrag,
  colourGLSL,
  contactFrag,
  contactVert,
  envMeshFrag,
  envMeshVert,
  fullscreenVert,
  gradeFrag,
  lightGLSL,
} from "./shaders";

const BG = 1;
const FG = 2;
const TILE = 256;
const ATLAS_COLS = 4;

export interface Controls {
  key: number;
  fill: number;
  soft: number;
  /** User override of the key direction (probe drag), or null for the fit. */
  dir: Vec3 | null;
}

interface LightUniforms {
  [name: string]: THREE.IUniform;
}

/** RGBA half-float texels from a per-pixel fill function. */
function half(count: number, fill: (i: number, out: Float32Array) => void): Uint16Array {
  const out = new Uint16Array(count * 4);
  const tmp = new Float32Array(4);
  for (let i = 0; i < count; i++) {
    tmp.fill(0);
    fill(i, tmp);
    for (let k = 0; k < 4; k++) out[i * 4 + k] = THREE.DataUtils.toHalfFloat(tmp[k]);
  }
  return out;
}

interface ContactUniforms {
  uContactMap: { value: THREE.Texture };
  uContactCount: { value: number };
  uContactMatrix: { value: THREE.Matrix4[] };
  uContactTile: { value: THREE.Vector4[] };
}

function dataTexture(data: Uint16Array | Uint8Array, w: number, h: number, format: THREE.PixelFormat, type: THREE.TextureDataType): THREE.DataTexture {
  const t = new THREE.DataTexture(data, w, h, format, type);
  t.minFilter = THREE.LinearFilter;
  t.magFilter = THREE.LinearFilter;
  t.wrapS = t.wrapT = THREE.ClampToEdgeWrapping;
  t.flipY = false;
  t.needsUpdate = true;
  return t;
}

export class Stage {
  readonly renderer: THREE.WebGLRenderer;
  readonly camera = new THREE.PerspectiveCamera(50, 1, 0.01, 1000);
  private scene = new THREE.Scene();
  private envScene = new THREE.Scene();
  private rtBg: THREE.WebGLRenderTarget;
  private rtFg: THREE.WebGLRenderTarget;
  private bgQuad: FullScreenQuad;
  private gradeQuad: FullScreenQuad;
  private blurQuad: FullScreenQuad;
  private bgMat: THREE.ShaderMaterial;
  private gradeMat: THREE.ShaderMaterial;
  private blurMat: THREE.ShaderMaterial;
  private contactMat: THREE.ShaderMaterial;
  private catcherMat: THREE.ShadowMaterial;
  private envMeshMat: THREE.ShaderMaterial;
  private backdropMat: THREE.ShaderMaterial;
  private light: LightUniforms;
  private contact: ContactUniforms;
  private key = new THREE.DirectionalLight(0xffffff, 1);
  private cubeRT = new THREE.WebGLCubeRenderTarget(128, { type: THREE.HalfFloatType, generateMipmaps: false });
  private cubeCamera = new THREE.CubeCamera(0.02, 1000, this.cubeRT);
  private pmrem: THREE.PMREMGenerator;
  private atlasRaw: THREE.WebGLRenderTarget;
  private atlasTmp: THREE.WebGLRenderTarget;
  private atlas: THREE.WebGLRenderTarget;
  private contactCamera = new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 1);

  private meshGeometry: THREE.BufferGeometry | null = null;
  private occluder: THREE.Mesh | null = null;
  private catcher: THREE.Mesh | null = null;
  private envMesh: THREE.Mesh | null = null;
  private backdrop: THREE.Mesh | null = null;
  private textures: THREE.Texture[] = [];

  geometry: Geometry | null = null;
  fit: LightFit | null = null;
  stats: PhotoStats | null = null;
  photo: ImageBitmap | null = null;
  objects: Placed[] = [];
  view: View = "image";
  mono = true;
  controls: Controls = { key: 1, fill: 1, soft: 0.35, dir: null };
  private near = 0.5;
  private far = 10;
  private envTint = new THREE.Vector3(1, 1, 1);
  private keyBoost = 3;
  private needsRender = true;
  /** Display pixels per photo pixel, for scaling grain and blur. */
  private displayScale = 1;

  constructor(canvas: HTMLCanvasElement) {
    this.renderer = new THREE.WebGLRenderer({ canvas, antialias: false, alpha: false, powerPreference: "high-performance" });
    this.renderer.autoClear = false;
    this.renderer.toneMapping = THREE.NoToneMapping;
    this.renderer.shadowMap.enabled = true;
    this.renderer.shadowMap.type = THREE.PCFShadowMap;
    this.renderer.shadowMap.autoUpdate = false;
    this.pmrem = new THREE.PMREMGenerator(this.renderer);

    const rt = (samples: number) =>
      new THREE.WebGLRenderTarget(1, 1, { type: THREE.HalfFloatType, samples, depthBuffer: true });
    this.rtBg = rt(0);
    this.rtFg = rt(4);
    const atlasRT = () => new THREE.WebGLRenderTarget(TILE * ATLAS_COLS, TILE * ATLAS_COLS, { depthBuffer: true });
    this.atlasRaw = atlasRT();
    this.atlasTmp = atlasRT();
    this.atlas = atlasRT();

    const lightUniforms: LightUniforms = {
      uAmb: { value: Array.from({ length: 9 }, () => new THREE.Vector3()) },
      uKey: { value: new THREE.Vector3() },
      uKeyFit: { value: new THREE.Vector3() },
      uKeyDir: { value: new THREE.Vector3(0, 1, 0) },
      uKeyPos: { value: new THREE.Vector3() },
      uKeyPositional: { value: 0 },
      uFill: { value: 1 },
      uSrcCount: { value: 0 },
      uSrcPos: { value: Array.from({ length: MAX_SOURCES }, () => new THREE.Vector3()) },
      uSrcK: { value: Array.from({ length: MAX_SOURCES }, () => new THREE.Vector3()) },
      uSrcD0: { value: new Array(MAX_SOURCES).fill(1) },
    };
    this.light = lightUniforms;

    this.bgMat = new THREE.ShaderMaterial({
      vertexShader: fullscreenVert,
      fragmentShader: backgroundFrag,
      uniforms: {
        ...lightUniforms,
        uPhoto: { value: null },
        uNormal: { value: null },
        uPosition: { value: null },
        uDepth: { value: null },
        uView: { value: 0 },
        uNear: { value: 0.5 },
        uFar: { value: 10 },
        uWhite: { value: 0.8 },
      },
      depthTest: false,
      depthWrite: false,
    });
    this.bgQuad = new FullScreenQuad(this.bgMat);

    this.gradeMat = new THREE.ShaderMaterial({
      vertexShader: fullscreenVert,
      fragmentShader: gradeFrag,
      uniforms: {
        tBg: { value: this.rtBg.texture },
        tFg: { value: this.rtFg.texture },
        uTexel: { value: new THREE.Vector2(1, 1) },
        uBlur: { value: 0 },
        uNoise: { value: 0 },
        uBlack: { value: new THREE.Vector3() },
        uMono: { value: 1 },
        uMatch: { value: 1 },
      },
      depthTest: false,
      depthWrite: false,
    });
    this.gradeQuad = new FullScreenQuad(this.gradeMat);

    this.blurMat = new THREE.ShaderMaterial({
      vertexShader: fullscreenVert,
      fragmentShader: blurFrag,
      uniforms: { tMap: { value: null }, uStep: { value: new THREE.Vector2() } },
      depthTest: false,
      depthWrite: false,
    });
    this.blurQuad = new FullScreenQuad(this.blurMat);

    this.contactMat = new THREE.ShaderMaterial({ vertexShader: contactVert, fragmentShader: contactFrag, side: THREE.DoubleSide });

    // Shadow catcher: a ShadowMaterial whose output is the fraction of light
    // that survives, multiplied into the photo. The object blocks the key
    // where its shadow falls and the fill where it touches the surface.
    this.catcherMat = new THREE.ShadowMaterial();
    this.catcherMat.transparent = true;
    this.catcherMat.depthWrite = false;
    this.catcherMat.blending = THREE.CustomBlending;
    this.catcherMat.blendEquation = THREE.AddEquation;
    this.catcherMat.blendSrc = THREE.ZeroFactor;
    this.catcherMat.blendDst = THREE.SrcColorFactor;
    const contactUniforms: ContactUniforms = {
      uContactMap: { value: this.atlas.texture },
      uContactCount: { value: 0 },
      uContactMatrix: { value: Array.from({ length: MAX_CONTACTS }, () => new THREE.Matrix4()) },
      uContactTile: { value: Array.from({ length: MAX_CONTACTS }, () => new THREE.Vector4()) },
    };
    this.contact = contactUniforms;
    this.catcherMat.onBeforeCompile = (shader) => {
      Object.assign(shader.uniforms, lightUniforms, contactUniforms);
      shader.vertexShader = shader.vertexShader
        .replace("#include <common>", "#include <common>\nvarying vec3 vCatchPos;\nvarying vec3 vCatchNormal;")
        .replace(
          "#include <shadowmap_vertex>",
          "#include <shadowmap_vertex>\n\tvCatchPos = (modelMatrix * vec4(transformed, 1.0)).xyz;\n\tvCatchNormal = normalize(mat3(modelMatrix) * objectNormal);",
        );
      shader.fragmentShader = shader.fragmentShader
        .replace(
          "#include <common>",
          `#include <common>
varying vec3 vCatchPos;
varying vec3 vCatchNormal;
uniform sampler2D uContactMap;
uniform int uContactCount;
uniform mat4 uContactMatrix[${MAX_CONTACTS}];
uniform vec4 uContactTile[${MAX_CONTACTS}];
${colourGLSL}
${lightGLSL}
float contactShadow(vec3 p) {
  float lit = 1.0;
  for (int i = 0; i < ${MAX_CONTACTS}; i++) {
    if (i >= uContactCount) break;
    vec3 q = (uContactMatrix[i] * vec4(p, 1.0)).xyz;
    if (any(lessThan(q.xy, vec2(0.0))) || any(greaterThan(q.xy, vec2(1.0)))) continue;
    float fade = 1.0 - smoothstep(0.05, 0.6, abs(q.z));
    float a = texture2D(uContactMap, uContactTile[i].xy + q.xy * uContactTile[i].zw).a;
    lit *= 1.0 - a * fade;
  }
  return 1.0 - lit;
}`,
        )
        .replace(
          "#include <shadowmask_pars_fragment>",
          `#include <shadowmask_pars_fragment>
// Soft shadow for the catcher: 32 PCF taps over the key's shadow map. The
// built-in 5 taps are tuned for temporal AA and dither without it.
float catcherShadow() {
  #if defined( USE_SHADOWMAP ) && NUM_DIR_LIGHT_SHADOWS > 0
    DirectionalLightShadow ls = directionalLightShadows[ 0 ];
    vec4 coord = vDirectionalShadowCoord[ 0 ];
    coord.xyz /= coord.w;
    coord.z += ls.shadowBias;
    if ( coord.x < 0.0 || coord.x > 1.0 || coord.y < 0.0 || coord.y > 1.0 || coord.z > 1.0 ) return 1.0;
    vec2 texel = ls.shadowRadius / ls.shadowMapSize;
    float phi = fract( 52.9829189 * fract( dot( gl_FragCoord.xy, vec2( 0.06711056, 0.00583715 ) ) ) ) * 6.2831853;
    float sum = 0.0;
    for ( int i = 0; i < 32; i ++ ) {
      float r = sqrt( ( float( i ) + 0.5 ) / 32.0 );
      float t = float( i ) * 2.4 + phi;
      sum += texture( directionalShadowMap[ 0 ], vec3( coord.xy + vec2( cos( t ), sin( t ) ) * r * texel, coord.z ) );
    }
    return sum / 32.0;
  #else
    return 1.0;
  #endif
}`,
        )
        .replace(
          "gl_FragColor = vec4( color, opacity * ( 1.0 - getShadowMask() ) );",
          `vec3 n = normalize(vCatchNormal);
  vec3 amb = ambientShade(n);
  float kc = keyCos(n, vCatchPos);
  vec3 src = sourceShade(n, vCatchPos);
  vec3 total = max(amb + uKeyFit * kc + src, vec3(1e-4));
  float shadow = 1.0 - catcherShadow();
  vec3 lost = uKey * kc * shadow + amb * uFill * contactShadow(vCatchPos);
  gl_FragColor = vec4(clamp(1.0 - lost / total, 0.0, 1.0), 1.0);`,
        );
    };
    this.catcherMat.customProgramCacheKey = () => "depth-light-catcher";

    this.envMeshMat = new THREE.ShaderMaterial({
      vertexShader: envMeshVert,
      fragmentShader: envMeshFrag,
      uniforms: { uPhoto: { value: null }, uClipped: { value: null }, uBoost: { value: 3 }, uMono: { value: 1 } },
      side: THREE.DoubleSide,
    });
    this.backdropMat = new THREE.ShaderMaterial({
      vertexShader: backdropVert,
      fragmentShader: backdropFrag,
      uniforms: {
        uPhoto: { value: null },
        uPosition: { value: null },
        uClipped: { value: null },
        uRad: { value: Array.from({ length: 9 }, () => new THREE.Vector3()) },
        uFx: { value: 1 },
        uFy: { value: 1 },
        uBoost: { value: 3 },
        uMono: { value: 1 },
      },
      side: THREE.BackSide,
      depthWrite: false,
    });

    this.key.castShadow = true;
    this.key.shadow.mapSize.set(2048, 2048);
    this.key.layers.enableAll();
    this.scene.add(this.key, this.key.target);
    this.camera.layers.enableAll();
  }

  /** Build everything for a new image. */
  load(photo: ImageBitmap, analysis: Analysis): void {
    this.clear();
    const { geometry: g, fit, stats, rgba } = analysis;
    this.photo = photo;
    this.geometry = g;
    this.fit = fit;
    this.stats = stats;

    const photoTex = new THREE.Texture(photo);
    photoTex.colorSpace = THREE.SRGBColorSpace;
    photoTex.flipY = false;
    photoTex.generateMipmaps = true;
    photoTex.minFilter = THREE.LinearMipmapLinearFilter;
    photoTex.anisotropy = this.renderer.capabilities.getMaxAnisotropy();
    photoTex.needsUpdate = true;

    const n = g.width * g.height;
    const normalTex = dataTexture(
      half(n, (i, o) => {
        o[0] = g.normals[i * 3];
        o[1] = g.normals[i * 3 + 1];
        o[2] = g.normals[i * 3 + 2];
      }),
      g.width, g.height, THREE.RGBAFormat, THREE.HalfFloatType,
    );
    const posTex = dataTexture(
      half(n, (i, o) => {
        o[0] = g.points[i * 3];
        o[1] = g.points[i * 3 + 1];
        o[2] = g.points[i * 3 + 2];
        o[3] = g.mask[i];
      }),
      g.width, g.height, THREE.RGBAFormat, THREE.HalfFloatType,
    );
    const depthTex = dataTexture(
      half(n, (i, o) => {
        o[0] = g.depth[i];
      }),
      g.width, g.height, THREE.RGBAFormat, THREE.HalfFloatType,
    );
    const clipped = new Uint8Array(n);
    let tr = 0, tg = 0, tb = 0;
    for (let p = 0; p < n; p++) {
      const r = rgba[p * 4], gg = rgba[p * 4 + 1], b = rgba[p * 4 + 2];
      clipped[p] = Math.max(r, gg, b) >= 250 ? 255 : 0;
      tr += (r / 255) ** 2.2;
      tg += (gg / 255) ** 2.2;
      tb += (b / 255) ** 2.2;
    }
    const tl = 0.2126 * tr + 0.7152 * tg + 0.0722 * tb || 1;
    this.envTint.set(tr / tl, tg / tl, tb / tl);
    const clippedTex = dataTexture(clipped, g.width, g.height, THREE.RedFormat, THREE.UnsignedByteType);
    this.textures = [photoTex, normalTex, posTex, depthTex, clippedTex];

    // Depth range for display and export: 1st to 99th percentile.
    const depths = Array.from(g.depth).filter((d) => d > 0).sort((a, b) => a - b);
    this.near = depths.length ? depths[Math.floor(depths.length * 0.01)] : 0.5;
    this.far = depths.length ? depths[Math.floor(depths.length * 0.99)] : 10;
    if (this.far - this.near < 1e-3) this.far = this.near + 1;

    this.camera.fov = g.fovY;
    this.camera.aspect = g.width / g.height;
    this.camera.near = Math.max(0.01, this.near * 0.05);
    this.camera.far = this.far * 50;
    this.camera.updateProjectionMatrix();

    const mesh = buildSceneMesh(g);
    const geo = new THREE.BufferGeometry();
    geo.setAttribute("position", new THREE.BufferAttribute(mesh.positions, 3));
    geo.setAttribute("normal", new THREE.BufferAttribute(mesh.normals, 3));
    geo.setAttribute("uv", new THREE.BufferAttribute(mesh.uvs, 2));
    geo.setIndex(new THREE.BufferAttribute(mesh.index, 1));
    geo.computeBoundingSphere();
    this.meshGeometry = geo;

    this.occluder = new THREE.Mesh(geo, new THREE.MeshBasicMaterial({ colorWrite: false, polygonOffset: true, polygonOffsetFactor: 1, polygonOffsetUnits: 4 }));
    this.occluder.layers.set(FG);
    this.occluder.renderOrder = -1;
    this.catcher = new THREE.Mesh(geo, this.catcherMat);
    this.catcher.layers.set(BG);
    this.catcher.receiveShadow = true;
    this.scene.add(this.occluder, this.catcher);

    this.bgMat.uniforms.uPhoto.value = photoTex;
    this.bgMat.uniforms.uNormal.value = normalTex;
    this.bgMat.uniforms.uPosition.value = posTex;
    this.bgMat.uniforms.uDepth.value = depthTex;
    this.bgMat.uniforms.uNear.value = this.near;
    this.bgMat.uniforms.uFar.value = this.far;
    this.bgMat.uniforms.uWhite.value = fit.white;

    this.envMeshMat.uniforms.uPhoto.value = photoTex;
    this.envMeshMat.uniforms.uClipped.value = clippedTex;
    this.envMesh = new THREE.Mesh(geo, this.envMeshMat);
    const bd = this.backdropMat.uniforms;
    bd.uPhoto.value = photoTex;
    bd.uPosition.value = posTex;
    bd.uClipped.value = clippedTex;
    bd.uFx.value = g.fx;
    bd.uFy.value = g.fy;
    this.backdrop = new THREE.Mesh(new THREE.SphereGeometry(500, 64, 32), this.backdropMat);
    this.envScene.add(this.envMesh, this.backdrop);

    // Boost clipped pixels in reflections so a visible key source is as
    // bright as the light it gives: radiance = π·k / solid angle.
    const model = fit.mono;
    const src = fit.detected.find((s) => s.position && model.key.source && s.position[0] === model.key.source[0]);
    if (src && src.area > 0) {
      const d = Math.hypot(src.position![0] - fit.refPoint[0], src.position![1] - fit.refPoint[1], src.position![2] - fit.refPoint[2]);
      this.keyBoost = Math.min(50, Math.max(2, (Math.PI * model.key.strength[0] * d * d) / src.area));
    } else {
      this.keyBoost = 3;
    }

    this.updateLights();
    this.needsRender = true;
  }

  get model(): LightModel | null {
    return this.fit ? (this.mono ? this.fit.mono : this.fit.colour) : null;
  }

  private vec3Of(values: number[]): THREE.Vector3 {
    return values.length === 1 ? new THREE.Vector3(values[0], values[0], values[0]) : new THREE.Vector3(values[0], values[1], values[2]);
  }

  /** Direction toward the key from a point. */
  keyDirection(from?: THREE.Vector3): THREE.Vector3 {
    const m = this.model!;
    if (this.controls.dir) return new THREE.Vector3(...this.controls.dir).normalize();
    if (m.key.source) {
      const p = from ?? new THREE.Vector3(...this.fit!.refPoint);
      return new THREE.Vector3(...m.key.source).sub(p).normalize();
    }
    return new THREE.Vector3(...m.key.dir);
  }

  /** Push the fitted light (plus sliders and overrides) into every consumer. */
  updateLights(): void {
    const m = this.model;
    if (!m || !this.fit) return;
    const c = this.controls;
    const L = this.light;
    const amb = L.uAmb.value as THREE.Vector3[];
    for (let i = 0; i < 9; i++) amb[i].copy(this.vec3Of(m.ambient.map((ch) => ch[i])));
    const k = this.vec3Of(m.key.strength);
    (L.uKeyFit.value as THREE.Vector3).copy(k);
    (L.uKey.value as THREE.Vector3).copy(k).multiplyScalar(c.key);
    L.uFill.value = c.fill;
    const positional = !c.dir && !!m.key.source;
    L.uKeyPositional.value = positional ? 1 : 0;
    if (m.key.source) (L.uKeyPos.value as THREE.Vector3).set(...m.key.source);
    (L.uKeyDir.value as THREE.Vector3).copy(this.keyDirection());
    const srcs = m.sources.slice(0, MAX_SOURCES);
    L.uSrcCount.value = srcs.length;
    sourceUniforms.uSrcCount.value = Math.min(srcs.length, MAX_OBJECT_SOURCES);
    srcs.forEach((s, i) => {
      (L.uSrcPos.value as THREE.Vector3[])[i].set(...s.position);
      (L.uSrcK.value as THREE.Vector3[])[i].copy(this.vec3Of(s.strength));
      (L.uSrcD0.value as number[])[i] = s.d0;
      if (i >= MAX_OBJECT_SOURCES) return;
      // Objects take irradiance in three.js units: E = π·S.
      sourceUniforms.uSrcPos.value[i].set(...s.position);
      sourceUniforms.uSrcE.value[i].copy(this.vec3Of(s.strength)).multiplyScalar(Math.PI);
      sourceUniforms.uSrcD0.value[i] = s.d0;
    });

    // Key light: three.js renders ρ/π·E, the fit uses ρ·S, so E = π·S.
    const peak = Math.max(k.x, k.y, k.z, 1e-9);
    this.key.color.setRGB(k.x / peak, k.y / peak, k.z / peak, THREE.LinearSRGBColorSpace);
    this.key.intensity = Math.PI * peak * c.key;
    this.key.shadow.radius = 1 + c.soft * 30;

    const rad = this.backdropMat.uniforms.uRad.value as THREE.Vector3[];
    const probe = m.ambient.map((ch) => toThreeProbe(ch));
    for (let i = 0; i < 9; i++) rad[i].copy(this.vec3Of(probe.map((ch) => ch[i] * c.fill)));
    const mono = this.mono ? 1 : 0;
    this.backdropMat.uniforms.uMono.value = mono;
    this.backdropMat.uniforms.uBoost.value = this.keyBoost;
    this.envMeshMat.uniforms.uMono.value = mono;
    this.envMeshMat.uniforms.uBoost.value = this.keyBoost;
    this.gradeMat.uniforms.uMono.value = mono;
    for (const o of this.objects) o.dirty = true;
    this.needsRender = true;
  }

  setView(view: View): void {
    this.view = view;
    this.bgMat.uniforms.uView.value = view === "image" ? 0 : view === "depth" ? 1 : 2;
    for (const o of this.objects) setView(o, view);
    this.needsRender = true;
  }

  setMono(mono: boolean): void {
    this.mono = mono;
    this.updateLights();
  }

  setControls(c: Partial<Controls>): void {
    Object.assign(this.controls, c);
    this.updateLights();
  }

  invalidate(): void {
    this.needsRender = true;
  }

  resize(width: number, height: number): void {
    this.renderer.setSize(width, height, false);
    this.rtBg.setSize(width, height);
    this.rtFg.setSize(width, height);
    this.gradeMat.uniforms.uTexel.value.set(1 / width, 1 / height);
    this.displayScale = this.photo ? width / this.photo.width : 1;
    this.needsRender = true;
  }

  /** Surface under a canvas position, uv with (0, 0) at the top left. */
  surface(u: number, v: number): { point: THREE.Vector3; normal: THREE.Vector3 } | null {
    if (!this.geometry) return null;
    const s = surfaceAt(this.geometry, u, v);
    return s ? { point: new THREE.Vector3(...s.point), normal: new THREE.Vector3(...s.normal) } : null;
  }

  addObject(kind: Kind, u: number, v: number, model?: THREE.Object3D): Placed | null {
    const s = this.surface(u, v);
    if (!s) return null;
    const o = createObject(kind, model);
    if (kind !== "model") {
      // Metric scale is honest but a fixed size is useless across a tabletop
      // and a valley, so primitives start at about an eighth of the view
      // height where they land. Scroll to resize.
      const visible = 2 * -s.point.z * Math.tan(((this.camera.fov / 2) * Math.PI) / 180);
      const size = Math.min(2, Math.max(0.05, 0.12 * visible));
      o.scale = Number(size.toPrecision(2)) / PRIMITIVE_SIZE;
    }
    place(o, s.point, s.normal);
    o.root.traverse((x) => x.layers.set(FG));
    setView(o, this.view);
    this.objects.push(o);
    this.scene.add(o.root);
    this.updateLights();
    return o;
  }

  moveObject(o: Placed, u: number, v: number): void {
    const s = this.surface(u, v);
    if (!s) return;
    place(o, s.point, s.normal);
    this.needsRender = true;
  }

  rotateObject(o: Placed, delta: number): void {
    o.angle += delta;
    updateTransform(o);
    this.needsRender = true;
  }

  scaleObject(o: Placed, factor: number): void {
    o.scale = Math.min(20, Math.max(0.05, o.scale * factor));
    updateTransform(o);
    this.needsRender = true;
  }

  removeObject(o: Placed): void {
    this.scene.remove(o.root);
    dispose(o);
    this.objects = this.objects.filter((x) => x !== o);
    for (const x of this.objects) x.dirty = true;
    this.needsRender = true;
  }

  /** Object under a canvas position, ignoring objects hidden by the room. */
  pick(u: number, v: number): Placed | null {
    const ray = new THREE.Raycaster();
    ray.layers.set(FG);
    ray.setFromCamera(new THREE.Vector2(u * 2 - 1, 1 - v * 2), this.camera);
    const hits = ray.intersectObjects(this.objects.map((o) => o.root), true);
    if (!hits.length) return null;
    const hit = hits[0];
    const s = this.surface(u, v);
    if (s && s.point.length() < hit.distance * 0.97) return null;
    return this.objects.find((o) => {
      let x: THREE.Object3D | null = hit.object;
      while (x) {
        if (x === o.root) return true;
        x = x.parent;
      }
      return false;
    }) ?? null;
  }

  /** Canvas uv of an object's contact point. */
  project(o: Placed): THREE.Vector2 {
    const p = o.contact.clone().project(this.camera);
    return new THREE.Vector2((p.x + 1) / 2, (1 - p.y) / 2);
  }

  private objectsCentre(): { c: THREE.Vector3; r: number } | null {
    if (!this.objects.length) return null;
    const box = new THREE.Box3();
    for (const o of this.objects) box.expandByObject(o.root);
    const s = box.getBoundingSphere(new THREE.Sphere());
    return { c: s.center, r: Math.max(s.radius, 0.05) };
  }

  private updateKeyAndShadow(): void {
    const oc = this.objectsCentre();
    const c = oc?.c ?? new THREE.Vector3(...this.fit!.refPoint);
    const r = oc?.r ?? 1;
    // A visible key source sits at its real position, so the angle follows the objects.
    const dir = this.keyDirection(c);
    this.key.position.copy(c).addScaledVector(dir, r * 4 + 2);
    this.key.target.position.copy(c);
    this.key.target.updateMatrixWorld();
    const cam = this.key.shadow.camera;
    const halfSize = r * 4 + 0.3;
    cam.left = -halfSize;
    cam.right = halfSize;
    cam.top = halfSize;
    cam.bottom = -halfSize;
    cam.near = 0.01;
    cam.far = r * 4 + 2 + Math.max(this.far * 3, 20);
    cam.updateProjectionMatrix();
    this.key.shadow.bias = -0.0002;
    this.key.shadow.normalBias = 0;
  }

  /** Local refit, reflections and contact shadows for objects that moved. */
  private updateObjects(): void {
    const m = this.model!;
    const fit = this.fit!;
    const prior = { channels: m.channels, ambient: m.ambient, keyStrength: m.key.strength };
    const tmp = new THREE.Vector3();
    const dirtyContacts: number[] = [];
    this.objects.forEach((o, i) => {
      if (!o.dirty) return;
      o.dirty = false;
      const c = center(o, tmp);
      const local = localRefit(fit.local, prior, [c.x, c.y, c.z], radius(o));
      const probe = local.ambient.map((ch) => toThreeProbe(ch));
      for (let k = 0; k < 9; k++) {
        const v = probe.length === 1 ? [probe[0][k], probe[0][k], probe[0][k]] : [probe[0][k], probe[1][k], probe[2][k]];
        o.uniforms.uAmbL.value[k].set(v[0], v[1], v[2]).multiplyScalar(this.controls.fill);
      }
      o.uniforms.uKeyScale.value = local.keyScale;
      o.uniforms.uBleed.value = this.mono ? 0 : 0.35;
      o.uniforms.uEnvTint.value.copy(this.envTint);
      o.materials.depth.uniforms.uNear.value = this.near;
      o.materials.depth.uniforms.uFar.value = this.far;
      this.updateEnv(o, c);
      if (i < MAX_CONTACTS) dirtyContacts.push(i);
    });
    if (dirtyContacts.length) this.updateContacts(dirtyContacts);
  }

  private updateEnv(o: Placed, c: THREE.Vector3): void {
    this.cubeCamera.position.copy(c);
    this.cubeCamera.update(this.renderer, this.envScene);
    o.env = this.pmrem.fromCubemap(this.cubeRT.texture, o.env);
    for (const mat of o.materials.image.values()) {
      for (const m of Array.isArray(mat) ? mat : [mat]) {
        const s = m as THREE.MeshStandardMaterial;
        if (s.isMeshStandardMaterial && s.envMap !== o.env.texture) {
          s.envMap = o.env.texture;
          s.needsUpdate = true;
        }
      }
    }
  }

  private updateContacts(indices: number[]): void {
    const r = this.renderer;
    const cam = this.contactCamera;
    const prevOverride = this.scene.overrideMaterial;
    const visible = new Map<THREE.Object3D, boolean>();
    for (const x of [this.occluder, this.catcher, ...this.objects.map((o) => o.root)]) if (x) visible.set(x, x.visible);
    this.scene.overrideMaterial = this.contactMat;
    r.setRenderTarget(this.atlasRaw);
    const bias = new THREE.Matrix4().set(0.5, 0, 0, 0.5, 0, 0.5, 0, 0.5, 0, 0, 0.5, 0.5, 0, 0, 0, 1);
    for (const i of indices) {
      const o = this.objects[i];
      const size = o.bounds.getSize(new THREE.Vector3()).multiplyScalar(o.scale);
      const halfSize = Math.max(size.x, size.z) * 0.5 * 1.6;
      const height = size.y * 0.5;
      cam.left = -halfSize;
      cam.right = halfSize;
      cam.top = halfSize;
      cam.bottom = -halfSize;
      cam.near = 0;
      cam.far = height;
      cam.position.copy(o.contact).addScaledVector(o.normal, -0.01);
      const side = Math.abs(o.normal.y) < 0.9 ? new THREE.Vector3(0, 1, 0) : new THREE.Vector3(1, 0, 0);
      cam.up.copy(side);
      cam.lookAt(o.contact.clone().add(o.normal));
      cam.updateProjectionMatrix();
      cam.updateMatrixWorld();
      cam.layers.set(FG);
      for (const x of visible.keys()) x.visible = x === o.root;
      const tx = (i % ATLAS_COLS) * TILE, ty = Math.floor(i / ATLAS_COLS) * TILE;
      r.setViewport(tx, ty, TILE, TILE);
      r.setScissor(tx, ty, TILE, TILE);
      r.setScissorTest(true);
      r.setClearColor(0x000000, 0);
      r.clear(true, true, false);
      r.render(this.scene, cam);
      r.setScissorTest(false);

      this.contact.uContactMatrix.value[i].copy(bias).multiply(cam.projectionMatrix).multiply(cam.matrixWorldInverse);
      this.contact.uContactTile.value[i].set(tx / (TILE * ATLAS_COLS), ty / (TILE * ATLAS_COLS), 1 / ATLAS_COLS, 1 / ATLAS_COLS);
    }
    for (const [x, v] of visible) x.visible = v;
    this.scene.overrideMaterial = prevOverride;
    r.setViewport(0, 0, this.rtBg.width, this.rtBg.height);

    // Blur raw → tmp (horizontal) → atlas (vertical).
    const px = 1 / (TILE * ATLAS_COLS);
    this.blurMat.uniforms.tMap.value = this.atlasRaw.texture;
    this.blurMat.uniforms.uStep.value.set(px * 1.5, 0);
    r.setRenderTarget(this.atlasTmp);
    this.blurQuad.render(r);
    this.blurMat.uniforms.tMap.value = this.atlasTmp.texture;
    this.blurMat.uniforms.uStep.value.set(0, px * 1.5);
    r.setRenderTarget(this.atlas);
    this.blurQuad.render(r);
    r.setRenderTarget(null);
  }

  /** Render if anything changed. Returns true if a frame was drawn. */
  frame(): boolean {
    if (!this.needsRender && !this.objects.some((o) => o.dirty)) return false;
    this.needsRender = false;
    this.draw(null, this.displayScale);
    return true;
  }

  private draw(target: THREE.WebGLRenderTarget | null, scale: number, bg = this.rtBg, fg = this.rtFg): void {
    if (!this.fit || !this.stats) return;
    const r = this.renderer;
    this.updateObjects();
    this.updateKeyAndShadow();
    this.contact.uContactCount.value = Math.min(this.objects.length, MAX_CONTACTS);

    const showShadows = this.view !== "depth" && this.objects.length > 0;
    if (this.catcher) this.catcher.visible = showShadows;
    this.key.castShadow = showShadows;

    // Objects first: three.js filters shadow casters by the main camera's
    // layers, so the shadow map has to be drawn while the camera sees them.
    r.setRenderTarget(fg);
    r.setClearColor(0x000000, 0);
    r.clear();
    this.camera.layers.set(FG);
    r.shadowMap.needsUpdate = showShadows;
    r.render(this.scene, this.camera);

    // Then the background, whose catcher reuses that shadow map.
    r.setRenderTarget(bg);
    r.setClearColor(0x000000, 1);
    r.clear();
    this.bgQuad.render(r);
    if (showShadows) {
      this.camera.layers.set(BG);
      r.render(this.scene, this.camera);
    }

    const s = this.stats;
    const gu = this.gradeMat.uniforms;
    gu.tBg.value = bg.texture;
    gu.tFg.value = fg.texture;
    gu.uTexel.value.set(1 / bg.width, 1 / bg.height);
    gu.uMatch.value = this.view === "image" ? 1 : 0;
    gu.uBlur.value = Math.max(0.5, s.blur * scale);
    gu.uNoise.value = s.noise * Math.min(1, scale);
    const black = 0.2126 * s.black[0] + 0.7152 * s.black[1] + 0.0722 * s.black[2];
    if (this.mono) gu.uBlack.value.set(black, black, black);
    else gu.uBlack.value.set(...s.black);
    r.setRenderTarget(target);
    this.gradeQuad.render(r);
    r.setRenderTarget(null);
  }

  /** Render the composite at a given size into RGBA bytes (top row first). */
  renderPixels(width: number, height: number, options: { view?: View; objects?: boolean } = {}): Uint8Array<ArrayBuffer> {
    const prevView = this.view;
    const hidden = options.objects === false ? this.objects.map((o) => o.root) : [];
    if (options.view && options.view !== prevView) this.setView(options.view);
    hidden.forEach((x) => (x.visible = false));
    const bg = new THREE.WebGLRenderTarget(width, height, { type: THREE.HalfFloatType, depthBuffer: true });
    const fg = new THREE.WebGLRenderTarget(width, height, { type: THREE.HalfFloatType, depthBuffer: true });
    const out = new THREE.WebGLRenderTarget(width, height, { type: THREE.UnsignedByteType, depthBuffer: false });
    const prevObjects = this.objects;
    if (hidden.length) this.objects = [];
    this.draw(out, width / (this.photo?.width ?? width), bg, fg);
    this.objects = prevObjects;
    const pixels = new Uint8Array(width * height * 4);
    this.renderer.readRenderTargetPixels(out, 0, 0, width, height, pixels);
    bg.dispose();
    fg.dispose();
    out.dispose();
    hidden.forEach((x) => (x.visible = true));
    if (options.view && options.view !== prevView) this.setView(prevView);
    this.needsRender = true;
    // WebGL reads bottom-up.
    const flipped = new Uint8Array(pixels.length);
    const row = width * 4;
    for (let y = 0; y < height; y++) flipped.set(pixels.subarray((height - 1 - y) * row, (height - y) * row), y * row);
    return flipped;
  }

  get depthRange(): { near: number; far: number } {
    return { near: this.near, far: this.far };
  }

  maxExportSide(): number {
    const gl = this.renderer.getContext();
    return Math.min(8192, gl.getParameter(gl.MAX_TEXTURE_SIZE), gl.getParameter(gl.MAX_RENDERBUFFER_SIZE));
  }

  private clear(): void {
    for (const o of [...this.objects]) this.removeObject(o);
    for (const x of [this.occluder, this.catcher]) if (x) this.scene.remove(x);
    sourceUniforms.uSrcCount.value = 0;
    if (this.envMesh) this.envScene.remove(this.envMesh);
    if (this.backdrop) {
      this.envScene.remove(this.backdrop);
      this.backdrop.geometry.dispose();
    }
    this.meshGeometry?.dispose();
    this.occluder?.material && (this.occluder.material as THREE.Material).dispose();
    for (const t of this.textures) t.dispose();
    this.textures = [];
    this.photo?.close();
    this.photo = null;
  }
}
