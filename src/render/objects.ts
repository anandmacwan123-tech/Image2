// Placed objects: primitives and dropped .glb files. Each object carries its
// own lighting uniforms, because the local refit gives every position its own
// ambient and key visibility; three.js light probes are per scene.

import * as THREE from "three";
import { GLTFLoader } from "three/addons/loaders/GLTFLoader.js";
import { depthObjFrag, depthObjVert } from "./shaders";

export type Kind = "sphere" | "cube" | "chrome" | "model";

export interface ObjectUniforms {
  uAmbL: { value: THREE.Vector3[] };
  uKeyScale: { value: number };
  uBleed: { value: number };
  uEnvTint: { value: THREE.Vector3 };
  /** Supporting surface, for occlusion where the object meets it. */
  uSupportPoint: { value: THREE.Vector3 };
  uSupportNormal: { value: THREE.Vector3 };
  uSupportReach: { value: number };
}

export interface Placed {
  id: number;
  kind: Kind;
  root: THREE.Group;
  /** Holds the model; rotated about the surface normal. */
  spin: THREE.Group;
  /** Surface point the object rests on, and the surface normal there. */
  contact: THREE.Vector3;
  normal: THREE.Vector3;
  angle: number;
  scale: number;
  /** Local bounds at scale 1 (for seating and contact shadows). */
  bounds: THREE.Box3;
  uniforms: ObjectUniforms;
  materials: { image: Map<THREE.Mesh, THREE.Material | THREE.Material[]>; light: THREE.Material; depth: THREE.ShaderMaterial };
  env: THREE.WebGLRenderTarget | null;
  dirty: boolean;
}

const UP = new THREE.Vector3(0, 1, 0);
let nextId = 1;

/** Base size of the primitives before scaling, in metres. */
export const PRIMITIVE_SIZE = 0.35;
export const MAX_OBJECT_SOURCES = 4;

/**
 * Visible light sources, shared by every object. They use the fit's own
 * model, cosine times a capped inverse square, rather than three.js point
 * lights, whose unbounded falloff blows out anything placed near a lamp.
 */
export const sourceUniforms = {
  uSrcCount: { value: 0 },
  uSrcPos: { value: Array.from({ length: MAX_OBJECT_SOURCES }, () => new THREE.Vector3()) },
  uSrcE: { value: Array.from({ length: MAX_OBJECT_SOURCES }, () => new THREE.Vector3()) },
  uSrcD0: { value: new Array<number>(MAX_OBJECT_SOURCES).fill(1) },
};

export function makeUniforms(): ObjectUniforms {
  return {
    uAmbL: { value: Array.from({ length: 9 }, () => new THREE.Vector3()) },
    uKeyScale: { value: 1 },
    uBleed: { value: 0 },
    uEnvTint: { value: new THREE.Vector3(1, 1, 1) },
    uSupportPoint: { value: new THREE.Vector3() },
    uSupportNormal: { value: new THREE.Vector3(0, 1, 0) },
    uSupportReach: { value: 0.1 },
  };
}

/**
 * Route diffuse ambient through the object's own SH (from the local refit),
 * scale the key by its local visibility, darken the crevice where the object
 * meets its surface, and use the reflection map for specular only, apart from
 * a colour bleed tint. The photo's camera sits at the origin looking down −z,
 * so view space is world space and the support uniforms need no transform.
 */
export function patchMaterial(material: THREE.Material, u: ObjectUniforms): void {
  const m = material as THREE.MeshStandardMaterial;
  if (!m.isMeshStandardMaterial) return;
  m.onBeforeCompile = (shader) => {
    Object.assign(shader.uniforms, u, sourceUniforms);
    const begin = THREE.ShaderChunk.lights_fragment_begin
      .replace(
        "getDirectionalLightInfo( directionalLight, directLight );",
        "getDirectionalLightInfo( directionalLight, directLight );\n\t\tdirectLight.color *= uKeyScale;",
      )
      .replace(
        "vec3 irradiance = getAmbientLightIrradiance( ambientLightColor );",
        `vec3 irradiance = getAmbientLightIrradiance( ambientLightColor ) + shGetIrradianceAt( transformNormalByInverseViewMatrix( geometryNormal, viewMatrix ), uAmbL );
        {
          float h = dot( geometryPosition - uSupportPoint, uSupportNormal );
          float facing = max( 0.0, - dot( geometryNormal, uSupportNormal ) );
          irradiance *= 1.0 - 0.85 * facing * ( 1.0 - smoothstep( 0.0, uSupportReach, h ) );
          for ( int i = 0; i < ${MAX_OBJECT_SOURCES}; i ++ ) {
            if ( i >= uSrcCount ) break;
            vec3 d = uSrcPos[ i ] - geometryPosition;
            float l2 = max( dot( d, d ), 1e-6 );
            float f = min( 4.0, uSrcD0[ i ] * uSrcD0[ i ] / l2 );
            irradiance += uSrcE[ i ] * max( dot( geometryNormal, d * inversesqrt( l2 ) ), 0.0 ) * f;
          }
        }`,
      );
    const maps = THREE.ShaderChunk.lights_fragment_maps.replace(
      "iblIrradiance += getIBLIrradiance( geometryNormal );",
      `{
        vec3 envE = getIBLIrradiance( geometryNormal );
        float envL = dot( envE, vec3( 0.2126, 0.7152, 0.0722 ) );
        vec3 tint = envL > 1e-5 ? envE / envL : vec3( 1.0 );
        irradiance *= mix( vec3( 1.0 ), tint / uEnvTint, uBleed );
      }`,
    );
    shader.fragmentShader = shader.fragmentShader
      .replace(
        "#include <common>",
        `#include <common>\nuniform vec3 uAmbL[ 9 ];\nuniform float uKeyScale;\nuniform float uBleed;\nuniform vec3 uEnvTint;\nuniform vec3 uSupportPoint;\nuniform vec3 uSupportNormal;\nuniform float uSupportReach;\nuniform int uSrcCount;\nuniform vec3 uSrcPos[ ${MAX_OBJECT_SOURCES} ];\nuniform vec3 uSrcE[ ${MAX_OBJECT_SOURCES} ];\nuniform float uSrcD0[ ${MAX_OBJECT_SOURCES} ];`,
      )
      .replace("#include <lights_fragment_begin>", begin)
      .replace("#include <lights_fragment_maps>", maps);
  };
  m.customProgramCacheKey = () => "depth-light-object";
  m.needsUpdate = true;
}

function white(): THREE.Color {
  return new THREE.Color().setRGB(0.8, 0.8, 0.8, THREE.LinearSRGBColorSpace);
}

function primitive(kind: Exclude<Kind, "model">): THREE.Mesh {
  const size = PRIMITIVE_SIZE;
  const geometry = kind === "cube" ? new THREE.BoxGeometry(size, size, size) : new THREE.SphereGeometry(size / 2, 96, 64);
  const material =
    kind === "chrome"
      ? new THREE.MeshStandardMaterial({ color: new THREE.Color(1, 1, 1), metalness: 1, roughness: 0.03 })
      : new THREE.MeshStandardMaterial({ color: white(), metalness: 0, roughness: 0.85 });
  return new THREE.Mesh(geometry, material);
}

export async function loadGLB(data: ArrayBuffer): Promise<THREE.Object3D> {
  const gltf = await new GLTFLoader().parseAsync(data, "");
  const scene = gltf.scene;
  // Normalise to a sensible size if the file has no meaningful units.
  const box = new THREE.Box3().setFromObject(scene);
  const size = box.getSize(new THREE.Vector3());
  const longest = Math.max(size.x, size.y, size.z);
  if (longest > 5 || longest < 0.02) scene.scale.multiplyScalar(0.5 / longest);
  return scene;
}

export function createObject(kind: Kind, model?: THREE.Object3D): Placed {
  const uniforms = makeUniforms();
  const content = kind === "model" ? model! : primitive(kind);
  const spin = new THREE.Group();
  spin.add(content);
  const root = new THREE.Group();
  root.add(spin);

  const image = new Map<THREE.Mesh, THREE.Material | THREE.Material[]>();
  content.traverse((o) => {
    const mesh = o as THREE.Mesh;
    if (!mesh.isMesh) return;
    mesh.castShadow = true;
    mesh.receiveShadow = true;
    const mats = Array.isArray(mesh.material) ? mesh.material : [mesh.material];
    for (const m of mats) patchMaterial(m, uniforms);
    image.set(mesh, mesh.material);
  });
  const light = new THREE.MeshStandardMaterial({ color: white(), roughness: 1, metalness: 0 });
  patchMaterial(light, uniforms);
  const depth = new THREE.ShaderMaterial({
    vertexShader: depthObjVert,
    fragmentShader: depthObjFrag,
    uniforms: { uNear: { value: 1 }, uFar: { value: 10 } },
  });

  root.updateMatrixWorld(true);
  const bounds = new THREE.Box3().setFromObject(spin);
  return {
    id: nextId++,
    kind,
    root,
    spin,
    contact: new THREE.Vector3(),
    normal: new THREE.Vector3(0, 1, 0),
    angle: 0,
    scale: 1,
    bounds,
    uniforms,
    materials: { image, light, depth },
    env: null,
    dirty: true,
  };
}

/** Seat the object on the surface: its bottom touches the point, up = normal. */
export function place(o: Placed, point: THREE.Vector3, normal: THREE.Vector3): void {
  o.contact.copy(point);
  o.normal.copy(normal).normalize();
  updateTransform(o);
}

export function updateTransform(o: Placed): void {
  o.root.quaternion.setFromUnitVectors(UP, o.normal);
  o.spin.rotation.set(0, o.angle, 0);
  o.root.scale.setScalar(o.scale);
  // Lift by the bottom of the bounds so the object sits on, not in, the surface.
  const lift = -o.bounds.min.y * o.scale + 0.002;
  o.root.position.copy(o.contact).addScaledVector(o.normal, lift);
  o.root.updateMatrixWorld(true);
  o.uniforms.uSupportPoint.value.copy(o.contact);
  o.uniforms.uSupportNormal.value.copy(o.normal);
  o.uniforms.uSupportReach.value = 0.35 * radius(o);
  o.dirty = true;
}

export function center(o: Placed, out = new THREE.Vector3()): THREE.Vector3 {
  return o.bounds.getCenter(out).applyMatrix4(o.spin.matrixWorld);
}

export function radius(o: Placed): number {
  return o.bounds.getSize(new THREE.Vector3()).length() * 0.5 * o.scale;
}

export type View = "image" | "depth" | "light";

export function setView(o: Placed, view: View): void {
  for (const [mesh, mat] of o.materials.image) {
    mesh.material = view === "image" ? mat : view === "light" ? o.materials.light : o.materials.depth;
    mesh.castShadow = view !== "depth";
  }
}

export function dispose(o: Placed): void {
  o.root.traverse((x) => {
    const mesh = x as THREE.Mesh;
    if (!mesh.isMesh) return;
    mesh.geometry.dispose();
  });
  for (const mat of o.materials.image.values()) (Array.isArray(mat) ? mat : [mat]).forEach((m) => m.dispose());
  o.materials.light.dispose();
  o.materials.depth.dispose();
  o.env?.dispose();
}
