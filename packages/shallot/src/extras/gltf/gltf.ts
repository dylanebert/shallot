import { compose, multiply } from "../../engine";
import { VERTEX_FLOATS } from "../../standard/render/core";
import type { DracoMesh } from "./draco";
import type { SkinChannel, SkinInput } from "./vat";

/** the injected Draco decode — {@link loadGltf} dynamic-imports the codec (`decodeDraco`) only when a
 *  primitive carries `KHR_draco_mesh_compression`, so the ~250KB decoder stays out of the base bundle. */
export type DracoDecode = (
    bytes: Uint8Array,
    attributes: DracoExtension["attributes"],
) => DracoMesh;

/** the injected meshopt decode — {@link loadGltf} dynamic-imports the codec (`decodeMeshopt`) only when a
 *  bufferView carries `EXT_meshopt_compression`. Decompresses one compressed bufferView's `source` into
 *  `count * size` plain (filtered) bytes the standard accessor read then consumes. */
export type MeshoptDecode = (
    source: Uint8Array,
    count: number,
    size: number,
    mode: MeshoptExtension["mode"],
    filter: MeshoptExtension["filter"],
) => Uint8Array;

// Minimal glTF 2.0 decode — the data-boundary half of the importer, pure and
// testable (no GPU, no State). It decodes positions, normals, UVs, indices, the
// node TRS hierarchy, `baseColorFactor`, and (for a skinned + animated mesh) the
// skin + a chosen clip into a {@link SkinInput} the GPU loader bakes to a VAT.
// Morph targets, tangents, vertex colors, and the second UV are dropped — none
// have a consumer yet. The output feeds `load` (index.ts), which owns the GPU
// buffers + spawns the entities.
//
// The accessor decode (interleaved `byteStride` + per-accessor `byteOffset`)
// and the node TRS→world bake are the misimplement-then-blame-the-file traps,
// so they live here behind unit tests, validated against three.js GLTFLoader
// (the decode authority).

/** the slice of the glTF JSON this importer reads. Loose by design — only the fields decoded (or detected
 *  as unsupported) appear. */
export interface GltfJson {
    accessors?: Accessor[];
    bufferViews?: BufferView[];
    meshes?: GltfJsonMesh[];
    materials?: Material[];
    images?: { uri?: string; bufferView?: number; mimeType?: string }[];
    textures?: { source?: number; sampler?: number; extensions?: Extensions }[];
    samplers?: object[];
    nodes?: Node[];
    scenes?: { nodes?: number[] }[];
    scene?: number;
    buffers?: { uri?: string; byteLength: number; extensions?: Record<string, unknown> }[];
    animations?: Animation[];
    skins?: Skin[];
    /** extensions the asset *requires* — a renderer that can't honor one renders the scene wrong */
    extensionsRequired?: string[];
    /** extensions the asset uses (a superset of required) — optional ones may be ignored */
    extensionsUsed?: string[];
}

/** an object's `extensions` map — keyed by extension name (`KHR_*`). The two the importer decodes are typed
 *  ({@link DracoExtension} on a primitive, {@link BasisuExtension} on a texture); the rest stay opaque so
 *  the unsupported-feature scan can name them. */
export interface Extensions {
    // biome-ignore lint/style/useNamingConvention: glTF extension names (KHR_*) are the JSON keys
    KHR_draco_mesh_compression?: DracoExtension;
    // biome-ignore lint/style/useNamingConvention: glTF extension names (KHR_*) are the JSON keys
    KHR_texture_basisu?: BasisuExtension;
    // biome-ignore lint/style/useNamingConvention: the ratified name on a bufferView
    EXT_meshopt_compression?: MeshoptExtension;
    // biome-ignore lint/style/useNamingConvention: the pre-ratification alias the Khronos sample assets ship
    KHR_meshopt_compression?: MeshoptExtension;
    [name: string]: unknown;
}

/** `KHR_draco_mesh_compression` on a primitive: the compressed bufferView + a semantic→Draco-attribute-id map. */
export interface DracoExtension {
    bufferView: number;
    attributes: {
        // biome-ignore lint/style/useNamingConvention: glTF attribute semantic names (uppercase)
        POSITION: number;
        [name: string]: number | undefined;
    };
}

/** `KHR_texture_basisu` on a texture: the KTX2/Basis image source replacing the texture's PNG/JPEG `source`. */
export interface BasisuExtension {
    source: number;
}

/**
 * `EXT_meshopt_compression` on a bufferView: the compressed source slice + how to decompress it. The
 * extension's fields are authoritative — the bufferView's own `buffer`/`byteOffset`/`byteLength` are the
 * (zero-filled) fallback per spec. `mode` selects the codec, `byteStride` is the decompressed element stride,
 * `count` the element count, `filter` the optional post-decode transform.
 */
export interface MeshoptExtension {
    buffer: number;
    byteOffset?: number;
    byteLength: number;
    byteStride: number;
    count: number;
    mode: "ATTRIBUTES" | "TRIANGLES" | "INDICES";
    filter?: "NONE" | "OCTAHEDRAL" | "QUATERNION" | "EXPONENTIAL";
}

interface Accessor {
    bufferView?: number;
    byteOffset?: number;
    componentType: number;
    /** KHR_mesh_quantization: an integer accessor maps to [-1,1] (signed) / [0,1] (unsigned) when set */
    normalized?: boolean;
    count: number;
    type: "SCALAR" | "VEC2" | "VEC3" | "VEC4" | "MAT2" | "MAT3" | "MAT4";
    sparse?: object;
}

interface BufferView {
    buffer: number;
    byteOffset?: number;
    byteLength: number;
    byteStride?: number;
    extensions?: Extensions;
}

interface GltfJsonMesh {
    name?: string;
    primitives: Primitive[];
}

interface Primitive {
    attributes: {
        // biome-ignore lint/style/useNamingConvention: glTF spec attribute semantic names (uppercase)
        POSITION: number;
        // biome-ignore lint/style/useNamingConvention: glTF spec attribute semantic names (uppercase)
        NORMAL?: number;
        // biome-ignore lint/style/useNamingConvention: glTF spec attribute semantic names (uppercase)
        TEXCOORD_0?: number;
        // biome-ignore lint/style/useNamingConvention: glTF spec attribute semantic names (uppercase)
        TEXCOORD_1?: number;
        // biome-ignore lint/style/useNamingConvention: glTF spec attribute semantic names (uppercase)
        COLOR_0?: number;
        // biome-ignore lint/style/useNamingConvention: glTF spec attribute semantic names (uppercase)
        JOINTS_0?: number;
        // biome-ignore lint/style/useNamingConvention: glTF spec attribute semantic names (uppercase)
        WEIGHTS_0?: number;
        // biome-ignore lint/style/useNamingConvention: glTF spec attribute semantic names (uppercase)
        TANGENT?: number;
    };
    indices?: number;
    material?: number;
    mode?: number;
    targets?: object[];
    extensions?: Extensions;
}

interface Material {
    name?: string;
    pbrMetallicRoughness?: {
        baseColorFactor?: number[];
        baseColorTexture?: { index: number };
        metallicFactor?: number;
        roughnessFactor?: number;
        metallicRoughnessTexture?: { index: number };
    };
    normalTexture?: { index: number; scale?: number };
    occlusionTexture?: { index: number; strength?: number };
    emissiveFactor?: number[];
    emissiveTexture?: { index: number };
    alphaMode?: "OPAQUE" | "MASK" | "BLEND";
    alphaCutoff?: number;
    extensions?: Extensions;
}

interface Node {
    name?: string;
    mesh?: number;
    skin?: number;
    children?: number[];
    matrix?: number[];
    translation?: number[];
    rotation?: number[];
    scale?: number[];
}

/** a glTF skin — the joint node list + their inverse-bind matrices (the VAT bake reads both). */
interface Skin {
    joints: number[];
    inverseBindMatrices?: number;
    skeleton?: number;
}

/** an animation sampler: an input accessor (keyframe times) + output accessor (values) + interpolation. */
interface AnimationSampler {
    input: number;
    output: number;
    interpolation?: "LINEAR" | "STEP" | "CUBICSPLINE";
}

/** one animation clip — samplers + channels (a channel targets a node's TRS path, `weights` = morph). */
interface Animation {
    name?: string;
    samplers: AnimationSampler[];
    channels: {
        sampler: number;
        target: { node?: number; path: "translation" | "rotation" | "scale" | "weights" };
    }[];
}

/** one decoded primitive's geometry + material, in shallot's vertex layout. Shared across instances. */
export interface GltfMesh {
    name: string;
    /** packed (px py pz u)(nx ny nz v), {@link VERTEX_FLOATS} floats per vertex */
    vertices: Float32Array;
    indices: Uint32Array;
    /** `baseColorFactor`, linear rgba (glTF stores it linear, as the engine's `Color` expects) */
    color: [number, number, number, number];
    /** index into {@link GltfScene.materials} of this primitive's material, or -1 if it has none */
    material: number;
}

/**
 * one decoded glTF material — the per-material record the importer keys textures + routing off of.
 * `image` indexes {@link GltfScene.images} (the baseColorTexture's source), or is undefined for a
 * factor-only material that needs no texture. `alphaMode` routes the surface (OPAQUE / MASK→cutout /
 * BLEND→translucent); `cutoff` is the MASK threshold (glTF `alphaCutoff`, default 0.5).
 */
export interface GltfMaterial {
    /** `baseColorFactor`, linear rgba — multiplies the sampled albedo */
    color: [number, number, number, number];
    /** index into {@link GltfScene.images} of the baseColorTexture source, undefined if untextured */
    image?: number;
    /** `metallicFactor` / `roughnessFactor` ([0,1], glTF default 1) — scale the metallicRoughness texture */
    metallic: number;
    roughness: number;
    /** metallicRoughness texture source (glTF packs roughness in G, metallic in B), undefined if none */
    mrImage?: number;
    /** tangent-space normal map source + its `scale`, undefined if none (derivative tangents, no TANGENT attr) */
    normalImage?: number;
    normalScale: number;
    /** occlusion texture source (R channel) + its `strength`, undefined if none */
    occImage?: number;
    occStrength: number;
    /** `emissiveFactor × KHR_materials_emissive_strength`, linear rgb — multiplies the emissive texture */
    emissive: [number, number, number];
    emissiveImage?: number;
    alphaMode: "OPAQUE" | "MASK" | "BLEND";
    cutoff: number;
}

/**
 * one image source, source-agnostic by design (so `.glb` is a small later add): either an external
 * `uri` resolved relative to the `.gltf`, or an embedded `bufferView` (a data-URI / `.glb` BIN slice)
 * with its `mimeType`. The GPU half (decode → array layer) resolves whichever is present.
 */
export interface GltfImage {
    uri?: string;
    bufferView?: number;
    mimeType?: string;
}

/** one scene-graph placement of a {@link GltfMesh} — its node's world transform, decomposed to TRS. */
export interface GltfInstance {
    /** index into {@link GltfScene.meshes} */
    mesh: number;
    pos: [number, number, number];
    rot: [number, number, number, number];
    scale: [number, number, number];
}

/**
 * one glTF feature the importer intentionally doesn't handle yet, surfaced so a scene that needs it is
 * diagnosable rather than silently wrong. `feature` is a stable key
 * — an extension name (`KHR_materials_clearcoat`) or a category (`skin` / `morph` / `vertex-color` /
 * `texcoord-1` / `sparse-accessor` / `primitive-mode`). {@link parse} collects them (deduped by key);
 * {@link loadGltf} logs each once. The conformance suite asserts the set per model.
 */
export interface GltfUnsupported {
    feature: string;
    detail?: string;
}

/** decoded glTF: unique geometries + their per-node placements. A mesh shared by N nodes is one
 *  {@link GltfMesh} and N {@link GltfInstance}s, so the import rides the engine's instanced Part path. */
export interface GltfScene {
    meshes: GltfMesh[];
    instances: GltfInstance[];
    /** every material, indexed by {@link GltfMesh.material}; drives texture binding + alpha routing */
    materials: GltfMaterial[];
    /** every image source, indexed by {@link GltfMaterial.image}; the GPU half decodes these to layers */
    images: GltfImage[];
    /**
     * the VAT bake input parallel to {@link meshes}: a {@link SkinInput} for a skinned-mesh entry, `null`
     * for a static one. The rest-pose geometry stays in {@link GltfMesh}; the GPU loader runs `bakeVat`
     * on the non-null entries (kept off the deviceless conformance walk). A skinned entry's instance is
     * identity-placed (the bake is in skeleton space, so crowd placement rides the instance matrix).
     */
    skinInputs: (SkinInput | null)[];
    /** features this importer skipped, deduped by key — empty for a fully-supported scene */
    unsupported: GltfUnsupported[];
}

const COMPONENTS = { SCALAR: 1, VEC2: 2, VEC3: 3, VEC4: 4, MAT2: 4, MAT3: 9, MAT4: 16 } as const;

// glTF accessor.componentType (UNSIGNED_BYTE 5121 is the readIndices fallback). The signed BYTE/SHORT only
// appear via KHR_mesh_quantization (quantized positions/normals/uvs).
const BYTE = 5120;
const UNSIGNED_BYTE = 5121;
const SHORT = 5122;
const UNSIGNED_SHORT = 5123;
const UNSIGNED_INT = 5125;
const FLOAT = 5126;
const TRIANGLES = 4;

/**
 * decompose a column-major affine matrix into translation, rotation
 * quaternion (xyzw), and scale. Exact for a TRS chain (translate · rotate ·
 * scale); a mirror folds into the x-axis scale via the determinant sign. Shear
 * (a non-uniform scale through a rotation) is not representable as TRS and is
 * silently dropped — no consumer authors it, and the engine's transform is TRS.
 */
export function decompose(m: ArrayLike<number>): {
    pos: [number, number, number];
    rot: [number, number, number, number];
    scale: [number, number, number];
} {
    const c0x = m[0];
    const c0y = m[1];
    const c0z = m[2];
    const c1x = m[4];
    const c1y = m[5];
    const c1z = m[6];
    const c2x = m[8];
    const c2y = m[9];
    const c2z = m[10];

    let sx = Math.hypot(c0x, c0y, c0z);
    const sy = Math.hypot(c1x, c1y, c1z);
    const sz = Math.hypot(c2x, c2y, c2z);
    const det =
        c0x * (c1y * c2z - c1z * c2y) +
        c0y * (c1z * c2x - c1x * c2z) +
        c0z * (c1x * c2y - c1y * c2x);
    if (det < 0) sx = -sx;

    // normalize the basis columns by scale → the pure rotation matrix R (row,col)
    const r00 = c0x / sx;
    const r10 = c0y / sx;
    const r20 = c0z / sx;
    const r01 = c1x / sy;
    const r11 = c1y / sy;
    const r21 = c1z / sy;
    const r02 = c2x / sz;
    const r12 = c2y / sz;
    const r22 = c2z / sz;

    // Shepperd: pick the largest diagonal term to avoid a near-zero divisor
    let qx: number;
    let qy: number;
    let qz: number;
    let qw: number;
    const trace = r00 + r11 + r22;
    if (trace > 0) {
        const s = 0.5 / Math.sqrt(trace + 1);
        qw = 0.25 / s;
        qx = (r21 - r12) * s;
        qy = (r02 - r20) * s;
        qz = (r10 - r01) * s;
    } else if (r00 > r11 && r00 > r22) {
        const s = 2 * Math.sqrt(1 + r00 - r11 - r22);
        qw = (r21 - r12) / s;
        qx = 0.25 * s;
        qy = (r01 + r10) / s;
        qz = (r02 + r20) / s;
    } else if (r11 > r22) {
        const s = 2 * Math.sqrt(1 + r11 - r00 - r22);
        qw = (r02 - r20) / s;
        qx = (r01 + r10) / s;
        qy = 0.25 * s;
        qz = (r12 + r21) / s;
    } else {
        const s = 2 * Math.sqrt(1 + r22 - r00 - r11);
        qw = (r10 - r01) / s;
        qx = (r02 + r20) / s;
        qy = (r12 + r21) / s;
        qz = 0.25 * s;
    }

    return {
        pos: [m[12], m[13], m[14]],
        rot: [qx, qy, qz, qw],
        scale: [sx, sy, sz],
    };
}

// a node's local matrix: its explicit `matrix`, else its TRS composed
function localMatrix(node: Node, out: Float32Array): Float32Array {
    if (node.matrix) {
        out.set(node.matrix);
        return out;
    }
    const t = node.translation ?? [0, 0, 0];
    const r = node.rotation ?? [0, 0, 0, 1];
    const s = node.scale ?? [1, 1, 1];
    return compose(t[0], t[1], t[2], r[0], r[1], r[2], r[3], s[0], s[1], s[2], out);
}

// a glTF accessor componentType's byte size
function compSize(componentType: number): number {
    switch (componentType) {
        case BYTE:
        case UNSIGNED_BYTE:
            return 1;
        case SHORT:
        case UNSIGNED_SHORT:
            return 2;
        default:
            return 4; // FLOAT / UNSIGNED_INT
    }
}

// read + dequantize one accessor component to f32. FLOAT is the base path; KHR_mesh_quantization stores
// BYTE/SHORT, dequantized by `normalized` (→ [-1,1] signed / [0,1] unsigned, glTF 3.6.2.2) or read raw (a
// non-normalized quantized position, dequantized downstream by the baked node scale). Little-endian always.
function readComponent(
    view: DataView,
    o: number,
    componentType: number,
    normalized: boolean,
): number {
    switch (componentType) {
        case BYTE: {
            const v = view.getInt8(o);
            return normalized ? Math.max(v / 127, -1) : v;
        }
        case UNSIGNED_BYTE: {
            const v = view.getUint8(o);
            return normalized ? v / 255 : v;
        }
        case SHORT: {
            const v = view.getInt16(o, true);
            return normalized ? Math.max(v / 32767, -1) : v;
        }
        case UNSIGNED_SHORT: {
            const v = view.getUint16(o, true);
            return normalized ? v / 65535 : v;
        }
        default:
            return view.getFloat32(o, true);
    }
}

// read a float accessor (positions/normals/uvs) into a tight [count × components] array, honoring the
// bufferView's interleave stride, the accessor's offset within it, and a quantized (BYTE/SHORT) componentType.
function readFloats(gltf: GltfJson, buffers: ArrayBuffer[], index: number): Float32Array {
    const acc = gltf.accessors![index];
    const bv = gltf.bufferViews![acc.bufferView!];
    const comps = COMPONENTS[acc.type];
    const size = compSize(acc.componentType);
    const stride = bv.byteStride ?? comps * size;
    const base = (bv.byteOffset ?? 0) + (acc.byteOffset ?? 0);
    const norm = acc.normalized ?? false;
    const view = new DataView(buffers[bv.buffer]);
    const out = new Float32Array(acc.count * comps);
    for (let i = 0; i < acc.count; i++) {
        const o = base + i * stride;
        for (let c = 0; c < comps; c++)
            out[i * comps + c] = readComponent(view, o + c * size, acc.componentType, norm);
    }
    return out;
}

// read an index accessor (ubyte/ushort/uint) widened to u32 — the engine's index format
function readIndices(gltf: GltfJson, buffers: ArrayBuffer[], index: number): Uint32Array {
    const acc = gltf.accessors![index];
    const bv = gltf.bufferViews![acc.bufferView!];
    const size =
        acc.componentType === UNSIGNED_INT ? 4 : acc.componentType === UNSIGNED_SHORT ? 2 : 1;
    const stride = bv.byteStride ?? size;
    const base = (bv.byteOffset ?? 0) + (acc.byteOffset ?? 0);
    const view = new DataView(buffers[bv.buffer]);
    const out = new Uint32Array(acc.count);
    for (let i = 0; i < acc.count; i++) {
        const o = base + i * stride;
        out[i] =
            acc.componentType === UNSIGNED_INT
                ? view.getUint32(o, true)
                : acc.componentType === UNSIGNED_SHORT
                  ? view.getUint16(o, true)
                  : view.getUint8(o);
    }
    return out;
}

// read a VEC4 joint-index accessor (ubyte/ushort) widened to u16 — the values are slots into the skin's
// `joints` array, not node indices (glTF 3.7.3.1)
function readJoints(gltf: GltfJson, buffers: ArrayBuffer[], index: number): Uint16Array {
    const acc = gltf.accessors![index];
    const bv = gltf.bufferViews![acc.bufferView!];
    const size = acc.componentType === UNSIGNED_SHORT ? 2 : 1;
    const stride = bv.byteStride ?? size * 4;
    const base = (bv.byteOffset ?? 0) + (acc.byteOffset ?? 0);
    const view = new DataView(buffers[bv.buffer]);
    const out = new Uint16Array(acc.count * 4);
    for (let i = 0; i < acc.count; i++) {
        const o = base + i * stride;
        for (let c = 0; c < 4; c++) {
            out[i * 4 + c] =
                acc.componentType === UNSIGNED_SHORT
                    ? view.getUint16(o + c * 2, true)
                    : view.getUint8(o + c);
        }
    }
    return out;
}

// read a VEC4 weight accessor — float, or normalized ubyte/ushort (glTF allows all three). Normalized
// integer weights decode by their max, so each lands in [0, 1]
function readWeights(gltf: GltfJson, buffers: ArrayBuffer[], index: number): Float32Array {
    const acc = gltf.accessors![index];
    const bv = gltf.bufferViews![acc.bufferView!];
    const size =
        acc.componentType === UNSIGNED_BYTE ? 1 : acc.componentType === UNSIGNED_SHORT ? 2 : 4;
    const stride = bv.byteStride ?? size * 4;
    const base = (bv.byteOffset ?? 0) + (acc.byteOffset ?? 0);
    const view = new DataView(buffers[bv.buffer]);
    const out = new Float32Array(acc.count * 4);
    for (let i = 0; i < acc.count; i++) {
        const o = base + i * stride;
        for (let c = 0; c < 4; c++) {
            out[i * 4 + c] =
                acc.componentType === FLOAT
                    ? view.getFloat32(o + c * 4, true)
                    : acc.componentType === UNSIGNED_SHORT
                      ? view.getUint16(o + c * 2, true) / 65535
                      : view.getUint8(o + c) / 255;
        }
    }
    return out;
}

/**
 * smooth per-vertex normals from positions + triangle indices — the glTF 2.0 fallback when a primitive omits
 * `NORMAL` (the spec mandates the client calculate them; the Khronos Fox is the common case). Area-weighted
 * face-normal accumulation: a triangle's un-normalized edge cross product scales with its area, so a larger
 * face weights its vertices more, then normalize per vertex. Reference: three.js
 * `BufferGeometry.computeVertexNormals` (smooth, not the spec's literal flat — flat would need de-indexing and
 * facet an organic model; smooth is the universal loader behavior). Pure, so the geometry is unit-tested.
 */
export function computeNormals(pos: Float32Array, indices: Uint32Array): Float32Array {
    const normals = new Float32Array(pos.length);
    for (let i = 0; i + 2 < indices.length; i += 3) {
        const a = indices[i] * 3;
        const b = indices[i + 1] * 3;
        const c = indices[i + 2] * 3;
        const e1x = pos[b] - pos[a];
        const e1y = pos[b + 1] - pos[a + 1];
        const e1z = pos[b + 2] - pos[a + 2];
        const e2x = pos[c] - pos[a];
        const e2y = pos[c + 1] - pos[a + 1];
        const e2z = pos[c + 2] - pos[a + 2];
        const nx = e1y * e2z - e1z * e2y;
        const ny = e1z * e2x - e1x * e2z;
        const nz = e1x * e2y - e1y * e2x;
        normals[a] += nx;
        normals[a + 1] += ny;
        normals[a + 2] += nz;
        normals[b] += nx;
        normals[b + 1] += ny;
        normals[b + 2] += nz;
        normals[c] += nx;
        normals[c + 1] += ny;
        normals[c + 2] += nz;
    }
    for (let i = 0; i < normals.length; i += 3) {
        const len = Math.hypot(normals[i], normals[i + 1], normals[i + 2]) || 1;
        normals[i] /= len;
        normals[i + 1] /= len;
        normals[i + 2] /= len;
    }
    return normals;
}

// pack deinterleaved attribute arrays into shallot's (posU)(normalV) vertex layout. The callers synthesize a
// missing normal via computeNormals (glTF requires it), so `normal` is only null when a producer truly has
// none; uv defaults to zero (an untextured material never samples it). Shared by the accessor + Draco paths.
function packVertices(
    pos: Float32Array,
    normal: Float32Array | null,
    uv: Float32Array | null,
): Float32Array {
    const count = pos.length / 3;
    const vertices = new Float32Array(count * VERTEX_FLOATS);
    for (let i = 0; i < count; i++) {
        const o = i * VERTEX_FLOATS;
        vertices[o] = pos[i * 3];
        vertices[o + 1] = pos[i * 3 + 1];
        vertices[o + 2] = pos[i * 3 + 2];
        vertices[o + 3] = uv ? uv[i * 2] : 0;
        vertices[o + 4] = normal ? normal[i * 3] : 0;
        vertices[o + 5] = normal ? normal[i * 3 + 1] : 0;
        vertices[o + 6] = normal ? normal[i * 3 + 2] : 0;
        vertices[o + 7] = uv ? uv[i * 2 + 1] : 0;
    }
    return vertices;
}

// a primitive's baseColorFactor as linear rgba, white when it has no material
function factorColor(gltf: GltfJson, prim: Primitive): [number, number, number, number] {
    const factor = gltf.materials?.[prim.material ?? -1]?.pbrMetallicRoughness?.baseColorFactor;
    return factor ? [factor[0] ?? 1, factor[1] ?? 1, factor[2] ?? 1, factor[3] ?? 1] : [1, 1, 1, 1];
}

// a bufferView's raw bytes (the Draco-compressed blob the extension points at)
function bufferViewBytes(gltf: GltfJson, buffers: ArrayBuffer[], index: number): Uint8Array {
    const bv = gltf.bufferViews![index];
    return new Uint8Array(buffers[bv.buffer], bv.byteOffset ?? 0, bv.byteLength);
}

// a bufferView's meshopt extension — EXT_meshopt_compression (ratified) or KHR_meshopt_compression (the
// pre-ratification alias the Khronos sample assets still ship), one extension under two names
function meshoptExt(bv: BufferView): MeshoptExtension | undefined {
    return bv.extensions?.EXT_meshopt_compression ?? bv.extensions?.KHR_meshopt_compression;
}

// decompress every meshopt-compressed bufferView up front, returning a normalized (gltf, buffers) the rest of
// parse reads unchanged: each compressed view becomes a plain view over a freshly-appended decompressed buffer
// (byteOffset 0, byteStride = the extension's). The input gltf/buffers are untouched — decode() reuses the json
// for texture decode — so only the bufferViews array is shallow-replaced, and the originals' bytes stay valid.
function inflateMeshopt(
    gltf: GltfJson,
    buffers: ArrayBuffer[],
    decode: MeshoptDecode,
): { gltf: GltfJson; buffers: ArrayBuffer[] } {
    const views = gltf.bufferViews;
    if (!views?.some(meshoptExt)) return { gltf, buffers };
    const bufs = buffers.slice();
    const out = views.map((bv): BufferView => {
        const ext = meshoptExt(bv);
        if (!ext) return bv;
        const src = new Uint8Array(buffers[ext.buffer], ext.byteOffset ?? 0, ext.byteLength);
        const data = decode(src, ext.count, ext.byteStride, ext.mode, ext.filter);
        const buffer = bufs.length;
        bufs.push(data.buffer as ArrayBuffer);
        return { buffer, byteOffset: 0, byteLength: data.byteLength, byteStride: ext.byteStride };
    });
    return { gltf: { ...gltf, bufferViews: out }, buffers: bufs };
}

// deinterleave a standard primitive's attributes (accessor-backed) into the vertex layout
function decodeMesh(
    gltf: GltfJson,
    buffers: ArrayBuffer[],
    prim: Primitive,
    name: string,
): GltfMesh {
    const pos = readFloats(gltf, buffers, prim.attributes.POSITION);
    const uv =
        prim.attributes.TEXCOORD_0 !== undefined
            ? readFloats(gltf, buffers, prim.attributes.TEXCOORD_0)
            : null;
    const indices =
        prim.indices !== undefined
            ? readIndices(gltf, buffers, prim.indices)
            : Uint32Array.from({ length: pos.length / 3 }, (_, i) => i);
    // glTF requires the client calculate normals when a primitive omits NORMAL (the Khronos Fox has none —
    // without this it bakes zero normals and renders unlit/black)
    const normal =
        prim.attributes.NORMAL !== undefined
            ? readFloats(gltf, buffers, prim.attributes.NORMAL)
            : computeNormals(pos, indices);
    return {
        name,
        vertices: packVertices(pos, normal, uv),
        indices,
        color: factorColor(gltf, prim),
        material: prim.material ?? -1,
    };
}

// decode a Draco-compressed primitive — the injected `decode` (loadGltf dynamic-imports the codec only when
// a Draco asset appears, so it stays out of the base bundle) turns the extension's bufferView into the same
// attribute arrays the accessor path yields, then shares packVertices
function decodeDracoMesh(
    gltf: GltfJson,
    buffers: ArrayBuffer[],
    prim: Primitive,
    ext: DracoExtension,
    name: string,
    decode: DracoDecode,
): GltfMesh {
    const { pos, normal, uv, indices } = decode(
        bufferViewBytes(gltf, buffers, ext.bufferView),
        ext.attributes,
    );
    return {
        name,
        // synthesize normals if the Draco primitive carries none (same glTF requirement as the accessor path)
        vertices: packVertices(pos, normal ?? computeNormals(pos, indices), uv),
        indices,
        color: factorColor(gltf, prim),
        material: prim.material ?? -1,
    };
}

// resolve a texture reference to an image index. A KTX2 texture sources its image through
// KHR_texture_basisu, not the texture's plain `source` (which may be absent or a PNG fallback) — prefer
// the extension when present. undefined when the ref is absent, so the surface falls back to the factor
function texImage(gltf: GltfJson, ref?: { index: number }): number | undefined {
    if (!ref) return undefined;
    const t = gltf.textures?.[ref.index];
    return t ? (t.extensions?.KHR_texture_basisu?.source ?? t.source) : undefined;
}

// decode each material's metallic-roughness PBR fields + texture refs + alpha routing. Each `*Image`
// resolves a texture → an index into gltf.images; undefined when the slot is absent so the surface uses the
// factor alone. glTF metallicFactor / roughnessFactor default to 1; a baseColor-only material is factor-only
function decodeMaterials(gltf: GltfJson): GltfMaterial[] {
    return (gltf.materials ?? []).map((m) => {
        const pbr = m.pbrMetallicRoughness;
        const factor = pbr?.baseColorFactor;
        const color: [number, number, number, number] = factor
            ? [factor[0] ?? 1, factor[1] ?? 1, factor[2] ?? 1, factor[3] ?? 1]
            : [1, 1, 1, 1];
        // emissiveFactor is scaled by KHR_materials_emissive_strength (HDR glow past 1) when present
        const ef = m.emissiveFactor;
        const eStrength =
            (
                m.extensions?.KHR_materials_emissive_strength as
                    | { emissiveStrength?: number }
                    | undefined
            )?.emissiveStrength ?? 1;
        const emissive: [number, number, number] = ef
            ? [(ef[0] ?? 0) * eStrength, (ef[1] ?? 0) * eStrength, (ef[2] ?? 0) * eStrength]
            : [0, 0, 0];
        return {
            color,
            image: texImage(gltf, pbr?.baseColorTexture),
            metallic: pbr?.metallicFactor ?? 1,
            roughness: pbr?.roughnessFactor ?? 1,
            mrImage: texImage(gltf, pbr?.metallicRoughnessTexture),
            normalImage: texImage(gltf, m.normalTexture),
            normalScale: m.normalTexture?.scale ?? 1,
            occImage: texImage(gltf, m.occlusionTexture),
            occStrength: m.occlusionTexture?.strength ?? 1,
            emissive,
            emissiveImage: texImage(gltf, m.emissiveTexture),
            alphaMode: m.alphaMode ?? "OPAQUE",
            cutoff: m.alphaCutoff ?? 0.5,
        };
    });
}

// the extensions this importer decodes. KHR_draco_mesh_compression is handled only when loadGltf injected a
// decoder (`dracoAvailable`, since the geometry decode lives in parse); KHR_texture_basisu is always handled
// — its transcode runs GPU-side in buildTextures, independent of parse, the same way PNG textures do.
// Anything else — required or merely used — is reported unsupported.
function supportedExtension(
    name: string,
    dracoAvailable: boolean,
    meshoptAvailable: boolean,
): boolean {
    if (name === "KHR_draco_mesh_compression") return dracoAvailable;
    if (name === "EXT_meshopt_compression" || name === "KHR_meshopt_compression")
        return meshoptAvailable;
    // quantized accessors are dequantized in readFloats (positions ride the baked node scale), so the
    // companion mesh-quantization extension needs no codec — it's handled whenever a quantized asset loads
    if (name === "KHR_mesh_quantization") return true;
    // KHR_materials_emissive_strength folds into the emissive factor at decode (a plain HDR scale)
    return name === "KHR_texture_basisu" || name === "KHR_materials_emissive_strength";
}

// a column-major identity mat4 per joint — the fallback when a skin omits inverseBindMatrices (glTF
// then defines them as identity)
function identityMats(n: number): Float32Array {
    const out = new Float32Array(n * 16);
    for (let i = 0; i < n; i++)
        out[i * 16] = out[i * 16 + 5] = out[i * 16 + 10] = out[i * 16 + 15] = 1;
    return out;
}

// a primitive is VAT-bakeable when its skin attributes + position are accessor-backed (not Draco, not
// sparse) — the bake reads them as plain accessors. Draco/sparse skinned primitives stay flagged.
function primSkinnable(gltf: GltfJson, prim: Primitive): boolean {
    const a = prim.attributes;
    if (a.JOINTS_0 === undefined || a.WEIGHTS_0 === undefined) return false;
    if (prim.extensions?.KHR_draco_mesh_compression) return false;
    const posAcc = gltf.accessors?.[a.POSITION];
    if (posAcc?.bufferView === undefined || posAcc.sparse) return false;
    // a still-compressed meshopt position (inflate failed) can't be read as the bake's plain accessor
    return !meshoptExt(gltf.bufferViews![posAcc.bufferView]);
}

// whether the document's skinned + animated content bakes to a VAT: a skin
// and an animation both exist, every sampler is LINEAR/STEP (no CUBICSPLINE), and every skinned node's
// primitives carry accessor-backed JOINTS_0 + WEIGHTS_0. Drives BOTH the unsupported-flag drop and the
// bake — the two must agree, or the conformance status lies about what renders.
export function skinBakeable(gltf: GltfJson): boolean {
    if (!gltf.skins?.length || !gltf.animations?.length) return false;
    for (const anim of gltf.animations) {
        for (const s of anim.samplers ?? []) {
            const interp = s.interpolation ?? "LINEAR";
            if (interp !== "LINEAR" && interp !== "STEP") return false;
        }
    }
    const skinned = (gltf.nodes ?? []).filter((n) => n.skin !== undefined && n.mesh !== undefined);
    if (skinned.length === 0) return false;
    for (const node of skinned) {
        for (const prim of gltf.meshes![node.mesh!].primitives) {
            if ((prim.mode ?? TRIANGLES) !== TRIANGLES) continue;
            if (!primSkinnable(gltf, prim)) return false;
        }
    }
    return true;
}

// decode the glTF-agnostic bake input for one skinned node's primitive: the node hierarchy's base TRS, the
// clip's channels, the skin's joints + inverse-bind matrices, and the primitive's per-vertex weights + rest
// geometry. Light (accessor reads only) — the heavy per-frame bake is vat.ts's bakeVat, run GPU-side.
function decodeSkinInput(
    gltf: GltfJson,
    buffers: ArrayBuffer[],
    node: Node,
    animIndex: number,
    prim: Primitive,
): SkinInput {
    const skin = gltf.skins![node.skin!];
    const inverseBind =
        skin.inverseBindMatrices !== undefined
            ? readFloats(gltf, buffers, skin.inverseBindMatrices)
            : identityMats(skin.joints.length);
    const nodes = (gltf.nodes ?? []).map((n) => {
        if (n.matrix) {
            const trs = decompose(n.matrix);
            return { t: trs.pos, r: trs.rot, s: trs.scale, children: n.children ?? [] };
        }
        return {
            t: (n.translation ?? [0, 0, 0]) as [number, number, number],
            r: (n.rotation ?? [0, 0, 0, 1]) as [number, number, number, number],
            s: (n.scale ?? [1, 1, 1]) as [number, number, number],
            children: n.children ?? [],
        };
    });
    const roots = gltf.scenes?.[gltf.scene ?? 0]?.nodes ?? gltf.nodes?.map((_, i) => i) ?? [];
    const anim = gltf.animations![animIndex] ?? gltf.animations![0];
    const channels: SkinChannel[] = [];
    let duration = 0;
    for (const ch of anim.channels) {
        if (ch.target.path === "weights" || ch.target.node === undefined) continue;
        const s = anim.samplers[ch.sampler];
        const times = readFloats(gltf, buffers, s.input);
        channels.push({
            node: ch.target.node,
            path: ch.target.path,
            times,
            values: readFloats(gltf, buffers, s.output),
            step: (s.interpolation ?? "LINEAR") === "STEP",
        });
        duration = Math.max(duration, times[times.length - 1] ?? 0);
    }
    const restPos = readFloats(gltf, buffers, prim.attributes.POSITION);
    // a skinned primitive that omits NORMAL (the Khronos Fox) gets synthesized rest normals — the bake skins
    // these per frame, so without them the VAT is all-zero normals and the mesh renders unlit/black
    const restIndices =
        prim.indices !== undefined
            ? readIndices(gltf, buffers, prim.indices)
            : Uint32Array.from({ length: restPos.length / 3 }, (_, i) => i);
    const restNormal =
        prim.attributes.NORMAL !== undefined
            ? readFloats(gltf, buffers, prim.attributes.NORMAL)
            : computeNormals(restPos, restIndices);
    return {
        nodes,
        roots,
        channels,
        joints: skin.joints,
        inverseBind,
        jointIndex: readJoints(gltf, buffers, prim.attributes.JOINTS_0!),
        weights: readWeights(gltf, buffers, prim.attributes.WEIGHTS_0!),
        restPos,
        restNormal,
        duration,
    };
}

// scan a document for features the importer skips, deduped by a stable key (extension name or category). The
// loader logs these once; the conformance suite asserts the set per model. Detection only — no decode. When
// the skinned+animated content is VAT-bakeable (skinBakeable), `skin`/`animation` are NOT flagged — the
// importer handles them; otherwise both stay flagged (e.g. CUBICSPLINE, or animation without a skin).
function scanUnsupported(
    gltf: GltfJson,
    dracoAvailable: boolean,
    meshoptAvailable: boolean,
): GltfUnsupported[] {
    const found = new Map<string, GltfUnsupported>();
    const flag = (feature: string, detail?: string) => {
        if (!found.has(feature)) found.set(feature, detail ? { feature, detail } : { feature });
    };
    const bakeable = skinBakeable(gltf);

    for (const ext of gltf.extensionsRequired ?? []) {
        if (!supportedExtension(ext, dracoAvailable, meshoptAvailable))
            flag(ext, "required extension");
    }
    for (const ext of gltf.extensionsUsed ?? []) {
        if (!supportedExtension(ext, dracoAvailable, meshoptAvailable) && !found.has(ext))
            flag(ext, "extension");
    }
    if (gltf.animations?.length && !bakeable)
        flag("animation", `${gltf.animations.length} animations`);
    if (gltf.skins?.length && !bakeable) flag("skin", `${gltf.skins.length} skins`);

    for (const m of gltf.meshes ?? []) {
        for (const prim of m.primitives) {
            if ((prim.mode ?? TRIANGLES) !== TRIANGLES) flag("primitive-mode", `mode ${prim.mode}`);
            const a = prim.attributes;
            if (a.TEXCOORD_1 !== undefined) flag("texcoord-1", "second UV set");
            if (a.COLOR_0 !== undefined) flag("vertex-color");
            if (!bakeable && (a.JOINTS_0 !== undefined || a.WEIGHTS_0 !== undefined))
                flag("skin", "skinning attributes");
            if (prim.targets?.length) flag("morph", `${prim.targets.length} targets`);
            for (const idx of [a.POSITION, prim.indices]) {
                if (idx !== undefined && gltf.accessors?.[idx]?.sparse) flag("sparse-accessor");
            }
        }
    }
    return [...found.values()];
}

// decode one primitive's geometry, or null when it can't be read yet. A Draco primitive needs the injected
// decoder (absent → skipped, its extension warning explains why); a standard primitive needs a POSITION
// accessor backed by a bufferView (a sparse-only / unbacked accessor is skipped, flagged as sparse-accessor)
function decodePrimitive(
    gltf: GltfJson,
    buffers: ArrayBuffer[],
    prim: Primitive,
    name: string,
    decodeDraco?: DracoDecode,
): GltfMesh | null {
    const ext = prim.extensions?.KHR_draco_mesh_compression;
    if (ext) {
        return decodeDraco ? decodeDracoMesh(gltf, buffers, prim, ext, name, decodeDraco) : null;
    }
    const posBv = gltf.accessors?.[prim.attributes.POSITION]?.bufferView;
    if (posBv === undefined) return null;
    // a position bufferView still carrying its meshopt extension means inflateMeshopt was absent or failed —
    // its bytes are compressed, not the float geometry decodeMesh expects, so skip it (scanUnsupported flags it)
    if (meshoptExt(gltf.bufferViews![posBv])) return null;
    return decodeMesh(gltf, buffers, prim, name);
}

/**
 * decode a glTF document + its resolved binary buffers into unique geometries
 * and their per-node placements. Triangle primitives only; the node hierarchy
 * is baked to a world transform per instance (the engine has no runtime parent
 * — scenes are flat). `buffers` is one `ArrayBuffer` per glTF `buffers[]` entry
 * (the `.bin`, a data-URI, or a `.glb` BIN chunk), resolved by the caller.
 *
 * `decodeDraco` is the dynamic-imported Draco codec, passed by {@link loadGltf} only when a primitive needs
 * it (so the codec stays out of the base bundle); omit it and Draco primitives are skipped + flagged.
 * `decodeMeshopt` is the same for `EXT_meshopt_compression` — passed only when a bufferView needs it, it
 * decompresses those views up front so the accessor decode reads them like any other.
 *
 * @example
 * const { meshes, instances } = parse(json, [bin]);
 */
export function parse(
    gltf: GltfJson,
    buffers: ArrayBuffer[],
    decodeDraco?: DracoDecode,
    decodeMeshopt?: MeshoptDecode,
    animIndex = 0,
): GltfScene {
    // decompress meshopt bufferViews up front so the accessor reads below see plain views (the companion
    // KHR_mesh_quantization is dequantized in readFloats); a no-op when no view carries the extension. A decode
    // failure (the bundled decoder can't read a newer bitstream version or an unsupported filter) degrades
    // gracefully: the views stay compressed, decodePrimitive skips them, and scanUnsupported flags the
    // extension — a clean skip, not a crash (the same shape as a Draco primitive with no codec).
    let meshoptOk = decodeMeshopt !== undefined;
    if (decodeMeshopt) {
        try {
            ({ gltf, buffers } = inflateMeshopt(gltf, buffers, decodeMeshopt));
        } catch {
            meshoptOk = false;
        }
    }

    const meshes: GltfMesh[] = [];
    const skinInputs: (SkinInput | null)[] = [];

    // a skinned mesh can't share geometry across poses, so its primitives bake to per-node VAT entries in
    // the scene walk (phase 2); the static dedup decode below skips them.
    const bakeable = skinBakeable(gltf);
    const skinnedMeshes = new Set<number>(
        bakeable
            ? (gltf.nodes ?? [])
                  .filter((n) => n.skin !== undefined && n.mesh !== undefined)
                  .map((n) => n.mesh!)
            : [],
    );

    // 1. decode each unique static (mesh, primitive) once; record which entries each glTF mesh expands to
    const meshEntries: number[][] = [];
    for (let mi = 0; mi < (gltf.meshes?.length ?? 0); mi++) {
        const entries: number[] = [];
        if (!skinnedMeshes.has(mi)) {
            const m = gltf.meshes![mi];
            for (let pi = 0; pi < m.primitives.length; pi++) {
                const prim = m.primitives[pi];
                if ((prim.mode ?? TRIANGLES) !== TRIANGLES) continue; // scanUnsupported flags the mode
                const mesh = decodePrimitive(
                    gltf,
                    buffers,
                    prim,
                    `${m.name ?? mi}#${pi}`,
                    decodeDraco,
                );
                if (!mesh) continue; // unsupported compression / sparse-only — flagged in scanUnsupported
                entries.push(meshes.length);
                meshes.push(mesh);
                skinInputs.push(null);
            }
        }
        meshEntries[mi] = entries;
    }

    // 2. walk the scene graph, baking each node's world transform; emit one instance per (node, primitive).
    // A skinned node emits its own VAT entries (rest geometry + a SkinInput) and an identity instance — the
    // bake places the skeleton in scene space, so a crowd rides the instance matrix on top.
    const instances: GltfInstance[] = [];
    const local = new Float32Array(16);
    const stack: { node: number; parent: Float32Array }[] = [];
    const roots = gltf.scenes?.[gltf.scene ?? 0]?.nodes ?? gltf.nodes?.map((_, i) => i) ?? [];
    const identity = compose(0, 0, 0, 0, 0, 0, 1, 1, 1, 1, new Float32Array(16));
    for (let i = roots.length - 1; i >= 0; i--) stack.push({ node: roots[i], parent: identity });
    while (stack.length > 0) {
        const { node, parent } = stack.pop()!;
        const n = gltf.nodes![node];
        const world = multiply(parent, localMatrix(n, local), new Float32Array(16));
        if (n.mesh !== undefined && skinnedMeshes.has(n.mesh) && n.skin !== undefined) {
            const m = gltf.meshes![n.mesh];
            for (let pi = 0; pi < m.primitives.length; pi++) {
                const prim = m.primitives[pi];
                if ((prim.mode ?? TRIANGLES) !== TRIANGLES) continue;
                const mesh = decodePrimitive(
                    gltf,
                    buffers,
                    prim,
                    `${m.name ?? n.mesh}#${pi}@${node}`,
                );
                if (!mesh) continue;
                const idx = meshes.length;
                meshes.push(mesh);
                skinInputs.push(decodeSkinInput(gltf, buffers, n, animIndex, prim));
                instances.push({ mesh: idx, pos: [0, 0, 0], rot: [0, 0, 0, 1], scale: [1, 1, 1] });
            }
        } else if (n.mesh !== undefined) {
            const trs = decompose(world);
            for (const entry of meshEntries[n.mesh] ?? []) instances.push({ mesh: entry, ...trs });
        }
        for (const child of n.children ?? []) stack.push({ node: child, parent: world });
    }

    const images: GltfImage[] = (gltf.images ?? []).map((i) => ({
        uri: i.uri,
        bufferView: i.bufferView,
        mimeType: i.mimeType,
    }));
    return {
        meshes,
        instances,
        materials: decodeMaterials(gltf),
        images,
        skinInputs,
        unsupported: scanUnsupported(gltf, decodeDraco !== undefined, meshoptOk),
    };
}
