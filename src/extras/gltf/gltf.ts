import { generateTangents } from "../../core/mesh";
import {
    ImageAddressMode,
    ImageFilterMode,
    type ImageSamplerDescriptor,
} from "../../core/rendering";
import {
    composeMat4 as compose,
    decomposeMat4,
    multiplyMat4 as multiply,
} from "../../engine/utils";
import type { TextureTransform } from "../../standard/rendering";
import type { GltfAnimation, GltfAnimationPath, GltfNode } from "./animation";

const VERTEX_FLOATS = 8;

export interface GltfJson {
    asset?: { version: string; minVersion?: string };
    accessors?: Accessor[];
    bufferViews?: BufferView[];
    meshes?: GltfJsonMesh[];
    materials?: Material[];
    images?: { uri?: string; bufferView?: number; mimeType?: string }[];
    textures?: { source?: number; sampler?: number; extensions?: Extensions }[];
    samplers?: { wrapS?: number; wrapT?: number; magFilter?: number; minFilter?: number }[];
    nodes?: Node[];
    scenes?: { nodes?: number[] }[];
    scene?: number;
    buffers?: { uri?: string; byteLength: number; extensions?: Record<string, unknown> }[];
    animations?: {
        name?: string;
        channels: { sampler: number; target: { node?: number; path: GltfAnimationPath } }[];
        samplers: {
            input: number;
            output: number;
            interpolation?: "STEP" | "LINEAR" | "CUBICSPLINE";
        }[];
    }[];
    skins?: { joints: number[]; inverseBindMatrices?: number; skeleton?: number }[];
    extensionsRequired?: string[];
    extensionsUsed?: string[];
}

type Extensions = Record<string, unknown>;

interface Accessor {
    bufferView?: number;
    byteOffset?: number;
    componentType: number;
    normalized?: boolean;
    count: number;
    type: "SCALAR" | "VEC2" | "VEC3" | "VEC4" | "MAT2" | "MAT3" | "MAT4";
    sparse?: {
        count: number;
        indices: { bufferView: number; byteOffset?: number; componentType: number };
        values: { bufferView: number; byteOffset?: number };
    };
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
    weights?: number[];
}

interface Primitive {
    // biome-ignore-start lint/style/useNamingConvention: glTF owns its attribute and extension names.
    attributes: {
        POSITION: number;

        NORMAL?: number;

        TEXCOORD_0?: number;

        TEXCOORD_1?: number;

        COLOR_0?: number;

        JOINTS_0?: number;

        WEIGHTS_0?: number;

        TANGENT?: number;
    };
    // biome-ignore-end lint/style/useNamingConvention: glTF owns its attribute and extension names.
    indices?: number;
    material?: number;
    mode?: number;
    targets?: Record<string, number>[];
    extensions?: Extensions;
}

interface TextureInfo {
    index: number;
    texCoord?: number;
    // biome-ignore lint/style/useNamingConvention: glTF owns its extension names.
    extensions?: { KHR_texture_transform?: TextureTransform & { texCoord?: number } };
}
function textureChannel(texture?: TextureInfo): number {
    return texture?.extensions?.KHR_texture_transform?.texCoord ?? texture?.texCoord ?? 0;
}

interface Material {
    name?: string;
    doubleSided?: boolean;
    pbrMetallicRoughness?: {
        baseColorFactor?: number[];
        baseColorTexture?: TextureInfo;
        metallicFactor?: number;
        roughnessFactor?: number;
        metallicRoughnessTexture?: TextureInfo;
    };
    normalTexture?: TextureInfo & { scale?: number };
    occlusionTexture?: TextureInfo & { strength?: number };
    emissiveFactor?: number[];
    emissiveTexture?: TextureInfo;
    alphaMode?: "OPAQUE" | "MASK" | "BLEND";
    alphaCutoff?: number;
    extensions?: Extensions;
}

interface Node {
    weights?: number[];
    name?: string;
    mesh?: number;
    skin?: number;
    children?: number[];
    matrix?: number[];
    translation?: number[];
    rotation?: number[];
    scale?: number[];
}

export interface GltfMesh {
    name: string;
    /** Packed (px py pz u)(nx ny nz v), eight floats per vertex. */
    vertices: Float32Array;
    indices: Uint32Array;
    uv1?: Float32Array;
    /** Four skin-local joint indices per vertex, in packed vertex order. */
    joints?: Float32Array;
    /** Four dequantized linear blend weights per vertex, in packed vertex order. */
    weights?: Float32Array;
    color?: Float32Array;
    tangent?: Float32Array;
    /** Mesh-local xyz deltas, in final vertex order; absent semantics are zero. */
    morphTargets?: GltfMorphTarget[];
    material: number;
}

export interface GltfMorphTarget {
    position?: Float32Array;
    normal?: Float32Array;
    tangent?: Float32Array;
}

export interface GltfMaterial {
    color: [number, number, number, number];
    texture?: number;
    baseColorChannel?: 0 | 1;
    baseColorTransform?: TextureTransform;
    normalMapTexture?: number;
    normalMapChannel?: 0 | 1;
    normalMapTransform?: TextureTransform;
    normalMapScale?: number;
    metallicRoughnessTexture?: number;
    metallicRoughnessChannel?: 0 | 1;
    metallicRoughnessTransform?: TextureTransform;
    occlusionTexture?: number;
    occlusionChannel?: 0 | 1;
    occlusionTransform?: TextureTransform;
    occlusionStrength?: number;
    emissiveTexture?: number;
    emissiveChannel?: 0 | 1;
    emissiveTransform?: TextureTransform;
    metallic: number;
    roughness: number;
    emissive: [number, number, number];
    alphaMode: "OPAQUE" | "MASK" | "BLEND";
    cutoff: number;
    unlit: boolean;
    doubleSided: boolean;
}
export interface GltfSkin {
    /** Source asset skin index; primitive skin indices instead address the decoded skins array. */
    index: number;
    /** Indices into the default scene's nodes, in skin joint order. */
    joints: number[];
    /** Column-major inverse bind matrices, sixteen floats per joint; identity when omitted. */
    inverseBindMatrices: Float32Array;
    /** Index into skeletons. Skins sharing or touching joints share a skeleton. */
    skeleton: number;
}

export interface GltfSkeleton {
    /** Scene-node indices, parents before children, including nodes between joints. */
    bones: number[];
}

export interface GltfScene {
    /** Default scene, with parents before children. */
    nodes: GltfNode[];
    animations: GltfAnimation[];
    meshes: GltfMesh[];
    /** Only skins referenced by default-scene nodes, in asset order. */
    skins: GltfSkin[];
    skeletons: GltfSkeleton[];
    instances: {
        mesh: number;
        node: string;
        nodeIndex: number;
        matrix: Float32Array;
        skin: number;
    }[];
    materials: GltfMaterial[];
    images: { uri?: string; bytes?: Uint8Array; mimeType?: string }[];
    textures: { image?: number; sampler: ImageSamplerDescriptor }[];
    unsupported: string[];
}
const COMPONENTS = { SCALAR: 1, VEC2: 2, VEC3: 3, VEC4: 4, MAT2: 4, MAT3: 9, MAT4: 16 } as const;

const BYTE = 5120;
const UNSIGNED_BYTE = 5121;
const SHORT = 5122;
const UNSIGNED_SHORT = 5123;
const UNSIGNED_INT = 5125;
const TRIANGLES = 4;

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
        case UNSIGNED_INT:
            return view.getUint32(o, true);
        default:
            return view.getFloat32(o, true);
    }
}

export function readAccessor(gltf: GltfJson, buffers: ArrayBuffer[], index: number): Float32Array;
export function readAccessor(
    gltf: GltfJson,
    buffers: ArrayBuffer[],
    index: number,
    indices: true,
): Uint32Array;
export function readAccessor(
    gltf: GltfJson,
    buffers: ArrayBuffer[],
    index: number,
    indices = false,
): Float32Array | Uint32Array {
    const acc = gltf.accessors![index];
    const comps = COMPONENTS[acc.type];
    const size = compSize(acc.componentType);
    const out = indices ? new Uint32Array(acc.count * comps) : new Float32Array(acc.count * comps);
    const read = (
        bufferView: number,
        byteOffset: number,
        count: number,
        destinations?: Uint32Array,
    ) => {
        const bv = gltf.bufferViews![bufferView];
        const view = new DataView(buffers[bv.buffer]);
        const base = (bv.byteOffset ?? 0) + byteOffset;
        const stride = bv.byteStride ?? comps * size;
        for (let i = 0; i < count; i++)
            for (let c = 0; c < comps; c++)
                out[(destinations ? destinations[i] : i) * comps + c] = readComponent(
                    view,
                    base + i * stride + c * size,
                    acc.componentType,
                    acc.normalized ?? false,
                );
    };
    if (acc.bufferView !== undefined) read(acc.bufferView, acc.byteOffset ?? 0, acc.count);
    if (acc.sparse) {
        const sparse = acc.sparse;
        const bv = gltf.bufferViews![sparse.indices.bufferView];
        const view = new DataView(buffers[bv.buffer]);
        const base = (bv.byteOffset ?? 0) + (sparse.indices.byteOffset ?? 0);
        const size = compSize(sparse.indices.componentType);
        const destinations = Uint32Array.from({ length: sparse.count }, (_, i) =>
            readComponent(view, base + i * size, sparse.indices.componentType, false),
        );
        read(sparse.values.bufferView, sparse.values.byteOffset ?? 0, sparse.count, destinations);
    }
    return out;
}

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

function expandCorners(values: Float32Array, indices: Uint32Array, width: number): Float32Array {
    const result = new Float32Array(indices.length * width);
    for (let i = 0; i < indices.length; i++)
        result.set(values.subarray(indices[i] * width, (indices[i] + 1) * width), i * width);
    return result;
}

function triangleIndices(gltf: GltfJson, buffers: ArrayBuffer[], prim: Primitive): Uint32Array {
    let indices =
        prim.indices !== undefined
            ? readAccessor(gltf, buffers, prim.indices, true)
            : Uint32Array.from(
                  { length: gltf.accessors![prim.attributes.POSITION].count },
                  (_, i) => i,
              );
    if (prim.mode === 5 || prim.mode === 6) {
        const source = indices;
        indices = new Uint32Array((source.length - 2) * 3);
        for (let i = 0; i < source.length - 2; i++) {
            const triangle =
                prim.mode === 5 ? [i, i + 1 + (i % 2), i + 2 - (i % 2)] : [i + 1, i + 2, 0];
            for (let c = 0; c < 3; c++) indices[i * 3 + c] = source[triangle[c]];
        }
    }
    return indices;
}

function decodeMesh(
    gltf: GltfJson,
    buffers: ArrayBuffer[],
    prim: Primitive,
    name: string,
): GltfMesh {
    let pos = readAccessor(gltf, buffers, prim.attributes.POSITION);
    let uv =
        prim.attributes.TEXCOORD_0 !== undefined
            ? readAccessor(gltf, buffers, prim.attributes.TEXCOORD_0)
            : null;
    let indices = triangleIndices(gltf, buffers, prim);
    const stream = (accessor: number | undefined) =>
        accessor === undefined ? undefined : readAccessor(gltf, buffers, accessor);
    let uv1 = stream(prim.attributes.TEXCOORD_1);
    let joints = stream(prim.attributes.JOINTS_0);
    let weights = stream(prim.attributes.WEIGHTS_0);
    let color = stream(prim.attributes.COLOR_0);
    const morphTargets = prim.targets?.map((target) => ({
        position: stream(target.POSITION),
        normal: stream(target.NORMAL),
        tangent: prim.attributes.NORMAL === undefined ? undefined : stream(target.TANGENT),
    }));
    if (color && gltf.accessors![prim.attributes.COLOR_0!].type === "VEC3") {
        const rgb = color;
        color = Float32Array.from({ length: (rgb.length / 3) * 4 }, (_, i) =>
            i % 4 === 3 ? 1 : rgb[Math.floor(i / 4) * 3 + (i % 4)],
        );
    }
    const tangent =
        prim.attributes.NORMAL === undefined ? undefined : stream(prim.attributes.TANGENT);
    const normalTexture =
        prim.material === undefined ? undefined : gltf.materials?.[prim.material]?.normalTexture;
    if (normalTexture) {
        const semantic = `TEXCOORD_${textureChannel(normalTexture)}`;
        if (!(semantic in prim.attributes))
            throw new Error(`glTF ${name}: normalTexture names missing ${semantic}`);
    }
    if (prim.attributes.NORMAL === undefined) {
        if (uv1) uv1 = expandCorners(uv1, indices, 2);
        if (joints) joints = expandCorners(joints, indices, 4);
        if (weights) weights = expandCorners(weights, indices, 4);
        if (color) color = expandCorners(color, indices, 4);
        for (const target of morphTargets ?? []) {
            if (target.position) target.position = expandCorners(target.position, indices, 3);
            if (target.normal) target.normal = expandCorners(target.normal, indices, 3);
            if (target.tangent) target.tangent = expandCorners(target.tangent, indices, 3);
        }
        pos = expandCorners(pos, indices, 3);
        if (uv) uv = expandCorners(uv, indices, 2);
        indices = Uint32Array.from({ length: indices.length }, (_, i) => i);
    }
    const normal =
        prim.attributes.NORMAL !== undefined
            ? readAccessor(gltf, buffers, prim.attributes.NORMAL)
            : computeNormals(pos, indices);
    if (prim.attributes.NORMAL === undefined)
        for (const target of morphTargets ?? []) {
            const displaced = pos.map((v, i) => v + (target.position?.[i] ?? 0));
            target.normal = computeNormals(displaced, indices).map((v, i) => v - normal[i]);
        }
    return {
        name,
        morphTargets,
        vertices: packVertices(pos, normal, uv),
        indices,
        ...(uv1 ? { uv1 } : {}),
        ...(joints ? { joints } : {}),
        ...(weights ? { weights } : {}),
        ...(color ? { color } : {}),
        ...(tangent ? { tangent } : {}),
        material: prim.material ?? -1,
    };
}

/** Completes normal-textured primitives after synchronous accessor decoding. */
export async function generateSceneTangents(
    scene: GltfScene,
    gltf: GltfJson,
    buffers: ArrayBuffer[],
): Promise<void> {
    let index = 0;
    for (const source of gltf.meshes ?? [])
        for (const prim of source.primitives) {
            if ((prim.mode ?? TRIANGLES) < TRIANGLES) continue;
            const mesh = scene.meshes[index++];
            const texture =
                prim.material === undefined
                    ? undefined
                    : gltf.materials?.[prim.material]?.normalTexture;
            if (mesh.tangent || !texture) continue;
            const set = textureChannel(texture);
            let uv =
                set === 0
                    ? undefined
                    : set === 1
                      ? mesh.uv1
                      : readAccessor(
                            gltf,
                            buffers,
                            (prim.attributes as Record<string, number>)[`TEXCOORD_${set}`],
                        );
            if (uv && set > 1 && prim.attributes.NORMAL === undefined)
                uv = expandCorners(uv, triangleIndices(gltf, buffers, prim), 2);
            const transform = texture.extensions?.KHR_texture_transform;
            if (transform) {
                const coordinates =
                    uv ??
                    Float32Array.from(
                        { length: mesh.vertices.length / 4 },
                        (_, i) => mesh.vertices[Math.floor(i / 2) * 8 + (i % 2 === 0 ? 3 : 7)],
                    );
                const [sx, sy] = transform.scale ?? [1, 1];
                const [ox, oy] = transform.offset ?? [0, 0];
                const c = Math.cos(transform.rotation ?? 0),
                    s = Math.sin(transform.rotation ?? 0);
                uv = Float32Array.from(coordinates, (_, i) => {
                    const v = i - (i % 2),
                        x = coordinates[v] * sx,
                        y = coordinates[v + 1] * sy;
                    return i % 2 === 0 ? c * x - s * y + ox : s * x + c * y + oy;
                });
            }
            if (mesh.morphTargets?.length) {
                scene.meshes[index - 1] = await generateMorphTangents(mesh, uv);
                continue;
            }
            const generated = await generateTangents(uv ? { ...mesh, uv1: uv } : mesh, uv ? 1 : 0);
            if (set > 1 || transform) {
                if (mesh.uv1) {
                    const remapped = new Float32Array(generated.vertices.length / 4);
                    remapped.set(mesh.uv1);
                    for (let corner = 0; corner < generated.indices.length; corner++) {
                        const dest = generated.indices[corner];
                        const src = mesh.indices[corner];
                        remapped.set(mesh.uv1.subarray(src * 2, src * 2 + 2), dest * 2);
                    }
                    generated.uv1 = remapped;
                } else delete generated.uv1;
            }
            for (const semantic of ["joints", "weights"] as const) {
                const source = mesh[semantic];
                if (!source) continue;
                const remapped = new Float32Array((generated.vertices.length / 8) * 4);
                for (let corner = 0; corner < generated.indices.length; corner++) {
                    const src = mesh.indices[corner];
                    remapped.set(
                        source.subarray(src * 4, src * 4 + 4),
                        generated.indices[corner] * 4,
                    );
                }
                (generated as GltfMesh)[semantic] = remapped;
            }
            scene.meshes[index - 1] = { ...mesh, ...generated };
        }
}

async function generateMorphTangents(mesh: GltfMesh, uv?: Float32Array): Promise<GltfMesh> {
    const input = uv ? { ...mesh, uv1: uv } : mesh;
    const runs = [await generateTangents(input, uv ? 1 : 0)];
    for (const target of mesh.morphTargets!) {
        const vertices = mesh.vertices.slice();
        for (let v = 0; v < vertices.length / 8; v++) {
            for (let a = 0; a < 3; a++) {
                vertices[v * 8 + a] += target.position?.[v * 3 + a] ?? 0;
                vertices[v * 8 + 4 + a] += target.normal?.[v * 3 + a] ?? 0;
            }
            const length =
                Math.hypot(vertices[v * 8 + 4], vertices[v * 8 + 5], vertices[v * 8 + 6]) || 1;
            for (let a = 0; a < 3; a++) vertices[v * 8 + 4 + a] /= length;
        }
        runs.push(await generateTangents({ ...input, vertices }, uv ? 1 : 0));
    }
    const bits = runs.map(
        (run) => new Uint32Array(run.tangent.buffer, run.tangent.byteOffset, run.tangent.length),
    );
    const keys = new Map<string, number>();
    const originals: number[] = [],
        corners: number[] = [];
    const indices = mesh.indices.map((original, corner) => {
        const key = [
            original,
            ...runs.flatMap((run, r) => [
                ...bits[r].subarray(run.indices[corner] * 4, run.indices[corner] * 4 + 4),
            ]),
        ].join(":");
        let vertex = keys.get(key);
        if (vertex === undefined) {
            vertex = originals.length;
            keys.set(key, vertex);
            originals.push(original);
            corners.push(corner);
        }
        return vertex;
    });
    const remap = (stream: Float32Array, width: number) =>
        Float32Array.from(
            { length: originals.length * width },
            (_, i) => stream[originals[Math.floor(i / width)] * width + (i % width)],
        );
    const tangent = Float32Array.from(
        { length: originals.length * 4 },
        (_, i) => runs[0].tangent[runs[0].indices[corners[Math.floor(i / 4)]] * 4 + (i % 4)],
    );
    const morphTargets = mesh.morphTargets!.map((target, r) => ({
        position: target.position && remap(target.position, 3),
        normal: target.normal && remap(target.normal, 3),
        tangent: Float32Array.from({ length: originals.length * 3 }, (_, i) => {
            const vertex = Math.floor(i / 3),
                a = i % 3;
            return (
                runs[r + 1].tangent[runs[r + 1].indices[corners[vertex]] * 4 + a] -
                tangent[vertex * 4 + a]
            );
        }),
    }));
    return {
        ...mesh,
        indices,
        tangent,
        morphTargets,
        vertices: remap(mesh.vertices, 8),
        uv1: mesh.uv1 && remap(mesh.uv1, 2),
        color: mesh.color && remap(mesh.color, 4),
        joints: mesh.joints && remap(mesh.joints, 4),
        weights: mesh.weights && remap(mesh.weights, 4),
    };
}

function mergeSkeletons(skins: GltfSkin[], nodes: GltfNode[]): GltfSkeleton[] {
    const parents = skins.map((_, i) => i);
    const root = (index: number): number => {
        while (parents[index] !== index) index = parents[index];
        return index;
    };
    const owners = new Map<number, number>();
    for (const [index, skin] of skins.entries())
        for (const joint of skin.joints) {
            if (joint < 0) throw new Error("glTF skin joint is outside the default scene");
            const owner = owners.get(joint);
            if (owner !== undefined) parents[root(index)] = root(owner);
            owners.set(joint, index);
        }
    for (const [joint, owner] of owners)
        for (
            let ancestor = nodes[joint].parent;
            ancestor !== -1;
            ancestor = nodes[ancestor].parent
        ) {
            const other = owners.get(ancestor);
            if (other !== undefined) parents[root(owner)] = root(other);
        }
    const groups = new Map<number, Set<number>>();
    for (const [index, skin] of skins.entries()) {
        const group = root(index);
        if (!groups.has(group)) groups.set(group, new Set());
        for (const joint of skin.joints) groups.get(group)!.add(joint);
    }
    const skeletons: GltfSkeleton[] = [];
    for (const [group, joints] of groups) {
        const bones = new Set(joints);
        for (const joint of joints) {
            const path: number[] = [];
            for (
                let ancestor = nodes[joint].parent;
                ancestor !== -1;
                ancestor = nodes[ancestor].parent
            ) {
                if (joints.has(ancestor)) {
                    for (const node of path) bones.add(node);
                    break;
                }
                path.push(ancestor);
            }
        }
        for (const [index, skin] of skins.entries())
            if (root(index) === group) skin.skeleton = skeletons.length;
        skeletons.push({ bones: [...bones].sort((a, b) => a - b) });
    }
    return skeletons;
}

function supportedExtension(name: string): boolean {
    return (
        name === "KHR_materials_unlit" ||
        name === "KHR_mesh_quantization" ||
        name === "KHR_texture_transform" ||
        name === "KHR_materials_emissive_strength"
    );
}

/** Refuses versions beyond 2.0 and unsupported required extensions before reading resources. */
export function validateAsset(gltf: GltfJson): void {
    for (const field of ["minVersion", "version"] as const) {
        const version = gltf.asset?.[field];
        if (version === undefined) continue;
        const [major, minor] = version.split(".").map(Number);
        if (major > 2 || (major === 2 && minor > 0))
            throw new Error(`Unsupported glTF asset.${field}: ${version}; supports 2.0`);
    }
    for (const name of gltf.extensionsRequired ?? []) {
        if (!supportedExtension(name))
            throw new Error(`Unsupported required glTF extension: ${name}`);
    }
}

/** Decodes resolved buffers without a device. Required unsupported extensions refuse by name.
 * Instances retain column-major world matrices, including shear; geometry stays mesh-local.
 */
export function parse(gltf: GltfJson, buffers: ArrayBuffer[]): GltfScene {
    validateAsset(gltf);
    const ignored = new Set<string>();
    const scan = (value: unknown): void => {
        if (!value || typeof value !== "object") return;
        for (const [key, child] of Object.entries(value)) {
            if (key === "extensions") {
                for (const name of Object.keys(child))
                    if (!supportedExtension(name)) ignored.add(name);
            } else scan(child);
        }
    };
    scan(gltf);
    for (const name of gltf.extensionsUsed ?? []) if (!supportedExtension(name)) ignored.add(name);
    if ("cameras" in gltf) ignored.add("camera");
    const materials = (gltf.materials ?? []).map((m): GltfMaterial => {
        const pbr = m.pbrMetallicRoughness;
        const channel = (texture?: TextureInfo): 0 | 1 => (textureChannel(texture) === 1 ? 1 : 0);
        const textureIndex = (name: string, texture?: TextureInfo) => {
            if (textureChannel(texture) > 1) {
                ignored.add(`${name}.texCoord`);
                return undefined;
            }
            return texture?.index;
        };
        return {
            color: (pbr?.baseColorFactor ?? [1, 1, 1, 1]) as GltfMaterial["color"],
            texture: textureIndex("baseColorTexture", pbr?.baseColorTexture),
            baseColorChannel: channel(pbr?.baseColorTexture),
            baseColorTransform: pbr?.baseColorTexture?.extensions?.KHR_texture_transform,
            normalMapTexture: textureIndex("normalTexture", m.normalTexture),
            normalMapChannel: channel(m.normalTexture),
            normalMapTransform: m.normalTexture?.extensions?.KHR_texture_transform,
            normalMapScale: m.normalTexture?.scale ?? 1,
            metallicRoughnessTexture: textureIndex(
                "metallicRoughnessTexture",
                pbr?.metallicRoughnessTexture,
            ),
            metallicRoughnessChannel: channel(pbr?.metallicRoughnessTexture),
            metallicRoughnessTransform:
                pbr?.metallicRoughnessTexture?.extensions?.KHR_texture_transform,
            occlusionTexture: textureIndex("occlusionTexture", m.occlusionTexture),
            occlusionChannel: channel(m.occlusionTexture),
            occlusionTransform: m.occlusionTexture?.extensions?.KHR_texture_transform,
            occlusionStrength: m.occlusionTexture?.strength ?? 1,
            emissiveTexture: textureIndex("emissiveTexture", m.emissiveTexture),
            emissiveChannel: channel(m.emissiveTexture),
            emissiveTransform: m.emissiveTexture?.extensions?.KHR_texture_transform,
            metallic: pbr?.metallicFactor ?? 1,
            roughness: pbr?.roughnessFactor ?? 1,
            emissive: (m.emissiveFactor ?? [0, 0, 0]).map(
                (value) =>
                    value *
                    ((
                        m.extensions?.KHR_materials_emissive_strength as
                            | { emissiveStrength?: number }
                            | undefined
                    )?.emissiveStrength ?? 1),
            ) as GltfMaterial["emissive"],
            alphaMode: m.alphaMode ?? "OPAQUE",
            cutoff: m.alphaCutoff ?? 0.5,
            unlit: m.extensions?.KHR_materials_unlit !== undefined,
            doubleSided: m.doubleSided ?? false,
        };
    });
    const addressMode = (wrap = 10497) =>
        wrap === 33071
            ? ImageAddressMode.ClampToEdge
            : wrap === 33648
              ? ImageAddressMode.MirrorRepeat
              : ImageAddressMode.Repeat;
    const textures = (gltf.textures ?? []).map((texture) => {
        const sampler =
            texture.sampler === undefined ? undefined : gltf.samplers?.[texture.sampler];
        const min = sampler?.minFilter ?? 9987;
        return {
            image: texture.source,
            sampler: {
                addressModeU: addressMode(sampler?.wrapS),
                addressModeV: addressMode(sampler?.wrapT),
                magFilter:
                    sampler?.magFilter === 9728 ? ImageFilterMode.Nearest : ImageFilterMode.Linear,
                minFilter:
                    min === 9728 || min === 9984 || min === 9986
                        ? ImageFilterMode.Nearest
                        : ImageFilterMode.Linear,
                mipmapFilter:
                    min === 9986 || min === 9987 ? ImageFilterMode.Linear : ImageFilterMode.Nearest,
            },
        };
    });
    const meshes: GltfMesh[] = [];
    const meshEntries = (gltf.meshes ?? []).map((mesh, meshIndex) => {
        const entries: number[] = [];
        for (const [primitiveIndex, prim] of mesh.primitives.entries()) {
            for (const semantic of Object.keys(prim.attributes)) {
                if (
                    ![
                        "POSITION",
                        "NORMAL",
                        "TEXCOORD_0",
                        "TEXCOORD_1",
                        "COLOR_0",
                        "TANGENT",
                        "JOINTS_0",
                        "WEIGHTS_0",
                    ].includes(semantic)
                )
                    ignored.add(semantic);
            }
            for (const target of prim.targets ?? [])
                for (const semantic of Object.keys(target))
                    if (semantic.startsWith("TEXCOORD_") || semantic.startsWith("COLOR_"))
                        ignored.add(`morph.${semantic}`);
            if ((prim.mode ?? TRIANGLES) < TRIANGLES) {
                ignored.add(["POINTS", "LINES", "LINE_LOOP", "LINE_STRIP"][prim.mode!]);
                continue;
            }
            entries.push(meshes.length);
            meshes.push(
                decodeMesh(
                    gltf,
                    buffers,
                    prim,
                    `${mesh.name ?? `Mesh${meshIndex}`}/Primitive${primitiveIndex}`,
                ),
            );
        }
        return entries;
    });
    const animations = (gltf.animations ?? []).map(
        (animation, index): GltfAnimation => ({
            name: animation.name ?? `Animation${index}`,
            channels: animation.channels
                .filter((channel) => channel.target.node !== undefined)
                .map((channel) => ({
                    sampler: channel.sampler,
                    node: channel.target.node!,
                    path: channel.target.path,
                })),
            samplers: animation.samplers.map((sampler) => ({
                input: readAccessor(gltf, buffers, sampler.input),
                output: readAccessor(gltf, buffers, sampler.output),
                interpolation: sampler.interpolation ?? "LINEAR",
            })),
        }),
    );
    const sceneNodes: GltfNode[] = [];
    const instances: GltfScene["instances"] = [];
    const nodes = gltf.nodes ?? [];
    const children = new Set(nodes.flatMap((n) => n.children ?? []));
    const roots =
        gltf.scenes?.[gltf.scene ?? 0]?.nodes ??
        nodes.map((_, i) => i).filter((i) => !children.has(i));
    const walk = (
        index: number,
        parent: Float32Array,
        ancestors: Set<number>,
        parentIndex = -1,
    ): void => {
        if (ancestors.has(index)) throw new Error("Cyclic glTF node hierarchy");
        const node = nodes[index];
        const local = localMatrix(node, new Float32Array(16));
        const trs = node.matrix ? decomposeMat4(local) : undefined;
        const sceneIndex = sceneNodes.length;
        sceneNodes.push({
            index,
            parent: parentIndex,
            name: node.name ?? `Node${index}`,
            translation: trs
                ? Array.from(trs.subarray(0, 3))
                : [...(node.translation ?? [0, 0, 0])],
            rotation: trs ? Array.from(trs.subarray(3, 7)) : [...(node.rotation ?? [0, 0, 0, 1])],
            scale: trs ? Array.from(trs.subarray(7, 10)) : [...(node.scale ?? [1, 1, 1])],
            weights: [
                ...(node.weights ??
                    (node.mesh === undefined ? undefined : gltf.meshes?.[node.mesh].weights) ??
                    Array(
                        gltf.meshes?.[node.mesh ?? -1]?.primitives.find((p) => p.targets?.length)
                            ?.targets?.length ?? 0,
                    ).fill(0)),
            ],
        });
        const matrix = multiply(parent, local);
        if (node.mesh !== undefined)
            for (const mesh of meshEntries[node.mesh])
                instances.push({
                    mesh,
                    node: node.name ?? `Node${index}`,
                    nodeIndex: index,
                    matrix,
                    skin: meshes[mesh].joints && meshes[mesh].weights ? (node.skin ?? -1) : -1,
                });
        for (const child of node.children ?? [])
            walk(child, matrix, new Set([...ancestors, index]), sceneIndex);
    };
    const identity = compose(0, 0, 0, 0, 0, 0, 1, 1, 1, 1);
    for (const root of roots) walk(root, identity, new Set());
    const referencedSkins = new Set(sceneNodes.map((node) => nodes[node.index].skin));
    const skins: GltfSkin[] = (gltf.skins ?? []).flatMap((skin, index) =>
        referencedSkins.has(index)
            ? [
                  {
                      index,
                      joints: skin.joints.map((joint) =>
                          sceneNodes.findIndex((node) => node.index === joint),
                      ),
                      inverseBindMatrices:
                          skin.inverseBindMatrices === undefined
                              ? Float32Array.from({ length: skin.joints.length * 16 }, (_, i) =>
                                    (i % 16) % 5 === 0 ? 1 : 0,
                                )
                              : readAccessor(gltf, buffers, skin.inverseBindMatrices).slice(
                                    0,
                                    skin.joints.length * 16,
                                ),
                      skeleton: -1,
                  },
              ]
            : [],
    );
    for (const instance of instances)
        if (instance.skin >= 0)
            instance.skin = skins.findIndex((skin) => skin.index === instance.skin);
    const skeletons = mergeSkeletons(skins, sceneNodes);
    return {
        nodes: sceneNodes,
        skins,
        skeletons,
        animations,
        meshes,
        instances,
        materials,
        textures,
        images: (gltf.images ?? []).map((image) => {
            if (image.bufferView === undefined) return { uri: image.uri, mimeType: image.mimeType };
            const view = gltf.bufferViews![image.bufferView];
            return {
                mimeType: image.mimeType,
                bytes: new Uint8Array(buffers[view.buffer], view.byteOffset ?? 0, view.byteLength),
            };
        }),
        unsupported: [...ignored].sort(),
    };
}
