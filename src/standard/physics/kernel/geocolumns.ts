import type { SurfaceMaterial } from "../common/types";
import type { CompoundData } from "../shapes/compound";
import type { HeightFieldData } from "../shapes/heightfield";
import type { HullData } from "../shapes/hull";
import { hash64NonZero, hullByteCount, hullImage, writeHullImage } from "../shapes/hullbytes";
import type { MeshData } from "../shapes/mesh";
import type { WorldState } from "../world/world";
import { kernel } from "./kernel";

/** World-owned upload scratch for authored bytes; retained geometry lives in kernel allocations. */
export class GeometryUploadScratch {
    u: Uint32Array = new Uint32Array(0);
    f: Float32Array = new Float32Array(0);
    bytes: Uint8Array = new Uint8Array(0);
    hashes: BigUint64Array = new BigUint64Array(0);
    views(buffer: ArrayBufferLike): void {
        if (this.u.buffer !== buffer) {
            this.u = new Uint32Array(buffer);
            this.f = new Float32Array(buffer);
            this.bytes = new Uint8Array(buffer);
            this.hashes = new BigUint64Array(buffer);
        }
    }
}
export function stageHullUpload(world: WorldState, hull: HullData): number {
    const k = kernel(world.ecsState);
    const bytes = hullByteCount(hull);
    const ptr = k.hullUploadBuffer(world.worldId, bytes);
    const s = (world.geometryUploadScratch ??= new GeometryUploadScratch());
    s.views(k.memory.buffer);
    writeHullImage(hull, s.bytes, s.u, s.f, s.hashes, ptr);
    return bytes;
}

const identities = new WeakMap<object, number>();
let nextIdentity = 1;
export function geometryIdentity(value: object): number {
    let id = identities.get(value);
    if (id === undefined) {
        id = nextIdentity++;
        identities.set(value, id);
    }
    return id;
}
export function acquireGeometryData<T extends object>(
    world: WorldState,
    kind: number,
    data: T,
    serialize: (data: T) => Uint8Array,
): number {
    const k = kernel(world.ecsState);
    const identity = geometryIdentity(data);
    let pointer = k.geometryDatabaseLookup(world.worldId, kind, identity) >>> 0;
    if (pointer !== 0) {
        pointer = k.geometryDatabaseAdd(world.worldId, kind, identity, 0, 1) >>> 0;
    } else {
        const image = serialize(data);
        const input = k.geometryUploadBuffer(world.worldId, image.byteLength);
        const scratch = (world.geometryUploadScratch ??= new GeometryUploadScratch());
        scratch.views(k.memory.buffer);
        scratch.bytes.set(image, input);
        pointer = k.geometryDatabaseAdd(world.worldId, kind, identity, image.byteLength, 1) >>> 0;
    }
    world.geometryIdentityValues.set(identity, data);
    return pointer;
}
export function acquireMeshData(world: WorldState, data: MeshData): number {
    return acquireGeometryData(world, 4, data, meshImage);
}
export function acquireHeightFieldData(world: WorldState, data: HeightFieldData): number {
    return acquireGeometryData(world, 2, data, heightImage);
}
export function acquireCompoundData(world: WorldState, data: CompoundData): number {
    return acquireGeometryData(world, 1, data, compoundImage);
}
export function hullDatabaseIndex(world: WorldState, hull: HullData): number {
    const bytes = stageHullUpload(world, hull);
    const index = kernel(world.ecsState).hullDatabaseLookup(world.worldId, bytes);
    if (index === -1) throw new Error("hull is not retained by this world");
    return index >>> 0;
}
function align8(n: number): number {
    return (n + 7) & ~7;
}
function meshImage(m: MeshData): Uint8Array {
    const nodeOffset = 96;
    const vertexOffset = align8(nodeOffset + m.nodes.length * 32);
    const triangleOffset = align8(vertexOffset + m.vertices.length * 12);
    const materialOffset = align8(triangleOffset + m.triangles.length * 12);
    const flagsOffset = align8(materialOffset + m.triangles.length);
    const byteCount = align8(flagsOffset + m.triangles.length);
    const bytes = new Uint8Array(byteCount),
        v = new DataView(bytes.buffer);
    const u = (o: number, x: number) => v.setUint32(o, x, true);
    const f = (o: number, x: number) => v.setFloat32(o, x, true);
    v.setBigUint64(0, 0xaaab9a00f1a8aaf7n, true);
    u(16, byteCount);
    for (let j = 0; j < 2; ++j) {
        const p = j === 0 ? m.bounds.lowerBound : m.bounds.upperBound;
        f(20 + j * 12, p.x);
        f(24 + j * 12, p.y);
        f(28 + j * 12, p.z);
    }
    f(44, m.surfaceArea);
    u(48, m.treeHeight);
    u(52, m.degenerateCount);
    u(56, nodeOffset);
    u(60, m.nodes.length);
    u(64, vertexOffset);
    u(68, m.vertices.length);
    u(72, triangleOffset);
    u(76, m.triangles.length);
    u(80, materialOffset);
    u(84, m.materialCount);
    u(88, flagsOffset);
    for (let i = 0; i < m.nodes.length; ++i) {
        const n = m.nodes[i],
            o = nodeOffset + i * 32;
        f(o, n.lowerBound.x);
        f(o + 4, n.lowerBound.y);
        f(o + 8, n.lowerBound.z);
        u(
            o + 12,
            n.leaf ? ((n.triangleCount << 2) | 3) >>> 0 : ((n.childOffset << 2) | n.axis) >>> 0,
        );
        f(o + 16, n.upperBound.x);
        f(o + 20, n.upperBound.y);
        f(o + 24, n.upperBound.z);
        u(o + 28, n.triangleOffset);
    }
    m.vertices.forEach((p, i) => {
        f(vertexOffset + i * 12, p.x);
        f(vertexOffset + i * 12 + 4, p.y);
        f(vertexOffset + i * 12 + 8, p.z);
    });
    m.triangles.forEach((t, i) => {
        u(triangleOffset + i * 12, t.index1);
        u(triangleOffset + i * 12 + 4, t.index2);
        u(triangleOffset + i * 12 + 8, t.index3);
    });
    for (let i = 0; i < m.materialIndices.length; ++i)
        v.setUint8(materialOffset + i, m.materialIndices[i]);
    for (let i = 0; i < m.flags.length; ++i) v.setUint8(flagsOffset + i, m.flags[i]);
    v.setBigUint64(8, hash64NonZero(bytes), true);
    return bytes;
}
function heightImage(h: HeightFieldData): Uint8Array {
    const heightsOffset = 96,
        materialOffset = align8(heightsOffset + h.compressedHeights.length * 2);
    const flagsOffset = align8(materialOffset + h.materialIndices.length),
        byteCount = align8(flagsOffset + h.flags.length);
    const bytes = new Uint8Array(byteCount),
        v = new DataView(bytes.buffer);
    const u = (o: number, x: number) => v.setUint32(o, x, true),
        f = (o: number, x: number) => v.setFloat32(o, x, true);
    v.setBigUint64(0, 0x8e41e5fb084848f8n, true);
    u(16, byteCount);
    for (let j = 0; j < 2; ++j) {
        const p = j === 0 ? h.aabb.lowerBound : h.aabb.upperBound;
        f(20 + j * 12, p.x);
        f(24 + j * 12, p.y);
        f(28 + j * 12, p.z);
    }
    f(44, h.minHeight);
    f(48, h.maxHeight);
    f(52, h.heightScale);
    f(56, h.scale.x);
    f(60, h.scale.y);
    f(64, h.scale.z);
    u(68, h.columnCount);
    u(72, h.rowCount);
    u(76, heightsOffset);
    u(80, materialOffset);
    u(84, flagsOffset);
    v.setUint8(88, Number(h.clockwise));
    for (let i = 0; i < h.compressedHeights.length; ++i)
        v.setUint16(heightsOffset + i * 2, h.compressedHeights[i], true);
    for (let i = 0; i < h.materialIndices.length; ++i)
        v.setUint8(materialOffset + i, h.materialIndices[i]);
    for (let i = 0; i < h.flags.length; ++i) v.setUint8(flagsOffset + i, h.flags[i]);
    v.setBigUint64(8, hash64NonZero(bytes), true);
    return bytes;
}
function treeImage(words: Uint32Array): Uint8Array {
    const bytes = new Uint8Array(words.byteLength),
        v = new DataView(bytes.buffer);
    for (let i = 0; i < words.length; i++) v.setUint32(i * 4, words[i], true);
    return bytes;
}
function materialImage(material: SurfaceMaterial): Uint8Array {
    const b = new Uint8Array(40),
        v = new DataView(b.buffer);
    const values = [
        material.friction,
        material.restitution,
        material.rollingResistance,
        material.tangentVelocity[0],
        material.tangentVelocity[1],
        material.tangentVelocity[2],
    ];
    for (let i = 0; i < values.length; ++i) v.setFloat32(i * 4, values[i], true);
    v.setBigUint64(24, material.userMaterialId, true);
    v.setUint32(32, material.customColor, true);
    return b;
}
function compoundImage(c: CompoundData): Uint8Array {
    const chunks: Uint8Array[] = [new Uint8Array(96)];
    let length = 96;
    const push = (bytes: Uint8Array): number => {
        const offset = align8(length);
        if (offset > length) chunks.push(new Uint8Array(offset - length));
        chunks.push(bytes);
        length = offset + bytes.length;
        return offset;
    };
    const treeOffset = push(treeImage(c.tree.compoundNodes!));
    const materialsOffset = push(concat(c.materials.map(materialImage)));
    const capsulesOffset = push(
        concat(
            c.capsules.map((x) => {
                const b = new Uint8Array(32),
                    v = new DataView(b.buffer);
                const values = [
                    x.capsule.center1.x,
                    x.capsule.center1.y,
                    x.capsule.center1.z,
                    x.capsule.center2.x,
                    x.capsule.center2.y,
                    x.capsule.center2.z,
                    x.capsule.radius,
                ];
                for (let i = 0; i < values.length; ++i) v.setFloat32(i * 4, values[i], true);
                v.setUint16(28, x.materialIndex, true);
                return b;
            }),
        ),
    );
    const hullOffset = push(new Uint8Array(c.hulls.length * 36));
    const hullView = new DataView(
        chunks.at(-1)!.buffer,
        chunks.at(-1)!.byteOffset,
        chunks.at(-1)!.byteLength,
    );
    const sharedHulls = sharedImages(hullImage, push);
    c.hulls.forEach((x, i) => {
        const o = i * 36;
        putTransform(hullView, o, x.transform);
        hullView.setUint16(o + 32, x.materialIndex, true);
        hullView.setUint32(o + 28, sharedHulls.offset(x.hull), true);
    });
    const meshOffset = push(new Uint8Array(c.meshes.length * 52));
    const meshView = new DataView(
        chunks.at(-1)!.buffer,
        chunks.at(-1)!.byteOffset,
        chunks.at(-1)!.byteLength,
    );
    const meshMaterials = new Uint8Array(
        c.meshes.reduce((n, x) => n + x.materialIndices.length, 0) * 2,
    );
    const meshMaterialView = new DataView(meshMaterials.buffer);
    const meshMaterialOffset = push(meshMaterials);
    let materialCursor = 0;
    const sharedMeshes = sharedImages(meshImage, push);
    c.meshes.forEach((x, i) => {
        const o = i * 52;
        putTransform(meshView, o, x.transform);
        meshView.setFloat32(o + 28, x.scale.x, true);
        meshView.setFloat32(o + 32, x.scale.y, true);
        meshView.setFloat32(o + 36, x.scale.z, true);
        meshView.setUint32(o + 40, sharedMeshes.offset(x.meshData), true);
        meshView.setUint32(o + 44, meshMaterialOffset + materialCursor * 2, true);
        meshView.setUint32(o + 48, x.materialIndices.length, true);
        for (const index of x.materialIndices)
            meshMaterialView.setUint16(materialCursor++ * 2, index, true);
    });
    const spheresOffset = push(
        concat(
            c.spheres.map((x) => {
                const b = new Uint8Array(20),
                    v = new DataView(b.buffer);
                v.setFloat32(0, x.sphere.center.x, true);
                v.setFloat32(4, x.sphere.center.y, true);
                v.setFloat32(8, x.sphere.center.z, true);
                v.setFloat32(12, x.sphere.radius, true);
                v.setUint16(16, x.materialIndex, true);
                return b;
            }),
        ),
    );
    const raw = concat(chunks);
    const bytes = new Uint8Array(align8(raw.length));
    bytes.set(raw);
    const v = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
    const u = (o: number, x: number) => v.setUint32(o, x, true);
    v.setBigUint64(0, 0x9e4b17d3a25c68f1n ^ 0xaaab9a00f1a8aaf7n ^ 0x6b1e39d4a87c25f3n, true);
    u(8, bytes.length);
    const nodes = c.tree.compoundNodes!;
    bytes.set(new Uint8Array(nodes.buffer, nodes.byteOffset, 24), 12);
    u(36, nodes[6] & 0x80000000 ? 0 : nodes[7]);
    u(40, treeOffset);
    u(44, c.tree.nodeCount);
    u(48, materialsOffset);
    u(52, c.materialCount);
    u(56, capsulesOffset);
    u(60, c.capsules.length);
    u(64, hullOffset);
    u(68, c.hulls.length);
    u(72, sharedHulls.distinct.length);
    u(76, meshOffset);
    u(80, c.meshes.length);
    u(84, sharedMeshes.distinct.length);
    u(88, spheresOffset);
    u(92, c.spheres.length);
    return bytes;
}
function concat(chunks: Uint8Array[]): Uint8Array {
    const out = new Uint8Array(chunks.reduce((n, b) => n + b.length, 0));
    let offset = 0;
    for (const b of chunks) {
        out.set(b, offset);
        offset += b.length;
    }
    return out;
}
// Each distinct object serializes once, and objects with equal bytes share one copy, as
// b3CreateCompound's hull and mesh maps share them.
function sharedImages<T extends object>(
    image: (value: T) => Uint8Array,
    push: (bytes: Uint8Array) => number,
) {
    const offsets = new Map<T, number>();
    const distinct: { image: Uint8Array; offset: number }[] = [];
    const offset = (value: T): number => {
        let at = offsets.get(value);
        if (at === undefined) {
            const bytes = image(value);
            at = distinct.find((y) => equalBytes(y.image, bytes))?.offset;
            if (at === undefined) {
                at = push(bytes);
                distinct.push({ image: bytes, offset: at });
            }
            offsets.set(value, at);
        }
        return at;
    };
    return { offset, distinct };
}
function equalBytes(a: Uint8Array, b: Uint8Array): boolean {
    return a.length === b.length && a.every((x, i) => x === b[i]);
}
function putTransform(v: DataView, o: number, t: import("../common/records").Transform): void {
    const values = [t.p.x, t.p.y, t.p.z, t.q.v.x, t.q.v.y, t.q.v.z, t.q.s];
    for (let i = 0; i < values.length; ++i) v.setFloat32(o + i * 4, values[i], true);
}
