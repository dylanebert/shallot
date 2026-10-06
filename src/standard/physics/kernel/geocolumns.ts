import type { World } from "../../../engine";
import type { Vec3 } from "../common/math";
import { ShapeType } from "../common/types";
import type { CompoundData } from "../shapes/compound";
import type { HeightFieldData } from "../shapes/heightfield";
import type { HullData } from "../shapes/hull";
import { hash64NonZero, hullByteCount, writeHullImage } from "../shapes/hullbytes";
import type { MeshData } from "../shapes/mesh";
import type { GeometryRecord, WorldState } from "../world/world";
import { kernel } from "./kernel";

/** World-owned authoring upload registers, not geometry storage. Contents clear after upload;
 * capacity and linear-memory views survive a database miss, as the kernel pools do. */
export class GeometryUploadScratch {
    words = new Uint32Array(16);
    count = 0;
    u: Uint32Array = new Uint32Array(0);
    f: Float32Array = new Float32Array(0);
    bytes: Uint8Array = new Uint8Array(0);
    hashes: BigUint64Array = new BigUint64Array(0);
    private readonly _float = new Float32Array(1);
    private readonly _uint = new Uint32Array(this._float.buffer);
    bits(x: number): number {
        this._float[0] = x;
        return this._uint[0];
    }
    put(x: number): void {
        if (this.count === this.words.length) {
            const next = new Uint32Array(this.words.length * 2);
            next.set(this.words);
            this.words = next;
        }
        this.words[this.count++] = x;
    }
    vec(p: Vec3): void {
        this.put(this.bits(p.x));
        this.put(this.bits(p.y));
        this.put(this.bits(p.z));
    }
    append(values: ArrayLike<number>): void {
        for (let i = 0; i < values.length; ++i) this.put(values[i]);
    }
    align(): void {
        if (this.count & 1) this.put(0);
    }
    packed(values: ArrayLike<number>, bits: 8 | 16): void {
        const lanes = 32 / bits;
        for (let i = 0; i < values.length; i += lanes) {
            let word = 0;
            for (let lane = 0; lane < lanes && i + lane < values.length; ++lane)
                word |= values[i + lane] << (lane * bits);
            this.put(word);
        }
        this.align();
    }
    finishData(record: number): void {
        this.align();
        const bytes = (this.count - record) * 4;
        this.words[record + 4] = bytes;
        const hash = hash64NonZero(new Uint8Array(this.words.buffer, record * 4, bytes));
        this.words[record + 2] = Number(hash & 0xffffffffn);
        this.words[record + 3] = Number(hash >> 32n);
    }
    views(buffer: ArrayBufferLike): void {
        if (this.u.buffer !== buffer) {
            this.u = new Uint32Array(buffer);
            this.f = new Float32Array(buffer);
            this.bytes = new Uint8Array(buffer);
            this.hashes = new BigUint64Array(buffer);
        }
    }
}
function uploadGeometry(world: World | undefined, scratch: GeometryUploadScratch): void {
    const k = kernel(world);
    k.reserveGeometry(scratch.count);
    scratch.views(k.memory.buffer);
    const u = scratch.u;
    const extraBase = u[(k.geoLayoutPtr() >>> 2) + 6] >>> 2;
    for (let i = 0; i < scratch.count; ++i) u[extraBase + i] = scratch.words[i];
}

/** Upload this world's changed authoring set. Nonconvex records retain their query layout until C2b;
 * their pool-relative references survive linear-memory growth. */
export function rebuildGeometry(world: WorldState): void {
    const s = (world.geometryUploadScratch ??= new GeometryUploadScratch());
    s.count = 0;
    world.meshDatabase.forEach(stageMesh, world);
    world.heightFieldDatabase.forEach(stageHeight, world);
    world.compoundDatabase.forEach(stageCompound, world);
    const k = kernel(world.ecsState);
    k.shapeSetActiveWorld(world.worldId);
    uploadGeometry(world.ecsState, s);
    s.words.fill(0, 0, s.count);
    s.count = 0;
    world.geometryUploadCount += 1;
    world.shapeStore.refreshViews();
    world.bodyStore.refreshViews();
    world.manifoldStore.refreshViews();
    for (let id = 0; id < world.shapeGeometry.length; ++id) {
        if (k.shapeAlive(world.worldId, id)) world.shapeStore.writeGeometryReference(world, id);
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

export function hullDatabaseIndex(world: WorldState, hull: HullData): number {
    const bytes = stageHullUpload(world, hull);
    const index = kernel(world.ecsState).hullDatabaseLookup(world.worldId, bytes);
    if (index === -1) throw new Error("hull is not retained by this world");
    return index >>> 0;
}
function stageMesh(this: WorldState, entry: GeometryRecord, m: MeshData): void {
    const s = this.geometryUploadScratch!;
    const record = s.count;
    entry.geoIndex = record;
    s.put(0xf1a8aaf7);
    s.put(0xaaab9a00);
    s.put(0);
    s.put(0);
    s.put(0);
    s.vec(m.bounds.lowerBound);
    s.vec(m.bounds.upperBound);
    s.put(s.bits(m.surfaceArea));
    s.put(m.treeHeight);
    s.put(m.degenerateCount);
    s.put(0);
    s.put(m.nodes.length);
    s.put(0);
    s.put(m.vertices.length);
    s.put(0);
    s.put(m.triangles.length);
    s.put(0);
    s.put(m.materialCount);
    s.put(0);
    s.put(0);
    s.words[record + 14] = (s.count - record) * 4;
    for (const n of m.nodes) {
        s.vec(n.lowerBound);
        s.put(n.leaf ? (n.triangleCount << 2) | 3 : (n.childOffset << 2) | n.axis);
        s.vec(n.upperBound);
        s.put(n.triangleOffset);
    }
    s.words[record + 16] = (s.count - record) * 4;
    for (const p of m.vertices) s.vec(p);
    s.align();
    s.words[record + 18] = (s.count - record) * 4;
    for (const t of m.triangles) {
        s.put(t.index1);
        s.put(t.index2);
        s.put(t.index3);
    }
    s.align();
    s.words[record + 20] = (s.count - record) * 4;
    s.packed(m.materialIndices, 8);
    s.words[record + 22] = (s.count - record) * 4;
    s.packed(m.flags, 8);
    s.finishData(record);
}
function stageHeight(this: WorldState, entry: GeometryRecord, h: HeightFieldData): void {
    const s = this.geometryUploadScratch!;
    const record = s.count;
    entry.geoIndex = record;
    s.put(0x084848f8);
    s.put(0x8e41e5fb);
    s.put(0);
    s.put(0);
    s.put(0);
    s.vec(h.aabb.lowerBound);
    s.vec(h.aabb.upperBound);
    s.put(s.bits(h.minHeight));
    s.put(s.bits(h.maxHeight));
    s.put(s.bits(h.heightScale));
    s.vec(h.scale);
    s.put(h.columnCount);
    s.put(h.rowCount);
    s.put(0);
    s.put(0);
    s.put(0);
    s.put(Number(h.clockwise));
    s.put(0);
    s.words[record + 19] = (s.count - record) * 4;
    s.packed(h.compressedHeights, 16);
    s.words[record + 20] = (s.count - record) * 4;
    s.packed(h.materialIndices, 8);
    s.words[record + 21] = (s.count - record) * 4;
    s.packed(h.flags, 8);
    s.finishData(record);
}
function stageCompound(this: WorldState, entry: GeometryRecord, c: CompoundData): void {
    const s = this.geometryUploadScratch!;
    const record = s.count;
    entry.geoIndex = record;
    const count = c.capsules.length + c.hulls.length + c.meshes.length + c.spheres.length;
    s.put(c.tree.root);
    s.put(c.tree.nodeCapacity);
    s.put(count);
    s.put(0);
    s.put(0);
    s.words[record + 3] = s.count;
    s.append(c.tree.ni);
    s.words[record + 4] = s.count;
    for (const child of c.capsules) {
        s.put(ShapeType.Capsule);
        identity(s);
        materials(s, child.materialIndex);
        s.vec(child.capsule.center1);
        s.vec(child.capsule.center2);
        s.put(s.bits(child.capsule.radius));
    }
    for (const child of c.hulls) {
        s.put(ShapeType.Hull);
        s.vec(child.transform.p);
        s.vec(child.transform.q.v);
        s.put(s.bits(child.transform.q.s));
        materials(s, child.materialIndex);
        s.put(hullDatabaseIndex(this, child.hull));
        s.put(s.bits(child.hull.innerRadius));
        for (let i = 0; i < 5; ++i) s.put(0);
    }
    for (const child of c.meshes) {
        s.put(ShapeType.Mesh);
        s.vec(child.transform.p);
        s.vec(child.transform.q.v);
        s.put(s.bits(child.transform.q.s));
        s.append(child.materialIndices);
        s.put(this.meshDatabase.get(child.meshData)!.geoIndex);
        s.vec(child.scale);
        s.put(0);
        s.put(0);
        s.put(0);
    }
    for (const child of c.spheres) {
        s.put(ShapeType.Sphere);
        identity(s);
        materials(s, child.materialIndex);
        s.vec(child.sphere.center);
        s.put(s.bits(child.sphere.radius));
        s.put(0);
        s.put(0);
        s.put(0);
    }
}
function identity(s: GeometryUploadScratch): void {
    for (let i = 0; i < 6; ++i) s.put(0);
    s.put(s.bits(1));
}
function materials(s: GeometryUploadScratch, material: number): void {
    s.put(material);
    s.put(0);
    s.put(0);
    s.put(0);
}
