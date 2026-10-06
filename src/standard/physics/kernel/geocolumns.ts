import type { World } from "../../../engine";
import type { Vec3 } from "../common/math";
import { ShapeType } from "../common/types";
import type { CompoundData } from "../shapes/compound";
import type { HeightFieldData } from "../shapes/heightfield";
import type { HullData } from "../shapes/hull";
import { hullByteCount, writeHullImage } from "../shapes/hullbytes";
import type { MeshData } from "../shapes/mesh";
import type { GeometryRecord, WorldState } from "../world/world";
import { kernel } from "./kernel";

const align8 = (words: number): number => (words + 1) & ~1;
const EMPTY: readonly number[] = [];
const EMPTY_HULLS: readonly UploadHull[] = [];
export type UploadHull = Pick<
    HullData,
    | "center"
    | "aabb"
    | "surfaceArea"
    | "volume"
    | "innerRadius"
    | "centralInertia"
    | "hash"
    | "vertexCount"
    | "edgeCount"
    | "faceCount"
    | "points"
    | "vertices"
    | "edges"
    | "faces"
    | "planes"
>;

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
    views(buffer: ArrayBufferLike): void {
        if (this.u.buffer !== buffer) {
            this.u = new Uint32Array(buffer);
            this.f = new Float32Array(buffer);
            this.bytes = new Uint8Array(buffer);
            this.hashes = new BigUint64Array(buffer);
        }
    }
}
function hullSize(h: UploadHull): number {
    return hullByteCount(h) / 4;
}

/** Upload Box3D b3HullData headers and their eight-byte-aligned relative-offset trailing arrays.
 * The directory addresses each shared hull by upload id. */
export function uploadGeometry(
    world: World | undefined,
    hulls: readonly (UploadHull | undefined)[],
    extra: ArrayLike<number> = EMPTY,
    scratch = new GeometryUploadScratch(),
    hullCount = hulls.length,
    extraCount = extra.length,
): void {
    let total = align8(hullCount);
    for (let i = 0; i < hullCount; ++i) total += hullSize(hulls[i]!);
    const k = kernel(world);
    k.reserveGeometry(total, extraCount);
    scratch.views(k.memory.buffer);
    const u = scratch.u,
        f = scratch.f;
    const layout = k.geoLayoutPtr() >>> 2;
    const pool = u[layout] >>> 2,
        extraBase = u[layout + 6] >>> 2;
    for (let i = 0; i < extraCount; ++i) u[extraBase + i] = extra[i];
    u.fill(0, pool, pool + total);
    let base = align8(hullCount);
    for (let i = 0; i < hullCount; ++i) {
        const h = hulls[i]!;
        u[pool + i] = base * 4;
        writeHullImage(h, scratch.bytes, u, f, scratch.hashes, (pool + base) * 4);
        base += hullSize(h);
    }
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
    uploadGeometry(world.ecsState, EMPTY_HULLS, s.words, s, 0, s.count);
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
    s.put(m.nodes.length);
    s.put(m.vertices.length);
    s.put(m.triangles.length);
    for (let i = 0; i < 5; ++i) s.put(0);
    s.words[record + 3] = s.count;
    for (const n of m.nodes) {
        s.vec(n.lowerBound);
        s.vec(n.upperBound);
        s.put(Number(n.leaf));
        s.put(n.axis);
        s.put(n.childOffset);
        s.put(n.triangleCount);
        s.put(n.triangleOffset);
    }
    s.words[record + 4] = s.count;
    for (const p of m.vertices) s.vec(p);
    s.words[record + 5] = s.count;
    for (const t of m.triangles) {
        s.put(t.index1);
        s.put(t.index2);
        s.put(t.index3);
    }
    s.words[record + 6] = s.count;
    s.append(m.flags);
    s.words[record + 7] = s.count;
    s.append(m.materialIndices);
}
function stageHeight(this: WorldState, entry: GeometryRecord, h: HeightFieldData): void {
    const s = this.geometryUploadScratch!;
    const record = s.count;
    entry.geoIndex = record;
    s.vec(h.aabb.lowerBound);
    s.vec(h.aabb.upperBound);
    s.put(s.bits(h.minHeight));
    s.put(s.bits(h.maxHeight));
    s.put(s.bits(h.heightScale));
    s.vec(h.scale);
    s.put(h.columnCount);
    s.put(h.rowCount);
    s.put(Number(h.clockwise));
    s.put(0);
    s.put(0);
    s.put(0);
    s.words[record + 15] = s.count;
    s.append(h.compressedHeights);
    s.words[record + 16] = s.count;
    s.append(h.materialIndices);
    s.words[record + 17] = s.count;
    s.append(h.flags);
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
