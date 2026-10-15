import type { AABB, Transform, Vec3 } from "../common/records";
import { type Kernel, kernel } from "../kernel/kernel";
import type { Capsule, Sphere } from "./geometry";
import type { HeightFieldData } from "./heightfield";
import type { HullData } from "./hull";
import type { Mesh, MeshData, Triangle } from "./mesh";

type Scratch = { k: Kernel; f: Float32Array; u: Uint32Array; output: DataView };
const scratch = new WeakMap<Kernel, Scratch>();
function prepare(words = 32): Scratch {
    const k = kernel(undefined);
    const ptr = k.geometryReadInputPtr(words);
    let s = scratch.get(k);
    if (!s) {
        s = {
            k,
            f: new Float32Array(0),
            u: new Uint32Array(0),
            output: new DataView(k.memory.buffer),
        };
        scratch.set(k, s);
    }
    if (s.f.buffer !== k.memory.buffer || s.f.byteOffset !== ptr || s.f.length !== words) {
        s.f = new Float32Array(k.memory.buffer, ptr, words);
        s.u = new Uint32Array(k.memory.buffer, ptr, words);
    }
    return s;
}
function vector(s: Scratch, offset: number, v: Vec3): void {
    s.f[offset] = v.x;
    s.f[offset + 1] = v.y;
    s.f[offset + 2] = v.z;
}
function transform(s: Scratch, t: Transform): void {
    vector(s, 0, t.p);
    vector(s, 3, t.q.v);
    s.f[6] = t.q.s;
}
function result(s: Scratch, operation: number): number {
    const ptr = s.k.geometryRead(operation);
    if (s.output.buffer !== s.k.memory.buffer) s.output = new DataView(s.k.memory.buffer);
    return ptr;
}
function readVector(s: Scratch, offset: number, out: Vec3): Vec3 {
    out.x = s.output.getFloat32(offset, true);
    out.y = s.output.getFloat32(offset + 4, true);
    out.z = s.output.getFloat32(offset + 8, true);
    return out;
}
function bounds(s: Scratch, operation: number, out: AABB): AABB {
    const p = result(s, operation);
    readVector(s, p, out.lowerBound);
    readVector(s, p + 12, out.upperBound);
    return out;
}
function emptyBounds(): AABB {
    return { lowerBound: { x: 0, y: 0, z: 0 }, upperBound: { x: 0, y: 0, z: 0 } };
}
function triangle(s: Scratch, operation: number): Triangle {
    const p = result(s, operation);
    return {
        vertices: [0, 12, 24].map((offset) => readVector(s, p + offset, { x: 0, y: 0, z: 0 })) as [
            Vec3,
            Vec3,
            Vec3,
        ],
        i1: s.output.getUint32(p + 36, true),
        i2: s.output.getUint32(p + 40, true),
        i3: s.output.getUint32(p + 44, true),
        flags: s.output.getUint32(p + 48, true),
    };
}
/** Enclosing world-space bounds of a transformed sphere (b3ComputeSphereAABB). */
export function computeSphereAABB(shape: Sphere, pose: Transform): AABB {
    const s = prepare();
    transform(s, pose);
    vector(s, 8, shape.center);
    s.f[11] = shape.radius;
    return bounds(s, 0, emptyBounds());
}
/** Enclosing world-space bounds of a transformed capsule (b3ComputeCapsuleAABB). */
export function computeCapsuleAABB(shape: Capsule, pose: Transform): AABB {
    const s = prepare();
    transform(s, pose);
    vector(s, 8, shape.center1);
    vector(s, 11, shape.center2);
    s.f[14] = shape.radius;
    return bounds(s, 1, emptyBounds());
}
/** Enclosing AABB of a transformed hull (b3ComputeHullAABB). */
export function computeHullAABB(shape: HullData, pose: Transform): AABB {
    const s = prepare();
    transform(s, pose);
    vector(s, 8, shape.aabb.lowerBound);
    vector(s, 11, shape.aabb.upperBound);
    return bounds(s, 2, emptyBounds());
}
/** World-space AABB enclosing a scaled, transformed mesh (b3ComputeMeshAABB). */
export function computeMeshAABB(
    data: MeshData,
    pose: Transform,
    scale: Vec3,
    out: AABB = emptyBounds(),
): AABB {
    const s = prepare();
    transform(s, pose);
    vector(s, 8, data.bounds.lowerBound);
    vector(s, 11, data.bounds.upperBound);
    vector(s, 14, scale);
    return bounds(s, 3, out);
}
/** Clamp scale magnitude away from zero, preserving sign (b3SafeScale). */
export function safeScale(value: Vec3, out: Vec3 = { x: 0, y: 0, z: 0 }): Vec3 {
    const s = prepare();
    vector(s, 8, value);
    return readVector(s, result(s, 4), out);
}
/** Fetch a scaled mesh triangle; reflected scales reverse winding and remap edge flags. */
export function getMeshTriangle(mesh: Mesh, index: number): Triangle {
    const s = prepare();
    const t = mesh.data.triangles[index];
    vector(s, 8, mesh.data.vertices[t.index1]);
    vector(s, 11, mesh.data.vertices[t.index2]);
    vector(s, 14, mesh.data.vertices[t.index3]);
    s.u[17] = t.index1;
    s.u[18] = t.index2;
    s.u[19] = t.index3;
    s.u[20] = mesh.data.flags[index];
    vector(s, 21, mesh.scale);
    return triangle(s, 5);
}
/** Fetch a decoded height-field triangle, with clockwise winding and edge flags applied. */
export function getHeightFieldTriangle(field: HeightFieldData, index: number): Triangle {
    const s = prepare(32 + field.compressedHeights.length);
    s.u[8] = field.columnCount;
    s.u[9] = index;
    s.f[10] = field.minHeight;
    s.f[11] = field.heightScale;
    vector(s, 12, field.scale);
    s.u[15] = +field.clockwise;
    s.u[16] = field.flags[index];
    s.u.set(field.compressedHeights, 32);
    return triangle(s, 6);
}
