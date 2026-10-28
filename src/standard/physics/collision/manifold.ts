import type { World } from "../../../engine/ecs";
import type { Transform, Vec3 } from "../common/records";
import { kernel } from "../kernel/kernel";
import type { Capsule, Sphere } from "../shapes/geometry";
import type { HullData } from "../shapes/hull";
import { hullByteCount, writeHullImage } from "../shapes/hullbytes";
import type { SimplexCache } from "./queries";

/** Box3D separating features, including the manual axes used by the Manifold samples. */
export const SeparatingFeature = {
    invalid: 0,
    backsideAxis: 1,
    faceAxisA: 2,
    faceAxisB: 3,
    edgePairAxis: 4,
    closestPointsAxis: 5,
    manualFaceAxisA: 6,
    manualFaceAxisB: 7,
    manualEdgePairAxis: 8,
} as const;
/** Zero-initialize on first use and when changing geometry. Calls update every field. */
export type SATCache = {
    separation: number;
    type: number;
    indexA: number;
    indexB: number;
    hit: number;
};
export type FeaturePair = { owner1: number; index1: number; owner2: number; index2: number };
export type LocalManifoldPoint = {
    point: Vec3;
    separation: number;
    pair: FeaturePair;
    triangleIndex: number;
};
/** Only points below pointCount are valid. Lengths are meters; squaredDistance is meters squared. */
export type LocalManifold = {
    normal: Vec3;
    triangleNormal: Vec3;
    points: LocalManifoldPoint[];
    pointCount: number;
    triangleIndex: number;
    vertexIndices: number[];
    triangleFlags: number;
    feature: number;
    squaredDistance: number;
};
const vector = (): Vec3 => ({ x: 0, y: 0, z: 0 });
const scratch = {
    create: () => ({
        buffer: null as ArrayBufferLike | null,
        f: new Float32Array(0),
        u: new Uint32Array(0),
        bytes: new Uint8Array(0),
        hashes: new BigUint64Array(0),
        o: new Float32Array(0),
        ou: new Uint32Array(0),
        result: {
            normal: vector(),
            triangleNormal: vector(),
            pointCount: 0,
            triangleIndex: 0,
            vertexIndices: [0, 0, 0],
            triangleFlags: 0,
            feature: 0,
            squaredDistance: 0,
            points: Array.from({ length: 64 }, () => ({
                point: vector(),
                separation: 0,
                pair: { owner1: 0, index1: 0, owner2: 0, index2: 0 },
                triangleIndex: 0,
            })),
        } satisfies LocalManifold,
    }),
};
type Scratch = ReturnType<typeof scratch.create>;
function put(s: Scratch, i: number, v: Vec3): void {
    s.f[i] = v.x;
    s.f[i + 1] = v.y;
    s.f[i + 2] = v.z;
}
function get(s: Scratch, i: number, v: Vec3): void {
    v.x = s.o[i];
    v.y = s.o[i + 1];
    v.z = s.o[i + 2];
}
function prepare(world: World, capacity: number): Scratch {
    if (!Number.isInteger(capacity) || capacity < 1 || capacity > 64)
        throw new RangeError("Manifold capacity must be 1–64");
    const k = kernel(world),
        s = world.resource(scratch);
    if (s.buffer !== k.memory.buffer) {
        s.buffer = k.memory.buffer;
        const p = k.manifoldInputPtr();
        s.f = new Float32Array(s.buffer, p, 16384);
        s.u = new Uint32Array(s.buffer, p, 16384);
        s.bytes = new Uint8Array(s.buffer, p, 65536);
        s.hashes = new BigUint64Array(s.buffer, p, 8192);
        s.o = new Float32Array(s.buffer, k.manifoldOutputPtr(), 512);
        s.ou = new Uint32Array(s.buffer, k.manifoldOutputPtr(), 512);
    }
    s.u.fill(0, 0, 128);
    s.u[0] = capacity;
    s.u[49] = 1;
    return s;
}
function pose(s: Scratch, t: Transform): void {
    put(s, 4, t.p);
    put(s, 7, t.q.v);
    s.f[10] = t.q.s;
}
function sphere(s: Scratch, i: number, v: Sphere): void {
    put(s, i, v.center);
    s.f[i + 3] = v.radius;
}
function capsule(s: Scratch, i: number, v: Capsule): void {
    put(s, i, v.center1);
    put(s, i + 3, v.center2);
    s.f[i + 6] = v.radius;
}
function hull(s: Scratch, i: number, h: HullData): void {
    if (hullByteCount(h) > 31744) throw new RangeError("Hull exceeds manifold scratch capacity");
    writeHullImage(h, s.bytes, s.u, s.f, s.hashes, i * 4);
}
function triangle(s: Scratch, t: readonly [Vec3, Vec3, Vec3]): void {
    put(s, 34, t[0]);
    put(s, 37, t[1]);
    put(s, 40, t[2]);
}
function simplex(s: Scratch, c: SimplexCache, a: number, b: number): void {
    if (
        !Number.isInteger(c.count) ||
        c.count < 0 ||
        c.count > 4 ||
        c.indexA.length < 4 ||
        c.indexB.length < 4
    )
        throw new RangeError("Invalid simplex cache");
    s.f[50] = c.metric;
    s.u[51] = c.count;
    for (let i = 0; i < 4; i++) {
        if (
            i < c.count &&
            (!Number.isInteger(c.indexA[i]) ||
                c.indexA[i] < 0 ||
                c.indexA[i] >= a ||
                !Number.isInteger(c.indexB[i]) ||
                c.indexB[i] < 0 ||
                c.indexB[i] >= b)
        )
            throw new RangeError("Invalid simplex cache index");
        s.u[52 + i] = c.indexA[i];
        s.u[56 + i] = c.indexB[i];
    }
}
function sat(s: Scratch, c: SATCache, a: number, b: number): void {
    if (
        !Number.isInteger(c.type) ||
        c.type < 0 ||
        c.type > 8 ||
        !Number.isInteger(c.indexA) ||
        c.indexA < 0 ||
        c.indexA >= a ||
        !Number.isInteger(c.indexB) ||
        c.indexB < 0 ||
        c.indexB >= b
    )
        throw new RangeError("Invalid SAT cache");
    s.f[44] = c.separation;
    s.u[45] = c.type;
    s.u[46] = c.indexA;
    s.u[47] = c.indexB;
    s.u[48] = c.hit;
}
function result(
    world: World,
    s: Scratch,
    op: number,
    c?: SimplexCache,
    cache?: SATCache,
): LocalManifold {
    kernel(world).manifoldQuery(op);
    const m = s.result,
        u = s.ou,
        f = s.o;
    m.pointCount = u[0];
    get(s, 1, m.normal);
    m.feature = u[4];
    m.squaredDistance = f[5];
    let n = 6;
    for (let i = 0; i < m.pointCount; i++, n += 6) {
        const p = m.points[i],
            id = u[n + 4];
        get(s, n, p.point);
        p.separation = f[n + 3];
        p.pair.owner1 = id >>> 24;
        p.pair.index1 = (id >>> 16) & 255;
        p.pair.owner2 = (id >>> 8) & 255;
        p.pair.index2 = id & 255;
        p.triangleIndex = u[n + 5] | 0;
    }
    get(s, n + 1, m.triangleNormal);
    m.triangleIndex = u[n + 4] | 0;
    for (let i = 0; i < 3; i++) m.vertexIndices[i] = u[n + 5 + i];
    m.triangleFlags = u[n + 8];
    n += 9;
    if (c) {
        c.metric = f[n];
        c.count = u[n + 1];
        for (let i = 0; i < 4; i++) {
            c.indexA[i] = u[n + 2 + i];
            c.indexB[i] = u[n + 6 + i];
        }
    }
    if (cache) {
        cache.separation = f[n + 10];
        cache.type = u[n + 11];
        cache.indexA = u[n + 12];
        cache.indexB = u[n + 13];
        cache.hit = u[n + 14];
    }
    return m;
}
/** Contact in A's frame; transform places B in A. Borrowed until any collide call on this world.
 * All collide calls allocate only on first use or kernel memory growth. Capacity is 1–64.
 */
export function collideSpheres(
    world: World,
    a: Sphere,
    b: Sphere,
    transform: Transform,
    capacity = 64,
): LocalManifold {
    const s = prepare(world, capacity);
    sphere(s, 12, a);
    sphere(s, 16, b);
    pose(s, transform);
    return result(world, s, 10);
}
/** Capsule A against sphere B; transform and borrowed-result lifetime as collideSpheres. */
export function collideCapsuleAndSphere(
    world: World,
    a: Capsule,
    b: Sphere,
    transform: Transform,
    capacity = 64,
): LocalManifold {
    const s = prepare(world, capacity);
    capsule(s, 20, a);
    sphere(s, 16, b);
    pose(s, transform);
    return result(world, s, 11);
}
/** Hull A against sphere B, mutating the caller's simplex cache. Result is borrowed as collideSpheres. */
export function collideHullAndSphere(
    world: World,
    a: HullData,
    b: Sphere,
    transform: Transform,
    cache: SimplexCache,
    capacity = 64,
): LocalManifold {
    const s = prepare(world, capacity);
    hull(s, 128, a);
    sphere(s, 16, b);
    pose(s, transform);
    simplex(s, cache, a.vertexCount, 1);
    return result(world, s, 12, cache);
}
/** Capsules in A's frame; transform and borrowed-result lifetime as collideSpheres. */
export function collideCapsules(
    world: World,
    a: Capsule,
    b: Capsule,
    transform: Transform,
    capacity = 64,
): LocalManifold {
    const s = prepare(world, capacity);
    capsule(s, 20, a);
    capsule(s, 27, b);
    pose(s, transform);
    return result(world, s, 13);
}
/** Hull A against capsule B, mutating the caller's simplex cache. Borrowed as collideSpheres. */
export function collideHullAndCapsule(
    world: World,
    a: HullData,
    b: Capsule,
    transform: Transform,
    cache: SimplexCache,
    capacity = 64,
): LocalManifold {
    const s = prepare(world, capacity);
    hull(s, 128, a);
    capsule(s, 27, b);
    pose(s, transform);
    simplex(s, cache, a.vertexCount, 2);
    return result(world, s, 14, cache);
}
/** Hulls in A's frame, mutating the caller's SAT cache. Borrowed as collideSpheres. */
export function collideHulls(
    world: World,
    a: HullData,
    b: HullData,
    transform: Transform,
    cache: SATCache,
    capacity = 64,
): LocalManifold {
    const s = prepare(world, capacity);
    hull(s, 128, a);
    hull(s, 8192, b);
    pose(s, transform);
    sat(
        s,
        cache,
        cache.type === 2 || cache.type === 6 ? a.faceCount : a.edgeCount,
        cache.type === 3 || cache.type === 7 ? b.faceCount : b.edgeCount,
    );
    return result(world, s, 15, undefined, cache);
}
/** Triangle vertices are already in the sphere's frame. Result is in that frame, borrowed as collideSpheres. */
export function collideTriangleAndSphere(
    world: World,
    vertices: readonly [Vec3, Vec3, Vec3],
    b: Sphere,
    capacity = 64,
): LocalManifold {
    const s = prepare(world, capacity);
    triangle(s, vertices);
    sphere(s, 12, b);
    return result(world, s, 16);
}
/** Triangle vertices and result are in the capsule's frame. Mutates cache; borrowed as collideSpheres. */
export function collideTriangleAndCapsule(
    world: World,
    vertices: readonly [Vec3, Vec3, Vec3],
    b: Capsule,
    cache: SimplexCache,
    capacity = 64,
): LocalManifold {
    const s = prepare(world, capacity);
    triangle(s, vertices);
    capsule(s, 20, b);
    simplex(s, cache, 3, 2);
    return result(world, s, 17, cache);
}
/** Triangle vertices and result are in the hull's frame. Flags are Box3D mesh edge flags.
 * Mutates cache; enableSpeculativePoints admits positive-separation contacts. Borrowed as collideSpheres.
 */
export function collideTriangleAndHull(
    world: World,
    vertices: readonly [Vec3, Vec3, Vec3],
    flags: number,
    b: HullData,
    cache: SATCache,
    enableSpeculativePoints = true,
    capacity = 64,
): LocalManifold {
    const s = prepare(world, capacity);
    triangle(s, vertices);
    hull(s, 128, b);
    s.u[43] = flags;
    s.u[49] = Number(enableSpeculativePoints);
    sat(s, cache, 3, cache.type === 3 || cache.type === 7 ? b.faceCount : b.edgeCount);
    return result(world, s, 18, undefined, cache);
}
