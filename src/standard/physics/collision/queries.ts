import type { World } from "../../../engine/ecs";
import type { Quat, Transform, Vec3 } from "../common/records";
import { kernel } from "../kernel/kernel";
import type { ShapeProxy } from "./distance";

/** Convex distance input; transform places B in A's frame. Lengths are meters. */
export type DistanceInput = {
    proxyA: ShapeProxy;
    proxyB: ShapeProxy;
    transform: Transform;
    useRadii: boolean;
};
/** GJK warm start. Zero count on first use and whenever either point cloud changes. */
export type SimplexCache = {
    metric: number;
    count: number;
    indexA: number[];
    indexB: number[];
};
export type SimplexVertex = {
    wA: Vec3;
    wB: Vec3;
    w: Vec3;
    a: number;
    indexA: number;
    indexB: number;
};
export type Simplex = { vertices: SimplexVertex[]; count: number };
export type DistanceOutput = {
    pointA: Vec3;
    pointB: Vec3;
    normal: Vec3;
    distance: number;
    iterations: number;
    simplexCount: number;
    simplexes: Simplex[];
};
/** Convex translation cast. Transform places B in A's frame; translationB is in A's axes. */
export type ShapeCastPairInput = {
    proxyA: ShapeProxy;
    proxyB: ShapeProxy;
    transform: Transform;
    translationB: Vec3;
    maxFraction: number;
    canEncroach: boolean;
};
export type ShapeCastOutput = {
    point: Vec3;
    normal: Vec3;
    fraction: number;
    iterations: number;
    hit: boolean;
};
/** Center-of-mass positions in meters and unit quaternions at the sweep endpoints. */
export type Sweep = { localCenter: Vec3; c1: Vec3; c2: Vec3; q1: Quat; q2: Quat };
export type TOIInput = {
    proxyA: ShapeProxy;
    proxyB: ShapeProxy;
    sweepA: Sweep;
    sweepB: Sweep;
    maxFraction: number;
};
/** Box3D TOI states: unknown, failed, overlapped, hit, separated. */
export const TOIState = { unknown: 0, failed: 1, overlapped: 2, hit: 3, separated: 4 } as const;
export type TOIOutput = {
    state: number;
    fraction: number;
    distance: number;
    point: Vec3;
    normal: Vec3;
    distanceIterations: number;
    pushBackIterations: number;
    rootIterations: number;
};

const vector = (): Vec3 => ({ x: 0, y: 0, z: 0 });
const queryState = {
    create: () => ({
        buffer: null as ArrayBufferLike | null,
        input: new Float32Array(0),
        output: new Float32Array(0),
        distance: {
            pointA: vector(),
            pointB: vector(),
            normal: vector(),
            distance: 0,
            iterations: 0,
            simplexCount: 0,
            simplexes: Array.from({ length: 32 }, () => ({
                count: 0,
                vertices: Array.from({ length: 4 }, () => ({
                    wA: vector(),
                    wB: vector(),
                    w: vector(),
                    a: 0,
                    indexA: 0,
                    indexB: 0,
                })),
            })),
        } satisfies DistanceOutput,
        cast: { point: vector(), normal: vector(), fraction: 0, iterations: 0, hit: false },
        toi: {
            point: vector(),
            normal: vector(),
            fraction: 0,
            state: 0,
            distance: 0,
            distanceIterations: 0,
            pushBackIterations: 0,
            rootIterations: 0,
        },
        transform: { p: vector(), q: { v: vector(), s: 1 } },
    }),
};
function prepare(world: World) {
    const k = kernel(world);
    const state = world.resource(queryState);
    if (state.buffer !== k.memory.buffer) {
        state.buffer = k.memory.buffer;
        state.input = new Float32Array(state.buffer, k.collisionInputPtr(), 850);
        state.output = new Float32Array(state.buffer, k.collisionOutputPtr(), 1600);
    }
    return state;
}
function put(r: Float32Array, i: number, v: Vec3): void {
    r[i] = v.x;
    r[i + 1] = v.y;
    r[i + 2] = v.z;
}
function get(r: Float32Array, i: number, v: Vec3): void {
    v.x = r[i];
    v.y = r[i + 1];
    v.z = r[i + 2];
}
function proxy(r: Float32Array, i: number, p: ShapeProxy): void {
    if (!Number.isInteger(p.count) || p.count < 1 || p.count > 128 || p.count > p.points.length) {
        throw new RangeError("Collision proxies require 1–128 points");
    }
    if (!Number.isFinite(p.radius) || p.radius < 0) throw new RangeError("Invalid proxy radius");
    r[i] = p.count;
    r[i + 1] = p.radius;
    for (let j = 0; j < p.count; j++) put(r, i + 2 + 3 * j, p.points[j]);
}
function transform(r: Float32Array, t: Transform): void {
    put(r, 800, t.p);
    quaternion(r, 803, t.q);
}
function quaternion(r: Float32Array, i: number, q: Quat): void {
    put(r, i, q.v);
    r[i + 3] = q.s;
}
function sweep(r: Float32Array, i: number, s: Sweep): void {
    put(r, i, s.localCenter);
    put(r, i + 3, s.c1);
    put(r, i + 6, s.c2);
    quaternion(r, i + 9, s.q1);
    quaternion(r, i + 13, s.q2);
}
/**
 * Convex distance in A's frame, in meters. Mutates caller-owned cache.
 * Result and simplex records are borrowed until the next shapeDistance on this world.
 * simplexCapacity is 0–32; only records below simplexCount are valid. Calls allocate only on first
 * use or after kernel memory growth. Reset cache.count when changing proxy point clouds.
 */
export function shapeDistance(
    world: World,
    input: DistanceInput,
    cache: SimplexCache,
    simplexCapacity = 0,
): DistanceOutput {
    if (!Number.isInteger(simplexCapacity) || simplexCapacity < 0 || simplexCapacity > 32)
        throw new RangeError("Invalid simplex capacity");
    if (
        !Number.isInteger(cache.count) ||
        cache.count < 0 ||
        cache.count > 4 ||
        cache.indexA.length < 4 ||
        cache.indexB.length < 4
    )
        throw new RangeError("Invalid simplex cache");
    for (let i = 0; i < cache.count; i++) {
        if (
            !Number.isInteger(cache.indexA[i]) ||
            cache.indexA[i] < 0 ||
            cache.indexA[i] >= input.proxyA.count ||
            !Number.isInteger(cache.indexB[i]) ||
            cache.indexB[i] < 0 ||
            cache.indexB[i] >= input.proxyB.count
        )
            throw new RangeError("Invalid simplex cache index");
    }
    const s = prepare(world),
        r = s.input;
    proxy(r, 0, input.proxyA);
    proxy(r, 386, input.proxyB);
    transform(r, input.transform);
    r[780] = cache.metric;
    r[781] = cache.count;
    for (let i = 0; i < 4; i++) {
        r[782 + i] = cache.indexA[i];
        r[786 + i] = cache.indexB[i];
    }
    r[807] = Number(input.useRadii);
    r[808] = simplexCapacity;
    kernel(world).collisionQuery(0);
    const o = s.output,
        d = s.distance;
    get(o, 0, d.pointA);
    get(o, 3, d.pointB);
    get(o, 6, d.normal);
    d.distance = o[9];
    d.iterations = o[10];
    d.simplexCount = o[11];
    cache.metric = o[12];
    cache.count = o[13];
    for (let i = 0; i < 4; i++) {
        cache.indexA[i] = o[14 + i];
        cache.indexB[i] = o[18 + i];
    }
    for (let i = 0; i < d.simplexCount; i++) {
        const simplex = d.simplexes[i],
            base = 22 + i * 49;
        simplex.count = o[base];
        for (let j = 0; j < 4; j++) {
            const v = simplex.vertices[j],
                b = base + 1 + j * 12;
            get(o, b, v.wA);
            get(o, b + 3, v.wB);
            get(o, b + 6, v.w);
            v.a = o[b + 9];
            v.indexA = o[b + 10];
            v.indexB = o[b + 11];
        }
    }
    return d;
}
/**
 * Cast B against A. Point and normal are in A's frame; fraction is along translationB.
 * Initial overlap is a miss; canEncroach permits an initially touching rounded shape to advance.
 * Borrowed until the next shapeCast on this world; allocates only on first use or memory growth.
 */
export function shapeCast(world: World, input: ShapeCastPairInput): ShapeCastOutput {
    const s = prepare(world),
        r = s.input;
    proxy(r, 0, input.proxyA);
    proxy(r, 386, input.proxyB);
    transform(r, input.transform);
    put(r, 807, input.translationB);
    r[810] = input.maxFraction;
    r[811] = Number(input.canEncroach);
    kernel(world).collisionQuery(1);
    const o = s.output,
        c = s.cast;
    get(o, 0, c.point);
    get(o, 3, c.normal);
    c.fraction = o[6];
    c.iterations = o[7];
    c.hit = o[8] !== 0;
    return c;
}
/**
 * Rotational convex sweep. Point and normal share the sweep positions' frame, in meters.
 * Borrowed until the next timeOfImpact on this world; allocates only on first use or memory growth.
 */
export function timeOfImpact(world: World, input: TOIInput): TOIOutput {
    const s = prepare(world),
        r = s.input;
    proxy(r, 0, input.proxyA);
    proxy(r, 386, input.proxyB);
    sweep(r, 800, input.sweepA);
    sweep(r, 817, input.sweepB);
    r[834] = input.maxFraction;
    kernel(world).collisionQuery(2);
    const o = s.output,
        t = s.toi;
    get(o, 0, t.point);
    get(o, 3, t.normal);
    t.fraction = o[6];
    t.state = o[7];
    t.distance = o[8];
    t.distanceIterations = o[9];
    t.pushBackIterations = o[10];
    t.rootIterations = o[11];
    return t;
}
/** Sweep placement at a dimensionless fraction. Borrowed until the next getSweepTransform on world. */
export function getSweepTransform(world: World, input: Sweep, fraction: number): Transform {
    const s = prepare(world);
    sweep(s.input, 800, input);
    s.input[834] = fraction;
    kernel(world).collisionQuery(3);
    get(s.output, 0, s.transform.p);
    get(s.output, 3, s.transform.q.v);
    s.transform.q.s = s.output[6];
    return s.transform;
}
