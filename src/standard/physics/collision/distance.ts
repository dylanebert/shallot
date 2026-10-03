// GJK distance.
// Ported op-for-op from Box3D's distance.c (Erin Catto, MIT; portions by Dirk Gregorius).
// fround discipline + scalar-branch mirroring (every f32 result rounds through Math.fround, keeping bit-exact f32 parity).
//
// The whole query runs in shape A's frame using the relative pose of B in A, keeping the math
// near the local origin. Results stay in frame A.

import { MAX_SHAPE_CAST_POINTS } from "../common/constants";
import {
    type AABB,
    FLT_EPSILON,
    FLT_MAX,
    FLT_MIN,
    f32,
    mat3,
    maxf,
    minInt,
    scalarTripleProduct as scalarTriple,
    type Transform,
    type Vec3,
    vec3,
    xf,
} from "../common/math";

const MAX_SIMPLEX_VERTICES = 4;
const MAX_GJK_ITERATIONS = 32;

/** A convex shape as a point cloud wrapped with a rounding radius (b3ShapeProxy). */
export type ShapeProxy = {
    /** The point cloud. */
    points: Vec3[];
    /** The number of points. */
    count: number;
    /** The external radius of the point cloud. */
    radius: number;
};

/** Warm-start data for the GJK simplex; zero-initialize on the first call (b3SimplexCache). */
export type SimplexCache = {
    /** Length/area/volume metric used to compare two simplexes. */
    metric: number;
    /** Number of stored simplex points. */
    count: number;
    /** Cached simplex indices on shape A. */
    indexA: number[];
    /** Cached simplex indices on shape B. */
    indexB: number[];
};

/** A fresh, empty simplex cache. */
export const emptyCache = (): SimplexCache => ({
    metric: 0,
    count: 0,
    indexA: [0, 0, 0, 0],
    indexB: [0, 0, 0, 0],
});

/**
 * Pull a proxy's point cloud into a shape's local frame (b3MakeLocalProxy). The C multiplies by the
 * inverse-transform rotation matrix (not the quaternion-rotate formula), so the port mirrors that to
 * stay bit-exact. Point count is clamped to the shape-cast maximum.
 */
export function makeLocalProxy(proxy: ShapeProxy, transform: Transform): ShapeProxy {
    const invTransform = xf.invert(transform);
    const m = mat3.fromQuat(invTransform.q);
    const count = minInt(proxy.count, MAX_SHAPE_CAST_POINTS);
    const points: Vec3[] = new Array(count);
    for (let i = 0; i < count; ++i) {
        points[i] = vec3.add(mat3.mulV(m, proxy.points[i]), invTransform.p);
    }
    return { points, count, radius: proxy.radius };
}

/** AABB enclosing a proxy's point cloud, grown by its radius (b3ComputeProxyAABB). */
export function computeProxyAABB(proxy: ShapeProxy): AABB {
    let lower = proxy.points[0];
    let upper = proxy.points[0];
    for (let i = 1; i < proxy.count; ++i) {
        lower = vec3.min(lower, proxy.points[i]);
        upper = vec3.max(upper, proxy.points[i]);
    }
    const r: Vec3 = { x: proxy.radius, y: proxy.radius, z: proxy.radius };
    return { lowerBound: vec3.sub(lower, r), upperBound: vec3.add(upper, r) };
}

/** One simplex vertex: the Minkowski support point and its barycentric weight (b3SimplexVertex). */
export type SimplexVertex = {
    /** Support point in proxy A. */
    wA: Vec3;
    /** Support point in proxy B. */
    wB: Vec3;
    /** wB - wA. */
    w: Vec3;
    /** Barycentric coordinate. */
    a: number;
    /** wA index. */
    indexA: number;
    /** wB index. */
    indexB: number;
};

/** The GJK simplex: up to four vertices (b3Simplex). */
export type Simplex = {
    vertices: SimplexVertex[];
    count: number;
};

/** Input for {@link shapeDistance} (b3DistanceInput). */
export type DistanceInput = {
    proxyA: ShapeProxy;
    proxyB: ShapeProxy;
    /** Transform of shape B in shape A's frame. */
    transform: Transform;
    /** Should the proxy radius be considered? */
    useRadii: boolean;
};

/** Output of {@link shapeDistance} (b3DistanceOutput). */
export type DistanceOutput = {
    /** Closest point on shape A, in shape A's frame. */
    pointA: Vec3;
    /** Closest point on shape B, in shape A's frame. */
    pointB: Vec3;
    /** A-to-B normal in shape A's frame. Invalid if distance is zero. */
    normal: Vec3;
    /** Final distance, zero if overlapped. */
    distance: number;
    /** Number of GJK iterations used. */
    iterations: number;
    /** Number of simplexes stored in the simplex array. */
    simplexCount: number;
};

/** Low level ray cast input (b3RayCastInput). */
export type RayCastInput = {
    /** Start point of the ray. */
    origin: Vec3;
    /** Ray displacement; end = origin + translation. */
    translation: Vec3;
    /** Maximum fraction of the translation to consider, typically 1. */
    maxFraction: number;
};

/** Low level shape-cast input: a point cloud + radius swept along a translation (b3ShapeCastInput). */
export type ShapeCastInput = {
    proxy: ShapeProxy;
    translation: Vec3;
    maxFraction: number;
    /** Allow an already-touching shape with radius to move slightly closer. */
    canEncroach: boolean;
};

/** Low level ray/shape-cast output (b3CastOutput). */
export type CastOutput = {
    normal: Vec3;
    point: Vec3;
    fraction: number;
    iterations: number;
    triangleIndex: number;
    childIndex: number;
    materialIndex: number;
    hit: boolean;
};

/** A zero-initialized cast output (C `b3CastOutput output = { 0 }`): a miss, all fields zero. */
export function emptyCastOutput(): CastOutput {
    return {
        normal: vec3.zero(),
        point: vec3.zero(),
        fraction: 0,
        iterations: 0,
        triangleIndex: 0,
        childIndex: 0,
        materialIndex: 0,
        hit: false,
    };
}

// --- simplex helpers ------------------------------------------------------------------------

const zeroVertex = (): SimplexVertex => ({
    wA: vec3.zero(),
    wB: vec3.zero(),
    w: vec3.zero(),
    a: 0,
    indexA: 0,
    indexB: 0,
});

const emptySimplex = (): Simplex => ({
    vertices: [zeroVertex(), zeroVertex(), zeroVertex(), zeroVertex()],
    count: 0,
});

// A vertex is a value type in C. Its Vec3 fields are never mutated in place (always whole-replaced),
// so a shallow field copy reproduces C's struct-copy semantics.
const cloneVertex = (v: SimplexVertex): SimplexVertex => ({
    wA: v.wA,
    wB: v.wB,
    w: v.w,
    a: v.a,
    indexA: v.indexA,
    indexB: v.indexB,
});

function assignVertex(dst: SimplexVertex, src: SimplexVertex): void {
    dst.wA = src.wA;
    dst.wB = src.wB;
    dst.w = src.w;
    dst.a = src.a;
    dst.indexA = src.indexA;
    dst.indexB = src.indexB;
}

const cloneSimplex = (s: Simplex): Simplex => ({
    vertices: [
        cloneVertex(s.vertices[0]),
        cloneVertex(s.vertices[1]),
        cloneVertex(s.vertices[2]),
        cloneVertex(s.vertices[3]),
    ],
    count: s.count,
});

// --- support functions ----------------------------------------------------------------------

/** Index of the proxy point furthest along `axis` (b3GetProxySupport). */
export function getProxySupport(proxy: ShapeProxy, axis: Vec3): number {
    const points = proxy.points;
    const origin = points[0];
    let maxIndex = 0;
    let maxProjection = 0;
    for (let index = 1; index < proxy.count; ++index) {
        const projection = vec3.dot(axis, vec3.sub(points[index], origin));
        if (projection > maxProjection) {
            maxIndex = index;
            maxProjection = projection;
        }
    }
    return maxIndex;
}

/** Index of the point furthest along `axis` in a raw point cloud (b3GetPointSupport). */
export function getPointSupport(points: Vec3[], count: number, axis: Vec3): number {
    const origin = points[0];
    let maxIndex = 0;
    let maxProjection = 0;
    for (let index = 1; index < count; ++index) {
        const projection = vec3.dot(axis, vec3.sub(points[index], origin));
        if (projection > maxProjection) {
            maxIndex = index;
            maxProjection = projection;
        }
    }
    return maxIndex;
}

// --- barycentric coordinates ----------------------------------------------------------------

function barycentricEdge(a: Vec3, b: Vec3): [number, number, number] {
    const ab = vec3.sub(b, a);
    const divisor = vec3.dot(ab, ab);
    return [vec3.dot(b, ab), -vec3.dot(a, ab), divisor];
}

function barycentricTri(a: Vec3, b: Vec3, c: Vec3): [number, number, number, number] {
    const ab = vec3.sub(b, a);
    const ac = vec3.sub(c, a);
    const bXC = vec3.cross(b, c);
    const cXA = vec3.cross(c, a);
    const aXB = vec3.cross(a, b);
    const abXAc = vec3.cross(ab, ac);
    const divisor = vec3.dot(abXAc, abXAc);
    return [vec3.dot(bXC, abXAc), vec3.dot(cXA, abXAc), vec3.dot(aXB, abXAc), divisor];
}

function barycentricTet(
    a: Vec3,
    b: Vec3,
    c: Vec3,
    d: Vec3,
): [number, number, number, number, number] {
    const ab = vec3.sub(b, a);
    const ac = vec3.sub(c, a);
    const ad = vec3.sub(d, a);
    const divisor = scalarTriple(ab, ac, ad);
    const sign = divisor < 0 ? -1 : 1;
    return [
        f32(sign * scalarTriple(b, c, d)),
        f32(sign * scalarTriple(a, d, c)),
        f32(sign * scalarTriple(a, b, d)),
        f32(sign * scalarTriple(a, c, b)),
        f32(sign * divisor),
    ];
}

// --- metric ---------------------------------------------------------------------------------

function getMetric(simplex: Simplex): number {
    const vs = simplex.vertices;
    switch (simplex.count) {
        case 1:
            return 0;
        case 2:
            return vec3.distance(vs[0].w, vs[1].w);
        case 3: {
            const cross = vec3.cross(vec3.sub(vs[1].w, vs[0].w), vec3.sub(vs[2].w, vs[0].w));
            return f32(vec3.length(cross) / 2);
        }
        case 4:
            return f32(
                scalarTriple(
                    vec3.sub(vs[1].w, vs[0].w),
                    vec3.sub(vs[2].w, vs[0].w),
                    vec3.sub(vs[3].w, vs[0].w),
                ) / 6,
            );
        default:
            return 0;
    }
}

function writeCache(cache: SimplexCache, simplex: Simplex): void {
    const count = simplex.count;
    cache.metric = getMetric(simplex);
    cache.count = count;
    for (let index = 0; index < count; ++index) {
        cache.indexA[index] = simplex.vertices[index].indexA;
        cache.indexB[index] = simplex.vertices[index].indexB;
    }
}

// --- simplex solvers ------------------------------------------------------------------------

function solveSimplex2(simplex: Simplex): boolean {
    const vs = simplex.vertices;
    const a = vs[0].w;
    const b = vs[1].w;
    const ab = vec3.sub(b, a);
    const divisor = vec3.dot(ab, ab);
    const u = vec3.dot(b, ab);
    const v = -vec3.dot(a, ab);

    // V( A )
    if (v <= 0) {
        simplex.count = 1;
        vs[0].a = 1;
        return true;
    }
    // V( B )
    if (u <= 0) {
        simplex.count = 1;
        assignVertex(vs[0], vs[1]);
        vs[0].a = 1;
        return true;
    }
    // Edge region
    if (divisor <= 0) return false;

    const denominator = f32(1 / divisor);
    vs[0].a = f32(denominator * u);
    vs[1].a = f32(denominator * v);
    return true;
}

function solveSimplex3(simplex: Simplex): boolean {
    const vs = simplex.vertices;
    // Snapshot the simplex (aliasing: the slots below get overwritten).
    const v1 = cloneVertex(vs[0]);
    const v2 = cloneVertex(vs[1]);
    const v3 = cloneVertex(vs[2]);

    const wAB = barycentricEdge(v1.w, v2.w);
    const wBC = barycentricEdge(v2.w, v3.w);
    const wCA = barycentricEdge(v3.w, v1.w);

    // VR( A )
    if (wAB[1] <= 0 && wCA[0] <= 0) {
        simplex.count = 1;
        assignVertex(vs[0], v1);
        vs[0].a = 1;
        return true;
    }
    // VR( B )
    if (wBC[1] <= 0 && wAB[0] <= 0) {
        simplex.count = 1;
        assignVertex(vs[0], v2);
        vs[0].a = 1;
        return true;
    }
    // VR( C )
    if (wCA[1] <= 0 && wBC[0] <= 0) {
        simplex.count = 1;
        assignVertex(vs[0], v3);
        vs[0].a = 1;
        return true;
    }

    const wABC = barycentricTri(v1.w, v2.w, v3.w);

    // VR( AB )
    if (wABC[2] <= 0 && wAB[0] > 0 && wAB[1] > 0) {
        simplex.count = 2;
        assignVertex(vs[0], v1);
        assignVertex(vs[1], v2);
        const divisor = wAB[2];
        if (divisor <= 0) return false;
        vs[0].a = f32(wAB[0] / divisor);
        vs[1].a = f32(wAB[1] / divisor);
        return true;
    }
    // VR( BC )
    if (wABC[0] <= 0 && wBC[0] > 0 && wBC[1] > 0) {
        simplex.count = 2;
        assignVertex(vs[0], v2);
        assignVertex(vs[1], v3);
        const divisor = wBC[2];
        if (divisor <= 0) return false;
        vs[0].a = f32(wBC[0] / divisor);
        vs[1].a = f32(wBC[1] / divisor);
        return true;
    }
    // VR( CA )
    if (wABC[1] <= 0 && wCA[0] > 0 && wCA[1] > 0) {
        simplex.count = 2;
        assignVertex(vs[0], v3);
        assignVertex(vs[1], v1);
        const divisor = wCA[2];
        if (divisor <= 0) return false;
        vs[0].a = f32(wCA[0] / divisor);
        vs[1].a = f32(wCA[1] / divisor);
        return true;
    }

    // Face region
    const divisor = wABC[3];
    if (divisor <= 0) return false;
    vs[0].a = f32(wABC[0] / divisor);
    vs[1].a = f32(wABC[1] / divisor);
    vs[2].a = f32(wABC[2] / divisor);
    return true;
}

function solveSimplex4(simplex: Simplex): boolean {
    const vs = simplex.vertices;
    const vA = cloneVertex(vs[0]);
    const vB = cloneVertex(vs[1]);
    const vC = cloneVertex(vs[2]);
    const vD = cloneVertex(vs[3]);

    const wAB = barycentricEdge(vA.w, vB.w);
    const wAC = barycentricEdge(vA.w, vC.w);
    const wAD = barycentricEdge(vA.w, vD.w);
    const wBC = barycentricEdge(vB.w, vC.w);
    const wCD = barycentricEdge(vC.w, vD.w);
    const wDB = barycentricEdge(vD.w, vB.w);

    // VR( A )
    if (wAB[1] <= 0 && wAC[1] <= 0 && wAD[1] <= 0) {
        simplex.count = 1;
        assignVertex(vs[0], vA);
        vs[0].a = 1;
        return true;
    }
    // VR( B )
    if (wAB[0] <= 0 && wDB[0] <= 0 && wBC[1] <= 0) {
        simplex.count = 1;
        assignVertex(vs[0], vB);
        vs[0].a = 1;
        return true;
    }
    // VR( C )
    if (wAC[0] <= 0 && wBC[0] <= 0 && wCD[1] <= 0) {
        simplex.count = 1;
        assignVertex(vs[0], vC);
        vs[0].a = 1;
        return true;
    }
    // VR( D )
    if (wAD[0] <= 0 && wCD[0] <= 0 && wDB[1] <= 0) {
        simplex.count = 1;
        assignVertex(vs[0], vD);
        vs[0].a = 1;
        return true;
    }

    const wACB = barycentricTri(vA.w, vC.w, vB.w);
    const wABD = barycentricTri(vA.w, vB.w, vD.w);
    const wADC = barycentricTri(vA.w, vD.w, vC.w);
    const wBCD = barycentricTri(vB.w, vC.w, vD.w);

    // VR( AB )
    if (wABD[2] <= 0 && wACB[1] <= 0 && wAB[0] > 0 && wAB[1] > 0) {
        simplex.count = 2;
        assignVertex(vs[0], vA);
        assignVertex(vs[1], vB);
        const divisor = wAB[2];
        if (divisor <= 0) return false;
        vs[0].a = f32(wAB[0] / divisor);
        vs[1].a = f32(wAB[1] / divisor);
        return true;
    }
    // VR( AC )
    if (wACB[2] <= 0 && wADC[1] <= 0 && wAC[0] > 0 && wAC[1] > 0) {
        simplex.count = 2;
        assignVertex(vs[0], vA);
        assignVertex(vs[1], vC);
        const divisor = wAC[2];
        if (divisor <= 0) return false;
        vs[0].a = f32(wAC[0] / divisor);
        vs[1].a = f32(wAC[1] / divisor);
        return true;
    }
    // VR( AD )
    if (wADC[2] <= 0 && wABD[1] <= 0 && wAD[0] > 0 && wAD[1] > 0) {
        simplex.count = 2;
        assignVertex(vs[0], vA);
        assignVertex(vs[1], vD);
        const divisor = wAD[2];
        if (divisor <= 0) return false;
        vs[0].a = f32(wAD[0] / divisor);
        vs[1].a = f32(wAD[1] / divisor);
        return true;
    }
    // VR( BC )
    if (wACB[0] <= 0 && wBCD[2] <= 0 && wBC[0] > 0 && wBC[1] > 0) {
        simplex.count = 2;
        assignVertex(vs[0], vB);
        assignVertex(vs[1], vC);
        const divisor = wBC[2];
        if (divisor <= 0) return false;
        vs[0].a = f32(wBC[0] / divisor);
        vs[1].a = f32(wBC[1] / divisor);
        return true;
    }
    // VR( CD )
    if (wADC[0] <= 0 && wBCD[0] <= 0 && wCD[0] > 0 && wCD[1] > 0) {
        simplex.count = 2;
        assignVertex(vs[0], vC);
        assignVertex(vs[1], vD);
        const divisor = wCD[2];
        if (divisor <= 0) return false;
        vs[0].a = f32(wCD[0] / divisor);
        vs[1].a = f32(wCD[1] / divisor);
        return true;
    }
    // VR( DB )
    if (wABD[0] <= 0 && wBCD[1] <= 0 && wDB[0] > 0 && wDB[1] > 0) {
        simplex.count = 2;
        assignVertex(vs[0], vD);
        assignVertex(vs[1], vB);
        const divisor = wDB[2];
        if (divisor <= 0) return false;
        vs[0].a = f32(wDB[0] / divisor);
        vs[1].a = f32(wDB[1] / divisor);
        return true;
    }

    const wABCD = barycentricTet(vA.w, vB.w, vC.w, vD.w);

    // VR( ACB )
    if (wABCD[3] < 0 && wACB[0] > 0 && wACB[1] > 0 && wACB[2] > 0) {
        simplex.count = 3;
        assignVertex(vs[0], vA);
        assignVertex(vs[1], vC);
        assignVertex(vs[2], vB);
        const divisor = wACB[3];
        if (divisor <= 0) return false;
        vs[0].a = f32(wACB[0] / divisor);
        vs[1].a = f32(wACB[1] / divisor);
        vs[2].a = f32(wACB[2] / divisor);
        return true;
    }
    // VR( ABD )
    if (wABCD[2] < 0 && wABD[0] > 0 && wABD[1] > 0 && wABD[2] > 0) {
        simplex.count = 3;
        assignVertex(vs[0], vA);
        assignVertex(vs[1], vB);
        assignVertex(vs[2], vD);
        const divisor = wABD[3];
        if (divisor <= 0) return false;
        vs[0].a = f32(wABD[0] / divisor);
        vs[1].a = f32(wABD[1] / divisor);
        vs[2].a = f32(wABD[2] / divisor);
        return true;
    }
    // VR( ADC )
    if (wABCD[1] < 0 && wADC[0] > 0 && wADC[1] > 0 && wADC[2] > 0) {
        simplex.count = 3;
        assignVertex(vs[0], vA);
        assignVertex(vs[1], vD);
        assignVertex(vs[2], vC);
        const divisor = wADC[3];
        if (divisor <= 0) return false;
        vs[0].a = f32(wADC[0] / divisor);
        vs[1].a = f32(wADC[1] / divisor);
        vs[2].a = f32(wADC[2] / divisor);
        return true;
    }
    // VR( BCD )
    if (wABCD[0] < 0 && wBCD[0] > 0 && wBCD[1] > 0 && wBCD[2] > 0) {
        simplex.count = 3;
        assignVertex(vs[0], vB);
        assignVertex(vs[1], vC);
        assignVertex(vs[2], vD);
        const divisor = wBCD[3];
        if (divisor <= 0) return false;
        vs[0].a = f32(wBCD[0] / divisor);
        vs[1].a = f32(wBCD[1] / divisor);
        vs[2].a = f32(wBCD[2] / divisor);
        return true;
    }

    // *** Inside tetrahedron ***
    const divisor = wABCD[4];
    if (divisor <= 0) return false;
    vs[0].a = f32(wABCD[0] / divisor);
    vs[1].a = f32(wABCD[1] / divisor);
    vs[2].a = f32(wABCD[2] / divisor);
    vs[3].a = f32(wABCD[3] / divisor);
    return true;
}

function computeWitnessPoints(simplex: Simplex): { vertexA: Vec3; vertexB: Vec3 } {
    const vs = simplex.vertices;
    switch (simplex.count) {
        case 1:
            return { vertexA: vs[0].wA, vertexB: vs[0].wB };
        case 2:
            return {
                vertexA: vec3.blend2(vs[0].a, vs[0].wA, vs[1].a, vs[1].wA),
                vertexB: vec3.blend2(vs[0].a, vs[0].wB, vs[1].a, vs[1].wB),
            };
        case 3:
            return {
                vertexA: vec3.blend3(vs[0].a, vs[0].wA, vs[1].a, vs[1].wA, vs[2].a, vs[2].wA),
                vertexB: vec3.blend3(vs[0].a, vs[0].wB, vs[1].a, vs[1].wB, vs[2].a, vs[2].wB),
            };
        case 4: {
            // Force identical points and zero distance.
            const sum = vec3.add(
                vec3.blend2(vs[0].a, vs[0].wA, vs[1].a, vs[1].wA),
                vec3.blend2(vs[2].a, vs[2].wA, vs[3].a, vs[3].wA),
            );
            return { vertexA: sum, vertexB: sum };
        }
        default:
            return { vertexA: vec3.zero(), vertexB: vec3.zero() };
    }
}

// --- shape distance -------------------------------------------------------------------------

/**
 * Closest points between two convex proxies via GJK (b3ShapeDistance).
 *
 * `cache` warm-starts the simplex and is updated in place; zero-initialize it on the first call.
 * The query runs in shape A's frame using `input.transform`, the relative pose of B in A.
 */
export function shapeDistance(input: DistanceInput, cache: SimplexCache): DistanceOutput {
    const xfT = input.transform;
    const m = mat3.fromQuat(xfT.q);
    const mt = mat3.transpose(m);

    const proxyA = input.proxyA;
    const proxyB = input.proxyB;

    let simplex = emptySimplex();
    let vs = simplex.vertices;

    simplex.count = cache.count;
    for (let i = 0; i < cache.count; ++i) {
        const index1 = cache.indexA[i];
        const index2 = cache.indexB[i];
        const vertex1 = proxyA.points[index1];
        const vertex2 = vec3.add(mat3.mulV(m, proxyB.points[index2]), xfT.p);
        vs[i].indexA = index1;
        vs[i].indexB = index2;
        vs[i].wA = vertex1;
        vs[i].wB = vertex2;
        vs[i].w = vec3.sub(vertex2, vertex1);
        vs[i].a = 0;
    }

    // Flush the simplex if its metric drifted substantially from the cached one.
    if (simplex.count > 0) {
        const metric1 = cache.metric;
        const metric2 = getMetric(simplex);
        if (f32(2 * metric1) < metric2 || metric2 < f32(0.5 * metric1) || metric2 < FLT_EPSILON) {
            simplex.count = 0;
        }
    }

    if (simplex.count === 0) {
        const vertex1 = proxyA.points[0];
        const vertex2 = vec3.add(mat3.mulV(m, proxyB.points[0]), xfT.p);
        simplex.count = 1;
        vs[0].indexA = 0;
        vs[0].indexB = 0;
        vs[0].wA = vertex1;
        vs[0].wB = vertex2;
        vs[0].w = vec3.sub(vertex2, vertex1);
        vs[0].a = 0;
    }

    let backup = emptySimplex();
    // The official implementation records the initial simplex and every solved simplex. The
    // diagnostic count is observable in the public distance oracle, even though the cache only
    // retains the final simplex.
    let simplexIndex = 1;

    const output: DistanceOutput = {
        pointA: vec3.zero(),
        pointB: vec3.zero(),
        normal: vec3.zero(),
        distance: 0,
        iterations: 0,
        simplexCount: 0,
    };

    let distanceSq = FLT_MAX;
    let normal = vec3.zero();

    let iteration = 0;
    for (; iteration < MAX_GJK_ITERATIONS; ++iteration) {
        let solved = false;
        switch (simplex.count) {
            case 1:
                simplex.vertices[0].a = 1;
                solved = true;
                break;
            case 2:
                solved = solveSimplex2(simplex);
                break;
            case 3:
                solved = solveSimplex3(simplex);
                break;
            case 4:
                solved = solveSimplex4(simplex);
                break;
        }

        if (solved === false) {
            simplex = backup;
            break;
        }

        simplexIndex += 1;

        if (simplex.count === MAX_SIMPLEX_VERTICES) {
            const w = computeWitnessPoints(simplex);
            output.pointA = w.vertexA;
            output.pointB = w.vertexB;
            output.normal = vec3.zero();
            output.distance = 0;
            output.iterations = iteration;
            output.simplexCount = simplexIndex;
            return output;
        }

        const oldDistanceSq = distanceSq;
        vs = simplex.vertices;

        let closestPoint = vec3.zero();
        switch (simplex.count) {
            case 1:
                closestPoint = vs[0].w;
                break;
            case 2:
                closestPoint = vec3.blend2(vs[0].a, vs[0].w, vs[1].a, vs[1].w);
                break;
            case 3:
                closestPoint = vec3.blend3(vs[0].a, vs[0].w, vs[1].a, vs[1].w, vs[2].a, vs[2].w);
                break;
        }

        distanceSq = vec3.dot(closestPoint, closestPoint);

        if (distanceSq >= oldDistanceSq) {
            simplex = backup;
            break;
        }

        let searchDirection = vec3.zero();
        switch (simplex.count) {
            case 1:
                searchDirection = vec3.neg(vs[0].w);
                break;
            case 2: {
                const a = vs[0].w;
                const b = vs[1].w;
                const ab = vec3.sub(b, a);
                searchDirection = vec3.cross(vec3.cross(ab, vec3.neg(a)), ab);
                break;
            }
            case 3: {
                const a = vs[0].w;
                const b = vs[1].w;
                const c = vs[2].w;
                const ab = vec3.sub(b, a);
                const ac = vec3.sub(c, a);
                const n = vec3.cross(ab, ac);
                searchDirection = vec3.dot(n, a) < 0 ? n : vec3.neg(n);
                break;
            }
        }

        if (vec3.lengthSq(searchDirection) < f32(1000 * FLT_MIN)) {
            // The origin is contained by a line segment or triangle: the shapes overlap.
            const w = computeWitnessPoints(simplex);
            output.pointA = w.vertexA;
            output.pointB = w.vertexB;
            output.normal = vec3.zero();
            output.distance = 0;
            output.iterations = iteration;
            output.simplexCount = simplexIndex;
            return output;
        }

        normal = vec3.neg(searchDirection);

        const indexA = getProxySupport(input.proxyA, vec3.neg(searchDirection));
        const supportA = input.proxyA.points[indexA];
        const searchDirection2 = mat3.mulV(mt, searchDirection);
        const indexB = getProxySupport(input.proxyB, searchDirection2);
        const supportB = vec3.add(mat3.mulV(m, input.proxyB.points[indexB]), xfT.p);

        backup = cloneSimplex(simplex);

        // Duplicate support point is the main termination criterion.
        let duplicate = false;
        for (let i = 0; i < simplex.count; ++i) {
            if (vs[i].indexA === indexA && vs[i].indexB === indexB) {
                duplicate = true;
                break;
            }
        }
        if (duplicate) break;

        const nv = vs[simplex.count];
        nv.indexA = indexA;
        nv.indexB = indexB;
        nv.wA = supportA;
        nv.wB = supportB;
        nv.w = vec3.sub(supportB, supportA);
        simplex.count += 1;
    }

    const w = computeWitnessPoints(simplex);
    output.pointA = w.vertexA;
    output.pointB = w.vertexB;
    output.iterations = iteration;
    output.simplexCount = simplexIndex;

    normal = vec3.normalize(normal);
    if (vec3.isNormalized(normal) === false) {
        // Treat as overlap.
        output.distance = 0;
        output.normal = vec3.zero();
        return output;
    }

    output.distance = vec3.distance(w.vertexA, w.vertexB);
    output.normal = normal;
    // Simplex is useful, cache it, but only after all overlap exits above.

    if (input.useRadii) {
        const rA = input.proxyA.radius;
        const rB = input.proxyB.radius;
        output.distance = maxf(0, f32(f32(output.distance - rA) - rB));
        // Keep closest points on the perimeter even if overlapped, so they move smoothly.
        output.pointA = vec3.mulAdd(output.pointA, rA, normal);
        output.pointB = vec3.mulSub(output.pointB, rB, normal);
    }

    writeCache(cache, simplex);
    return output;
}
