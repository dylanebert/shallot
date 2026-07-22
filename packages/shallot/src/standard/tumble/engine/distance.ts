// GJK distance, shape casting, and conservative-advancement time of impact.
// Ported op-for-op from Box3D's distance.c (Erin Catto, MIT; portions by Dirk Gregorius).
// fround discipline + scalar-branch mirroring per the README.
//
// The whole query runs in shape A's frame using the relative pose of B in A, keeping the math
// near the local origin. Results stay in frame A.

import { MAX_SHAPE_CAST_POINTS } from "./core";
import {
    type AABB,
    absf,
    FLT_EPSILON,
    FLT_MAX,
    FLT_MIN,
    f32,
    mat3,
    maxf,
    minInt,
    type Quat,
    quat,
    scalarTripleProduct as scalarTriple,
    type Transform,
    type Vec3,
    vec3,
    xf,
} from "./math";

const MAX_SIMPLEX_VERTICES = 4;
const MAX_GJK_ITERATIONS = 32;
const NULL_INDEX = -1;

// B3_LINEAR_SLOP = 0.005 * lengthUnitsPerMeter; the port targets the default length unit of 1.0,
// so the multiply is identity and bit-exact.
const LINEAR_SLOP = f32(0.005);

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

/** Input for {@link shapeCast} (b3ShapeCastPairInput). */
export type ShapeCastPairInput = {
    proxyA: ShapeProxy;
    proxyB: ShapeProxy;
    /** Transform of shape B in shape A's frame. */
    transform: Transform;
    /** Translation of shape B, in A's frame. */
    translationB: Vec3;
    /** Fraction of the translation to consider, typically 1. */
    maxFraction: number;
    /** Allow shapes with a radius to move slightly closer if already touching. */
    canEncroach: boolean;
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

/** Body/shape motion for TOI: the center of mass sweep plus rotations (b3Sweep). */
export type Sweep = {
    /** Local center of mass position. */
    localCenter: Vec3;
    /** Starting center of mass world position. */
    c1: Vec3;
    /** Ending center of mass world position. */
    c2: Vec3;
    /** Starting world rotation. */
    q1: Quat;
    /** Ending world rotation. */
    q2: Quat;
};

/** Input for {@link timeOfImpact} (b3TOIInput). */
export type TOIInput = {
    proxyA: ShapeProxy;
    proxyB: ShapeProxy;
    sweepA: Sweep;
    sweepB: Sweep;
    /** The sweep interval is [0, maxFraction]. */
    maxFraction: number;
};

/** Time of impact result state (b3TOIState). */
export const TOIState = {
    Unknown: 0,
    Failed: 1,
    Overlapped: 2,
    Hit: 3,
    Separated: 4,
} as const;
export type TOIStateValue = (typeof TOIState)[keyof typeof TOIState];

/** Output of {@link timeOfImpact} (b3TOIOutput). */
export type TOIOutput = {
    state: TOIStateValue;
    point: Vec3;
    normal: Vec3;
    fraction: number;
    distance: number;
    distanceIterations: number;
    pushBackIterations: number;
    rootIterations: number;
    usedFallback: boolean;
};

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

        if (simplex.count === MAX_SIMPLEX_VERTICES) {
            const w = computeWitnessPoints(simplex);
            output.pointA = w.vertexA;
            output.pointB = w.vertexB;
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

    normal = vec3.normalize(normal);
    if (vec3.isNormalized(normal) === false) {
        // Treat as overlap.
        return output;
    }

    const w = computeWitnessPoints(simplex);
    writeCache(cache, simplex);

    output.pointA = w.vertexA;
    output.pointB = w.vertexB;
    output.distance = vec3.distance(w.vertexA, w.vertexB);
    output.normal = normal;
    output.iterations = iteration;

    if (input.useRadii) {
        const rA = input.proxyA.radius;
        const rB = input.proxyB.radius;
        output.distance = maxf(0, f32(f32(output.distance - rA) - rB));
        // Keep closest points on the perimeter even if overlapped, so they move smoothly.
        output.pointA = vec3.mulAdd(output.pointA, rA, normal);
        output.pointB = vec3.mulSub(output.pointB, rB, normal);
    }

    return output;
}

// --- shape cast -----------------------------------------------------------------------------

/**
 * Cast proxy B (translated by `translationB`) against proxy A (b3ShapeCast). Initial overlap is a
 * hit at fraction zero. Returns the fraction, contact point, and normal.
 */
export function shapeCast(input: ShapeCastPairInput): CastOutput {
    const linearSlop = LINEAR_SLOP;
    const totalRadius = f32(input.proxyA.radius + input.proxyB.radius);
    let target = maxf(linearSlop, f32(totalRadius - linearSlop));
    const tolerance = f32(0.25 * linearSlop);

    const cache = emptyCache();
    let alpha = 0;

    const distanceInput: DistanceInput = {
        proxyA: input.proxyA,
        proxyB: input.proxyB,
        useRadii: false,
        transform: { p: input.transform.p, q: input.transform.q },
    };

    const delta2 = input.translationB;
    const output: CastOutput = {
        normal: vec3.zero(),
        point: vec3.zero(),
        fraction: 0,
        iterations: 0,
        triangleIndex: NULL_INDEX,
        childIndex: 0,
        materialIndex: 0,
        hit: false,
    };

    const maxIterations = 20;
    for (let iteration = 0; iteration < maxIterations; ++iteration) {
        output.iterations += 1;
        const distanceOutput = shapeDistance(distanceInput, cache);

        if (distanceOutput.distance < f32(target + tolerance)) {
            if (iteration === 0) {
                if (input.canEncroach && distanceOutput.distance > f32(2 * linearSlop)) {
                    target = f32(distanceOutput.distance - linearSlop);
                } else {
                    // Initial overlap.
                    output.hit = true;
                    const c1 = vec3.mulAdd(
                        distanceOutput.pointA,
                        input.proxyA.radius,
                        distanceOutput.normal,
                    );
                    const c2 = vec3.mulAdd(
                        distanceOutput.pointB,
                        -input.proxyB.radius,
                        distanceOutput.normal,
                    );
                    output.point = vec3.lerp(c1, c2, 0.5);
                    return output;
                }
            } else {
                output.fraction = alpha;
                output.point = vec3.mulAdd(
                    distanceOutput.pointA,
                    input.proxyA.radius,
                    distanceOutput.normal,
                );
                output.normal = distanceOutput.normal;
                output.hit = true;
                return output;
            }
        }

        // Check if the shapes are approaching each other.
        const denominator = vec3.dot(delta2, distanceOutput.normal);
        if (denominator >= 0) {
            // Miss.
            return output;
        }

        alpha = f32(alpha + f32(f32(target - distanceOutput.distance) / denominator));
        if (alpha >= input.maxFraction) {
            // Success!
            return output;
        }

        distanceInput.transform = {
            p: vec3.mulAdd(input.transform.p, alpha, delta2),
            q: distanceInput.transform.q,
        };
    }

    // Failure!
    return output;
}

// --- sweep + time of impact -----------------------------------------------------------------

/** Interpolated transform of a sweep at `time` in [0, 1] (b3GetSweepTransform). */
export function getSweepTransform(sweep: Sweep, time: number): Transform {
    const q = quat.nlerp(sweep.q1, sweep.q2, time);
    const p = vec3.sub(vec3.lerp(sweep.c1, sweep.c2, time), quat.rotate(q, sweep.localCenter));
    return { p, q };
}

function getFinalSweepTransform(sweep: Sweep): Transform {
    const q = sweep.q2;
    const p = vec3.sub(sweep.c2, quat.rotate(q, sweep.localCenter));
    return { p, q };
}

function uniqueCount(vertexCount: number, vertices: number[]): number {
    switch (vertexCount) {
        case 1:
            return 1;
        case 2:
            return vertices[0] !== vertices[1] ? 2 : 1;
        case 3:
            if (
                vertices[0] !== vertices[1] &&
                vertices[0] !== vertices[2] &&
                vertices[1] !== vertices[2]
            ) {
                return 3;
            }
            if (
                vertices[0] === vertices[1] &&
                vertices[0] === vertices[2] &&
                vertices[1] === vertices[2]
            ) {
                return 1;
            }
            return 2;
        default:
            return 0;
    }
}

// Does the cross product of two edges switch direction? (b3CheckFastEdges)
function checkFastEdges(
    xfA: Transform,
    localEdgeA: Vec3,
    xfB: Transform,
    localEdgeB: Vec3,
    axis0: Vec3,
): boolean {
    const edgeA = quat.rotate(xfA.q, localEdgeA);
    const edgeB = quat.rotate(xfB.q, localEdgeB);
    const axis = vec3.cross(edgeA, edgeB);
    return vec3.dot(axis, axis0) < 0;
}

const SeparationType = {
    Unknown: 0,
    Vertices: 1,
    Edges: 2,
    FaceA: 3,
    FaceB: 4,
} as const;
type SeparationTypeValue = (typeof SeparationType)[keyof typeof SeparationType];

type SeparationFunction = {
    proxyA: ShapeProxy;
    proxyB: ShapeProxy;
    sweepA: Sweep;
    sweepB: Sweep;
    witness1: Vec3;
    witness2: Vec3;
    type: SeparationTypeValue;
};

function makeSeparationFunction(
    cache: SimplexCache,
    proxyA: ShapeProxy,
    sweepA: Sweep,
    proxyB: ShapeProxy,
    sweepB: Sweep,
    worldNormal: Vec3,
    t1: number,
): SeparationFunction {
    const fcn: SeparationFunction = {
        proxyA,
        proxyB,
        sweepA,
        sweepB,
        witness1: vec3.zero(),
        witness2: vec3.zero(),
        type: SeparationType.Unknown,
    };

    const indexA = [cache.indexA[0], cache.indexA[1], cache.indexA[2]];
    const indexB = [cache.indexB[0], cache.indexB[1], cache.indexB[2]];

    const uniqueCountA = uniqueCount(cache.count, indexA);
    const uniqueCountB = uniqueCount(cache.count, indexB);

    const xfA1 = getSweepTransform(sweepA, t1);
    const xfB1 = getSweepTransform(sweepB, t1);
    const qA = xfA1.q;
    const qB = xfB1.q;
    const deltaP = vec3.sub(xfB1.p, xfA1.p);

    switch (cache.count) {
        case 1:
            fcn.type = SeparationType.Vertices;
            fcn.witness1 = worldNormal;
            break;

        case 2: {
            if (uniqueCountA === 2 && uniqueCountB === 2) {
                // Edge/Edge
                const vA1 = proxyA.points[indexA[0]];
                let localEdgeA = vec3.sub(proxyA.points[indexA[1]], vA1);
                localEdgeA = vec3.normalize(localEdgeA);
                const edgeA = quat.rotate(qA, localEdgeA);

                const vB1 = proxyB.points[indexB[0]];
                let localEdgeB = vec3.sub(proxyB.points[indexB[1]], vB1);
                localEdgeB = vec3.normalize(localEdgeB);
                const edgeB = quat.rotate(qB, localEdgeB);

                let axis = vec3.cross(edgeA, edgeB);
                const lengthSquared = vec3.lengthSq(axis);

                const kToleranceSquared = f32(0.05 * 0.05);
                if (lengthSquared < kToleranceSquared) {
                    fcn.type = SeparationType.Vertices;
                    fcn.witness1 = worldNormal;
                } else {
                    const delta = vec3.add(
                        vec3.sub(quat.rotate(qB, vB1), quat.rotate(qA, vA1)),
                        deltaP,
                    );
                    if (vec3.dot(delta, axis) < 0) {
                        axis = vec3.neg(axis);
                        localEdgeB = vec3.neg(localEdgeB);
                    }
                    const xfA2 = getFinalSweepTransform(sweepA);
                    const xfB2 = getFinalSweepTransform(sweepB);
                    const fastEdges = checkFastEdges(xfA2, localEdgeA, xfB2, localEdgeB, axis);
                    if (fastEdges === true) {
                        fcn.type = SeparationType.Vertices;
                        fcn.witness1 = vec3.normalize(axis);
                    } else {
                        fcn.type = SeparationType.Edges;
                        fcn.witness1 = localEdgeA;
                        fcn.witness2 = localEdgeB;
                    }
                }
            } else {
                // Vertex versus edge, use world axis witness.
                fcn.type = SeparationType.Vertices;
                fcn.witness1 = worldNormal;
            }
            break;
        }

        case 3: {
            if (uniqueCountA === 3) {
                const vA1 = proxyA.points[indexA[0]];
                const vA2 = proxyA.points[indexA[1]];
                const vA3 = proxyA.points[indexA[2]];
                let localAxisA = vec3.cross(vec3.sub(vA2, vA1), vec3.sub(vA3, vA1));
                localAxisA = vec3.normalize(localAxisA);
                const axisA = quat.rotate(qA, localAxisA);

                const localPointA = vec3.scale(f32(1 / 3), vec3.add(vec3.add(vA1, vA2), vA3));
                const localPointB = proxyB.points[indexB[0]];
                const delta = vec3.add(
                    vec3.sub(quat.rotate(qB, localPointB), quat.rotate(qA, localPointA)),
                    deltaP,
                );
                if (vec3.dot(delta, axisA) < 0) {
                    localAxisA = vec3.neg(localAxisA);
                }
                fcn.type = SeparationType.FaceA;
                fcn.witness1 = localAxisA;
                fcn.witness2 = localPointA;
            } else if (uniqueCountB === 3) {
                const vB1 = proxyB.points[indexB[0]];
                const vB2 = proxyB.points[indexB[1]];
                const vB3 = proxyB.points[indexB[2]];
                let localAxisB = vec3.cross(vec3.sub(vB2, vB1), vec3.sub(vB3, vB1));
                localAxisB = vec3.normalize(localAxisB);
                const axisB = quat.rotate(qB, localAxisB);

                const localPointA = proxyA.points[indexA[0]];
                const localPointB = vec3.scale(f32(1 / 3), vec3.add(vec3.add(vB1, vB2), vB3));
                const delta = vec3.sub(
                    vec3.sub(quat.rotate(qA, localPointA), quat.rotate(qB, localPointB)),
                    deltaP,
                );
                if (vec3.dot(delta, axisB) < 0) {
                    localAxisB = vec3.neg(localAxisB);
                }
                fcn.type = SeparationType.FaceB;
                fcn.witness1 = localAxisB;
                fcn.witness2 = localPointB;
            } else {
                // uniqueCountA === 2 && uniqueCountB === 2
                if (indexA[0] === indexA[1]) {
                    indexA[1] = indexA[2];
                }
                const vA1 = proxyA.points[indexA[0]];
                const vA2 = proxyA.points[indexA[1]];
                const localEdgeA = vec3.normalize(vec3.sub(vA2, vA1));
                const edgeA = quat.rotate(qA, localEdgeA);

                if (indexB[0] === indexB[1]) {
                    indexB[1] = indexB[2];
                }
                const vB1 = proxyB.points[indexB[0]];
                const vB2 = proxyB.points[indexB[1]];
                let localEdgeB = vec3.normalize(vec3.sub(vB2, vB1));
                const edgeB = quat.rotate(qB, localEdgeB);

                let axis = vec3.cross(edgeA, edgeB);
                const lengthSquared = vec3.lengthSq(axis);

                const kToleranceSquared = f32(0.005 * 0.005);
                if (lengthSquared < kToleranceSquared) {
                    fcn.type = SeparationType.Vertices;
                    fcn.witness1 = worldNormal;
                } else {
                    const delta = vec3.add(
                        vec3.sub(quat.rotate(qB, vB1), quat.rotate(qA, vA1)),
                        deltaP,
                    );
                    if (vec3.dot(delta, axis) < 0) {
                        axis = vec3.neg(axis);
                        localEdgeB = vec3.neg(localEdgeB);
                    }
                    const xfA2 = getFinalSweepTransform(sweepA);
                    const xfB2 = getFinalSweepTransform(sweepB);
                    const fastEdges = checkFastEdges(xfA2, localEdgeA, xfB2, localEdgeB, axis);
                    if (fastEdges) {
                        fcn.type = SeparationType.Vertices;
                        fcn.witness1 = vec3.normalize(axis);
                    } else {
                        fcn.type = SeparationType.Edges;
                        fcn.witness1 = localEdgeA;
                        fcn.witness2 = localEdgeB;
                    }
                }
            }
            break;
        }
    }

    return fcn;
}

function findMinSeparation(
    fcn: SeparationFunction,
    t: number,
): { s: number; indexA: number; indexB: number } {
    const xfA = getSweepTransform(fcn.sweepA, t);
    const xfB = getSweepTransform(fcn.sweepB, t);

    switch (fcn.type) {
        case SeparationType.Vertices: {
            const axis = fcn.witness1;
            const localAxisA = quat.invRotate(xfA.q, axis);
            const localAxisB = quat.invRotate(xfB.q, vec3.neg(axis));
            const indexA = getPointSupport(fcn.proxyA.points, fcn.proxyA.count, localAxisA);
            const indexB = getPointSupport(fcn.proxyB.points, fcn.proxyB.count, localAxisB);
            const deltaP = vec3.sub(xfB.p, xfA.p);
            const localPointA = fcn.proxyA.points[indexA];
            const localPointB = fcn.proxyB.points[indexB];
            const delta = vec3.add(
                vec3.sub(quat.rotate(xfB.q, localPointB), quat.rotate(xfA.q, localPointA)),
                deltaP,
            );
            return { s: vec3.dot(delta, axis), indexA, indexB };
        }
        case SeparationType.Edges: {
            const edgeA = quat.rotate(xfA.q, fcn.witness1);
            const edgeB = quat.rotate(xfB.q, fcn.witness2);
            let axis = vec3.cross(edgeA, edgeB);
            axis = vec3.normalize(axis);
            const axisA = quat.invRotate(xfA.q, axis);
            const indexA = getPointSupport(fcn.proxyA.points, fcn.proxyA.count, axisA);
            const axisB = quat.invRotate(xfB.q, axis);
            const indexB = getPointSupport(fcn.proxyB.points, fcn.proxyB.count, vec3.neg(axisB));
            const deltaP = vec3.sub(xfB.p, xfA.p);
            const localPointA = fcn.proxyA.points[indexA];
            const localPointB = fcn.proxyB.points[indexB];
            const delta = vec3.add(
                vec3.sub(quat.rotate(xfB.q, localPointB), quat.rotate(xfA.q, localPointA)),
                deltaP,
            );
            return { s: vec3.dot(delta, axis), indexA, indexB };
        }
        case SeparationType.FaceA: {
            const normal = quat.rotate(xfA.q, fcn.witness1);
            const pointA = xf.point(xfA, fcn.witness2);
            const axisB = quat.invRotate(xfB.q, normal);
            const indexB = getPointSupport(fcn.proxyB.points, fcn.proxyB.count, vec3.neg(axisB));
            const pointB = xf.point(xfB, fcn.proxyB.points[indexB]);
            return { s: vec3.dot(vec3.sub(pointB, pointA), normal), indexA: -1, indexB };
        }
        case SeparationType.FaceB: {
            const normal = quat.rotate(xfB.q, fcn.witness1);
            const axisA = quat.invRotate(xfA.q, normal);
            const indexA = getPointSupport(fcn.proxyA.points, fcn.proxyA.count, vec3.neg(axisA));
            const pointA = xf.point(xfA, fcn.proxyA.points[indexA]);
            const pointB = xf.point(xfB, fcn.witness2);
            return { s: vec3.dot(vec3.sub(pointA, pointB), normal), indexA, indexB: -1 };
        }
        default:
            return { s: 0, indexA: 0, indexB: 0 };
    }
}

function evaluateSeparation(
    fcn: SeparationFunction,
    index1: number,
    index2: number,
    beta: number,
): number {
    const transform1 = getSweepTransform(fcn.sweepA, beta);
    const transform2 = getSweepTransform(fcn.sweepB, beta);

    switch (fcn.type) {
        case SeparationType.Vertices: {
            const axis = fcn.witness1;
            const point1 = xf.point(transform1, fcn.proxyA.points[index1]);
            const point2 = xf.point(transform2, fcn.proxyB.points[index2]);
            return vec3.dot(vec3.sub(point2, point1), axis);
        }
        case SeparationType.Edges: {
            const edge1 = quat.rotate(transform1.q, fcn.witness1);
            const edge2 = quat.rotate(transform2.q, fcn.witness2);
            let axis = vec3.cross(edge1, edge2);
            axis = vec3.normalize(axis);
            const point1 = xf.point(transform1, fcn.proxyA.points[index1]);
            const point2 = xf.point(transform2, fcn.proxyB.points[index2]);
            return vec3.dot(vec3.sub(point2, point1), axis);
        }
        case SeparationType.FaceA: {
            const axis = quat.rotate(transform1.q, fcn.witness1);
            const point1 = xf.point(transform1, fcn.witness2);
            const point2 = xf.point(transform2, fcn.proxyB.points[index2]);
            return vec3.dot(vec3.sub(point2, point1), axis);
        }
        case SeparationType.FaceB: {
            const axis = quat.rotate(transform2.q, fcn.witness1);
            const point1 = xf.point(transform1, fcn.proxyA.points[index1]);
            const point2 = xf.point(transform2, fcn.witness2);
            return vec3.dot(vec3.sub(point1, point2), axis);
        }
        default:
            return 0;
    }
}

function forceFixedAxis(fcn: SeparationFunction, beta: number): void {
    const transform1 = getSweepTransform(fcn.sweepA, beta);
    const transform2 = getSweepTransform(fcn.sweepB, beta);
    const edge1 = quat.rotate(transform1.q, fcn.witness1);
    const edge2 = quat.rotate(transform2.q, fcn.witness2);
    let axis = vec3.cross(edge1, edge2);
    axis = vec3.normalize(axis);
    fcn.type = SeparationType.Vertices;
    fcn.witness1 = axis;
    fcn.witness2 = vec3.zero();
}

/**
 * Conservative-advancement time of impact between two swept convex proxies (b3TimeOfImpact).
 * Returns the sweep fraction of first contact and the classification state.
 */
export function timeOfImpact(input: TOIInput): TOIOutput {
    const output: TOIOutput = {
        state: TOIState.Unknown,
        point: vec3.zero(),
        normal: vec3.zero(),
        fraction: -1,
        distance: 0,
        distanceIterations: 0,
        pushBackIterations: 0,
        rootIterations: 0,
        usedFallback: false,
    };

    // Shift to origin (mutable copies of the sweeps).
    const origin = input.sweepA.c1;
    const sweepA: Sweep = {
        localCenter: input.sweepA.localCenter,
        c1: vec3.zero(),
        c2: vec3.sub(input.sweepA.c2, origin),
        q1: input.sweepA.q1,
        q2: input.sweepA.q2,
    };
    const sweepB: Sweep = {
        localCenter: input.sweepB.localCenter,
        c1: vec3.sub(input.sweepB.c1, origin),
        c2: vec3.sub(input.sweepB.c2, origin),
        q1: input.sweepB.q1,
        q2: input.sweepB.q2,
    };

    const proxyA = input.proxyA;
    const proxyB = input.proxyB;

    const maxPushBackIterations = proxyA.count + proxyB.count;
    const tMax = input.maxFraction;

    const linearSlop = LINEAR_SLOP;
    const totalRadius = f32(proxyA.radius + proxyB.radius);
    const target = maxf(linearSlop, f32(totalRadius - linearSlop));
    const tolerance = f32(0.25 * linearSlop);

    let t1 = 0;
    const maxIterations = 25;
    let distanceIterations = 0;

    const cache = emptyCache();
    const distanceInput: DistanceInput = {
        proxyA,
        proxyB,
        useRadii: false,
        transform: xf.identity(),
    };

    for (;;) {
        const xfA = getSweepTransform(sweepA, t1);
        const xfB = getSweepTransform(sweepB, t1);
        distanceInput.transform = xf.invMul(xfA, xfB);
        const distanceOutput = shapeDistance(distanceInput, cache);
        output.distance = distanceOutput.distance;

        const worldNormal = quat.rotate(xfA.q, distanceOutput.normal);
        const worldPointA = xf.point(xfA, distanceOutput.pointA);
        const worldPointB = xf.point(xfA, distanceOutput.pointB);

        output.distanceIterations += 1;
        distanceIterations += 1;

        if (distanceOutput.distance <= 0) {
            output.state = TOIState.Overlapped;
            output.fraction = 0;
            break;
        }

        if (distanceOutput.distance <= f32(target + tolerance)) {
            output.state = TOIState.Hit;
            const pA = vec3.mulAdd(worldPointA, proxyA.radius, worldNormal);
            const pB = vec3.mulAdd(worldPointB, -proxyB.radius, worldNormal);
            output.point = vec3.add(vec3.lerp(pA, pB, 0.5), origin);
            output.normal = worldNormal;
            output.fraction = t1;
            break;
        }

        if (distanceIterations === maxIterations) {
            // Progress too slow (e.g. a capsule rotating around a triangle vertex).
            output.state = TOIState.Failed;
            output.fraction = t1;
            const pA = vec3.mulAdd(worldPointA, input.proxyA.radius, worldNormal);
            const pB = vec3.mulAdd(worldPointB, -input.proxyB.radius, worldNormal);
            output.point = vec3.add(vec3.lerp(pA, pB, 0.5), origin);
            output.normal = worldNormal;
            break;
        }

        const fcn = makeSeparationFunction(cache, proxyA, sweepA, proxyB, sweepB, worldNormal, t1);

        let done = false;
        let t2 = tMax;
        let pushBackIterations = 0;
        for (;;) {
            const min = findMinSeparation(fcn, t2);
            let s2 = min.s;
            const indexA = min.indexA;
            const indexB = min.indexB;

            // Is the final configuration separated?
            if (f32(s2 - target) > tolerance) {
                output.state = TOIState.Separated;
                output.fraction = input.maxFraction;
                done = true;
                break;
            }

            // Has the separation reached tolerance?
            if (s2 >= f32(target - tolerance)) {
                t1 = t2;
                break;
            }

            let s1 = evaluateSeparation(fcn, indexA, indexB, t1);

            // Overlap: the root finder may have run out of iterations.
            if (s1 < f32(target - tolerance)) {
                output.state = TOIState.Failed;
                output.fraction = t1;
                done = true;
                break;
            }

            if (s1 <= f32(target + tolerance)) {
                // t1 holds the TOI (could be 0).
                output.state = TOIState.Hit;
                output.fraction = t1;
                done = true;
                break;
            }

            // 1D root of f(x) - target = 0.
            let rootIterationCount = 0;
            const maxRootIterations = 50;
            let a1 = t1;
            let a2 = t2;
            for (;;) {
                let t: number;
                if (rootIterationCount & 1) {
                    // False position to improve convergence.
                    t = f32(a1 + f32(f32(f32(target - s1) * f32(a2 - a1)) / f32(s2 - s1)));
                } else {
                    // Bisection to guarantee progress.
                    t = f32(0.5 * f32(a1 + a2));
                }

                output.rootIterations += 1;
                rootIterationCount += 1;

                const s = evaluateSeparation(fcn, indexA, indexB, t);

                if (absf(f32(s - target)) <= tolerance) {
                    // t2 holds a tentative value for t1.
                    t2 = t;
                    break;
                }

                // Keep bracketing the root.
                if (s > target) {
                    a1 = t;
                    s1 = s;
                } else {
                    a2 = t;
                    s2 = s;
                }

                if (rootIterationCount === maxRootIterations) {
                    break;
                }
            }

            // Restart the inner loop on a failing edge case.
            if (rootIterationCount === maxRootIterations - 1 && fcn.type === SeparationType.Edges) {
                rootIterationCount = 0;
                t2 = input.maxFraction;
                forceFixedAxis(fcn, t1);
            }

            output.pushBackIterations += 1;
            pushBackIterations += 1;

            if (pushBackIterations === maxPushBackIterations) {
                break;
            }
        }

        if (done) {
            const pA = vec3.mulAdd(worldPointA, input.proxyA.radius, worldNormal);
            const pB = vec3.mulAdd(worldPointB, -input.proxyB.radius, worldNormal);
            output.point = vec3.add(vec3.lerp(pA, pB, 0.5), origin);
            output.normal = worldNormal;
            break;
        }
    }

    return output;
}
