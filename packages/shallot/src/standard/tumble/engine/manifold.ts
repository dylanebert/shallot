// Narrowphase manifold generation — ports manifold.c (shared clip/query helpers) and
// convex_manifold.c (sphere/capsule/hull pair collision) from Box3D. Every collide function
// takes a caller-owned `manifold.points` buffer of `capacity` slots and writes into it, mirroring
// the C API. Frame convention: results are in shape A's local frame; `transformBtoA` places shape
// B in shape A's frame. See the README for the fround discipline.

import type { DistanceInput, SimplexCache } from "./distance";
import { getPointSupport, shapeDistance } from "./distance";
import type { Capsule, Sphere } from "./geometry";
import type { HullData } from "./hull";
import { findHullSupportFace, findHullSupportVertex } from "./hull";
import {
    absf,
    arbitraryPerp,
    FLT_EPSILON,
    FLT_MAX,
    FLT_MIN,
    f32,
    getLengthAndNormalize,
    isWithinSegments,
    lineDistance,
    mat3,
    maxf,
    minf,
    minInt,
    type Plane,
    plane,
    pointToSegmentDistance,
    quat,
    segmentDistance,
    type Transform,
    type Vec3,
    vec3,
    xf,
} from "./math";

const NULL_INDEX = -1;
// B3_LINEAR_SLOP = 0.005 * lengthUnitsPerMeter; length units default to 1 (no world here).
const LINEAR_SLOP = f32(0.005);
const SPECULATIVE_DISTANCE = f32(4 * LINEAR_SLOP);
const MIN_CAPSULE_LENGTH = LINEAR_SLOP;
const MAX_CLIP_POINTS = 64;

/** Which shape owns a feature in a {@link FeaturePair} (b3FeatureOwner). */
export const FeatureOwner = { ShapeA: 0, ShapeB: 1 } as const;

/**
 * Identifies a contact point by the two intersecting edges that produced it (b3FeaturePair). Used
 * for temporal coherence and warm starting. Each field is a uint8.
 */
export type FeaturePair = {
    owner1: number;
    index1: number;
    owner2: number;
    index2: number;
};

/** A local manifold point in shape A's frame (b3LocalManifoldPoint). */
export type LocalManifoldPoint = {
    /** Contact point in frame A. */
    point: Vec3;
    /** Separation, negative for overlap. */
    separation: number;
    /** Feature pair identifying this point. */
    pair: FeaturePair;
    /** Triangle index when colliding with a mesh or height field. */
    triangleIndex: number;
};

/**
 * A local manifold with no dynamic info (b3LocalManifold). `points` is a caller-owned buffer of
 * `capacity` slots; the collide functions write into it and set `pointCount`.
 */
export type LocalManifold = {
    /** Local normal in frame A, pointing from shape A to shape B. */
    normal: Vec3;
    /** The triangle normal (mesh/height-field collision). */
    triangleNormal: Vec3;
    /** Manifold points, from the caller's buffer. */
    points: LocalManifoldPoint[];
    /** Number of valid points. */
    pointCount: number;
    /** Index of the triangle (mesh/height-field collision). */
    triangleIndex: number;
    /** Triangle vertex indices (mesh/height-field collision). */
    i1: number;
    i2: number;
    i3: number;
    /** Squared distance of a sphere from a triangle, for ghost-collision reduction. */
    squaredDistance: number;
    /** The triangle feature involved (b3TriangleFeature). */
    feature: number;
    /** b3MeshEdgeFlags of the triangle. */
    triangleFlags: number;
};

/** Result of a face-direction SAT query (b3FaceQuery). */
export type FaceQuery = { separation: number; faceIndex: number; vertexIndex: number };

/** Result of an edge-direction SAT query (b3EdgeQuery). */
export type EdgeQuery = { separation: number; indexA: number; indexB: number };

/** Cached triangle feature (b3TriangleFeature). Indexes into a triangle's vertices/edges/face. */
export const TriangleFeature = {
    None: 0,
    TriangleFace: 1,
    HullFace: 2,
    Edge1: 3, // v1-v2
    Edge2: 4, // v2-v3
    Edge3: 5, // v3-v1
    Vertex1: 6,
    Vertex2: 7,
    Vertex3: 8,
} as const;
export type TriangleFeature = (typeof TriangleFeature)[keyof typeof TriangleFeature];

/** Cached separating-axis feature type (b3SeparatingFeature). */
export const SeparatingFeature = {
    Invalid: 0,
    Backside: 1,
    FaceAxisA: 2,
    FaceAxisB: 3,
    EdgePairAxis: 4,
    ClosestPointsAxis: 5,
    ManualFaceAxisA: 6,
    ManualFaceAxisB: 7,
    ManualEdgePairAxis: 8,
} as const;

/** Separating-axis test cache for temporal acceleration of hull-hull collision (b3SATCache). */
export type SATCache = {
    separation: number;
    type: number;
    indexA: number;
    indexB: number;
    hit: number;
};

/** A fresh, empty SAT cache. */
export const emptySATCache = (): SATCache => ({
    separation: 0,
    type: 0,
    indexA: 0,
    indexB: 0,
    hit: 0,
});

/** A fresh, empty local manifold with a `capacity`-slot point buffer. */
export function makeLocalManifold(capacity: number): LocalManifold {
    const points: LocalManifoldPoint[] = [];
    for (let i = 0; i < capacity; ++i) points.push(emptyPoint());
    return {
        normal: { x: 0, y: 0, z: 0 },
        triangleNormal: { x: 0, y: 0, z: 0 },
        points,
        pointCount: 0,
        triangleIndex: 0,
        i1: 0,
        i2: 0,
        i3: 0,
        squaredDistance: 0,
        feature: TriangleFeature.None,
        triangleFlags: 0,
    };
}

// --- internal value-type helpers (JS objects are references; C copies structs by value) --------

export type ClipVertex = { position: Vec3; separation: number; pair: FeaturePair };

const clonePair = (p: FeaturePair): FeaturePair => ({
    owner1: p.owner1,
    index1: p.index1,
    owner2: p.owner2,
    index2: p.index2,
});

/** Copy `src`'s fields into `dst` in place (zero-alloc clip path). */
const copyPair = (dst: FeaturePair, src: FeaturePair): void => {
    dst.owner1 = src.owner1;
    dst.index1 = src.index1;
    dst.owner2 = src.owner2;
    dst.index2 = src.index2;
};

// b3FeaturePair_single = {0}: a fresh single-point feature pair.
export const singlePair = (): FeaturePair => ({ owner1: 0, index1: 0, owner2: 0, index2: 0 });

// Vec3 fields are never mutated in place (always whole-replaced), so the position ref is shared;
// the pair can be field-mutated (clip re-owns endpoints), so it is cloned.
export const cloneClipVertex = (v: ClipVertex): ClipVertex => ({
    position: v.position,
    separation: v.separation,
    pair: clonePair(v.pair),
});

// Two ping-pong scratch buffers for the sutherland-hodgman clip loop. `clipPolygon` reads its input
// buffer and writes the next result into a distinct buffer, so a fresh array is never allocated per
// reference-face edge (the pyramid's dominant narrow-phase churn). Positions are copied into the
// buffer's own Vec3 slots — the shared-position invariant above (cloneClipVertex) doesn't hold across
// buffers, so the clip must own its point storage. The widest input any caller builds is the
// triangle-vs-hull incident face (2·MAX_CLIP_POINTS); clipping a convex polygon against a half-plane
// adds at most one vertex per pass, so 4·MAX_CLIP_POINTS covers every clip result with headroom.
const CLIP_BUFFER_SIZE = 4 * MAX_CLIP_POINTS;
const makeClipBuffer = (): ClipVertex[] => {
    const buf: ClipVertex[] = [];
    for (let i = 0; i < CLIP_BUFFER_SIZE; ++i) {
        buf.push({ position: { x: 0, y: 0, z: 0 }, separation: 0, pair: singlePair() });
    }
    return buf;
};
const clipBufferA = makeClipBuffer();
const clipBufferB = makeClipBuffer();
/** The clip scratch buffer that isn't `input` — the ping-pong target for the next clip pass. */
export const otherClipBuffer = (input: ClipVertex[]): ClipVertex[] =>
    input === clipBufferA ? clipBufferB : clipBufferA;

const tmpClipDelta: Vec3 = { x: 0, y: 0, z: 0 };

// Module-scope scratch for the hull-hull manifold build. `buildPolygon` writes the incident face
// into `polyBuffer`, distinct from the two clip buffers so the first `otherClipBuffer` ping-pong
// lands on clipBufferA; `polyMatrix` holds its rotation. Both are caller-owned scratch fully
// consumed within one contact, so a single module buffer serves every convex contact. Sized to
// MAX_CLIP_POINTS, where `buildPolygon` caps its output. (`reducePoints` + `scratchEdgeManifold`
// are defined below, after `emptyPoint`/`makeLocalManifold`, to avoid a const-TDZ at module eval.)
const polyBuffer = makeClipBuffer();
const polyMatrix = mat3.zero();

const emptyPoint = (): LocalManifoldPoint => ({
    point: { x: 0, y: 0, z: 0 },
    separation: 0,
    pair: singlePair(),
    triangleIndex: 0,
});

// `reducePoints` receives the clipped face points before reduction (see the polyBuffer note above);
// its input count is clamped to MAX_CLIP_POINTS by buildFaceAContact's `minInt`. `scratchEdgeManifold`
// is the 1-point manifold for the hull-hull edge fallback (b3CollideHulls). Both reused per contact.
const reducePoints: LocalManifoldPoint[] = [];
for (let i = 0; i < MAX_CLIP_POINTS; ++i) reducePoints.push(emptyPoint());
// Index indirection for reduceManifoldPoints' swap-remove. The C reference swap-removes the point
// array itself; doing that to the reused `reducePoints` pool would scramble its slot→object mapping
// and corrupt the next contact's writes. Swap-remove `reduceIdx` instead, keeping the pool pristine.
const reduceIdx: number[] = new Array(MAX_CLIP_POINTS);
const scratchEdgeManifold = makeLocalManifold(1);

/** Copy `src`'s fields into the pre-allocated `dst` slot in place (zero-alloc reduce path). */
const copyManifoldPointInto = (dst: LocalManifoldPoint, src: LocalManifoldPoint): void => {
    vec3.copy(src.point, dst.point);
    dst.separation = src.separation;
    copyPair(dst.pair, src.pair);
    dst.triangleIndex = src.triangleIndex;
};

function resetSATCache(cache: SATCache): void {
    cache.separation = 0;
    cache.type = 0;
    cache.indexA = 0;
    cache.indexB = 0;
    cache.hit = 0;
}

function resetSimplexCache(cache: SimplexCache): void {
    cache.metric = 0;
    cache.count = 0;
    for (let i = 0; i < 4; ++i) {
        cache.indexA[i] = 0;
        cache.indexB[i] = 0;
    }
}

// --- shared helpers (manifold.c) ---------------------------------------------------------------

/** b3MakeFeaturePair — pack two features into a pair (each index truncated to uint8). */
export function makeFeaturePair(
    owner1: number,
    index1: number,
    owner2: number,
    index2: number,
): FeaturePair {
    return {
        owner1: owner1 & 0xff,
        index1: index1 & 0xff,
        owner2: owner2 & 0xff,
        index2: index2 & 0xff,
    };
}

/** b3MakeFeatureId — pack a feature pair into a uint32 id for warm-start matching. */
export function makeFeatureId(pair: FeaturePair): number {
    return ((pair.owner1 << 24) | (pair.index1 << 16) | (pair.owner2 << 8) | pair.index2) >>> 0;
}

// b3FlipPair — swap owners (and flip each) and indices so the feature pair is independent of which
// hull is chosen as the reference face. Returns a new pair.
export function flipPair(pair: FeaturePair): FeaturePair {
    return {
        owner1: 1 - pair.owner2,
        index1: pair.index2,
        owner2: 1 - pair.owner1,
        index2: pair.index1,
    };
}

// b3EdgeEdgeSeparation — separation along the cross product of two edges, oriented outward from
// the more significant shape center.
export function edgeEdgeSeparation(
    p1: Vec3,
    e1: Vec3,
    c1: Vec3,
    p2: Vec3,
    e2: Vec3,
    c2: Vec3,
): number {
    const u = vec3.cross(e1, e2);
    const length = vec3.length(u);

    // Skip near-parallel edges: |e1 x e2| = sin(alpha) * |e1| * |e2|.
    const kTolerance = f32(0.005);
    if (length < f32(kTolerance * f32(Math.sqrt(f32(vec3.lengthSq(e1) * vec3.lengthSq(e2)))))) {
        return -FLT_MAX;
    }
    if (f32(length * length) < f32(1000 * FLT_MIN)) {
        return -FLT_MAX;
    }

    let n = vec3.scale(f32(1 / length), u);

    // Orient n away from the shape with the most significant sign value.
    const sign1 = vec3.dot(n, vec3.sub(p1, c1));
    const sign2 = vec3.dot(n, vec3.sub(p2, c2));
    if (absf(sign1) > absf(sign2)) {
        if (sign1 < 0) n = vec3.neg(n);
    } else {
        if (sign2 > 0) n = vec3.neg(n);
    }

    return vec3.dot(n, vec3.sub(p2, p1));
}

// b3FindIncidentFace — the face on `hull` most anti-parallel to `refNormal`, found via the edge
// out of `vertexIndex` most perpendicular to the reference normal (handles wedge shapes).
export function findIncidentFace(hull: HullData, refNormal: Vec3, vertexIndex: number): number {
    const edges = hull.edges;
    const planes = hull.planes;
    const points = hull.points;

    let minEdgeIndex = -1;
    let minEdgeProjection = FLT_MAX;

    const vertex = hull.vertices[vertexIndex];
    let edgeIndex = vertex.edge;
    let edge = edges[edgeIndex];
    const edgeOrigin = points[edge.origin];

    do {
        const twin = edges[edge.twin];
        const twinOrigin = points[twin.origin];
        const axis = vec3.normalize(vec3.sub(twinOrigin, edgeOrigin));
        const edgeProjection = absf(vec3.dot(axis, refNormal));
        if (edgeProjection < minEdgeProjection) {
            minEdgeIndex = edgeIndex;
            minEdgeProjection = edgeProjection;
        }
        edgeIndex = twin.next;
        edge = edges[edgeIndex];
    } while (edgeIndex !== vertex.edge);

    const minEdge = edges[minEdgeIndex];
    const minFaceIndex1 = minEdge.face;
    const minPlane1 = planes[minFaceIndex1];
    const minTwin = edges[minEdge.twin];
    const minFaceIndex2 = minTwin.face;
    const minPlane2 = planes[minFaceIndex2];

    return vec3.dot(minPlane1.normal, refNormal) < vec3.dot(minPlane2.normal, refNormal)
        ? minFaceIndex1
        : minFaceIndex2;
}

// b3ClipPolygon — Sutherland-Hodgman clip of `polygon` against `clipPlane`; intersection points
// re-own their cut edge to `edge` on shape A. Writes the clipped polygon into the `out` scratch buffer
// (distinct from `polygon`; use {@link otherClipBuffer}) and returns the new vertex count.
export function clipPolygon(
    polygon: ClipVertex[],
    count: number,
    clipPlane: Plane,
    edge: number,
    refPlane: Plane,
    out: ClipVertex[],
): number {
    let vertex1 = polygon[count - 1];
    let distance1 = plane.separation(clipPlane, vertex1.position);
    let k = 0;

    for (let index = 0; index < count; ++index) {
        const vertex2 = polygon[index];
        const distance2 = plane.separation(clipPlane, vertex2.position);

        if (distance1 <= 0 && distance2 <= 0) {
            // Both behind: keep vertex2.
            const o = out[k++];
            vec3.copy(vertex2.position, o.position);
            o.separation = vertex2.separation;
            copyPair(o.pair, vertex2.pair);
        } else if (distance1 <= 0 && distance2 > 0) {
            // Leaving: keep intersection, adjust outgoing edge.
            const fraction = f32(distance1 / f32(distance1 - distance2));
            const o = out[k++];
            vec3.mulAddOut(
                vertex1.position,
                fraction,
                vec3.subOut(vertex2.position, vertex1.position, tmpClipDelta),
                o.position,
            );
            o.separation = plane.separation(refPlane, o.position);
            copyPair(o.pair, vertex2.pair);
            o.pair.owner2 = FeatureOwner.ShapeA;
            o.pair.index2 = edge & 0xff;
        } else if (distance2 <= 0 && distance1 > 0) {
            // Entering: keep intersection (adjust incoming edge) then vertex2.
            const fraction = f32(distance1 / f32(distance1 - distance2));
            const o = out[k++];
            vec3.mulAddOut(
                vertex1.position,
                fraction,
                vec3.subOut(vertex2.position, vertex1.position, tmpClipDelta),
                o.position,
            );
            o.separation = plane.separation(refPlane, o.position);
            copyPair(o.pair, vertex1.pair);
            o.pair.owner1 = FeatureOwner.ShapeA;
            o.pair.index1 = edge & 0xff;
            const o2 = out[k++];
            vec3.copy(vertex2.position, o2.position);
            o2.separation = vertex2.separation;
            copyPair(o2.pair, vertex2.pair);
        }

        vertex1 = vertex2;
        distance1 = distance2;
    }

    return k;
}

// --- convex_manifold.c: Gauss-map / clip helpers -----------------------------------------------

function isMinkowskiFaceIsolated(a: Vec3, b: Vec3, n: Vec3): boolean {
    const an = vec3.dot(a, n);
    const bn = vec3.dot(b, n);
    return f32(an * bn) <= 0;
}

function isMinkowskiFace(a: Vec3, b: Vec3, bxa: Vec3, c: Vec3, d: Vec3, dxc: Vec3): boolean {
    const cba = vec3.dot(c, bxa);
    const dba = vec3.dot(d, bxa);
    const adc = vec3.dot(a, dxc);
    const bdc = vec3.dot(b, dxc);
    return f32(cba * dba) < 0 && f32(adc * bdc) < 0 && f32(cba * bdc) > 0;
}

// b3ClipSegment — clip a 2-vertex segment against `plane`, in place. Returns the vertex count.
function clipSegment(segment: ClipVertex[], pl: Plane): number {
    const vertex1 = cloneClipVertex(segment[0]);
    const vertex2 = cloneClipVertex(segment[1]);

    const distance1 = plane.separation(pl, vertex1.position);
    const distance2 = plane.separation(pl, vertex2.position);

    let vertexCount = 0;
    if (distance1 <= 0) segment[vertexCount++] = vertex1;
    if (distance2 <= 0) segment[vertexCount++] = vertex2;

    if (f32(distance1 * distance2) < 0) {
        const t = f32(distance1 / f32(distance1 - distance2));
        const position = vec3.add(
            vec3.scale(f32(1 - t), vertex1.position),
            vec3.scale(t, vertex2.position),
        );
        const src = distance1 > 0 ? vertex1 : vertex2;
        segment[vertexCount] = { position, separation: 0, pair: clonePair(src.pair) };
        vertexCount++;
    }

    return vertexCount;
}

// b3ClipSegmentToHullFace — clip a segment against every side plane of the reference face.
function clipSegmentToHullFace(segment: ClipVertex[], hull: HullData, refFace: number): number {
    const faces = hull.faces;
    const planes = hull.planes;
    const edges = hull.edges;
    const points = hull.points;

    const refPlane = planes[refFace];
    const face = faces[refFace];
    let edgeIndex = face.edge;

    do {
        const edge = edges[edgeIndex];
        const nextEdgeIndex = edge.next;
        const next = edges[nextEdgeIndex];

        const vertex1 = points[edge.origin];
        const vertex2 = points[next.origin];
        const tangent = vec3.normalize(vec3.sub(vertex2, vertex1));
        const binormal = vec3.cross(tangent, refPlane.normal);

        const pointCount = clipSegment(segment, plane.fromNormalAndPoint(binormal, vertex1));
        if (pointCount < 2) {
            return 0;
        }
        edgeIndex = nextEdgeIndex;
    } while (edgeIndex !== face.edge);

    return 2;
}

// --- convex_manifold.c: SAT queries ------------------------------------------------------------

function queryFaceDirectionHullAndCapsule(
    hull: HullData,
    capsule: Capsule,
    capsuleTransform: Transform,
): FaceQuery {
    let maxFaceIndex = -1;
    let maxVertexIndex = -1;
    let maxFaceSeparation = -FLT_MAX;
    const planes = hull.planes;

    const capsulePoints = [
        xf.point(capsuleTransform, capsule.center1),
        xf.point(capsuleTransform, capsule.center2),
    ];

    for (let faceIndex = 0; faceIndex < hull.faceCount; ++faceIndex) {
        const pl = planes[faceIndex];
        const vertexIndex = getPointSupport(capsulePoints, 2, vec3.neg(pl.normal));
        const support = capsulePoints[vertexIndex];
        const separation = plane.separation(pl, support);
        if (separation > maxFaceSeparation) {
            maxVertexIndex = vertexIndex;
            maxFaceIndex = faceIndex;
            maxFaceSeparation = separation;
        }
    }

    return {
        separation: maxFaceSeparation,
        faceIndex: maxFaceIndex & 0xff,
        vertexIndex: maxVertexIndex & 0xff,
    };
}

function queryFaceDirections(
    hullA: HullData,
    hullB: HullData,
    relativeTransform: Transform,
): FaceQuery {
    // All computations in local space of the second hull.
    const transform = xf.invert(relativeTransform);
    const planesA = hullA.planes;
    const pointsB = hullB.points;

    let maxFaceIndex = -1;
    let maxVertexIndex = -1;
    let maxFaceSeparation = -FLT_MAX;

    for (let faceIndex = 0; faceIndex < hullA.faceCount; ++faceIndex) {
        const pl = plane.transform(transform, planesA[faceIndex]);
        const vertexIndex = findHullSupportVertex(hullB, vec3.neg(pl.normal));
        const support = pointsB[vertexIndex];
        const separation = plane.separation(pl, support);
        if (separation > maxFaceSeparation) {
            maxFaceIndex = faceIndex;
            maxVertexIndex = vertexIndex;
            maxFaceSeparation = separation;
        }
    }

    return {
        separation: maxFaceSeparation,
        faceIndex: maxFaceIndex & 0xff,
        vertexIndex: maxVertexIndex & 0xff,
    };
}

function queryEdgeDirectionHullAndCapsule(
    hull: HullData,
    capsule: Capsule,
    capsuleTransform: Transform,
): EdgeQuery {
    let maxSeparation = -FLT_MAX;
    let maxIndex1 = -1;
    let maxIndex2 = -1;

    // All computations in local space of the hull.
    const p1 = xf.point(capsuleTransform, capsule.center1);
    const q1 = xf.point(capsuleTransform, capsule.center2);
    const e1 = vec3.sub(q1, p1);

    const edges = hull.edges;
    const points = hull.points;
    const planes = hull.planes;

    for (let index = 0; index < hull.edgeCount; index += 2) {
        const edge = edges[index];
        const twin = edges[index + 1];

        const p2 = points[edge.origin];
        const q2 = points[twin.origin];
        const e2 = vec3.sub(q2, p2);

        const u2 = planes[edge.face].normal;
        const v2 = planes[twin.face].normal;

        if (isMinkowskiFaceIsolated(u2, v2, e1)) {
            const c1 = vec3.scale(0.5, vec3.add(q1, p1));
            const c2 = hull.center;
            const separation = edgeEdgeSeparation(q1, e1, c1, q2, e2, c2);
            if (separation > maxSeparation) {
                maxSeparation = separation;
                maxIndex1 = 0;
                maxIndex2 = index;
            }
        }
    }

    return {
        separation: maxSeparation,
        indexA: maxIndex1 & 0xff,
        indexB: maxIndex2 & 0xff,
    };
}

function queryEdgeDirections(
    hullA: HullData,
    hullB: HullData,
    transformBtoA: Transform,
): EdgeQuery {
    let maxSeparation = -FLT_MAX;
    let maxIndexA = NULL_INDEX;
    let maxIndexB = NULL_INDEX;

    const edgesA = hullA.edges;
    const pointsA = hullA.points;
    const planesA = hullA.planes;
    const edgesB = hullB.edges;
    const pointsB = hullB.points;
    const planesB = hullB.planes;

    // Work in frame A.
    const matrix = mat3.fromQuat(transformBtoA.q);

    for (let indexB = 0; indexB < hullB.edgeCount; indexB += 2) {
        const edgeB = edgesB[indexB];
        const twinB = edgesB[indexB + 1];

        let qB = pointsB[twinB.origin];
        const eB = mat3.mulV(matrix, vec3.sub(qB, pointsB[edgeB.origin]));
        qB = vec3.add(mat3.mulV(matrix, qB), transformBtoA.p);

        const uB = mat3.mulV(matrix, planesB[edgeB.face].normal);
        const vB = mat3.mulV(matrix, planesB[twinB.face].normal);

        for (let indexA = 0; indexA < hullA.edgeCount; indexA += 2) {
            const edgeA = edgesA[indexA];
            const twinA = edgesA[indexA + 1];

            const qA = pointsA[twinA.origin];
            const eA = vec3.sub(qA, pointsA[edgeA.origin]);
            const uA = planesA[edgeA.face].normal;
            const vA = planesA[twinA.face].normal;

            const cba = vec3.dot(uB, eA);
            const dba = vec3.dot(vB, eA);
            const adc = -vec3.dot(uA, eB);
            const bdc = -vec3.dot(vA, eB);
            const isMink = f32(cba * dba) < 0 && f32(adc * bdc) < 0 && f32(cba * bdc) > 0;

            if (isMink) {
                const centerA = hullA.center;
                const centerB = xf.point(transformBtoA, hullB.center);
                const separation = edgeEdgeSeparation(qA, eA, centerA, qB, eB, centerB);
                if (separation > maxSeparation) {
                    maxSeparation = separation;
                    maxIndexA = indexA;
                    maxIndexB = indexB;
                }
            }
        }
    }

    return { separation: maxSeparation, indexA: maxIndexA, indexB: maxIndexB };
}

// b3ReduceManifoldPoints — reduce a clipped point set to at most 4 points using a biased extremum
// search. The C reference swap-removes the point array; this swap-removes an index array over
// `points` (see `reduceIdx`) so a reused input pool keeps its slot→object mapping.
function reduceManifoldPoints(
    manifold: LocalManifold,
    capacity: number,
    points: LocalManifoldPoint[],
    count: number,
): void {
    if (capacity < 4) {
        return;
    }

    if (count <= 4) {
        for (let i = 0; i < count; ++i) copyManifoldPointInto(manifold.points[i], points[i]);
        manifold.pointCount = count;
        return;
    }

    const normal = manifold.normal;
    const speculativeDistance = SPECULATIVE_DISTANCE;
    const tolSqr = f32(speculativeDistance * speculativeDistance);

    // A pecking-order bias for contact point consistency across time steps.
    const bias = f32(0.95);

    // Swap-remove over `reduceIdx` (not `points`) to keep the reused pool's slot→object map intact.
    const idx = reduceIdx;
    for (let i = 0; i < count; ++i) idx[i] = i;

    // Step 1: extreme point that is touching.
    let bestIndex = NULL_INDEX;
    let bestScore = -FLT_MAX;
    const searchDirection = arbitraryPerp(normal);
    for (let index = 0; index < count; ++index) {
        const pt = points[idx[index]];
        if (pt.separation > speculativeDistance) {
            continue;
        }
        // The deeper the better.
        const score = f32(-pt.separation + vec3.dot(searchDirection, pt.point));
        if (f32(bias * score) > bestScore) {
            bestIndex = index;
            bestScore = score;
        }
    }

    if (bestIndex === NULL_INDEX) {
        manifold.pointCount = 0;
        return;
    }

    copyManifoldPointInto(manifold.points[0], points[idx[bestIndex]]);
    manifold.pointCount = 1;
    idx[bestIndex] = idx[count - 1];
    count -= 1;

    const a = manifold.points[0].point;

    // Step 2: farthest point in 2D.
    bestScore = 0;
    bestIndex = NULL_INDEX;
    for (let index = 0; index < count; ++index) {
        const p = points[idx[index]].point;
        const d = vec3.sub(p, a);
        const v = vec3.mulSub(d, vec3.dot(d, normal), normal);
        const distanceSquared = vec3.lengthSq(v);
        const separation = maxf(0, -points[idx[index]].separation);
        const score = f32(distanceSquared + f32(f32(4 * separation) * separation));
        if (f32(bias * score) > bestScore) {
            bestScore = score;
            bestIndex = index;
        }
    }

    if (bestScore < tolSqr) {
        return;
    }

    copyManifoldPointInto(manifold.points[1], points[idx[bestIndex]]);
    manifold.pointCount = 2;
    idx[bestIndex] = idx[count - 1];
    count -= 1;

    const b = manifold.points[1].point;

    // Step 3: point with the maximum triangular area.
    bestScore = tolSqr;
    bestIndex = NULL_INDEX;
    let bestSignedArea = 0;
    const ba = vec3.sub(b, a);
    for (let index = 0; index < count; ++index) {
        const p = points[idx[index]].point;
        const signedArea = vec3.dot(normal, vec3.cross(ba, vec3.sub(p, a)));
        const score = absf(signedArea);
        if (f32(bias * score) >= bestScore) {
            bestScore = score;
            bestIndex = index;
            bestSignedArea = signedArea;
        }
    }

    if (bestIndex === NULL_INDEX) {
        return;
    }

    copyManifoldPointInto(manifold.points[2], points[idx[bestIndex]]);
    manifold.pointCount = 3;
    idx[bestIndex] = idx[count - 1];
    count -= 1;

    const c = manifold.points[2].point;

    // Step 4: point adding the most area outside the current triangle.
    bestScore = tolSqr;
    bestIndex = NULL_INDEX;
    const sign = bestSignedArea < 0 ? -1 : 1;
    for (let index = 0; index < count; ++index) {
        const p = points[idx[index]].point;
        const u1 = f32(sign * vec3.dot(normal, vec3.cross(vec3.sub(p, a), ba)));
        const u2 = f32(sign * vec3.dot(normal, vec3.cross(vec3.sub(p, b), vec3.sub(c, b))));
        const u3 = f32(sign * vec3.dot(normal, vec3.cross(vec3.sub(p, c), vec3.sub(a, c))));
        const score = maxf(u1, maxf(u2, u3));
        if (f32(bias * score) > bestScore) {
            bestScore = score;
            bestIndex = index;
        }
    }

    if (bestIndex !== NULL_INDEX) {
        copyManifoldPointInto(manifold.points[manifold.pointCount], points[idx[bestIndex]]);
        manifold.pointCount += 1;
    }
}

// --- sphere / capsule pair collision -----------------------------------------------------------

/** b3CollideSpheres — one-point manifold for two spheres, in frame A. */
export function collideSpheres(
    manifold: LocalManifold,
    capacity: number,
    sphereA: Sphere,
    sphereB: Sphere,
    transformBtoA: Transform,
): void {
    if (capacity === 0) {
        return;
    }

    const center1 = sphereA.center;
    const center2 = xf.point(transformBtoA, sphereB.center);

    const totalRadius = f32(sphereA.radius + sphereB.radius);
    const offset = vec3.sub(center2, center1);
    const distanceSq = vec3.lengthSq(offset);

    if (distanceSq > f32(totalRadius * totalRadius)) {
        return;
    }

    let normal: Vec3 = { x: 0, y: 1, z: 0 };
    const distance = f32(Math.sqrt(distanceSq));
    if (f32(distance * distance) > f32(1000 * FLT_MIN)) {
        normal = vec3.scale(f32(1 / distance), offset);
    }

    // Contact at the midpoint: 0.5 * (((c1 + rA*n) + c2) - rB*n).
    const point = vec3.scale(
        0.5,
        vec3.mulSub(
            vec3.add(vec3.mulAdd(center1, sphereA.radius, normal), center2),
            sphereB.radius,
            normal,
        ),
    );

    manifold.normal = normal;
    manifold.pointCount = 1;

    const pt = manifold.points[0];
    pt.point = point;
    pt.separation = f32(distance - totalRadius);
    pt.pair = singlePair();
}

/** b3CollideCapsuleAndSphere — one-point manifold for a capsule (A) and sphere (B), in frame A. */
export function collideCapsuleAndSphere(
    manifold: LocalManifold,
    capacity: number,
    capsuleA: Capsule,
    sphereB: Sphere,
    transformBtoA: Transform,
): void {
    manifold.pointCount = 0;

    if (capacity < 1) {
        return;
    }

    const center = xf.point(transformBtoA, sphereB.center);
    const center1 = capsuleA.center1;
    const center2 = capsuleA.center2;

    const totalRadius = f32(sphereB.radius + capsuleA.radius);

    const closestPoint = pointToSegmentDistance(center1, center2, center);
    const offset = vec3.sub(center, closestPoint);
    const distanceSq = vec3.lengthSq(offset);

    if (distanceSq > f32(totalRadius * totalRadius)) {
        return;
    }

    let normal: Vec3 = { x: 0, y: 1, z: 0 };
    const distance = f32(Math.sqrt(distanceSq));
    if (f32(distance * distance) > f32(1000 * FLT_MIN)) {
        normal = vec3.scale(f32(1 / distance), offset);
    }

    // Contact at the midpoint: 0.5 * (((center - sB*n) + closestPoint) + cA*n).
    const point = vec3.scale(
        0.5,
        vec3.mulAdd(
            vec3.add(vec3.mulSub(center, sphereB.radius, normal), closestPoint),
            capsuleA.radius,
            normal,
        ),
    );

    manifold.normal = normal;
    manifold.pointCount = 1;

    const pt = manifold.points[0];
    pt.point = point;
    pt.separation = f32(distance - totalRadius);
    pt.pair = singlePair();
}

/** b3CollideHullAndSphere — one-point manifold for a hull (A) and sphere (B), in frame A. */
export function collideHullAndSphere(
    manifold: LocalManifold,
    capacity: number,
    hullA: HullData,
    sphereB: Sphere,
    transformBtoA: Transform,
    cache: SimplexCache,
): void {
    manifold.pointCount = 0;

    if (capacity === 0) {
        return;
    }

    const center = xf.point(transformBtoA, sphereB.center);
    const speculativeDistance = SPECULATIVE_DISTANCE;

    const distanceInput: DistanceInput = {
        proxyA: { points: hullA.points, count: hullA.vertexCount, radius: 0 },
        proxyB: { points: [center], count: 1, radius: 0 },
        transform: xf.identity(),
        useRadii: false,
    };

    const radiusA = 0;
    const radiusB = sphereB.radius;
    const radius = f32(radiusA + radiusB);

    const distanceOutput = shapeDistance(distanceInput, cache);

    if (distanceOutput.distance > f32(radius + speculativeDistance)) {
        resetSimplexCache(cache);
        return;
    }

    if (distanceOutput.distance > f32(100 * FLT_EPSILON)) {
        // Shallow penetration.
        const normal = vec3.normalize(vec3.sub(distanceOutput.pointB, distanceOutput.pointA));
        const cA = vec3.mulAdd(
            center,
            f32(radiusA - vec3.dot(vec3.sub(center, distanceOutput.pointA), normal)),
            normal,
        );
        const cB = vec3.mulSub(center, radiusB, normal);
        const point = vec3.lerp(cA, cB, 0.5);

        manifold.normal = normal;
        manifold.pointCount = 1;

        const pt = manifold.points[0];
        pt.point = point;
        pt.separation = f32(distanceOutput.distance - radius);
        pt.pair = singlePair();
    } else {
        // Deep penetration: pick the hull face the sphere center is least behind.
        let bestIndex = -1;
        let bestDistance = -FLT_MAX;
        const planes = hullA.planes;

        for (let index = 0; index < hullA.faceCount; ++index) {
            const distance = plane.separation(planes[index], center);
            if (distance > bestDistance) {
                bestIndex = index;
                bestDistance = distance;
            }
        }

        const normal = planes[bestIndex].normal;
        const cA = vec3.mulAdd(
            center,
            f32(radiusA - vec3.dot(vec3.sub(center, distanceOutput.pointA), normal)),
            normal,
        );
        const cB = vec3.mulSub(center, radiusB, normal);
        const point = vec3.lerp(cA, cB, 0.5);

        manifold.normal = normal;
        manifold.pointCount = 1;

        const pt = manifold.points[0];
        pt.point = point;
        pt.separation = f32(bestDistance - radius);
        pt.pair = singlePair();
    }
}

/** b3CollideCapsules — up to two-point manifold for two capsules, in frame A. */
export function collideCapsules(
    manifold: LocalManifold,
    capacity: number,
    capsuleA: Capsule,
    capsuleB: Capsule,
    transformBtoA: Transform,
): void {
    manifold.pointCount = 0;

    if (capacity < 2) {
        return;
    }

    const centerA1 = capsuleA.center1;
    const centerA2 = capsuleA.center2;
    const centerB1 = xf.point(transformBtoA, capsuleB.center1);
    const centerB2 = xf.point(transformBtoA, capsuleB.center2);

    const radius = f32(capsuleA.radius + capsuleB.radius);
    const maxDistance = f32(radius + SPECULATIVE_DISTANCE);

    const result = segmentDistance(centerA1, centerA2, centerB1, centerB2);
    const offset = vec3.sub(result.point2, result.point1);
    const distanceSquared = vec3.lengthSq(offset);
    const linearSlop = LINEAR_SLOP;
    const minDistance = f32(0.01 * linearSlop);

    if (
        distanceSquared > f32(maxDistance * maxDistance) ||
        distanceSquared < f32(minDistance * minDistance)
    ) {
        return;
    }

    const segmentA = vec3.sub(centerA2, centerA1);
    const edgeAResult = getLengthAndNormalize(segmentA);
    const edgeA = edgeAResult.v;
    if (edgeAResult.length < MIN_CAPSULE_LENGTH) {
        return;
    }

    const segmentB = vec3.sub(centerB2, centerB1);
    const edgeBResult = getLengthAndNormalize(segmentB);
    const edgeB = edgeBResult.v;
    if (edgeBResult.length < MIN_CAPSULE_LENGTH) {
        return;
    }

    // Parallel edges: |eA x eB| = sin(alpha).
    const alphaTol = f32(0.05);
    const alphaTolSqr = f32(alphaTol * alphaTol);
    const axis = vec3.cross(edgeA, edgeB);

    if (vec3.lengthSq(axis) < alphaTolSqr) {
        // Clip segment B against the side planes of segment A.
        const planesA0: Plane = {
            normal: vec3.neg(edgeA),
            offset: -vec3.dot(edgeA, capsuleA.center1),
        };
        const planesA1: Plane = {
            normal: edgeA,
            offset: vec3.dot(edgeA, capsuleA.center2),
        };

        const verticesB: ClipVertex[] = [
            {
                position: centerB1,
                separation: 0,
                pair: makeFeaturePair(FeatureOwner.ShapeA, 0, FeatureOwner.ShapeA, 0),
            },
            {
                position: centerB2,
                separation: 0,
                pair: makeFeaturePair(FeatureOwner.ShapeA, 1, FeatureOwner.ShapeA, 1),
            },
        ];

        let pointCount = clipSegment(verticesB, planesA0);
        if (pointCount === 2) {
            pointCount = clipSegment(verticesB, planesA1);
        }

        if (pointCount === 2) {
            const closestPoint1 = pointToSegmentDistance(centerA1, centerA2, verticesB[0].position);
            const closestPoint2 = pointToSegmentDistance(centerA1, centerA2, verticesB[1].position);

            const distance1 = vec3.distance(closestPoint1, verticesB[0].position);
            const distance2 = vec3.distance(closestPoint2, verticesB[1].position);
            if (distance1 <= radius && distance2 <= radius) {
                if (distance1 < minDistance || distance2 < minDistance) {
                    // Avoid divide by zero.
                    return;
                }

                const normal1 = vec3.scale(
                    f32(1 / distance1),
                    vec3.sub(verticesB[0].position, closestPoint1),
                );
                const normal2 = vec3.scale(
                    f32(1 / distance2),
                    vec3.sub(verticesB[1].position, closestPoint2),
                );
                const normal = vec3.normalize(vec3.add(normal1, normal2));
                const radiusA = capsuleA.radius;
                const radiusB = capsuleB.radius;

                // Contact at the midpoint: 0.5 * (((vB.pos + rA*nK) + cP) - rB*n).
                const point1 = vec3.scale(
                    0.5,
                    vec3.mulSub(
                        vec3.add(
                            vec3.mulAdd(verticesB[0].position, radiusA, normal1),
                            closestPoint1,
                        ),
                        radiusB,
                        normal,
                    ),
                );
                const point2 = vec3.scale(
                    0.5,
                    vec3.mulSub(
                        vec3.add(
                            vec3.mulAdd(verticesB[1].position, radiusA, normal2),
                            closestPoint2,
                        ),
                        radiusB,
                        normal,
                    ),
                );

                manifold.normal = normal;
                manifold.pointCount = 2;

                const pt1 = manifold.points[0];
                pt1.point = point1;
                pt1.separation = f32(distance1 - radius);
                pt1.pair = clonePair(verticesB[0].pair);

                const pt2 = manifold.points[1];
                pt2.point = point2;
                pt2.separation = f32(distance2 - radius);
                pt2.pair = clonePair(verticesB[1].pair);

                return;
            }
        }
    }

    const distanceResult = getLengthAndNormalize(offset);
    const normal = distanceResult.v;
    const distance = distanceResult.length;
    // Contact at the midpoint 0.5 * (((p1 + rA*n) + p2) - rB*n).
    const point = vec3.scale(
        0.5,
        vec3.mulSub(
            vec3.add(vec3.mulAdd(result.point1, capsuleA.radius, normal), result.point2),
            capsuleB.radius,
            normal,
        ),
    );

    manifold.normal = normal;
    manifold.pointCount = 1;

    const pt = manifold.points[0];
    pt.point = point;
    pt.separation = f32(distance - radius);
    pt.pair = singlePair();
}

// --- hull / capsule collision ------------------------------------------------------------------

function buildHullFaceAndCapsuleContact(
    manifold: LocalManifold,
    hullA: HullData,
    capsuleB: Capsule,
    transformBtoA: Transform,
    query: FaceQuery,
): boolean {
    const planes = hullA.planes;

    const refFace = query.faceIndex;
    const refPlane = planes[refFace];

    const segmentB: ClipVertex[] = [
        {
            position: xf.point(transformBtoA, capsuleB.center1),
            separation: 0,
            pair: makeFeaturePair(FeatureOwner.ShapeA, 0, FeatureOwner.ShapeA, 0),
        },
        {
            position: xf.point(transformBtoA, capsuleB.center2),
            separation: 0,
            pair: makeFeaturePair(FeatureOwner.ShapeA, 1, FeatureOwner.ShapeA, 1),
        },
    ];

    const pointCount = clipSegmentToHullFace(segmentB, hullA, refFace);
    if (pointCount < 2) {
        return false;
    }

    const distance1 = plane.separation(refPlane, segmentB[0].position);
    const distance2 = plane.separation(refPlane, segmentB[1].position);
    const speculativeDistance = SPECULATIVE_DISTANCE;

    if (distance1 <= speculativeDistance || distance2 <= speculativeDistance) {
        const normal = refPlane.normal;
        const point1 = vec3.mulSub(
            segmentB[0].position,
            f32(0.5 * f32(distance1 + capsuleB.radius)),
            normal,
        );
        const point2 = vec3.mulSub(
            segmentB[1].position,
            f32(0.5 * f32(distance2 + capsuleB.radius)),
            normal,
        );

        manifold.normal = normal;
        manifold.pointCount = 2;

        const pt1 = manifold.points[0];
        pt1.point = point1;
        pt1.separation = f32(distance1 - capsuleB.radius);
        pt1.pair = clonePair(segmentB[0].pair);

        const pt2 = manifold.points[1];
        pt2.point = point2;
        pt2.separation = f32(distance2 - capsuleB.radius);
        pt2.pair = clonePair(segmentB[1].pair);

        return true;
    }

    return false;
}

function deepestPointSeparation(manifold: LocalManifold): number {
    let minSeparation = FLT_MAX;
    const pointCount = manifold.pointCount;
    for (let i = 0; i < pointCount; ++i) {
        minSeparation = minf(minSeparation, manifold.points[i].separation);
    }
    return minSeparation;
}

function buildHullAndCapsuleEdgeContact(
    manifold: LocalManifold,
    capacity: number,
    hullA: HullData,
    capsuleB: Capsule,
    transformBtoA: Transform,
    query: EdgeQuery,
): boolean {
    if (capacity < 1) {
        return false;
    }

    const pc = xf.point(transformBtoA, capsuleB.center1);
    const qc = xf.point(transformBtoA, capsuleB.center2);
    const ec = vec3.sub(qc, pc);

    const edges = hullA.edges;
    const points = hullA.points;

    const edge2 = edges[query.indexB];
    const twin2 = edges[edge2.twin];
    const ch = hullA.center;
    const ph = points[edge2.origin];
    const qh = points[twin2.origin];
    const eh = vec3.sub(qh, ph);

    let normal = vec3.cross(ec, eh);
    normal = vec3.normalize(normal);

    // Normal should point outward from hull.
    if (vec3.dot(normal, vec3.sub(ph, ch)) < 0) {
        normal = vec3.neg(normal);
    }

    const result = lineDistance(ph, eh, pc, ec);
    if (isWithinSegments(result) === false) {
        // Closest point beyond end points.
        return false;
    }

    const point = vec3.scale(
        0.5,
        vec3.add(vec3.mulSub(result.point1, capsuleB.radius, normal), result.point2),
    );

    const separation = vec3.dot(normal, vec3.sub(result.point2, result.point1));

    manifold.normal = normal;
    manifold.pointCount = 1;

    const pt = manifold.points[0];
    pt.point = point;
    pt.separation = f32(separation - capsuleB.radius);
    pt.pair = makeFeaturePair(FeatureOwner.ShapeA, query.indexA, FeatureOwner.ShapeB, query.indexB);
    return true;
}

/** b3CollideHullAndCapsule — up to two-point manifold for a hull (A) and capsule (B), in frame A. */
export function collideHullAndCapsule(
    manifold: LocalManifold,
    capacity: number,
    hullA: HullData,
    capsuleB: Capsule,
    transformBtoA: Transform,
    cache: SimplexCache,
): void {
    manifold.pointCount = 0;

    if (capacity < 2) {
        return;
    }

    const distanceInput: DistanceInput = {
        proxyA: { points: hullA.points, count: hullA.vertexCount, radius: 0 },
        proxyB: { points: [capsuleB.center1, capsuleB.center2], count: 2, radius: 0 },
        transform: transformBtoA,
        useRadii: false,
    };

    const distanceOutput = shapeDistance(distanceInput, cache);
    const speculativeDistance = SPECULATIVE_DISTANCE;

    if (distanceOutput.distance > f32(capsuleB.radius + speculativeDistance)) {
        resetSimplexCache(cache);
        return;
    }

    if (distanceOutput.distance > f32(100 * FLT_EPSILON)) {
        const planes = hullA.planes;

        // Shallow penetration.
        const delta = distanceOutput.normal;
        const refFace = findHullSupportFace(hullA, delta);
        const refPlane = planes[refFace];

        // Try two contact points if the closest-points difference is nearly parallel to the face.
        const kTolerance = f32(0.998);
        if (absf(vec3.dot(refPlane.normal, delta)) > kTolerance) {
            const verticesB: ClipVertex[] = [
                {
                    position: xf.point(transformBtoA, capsuleB.center1),
                    separation: 0,
                    pair: makeFeaturePair(FeatureOwner.ShapeA, 0, FeatureOwner.ShapeA, 0),
                },
                {
                    position: xf.point(transformBtoA, capsuleB.center2),
                    separation: 0,
                    pair: makeFeaturePair(FeatureOwner.ShapeA, 1, FeatureOwner.ShapeA, 1),
                },
            ];

            const pointCount = clipSegmentToHullFace(verticesB, hullA, refFace);

            if (pointCount === 2) {
                const distance1 = plane.separation(refPlane, verticesB[0].position);
                const distance2 = plane.separation(refPlane, verticesB[1].position);
                if (
                    distance1 <= f32(capsuleB.radius + speculativeDistance) ||
                    distance2 <= f32(capsuleB.radius + speculativeDistance)
                ) {
                    const normal = refPlane.normal;
                    const point1 = vec3.mulSub(
                        verticesB[0].position,
                        f32(0.5 * f32(capsuleB.radius + distance1)),
                        normal,
                    );
                    const point2 = vec3.mulSub(
                        verticesB[1].position,
                        f32(0.5 * f32(capsuleB.radius + distance2)),
                        normal,
                    );

                    manifold.normal = normal;
                    manifold.pointCount = 2;

                    const pt1 = manifold.points[0];
                    pt1.point = point1;
                    pt1.separation = f32(distance1 - capsuleB.radius);
                    pt1.pair = clonePair(verticesB[0].pair);

                    const pt2 = manifold.points[1];
                    pt2.point = point2;
                    pt2.separation = f32(distance2 - capsuleB.radius);
                    pt2.pair = clonePair(verticesB[1].pair);

                    return;
                }
            }
        }

        // Create contact from closest points.
        const point = vec3.scale(
            0.5,
            vec3.add(
                vec3.mulSub(distanceOutput.pointA, capsuleB.radius, delta),
                distanceOutput.pointB,
            ),
        );

        manifold.normal = delta;
        manifold.pointCount = 1;

        const pt = manifold.points[0];
        pt.point = point;
        pt.separation = f32(distanceOutput.distance - capsuleB.radius);
        pt.pair = singlePair();
        return;
    }

    // Deep penetration.
    const faceQuery = queryFaceDirectionHullAndCapsule(hullA, capsuleB, transformBtoA);
    if (faceQuery.separation > capsuleB.radius) {
        return;
    }

    const edgeQuery = queryEdgeDirectionHullAndCapsule(hullA, capsuleB, transformBtoA);
    if (edgeQuery.separation > capsuleB.radius) {
        return;
    }

    // Create face contact.
    let faceSeparation = f32(faceQuery.separation - capsuleB.radius);
    buildHullFaceAndCapsuleContact(manifold, hullA, capsuleB, transformBtoA, faceQuery);
    if (manifold.pointCount > 1) {
        faceSeparation = deepestPointSeparation(manifold);
    }

    // Create edge contact if face contact fails or edge contact is significantly better.
    const kRelEdgeTolerance = f32(0.9);
    const kAbsTolerance = f32(0.5 * LINEAR_SLOP);
    const edgeSeparation = f32(edgeQuery.separation - capsuleB.radius);
    if (
        manifold.pointCount === 0 ||
        edgeSeparation > f32(f32(kRelEdgeTolerance * faceSeparation) + kAbsTolerance)
    ) {
        buildHullAndCapsuleEdgeContact(
            manifold,
            capacity,
            hullA,
            capsuleB,
            transformBtoA,
            edgeQuery,
        );
    }
}

// --- hull / hull collision ---------------------------------------------------------------------

// b3BuildPolygon — the incident face of `hull` transformed into frame A as a clip polygon, written
// into the `polyBuffer` scratch. Returns the vertex count.
function buildPolygon(
    transform: Transform,
    hull: HullData,
    incFace: number,
    refPlane: Plane,
): number {
    const faces = hull.faces;
    const edges = hull.edges;
    const points = hull.points;

    const face = faces[incFace];
    let edgeIndex = face.edge;

    const matrix = mat3.fromQuatOut(transform.q, polyMatrix);
    let k = 0;

    do {
        const edge = edges[edgeIndex];
        const nextEdgeIndex = edge.next;
        const next = edges[nextEdgeIndex];

        const v = polyBuffer[k];
        const position = v.position;
        mat3.mulVOut(matrix, points[next.origin], position);
        vec3.addOut(position, transform.p, position);
        v.separation = plane.separation(refPlane, position);
        v.pair.owner1 = FeatureOwner.ShapeB;
        v.pair.index1 = edgeIndex & 0xff;
        v.pair.owner2 = FeatureOwner.ShapeB;
        v.pair.index2 = nextEdgeIndex & 0xff;
        ++k;

        edgeIndex = nextEdgeIndex;
    } while (edgeIndex !== face.edge && k < MAX_CLIP_POINTS);

    return k;
}

function buildFaceAContact(
    manifold: LocalManifold,
    capacity: number,
    hullA: HullData,
    hullB: HullData,
    transformBtoA: Transform,
    query: FaceQuery,
    cache: SATCache,
): boolean {
    const facesA = hullA.faces;
    const edgesA = hullA.edges;
    const planesA = hullA.planes;
    const pointsA = hullA.points;

    // Reference face.
    const refFace = query.faceIndex;
    const refPlane = planesA[refFace];

    // Find incident face.
    const refNormalInB = quat.invRotate(transformBtoA.q, refPlane.normal);
    const incFace = findIncidentFace(hullB, refNormalInB, query.vertexIndex);

    // Build clip polygon from incident face in frame A.
    let pointCount = buildPolygon(transformBtoA, hullB, incFace, refPlane);
    let input = polyBuffer;

    // Clip incident face against side planes of the reference face.
    const face = facesA[refFace];
    let edgeIndex = face.edge;

    do {
        const edge = edgesA[edgeIndex];
        const nextEdgeIndex = edge.next;
        const next = edgesA[nextEdgeIndex];
        const vertex1 = pointsA[edge.origin];
        const vertex2 = pointsA[next.origin];
        const tangent = vec3.normalize(vec3.sub(vertex2, vertex1));
        const binormal = vec3.cross(tangent, refPlane.normal);

        const clipPlane = plane.fromNormalAndPoint(binormal, vertex1);

        const out = otherClipBuffer(input);
        pointCount = clipPolygon(input, pointCount, clipPlane, edgeIndex, refPlane, out);
        input = out;

        if (pointCount < 3) {
            resetSATCache(cache);
            return false;
        }

        edgeIndex = nextEdgeIndex;
    } while (edgeIndex !== face.edge);

    pointCount = minInt(pointCount, MAX_CLIP_POINTS);

    let minSeparation = FLT_MAX;

    manifold.normal = refPlane.normal;

    for (let i = 0; i < pointCount; ++i) {
        const clipPoint = input[i];
        const dst = reducePoints[i];
        // Half-way point keeps positions stable when swapping the reference face from A to B.
        vec3.mulSubOut(
            clipPoint.position,
            f32(0.5 * clipPoint.separation),
            refPlane.normal,
            dst.point,
        );
        dst.separation = clipPoint.separation;
        copyPair(dst.pair, clipPoint.pair);
        dst.triangleIndex = 0;
        minSeparation = minf(minSeparation, clipPoint.separation);
    }

    if (minSeparation >= SPECULATIVE_DISTANCE) {
        resetSATCache(cache);
        return false;
    }

    reduceManifoldPoints(manifold, capacity, reducePoints, pointCount);

    cache.separation = minSeparation;
    cache.type = SeparatingFeature.FaceAxisA;
    cache.indexA = query.faceIndex & 0xff;
    cache.indexB = query.vertexIndex & 0xff;

    return true;
}

function buildFaceBContact(
    manifold: LocalManifold,
    capacity: number,
    hullA: HullData,
    hullB: HullData,
    transformBtoA: Transform,
    query: FaceQuery,
    cache: SATCache,
): boolean {
    const transformAtoB = xf.invert(transformBtoA);
    const touching = buildFaceAContact(
        manifold,
        capacity,
        hullB,
        hullA,
        transformAtoB,
        query,
        cache,
    );
    if (touching === false) {
        return false;
    }

    // Results are in frame B; transform them into frame A.
    const matrix = mat3.fromQuat(transformBtoA.q);

    // Flip normal so it points from A to B, even though B owns the reference face.
    manifold.normal = vec3.neg(mat3.mulV(matrix, manifold.normal));
    cache.type = SeparatingFeature.FaceAxisB;
    cache.indexA = query.vertexIndex & 0xff;
    cache.indexB = query.faceIndex & 0xff;

    for (let i = 0; i < manifold.pointCount; ++i) {
        const pt = manifold.points[i];
        pt.point = vec3.add(mat3.mulV(matrix, pt.point), transformBtoA.p);
        pt.pair = flipPair(pt.pair);
    }

    return true;
}

function buildEdgeContact(
    manifold: LocalManifold,
    hullA: HullData,
    hullB: HullData,
    transformBtoA: Transform,
    query: EdgeQuery,
    cache: SATCache,
): boolean {
    const edgesA = hullA.edges;
    const pointsA = hullA.points;
    const edgesB = hullB.edges;
    const pointsB = hullB.points;

    const edgeA = edgesA[query.indexA];
    const twinA = edgesA[edgeA.twin];
    const centerA = hullA.center;
    const pA = pointsA[edgeA.origin];
    const qA = pointsA[twinA.origin];
    const eA = vec3.sub(qA, pA);

    const edgeB = edgesB[query.indexB];
    const twinB = edgesB[edgeB.twin];
    const pB = xf.point(transformBtoA, pointsB[edgeB.origin]);
    const qB = xf.point(transformBtoA, pointsB[twinB.origin]);
    const eB = vec3.sub(qB, pB);

    let normal = vec3.cross(eA, eB);
    normal = vec3.normalize(normal);

    if (vec3.dot(normal, vec3.sub(pA, centerA)) < 0) {
        normal = vec3.neg(normal);
    }

    const result = lineDistance(pA, eA, pB, eB);

    if (isWithinSegments(result) === false) {
        resetSATCache(cache);
        return false;
    }

    // This can slide off the end from caching.
    const separation = vec3.dot(normal, vec3.sub(result.point2, result.point1));
    const point = vec3.scale(0.5, vec3.add(result.point1, result.point2));

    manifold.normal = normal;
    manifold.pointCount = 1;

    const pt = manifold.points[0];
    pt.point = point;
    pt.separation = separation;
    pt.pair = makeFeaturePair(FeatureOwner.ShapeA, query.indexA, FeatureOwner.ShapeB, query.indexB);

    cache.separation = separation;
    cache.type = SeparatingFeature.EdgePairAxis;
    cache.indexA = query.indexA & 0xff;
    cache.indexB = query.indexB & 0xff;

    return true;
}

/** b3CollideHulls — up to four-point manifold for two convex hulls, in frame A, with SAT cache. */
export function collideHulls(
    manifold: LocalManifold,
    capacity: number,
    hullA: HullData,
    hullB: HullData,
    transformBtoA: Transform,
    cache: SATCache,
): void {
    manifold.pointCount = 0;

    if (capacity < 4) {
        return;
    }

    const speculativeDistance = SPECULATIVE_DISTANCE;
    const linearSlop = LINEAR_SLOP;
    const edgesA = hullA.edges;
    const planesA = hullA.planes;
    const pointsA = hullA.points;
    const edgesB = hullB.edges;
    const planesB = hullB.planes;
    const pointsB = hullB.points;

    // Attempt to use the cache to speed up collision.
    switch (cache.type) {
        case SeparatingFeature.Invalid:
            resetSATCache(cache);
            break;

        case SeparatingFeature.FaceAxisA: {
            const pl = planesA[cache.indexA];
            const searchDirectionInB = vec3.neg(quat.invRotate(transformBtoA.q, pl.normal));
            const vertexIndex = findHullSupportVertex(hullB, searchDirectionInB);
            const support = xf.point(transformBtoA, pointsB[vertexIndex]);
            const separation = plane.separation(pl, support);

            if (separation >= speculativeDistance) {
                return;
            }

            {
                const faceQuery: FaceQuery = {
                    separation: 0,
                    faceIndex: cache.indexA,
                    vertexIndex,
                };
                const localCache = emptySATCache();
                const touching = buildFaceAContact(
                    manifold,
                    capacity,
                    hullA,
                    hullB,
                    transformBtoA,
                    faceQuery,
                    localCache,
                );
                if (
                    touching === true &&
                    absf(f32(cache.separation - localCache.separation)) < linearSlop
                ) {
                    return;
                }
            }
            break;
        }

        case SeparatingFeature.FaceAxisB: {
            const pl = planesB[cache.indexB];
            const searchDirectionInA = vec3.neg(quat.rotate(transformBtoA.q, pl.normal));
            const vertexIndex = findHullSupportVertex(hullA, searchDirectionInA);
            const support = xf.invPoint(transformBtoA, pointsA[vertexIndex]);
            const separation = plane.separation(pl, support);

            if (separation >= speculativeDistance) {
                return;
            }

            {
                const faceQuery: FaceQuery = {
                    separation: 0,
                    faceIndex: cache.indexB,
                    vertexIndex,
                };
                const localCache = emptySATCache();
                const touching = buildFaceBContact(
                    manifold,
                    capacity,
                    hullA,
                    hullB,
                    transformBtoA,
                    faceQuery,
                    localCache,
                );
                if (
                    touching === true &&
                    absf(f32(cache.separation - localCache.separation)) < linearSlop
                ) {
                    return;
                }
            }
            break;
        }

        case SeparatingFeature.EdgePairAxis: {
            const index1 = cache.indexA;
            const edge1 = edgesA[index1];
            const twin1 = edgesA[index1 + 1];

            const p1 = pointsA[edge1.origin];
            const q1 = pointsA[twin1.origin];
            const e1 = vec3.sub(q1, p1);

            const u1 = planesA[edge1.face].normal;
            const v1 = planesA[twin1.face].normal;

            const index2 = cache.indexB;
            const edge2 = edgesB[index2];
            const twin2 = edgesB[index2 + 1];

            const p2 = xf.point(transformBtoA, pointsB[edge2.origin]);
            const q2 = xf.point(transformBtoA, pointsB[twin2.origin]);
            const e2 = vec3.sub(q2, p2);

            const u2 = quat.rotate(transformBtoA.q, planesB[edge2.face].normal);
            const v2 = quat.rotate(transformBtoA.q, planesB[twin2.face].normal);

            const isMink = isMinkowskiFace(u1, v1, e1, vec3.neg(u2), vec3.neg(v2), e2);
            if (isMink === true) {
                const c1 = hullA.center;
                const c2 = xf.point(transformBtoA, hullB.center);

                const separation = edgeEdgeSeparation(p1, e1, c1, p2, e2, c2);
                if (separation > speculativeDistance) {
                    return;
                }

                {
                    const edgeQuery: EdgeQuery = {
                        indexA: cache.indexA,
                        indexB: cache.indexB,
                        separation: 0,
                    };
                    const localCache = emptySATCache();
                    const touching = buildEdgeContact(
                        manifold,
                        hullA,
                        hullB,
                        transformBtoA,
                        edgeQuery,
                        localCache,
                    );
                    if (
                        touching &&
                        absf(f32(cache.separation - localCache.separation)) < linearSlop
                    ) {
                        return;
                    }
                }
            }
            break;
        }

        // Manual axes are for testing.
        case SeparatingFeature.ManualFaceAxisA: {
            const faceQueryA = queryFaceDirections(hullA, hullB, transformBtoA);
            buildFaceAContact(manifold, capacity, hullA, hullB, transformBtoA, faceQueryA, cache);
            return;
        }

        case SeparatingFeature.ManualFaceAxisB: {
            const faceQueryB = queryFaceDirections(hullB, hullA, xf.invert(transformBtoA));
            buildFaceBContact(manifold, capacity, hullA, hullB, transformBtoA, faceQueryB, cache);
            return;
        }

        case SeparatingFeature.ManualEdgePairAxis: {
            const edgeQuery = queryEdgeDirections(hullA, hullB, transformBtoA);
            if (edgeQuery.indexA !== NULL_INDEX) {
                buildEdgeContact(manifold, hullA, hullB, transformBtoA, edgeQuery, cache);
            }
            return;
        }

        default:
            break;
    }

    manifold.pointCount = 0;
    resetSATCache(cache);

    // Find axis of minimum penetration.
    const faceQueryA = queryFaceDirections(hullA, hullB, transformBtoA);
    if (faceQueryA.separation > speculativeDistance) {
        cache.separation = faceQueryA.separation;
        cache.type = SeparatingFeature.FaceAxisA;
        cache.indexA = faceQueryA.faceIndex & 0xff;
        cache.indexB = faceQueryA.vertexIndex & 0xff;
        return;
    }

    const faceQueryB = queryFaceDirections(hullB, hullA, xf.invert(transformBtoA));
    if (faceQueryB.separation > speculativeDistance) {
        cache.separation = faceQueryB.separation;
        cache.type = SeparatingFeature.FaceAxisB;
        cache.indexA = faceQueryB.vertexIndex & 0xff;
        cache.indexB = faceQueryB.faceIndex & 0xff;
        return;
    }

    const edgeQuery = queryEdgeDirections(hullA, hullB, transformBtoA);
    if (edgeQuery.separation > speculativeDistance) {
        cache.separation = edgeQuery.separation;
        cache.type = SeparatingFeature.EdgePairAxis;
        cache.indexA = edgeQuery.indexA & 0xff;
        cache.indexB = edgeQuery.indexB & 0xff;
        return;
    }

    // Always build a face contact (e.g. Jenga problem).
    const faceSeparationA = faceQueryA.separation;
    const faceSeparationB = faceQueryB.separation;

    if (faceSeparationB > f32(faceSeparationA + f32(0.5 * linearSlop))) {
        buildFaceBContact(manifold, capacity, hullA, hullB, transformBtoA, faceQueryB, cache);
    } else {
        buildFaceAContact(manifold, capacity, hullA, hullB, transformBtoA, faceQueryA, cache);
    }

    if (edgeQuery.indexA === NULL_INDEX) {
        // No valid edge pairs (all edges parallel).
        return;
    }

    const clippedFaceSeparation = cache.separation;

    // Create edge contact if face contact fails or edge contact is significantly better.
    const kRelEdgeTolerance = f32(0.9);
    const kAbsTolerance = f32(0.5 * linearSlop);

    if (
        manifold.pointCount === 0 ||
        edgeQuery.separation > f32(f32(kRelEdgeTolerance * clippedFaceSeparation) + kAbsTolerance)
    ) {
        const edgeManifold = scratchEdgeManifold;
        // buildEdgeContact leaves pointCount untouched on its miss path, so clear the reused buffer.
        edgeManifold.pointCount = 0;

        buildEdgeContact(edgeManifold, hullA, hullB, transformBtoA, edgeQuery, cache);

        if (edgeManifold.pointCount === 1) {
            // Copy the edge manifold out, preserving the caller's point buffer.
            manifold.normal = edgeManifold.normal;
            manifold.pointCount = edgeManifold.pointCount;
            copyManifoldPointInto(manifold.points[0], edgeManifold.points[0]);
        }
    }
}
