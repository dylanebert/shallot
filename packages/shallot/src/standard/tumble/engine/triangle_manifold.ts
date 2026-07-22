// Triangle narrowphase — ports triangle_manifold.c from Box3D (Erin Catto, MIT). Sphere/capsule/
// hull versus a single triangle, producing a manifold in the primary shape's local frame. These
// feed the mesh/height-field multi-manifold driver (mesh_contact). The hull path reuses the SAT
// cache and the shared clip/query helpers from manifold.ts. fround discipline per the README.

import { type DistanceInput, emptyCache, type SimplexCache, shapeDistance } from "./distance";
import type { Capsule, Sphere } from "./geometry";
import type { HullData } from "./hull";
import { findHullSupportVertex } from "./hull";
import {
    type ClipVertex,
    clipPolygon,
    cloneClipVertex,
    type EdgeQuery,
    edgeEdgeSeparation,
    type FaceQuery,
    FeatureOwner,
    findIncidentFace,
    flipPair,
    type LocalManifold,
    makeFeaturePair,
    otherClipBuffer,
    type SATCache,
    SeparatingFeature,
    singlePair,
    TriangleFeature,
} from "./manifold";
import {
    absf,
    FLT_EPSILON,
    FLT_MAX,
    FLT_MIN,
    f32,
    lineDistance,
    maxf,
    minf,
    minInt,
    type Plane,
    plane,
    type Vec3,
    vec3,
    xf,
} from "./math";

// B3_LINEAR_SLOP with the default unit length of 1 (mirrors manifold.ts — the narrowphase frame).
const LINEAR_SLOP = f32(0.005);
const SPECULATIVE_DISTANCE = f32(4 * LINEAR_SLOP);
const MAX_CLIP_POINTS = 64;
const NULL_INDEX = -1;

// A triangle prepared for hull SAT (b3TriangleData): the three vertices, the three edges, the
// centroid, and the plane.
type TriangleData = {
    v1: Vec3;
    v2: Vec3;
    v3: Vec3;
    e1: Vec3;
    e2: Vec3;
    e3: Vec3;
    center: Vec3;
    plane: Plane;
};

// b3SATCache is caller-owned; the C reset is `*cache = {0}` (in place). Mirror that, don't replace.
function resetCache(cache: SATCache): void {
    cache.separation = 0;
    cache.type = 0;
    cache.indexA = 0;
    cache.indexB = 0;
    cache.hit = 0;
}

// Value copy of a SAT cache — for the "read the cache but don't modify it" reuse attempts.
function copyCache(c: SATCache): SATCache {
    return {
        separation: c.separation,
        type: c.type,
        indexA: c.indexA,
        indexB: c.indexB,
        hit: c.hit,
    };
}

// --- closest point on triangle (math_functions.c) ----------------------------------------------

type TrianglePoint = { point: Vec3; feature: number };

// b3ClosestPointOnTriangle — Voronoi-region closest point of `q` on triangle abc, plus the feature.
function closestPointOnTriangle(a: Vec3, b: Vec3, c: Vec3, q: Vec3): TrianglePoint {
    const ab = vec3.sub(b, a);
    const ac = vec3.sub(c, a);
    const aq = vec3.sub(q, a);

    const d1 = vec3.dot(ab, aq);
    const d2 = vec3.dot(ac, aq);
    if (d1 <= 0 && d2 <= 0) {
        return { point: a, feature: TriangleFeature.Vertex1 };
    }

    const bq = vec3.sub(q, b);
    const d3 = vec3.dot(ab, bq);
    const d4 = vec3.dot(ac, bq);
    if (d3 > 0 && d4 <= d3) {
        return { point: b, feature: TriangleFeature.Vertex2 };
    }

    const vc = f32(f32(d1 * d4) - f32(d3 * d2));
    if (vc <= 0 && d1 >= 0 && d3 <= 0) {
        const t = f32(d1 / f32(d1 - d3));
        return { point: vec3.mulAdd(a, t, ab), feature: TriangleFeature.Edge1 };
    }

    const cq = vec3.sub(q, c);
    const d5 = vec3.dot(ab, cq);
    const d6 = vec3.dot(ac, cq);
    if (d6 >= 0 && d5 <= d6) {
        return { point: c, feature: TriangleFeature.Vertex3 };
    }

    const vb = f32(f32(d5 * d2) - f32(d1 * d6));
    if (vb <= 0 && d2 >= 0 && d6 <= 0) {
        const t = f32(d2 / f32(d2 - d6));
        return { point: vec3.mulAdd(a, t, ac), feature: TriangleFeature.Edge3 };
    }

    const va = f32(f32(d3 * d6) - f32(d5 * d4));
    if (va <= 0 && d4 >= d3 && d5 >= d6) {
        const bc = vec3.sub(c, b);
        const t = f32(f32(d4 - d3) / f32(f32(d4 - d3) + f32(d5 - d6)));
        return { point: vec3.mulAdd(b, t, bc), feature: TriangleFeature.Edge2 };
    }

    const denom = f32(1 / f32(f32(va + vb) + vc));
    const t1 = f32(vb * denom);
    const t2 = f32(vc * denom);
    let p = vec3.mulAdd(a, t1, ab);
    p = vec3.mulAdd(p, t2, ac);
    return { point: p, feature: TriangleFeature.TriangleFace };
}

// --- triangle feature from a simplex cache -----------------------------------------------------

// Indexed by the 3-bit vertex mask (b3GetTriangleFeature's s_triangleFeatures).
const TRIANGLE_FEATURES = [
    TriangleFeature.None, // 000 (unreachable)
    TriangleFeature.Vertex1, // 001
    TriangleFeature.Vertex2, // 010
    TriangleFeature.Edge1, // 011  v1,v2
    TriangleFeature.Vertex3, // 100
    TriangleFeature.Edge3, // 101  v1,v3
    TriangleFeature.Edge2, // 110  v2,v3
    TriangleFeature.TriangleFace, // 111
] as const;

function getTriangleFeature(cache: SimplexCache): number {
    const count = cache.count;
    let mask = 0;
    for (let i = 0; i < count; ++i) {
        mask |= 1 << cache.indexA[i];
    }
    return TRIANGLE_FEATURES[mask];
}

// --- sphere vs triangle ------------------------------------------------------------------------

/** b3CollideSphereAndTriangle — sphere A against triangle B, manifold in A's frame. */
export function collideSphereAndTriangle(
    manifold: LocalManifold,
    capacity: number,
    sphereA: Sphere,
    v1: Vec3,
    v2: Vec3,
    v3: Vec3,
): void {
    manifold.pointCount = 0;

    if (capacity === 0) {
        return;
    }

    const center = sphereA.center;
    const pl = plane.fromPoints(v1, v2, v3);

    const offset = plane.separation(pl, center);
    if (offset < 0) {
        // Cull back side collision
        return;
    }

    const closest = closestPointOnTriangle(v1, v2, v3, center);

    const squaredDistance = vec3.distanceSquared(closest.point, center);
    const maxDistance = f32(sphereA.radius + SPECULATIVE_DISTANCE);
    if (squaredDistance > f32(maxDistance * maxDistance)) {
        return;
    }

    const distance = f32(Math.sqrt(squaredDistance));
    let normal: Vec3;
    if (f32(distance * distance) > f32(1000 * FLT_MIN)) {
        normal = vec3.scale(f32(1 / distance), vec3.sub(center, closest.point));
    } else {
        normal = vec3.normalize(vec3.cross(vec3.sub(v2, v1), vec3.sub(v3, v1)));
    }

    // contact point mid-way
    const contactPoint = vec3.scale(
        f32(0.5),
        vec3.add(vec3.sub(center, vec3.scale(sphereA.radius, normal)), closest.point),
    );

    manifold.normal = normal;
    manifold.pointCount = 1;
    manifold.feature = closest.feature;
    manifold.squaredDistance = squaredDistance;

    const mp = manifold.points[0];
    mp.point = contactPoint;
    mp.separation = f32(distance - sphereA.radius);
    mp.pair = singlePair();
}

// --- capsule vs triangle -----------------------------------------------------------------------

// b3ClipSegmentToTriangleFace — clip a 2-vertex segment against the triangle's three side planes,
// in place. Returns false if any clip leaves other than two points.
function clipSegmentToTriangleFace(segment: ClipVertex[], points: Vec3[], pl: Plane): boolean {
    let vertex1 = points[2];
    for (let i = 0; i < 3; ++i) {
        const vertex2 = points[i];
        const tangent = vec3.normalize(vec3.sub(vertex2, vertex1));
        const binormal = vec3.cross(tangent, pl.normal);
        const clipPlane = plane.fromNormalAndPoint(binormal, vertex1);

        let vertexCount = 0;
        const p1 = cloneClipVertex(segment[0]);
        const p2 = cloneClipVertex(segment[1]);

        const distance1 = plane.separation(clipPlane, p1.position);
        const distance2 = plane.separation(clipPlane, p2.position);

        if (distance1 <= 0) {
            segment[vertexCount++] = p1;
        }
        if (distance2 <= 0) {
            segment[vertexCount++] = p2;
        }

        if (f32(distance1 * distance2) < 0) {
            const t = f32(distance1 / f32(distance1 - distance2));
            segment[vertexCount] = {
                position: vec3.lerp(p1.position, p2.position, t),
                separation: 0,
                pair: distance1 > 0 ? p1.pair : p2.pair,
            };
            vertexCount++;
        }

        if (vertexCount !== 2) {
            return false;
        }

        vertex1 = vertex2;
    }

    return true;
}

function queryTriangleFaceAndCapsule(pl: Plane, capsule: Capsule): FaceQuery {
    const separation1 = plane.separation(pl, capsule.center1);
    const separation2 = plane.separation(pl, capsule.center2);

    if (separation1 < separation2) {
        return { separation: separation1, faceIndex: 0, vertexIndex: 0 };
    }
    return { separation: separation2, faceIndex: 0, vertexIndex: 1 };
}

function queryTriangleAndCapsuleEdges(vertices: Vec3[], capsule: Capsule): EdgeQuery {
    const p1 = capsule.center1;
    const p2 = capsule.center2;
    const capsuleEdge = vec3.sub(p2, p1);
    const capsuleCenter = vec3.lerp(p1, p2, f32(0.5));

    const triangleCenter = vec3.scale(
        f32(1 / 3),
        vec3.add(vertices[0], vec3.add(vertices[1], vertices[2])),
    );

    let maxSeparation = -FLT_MAX;
    let maxIndex1 = 0xff;
    const maxIndex2 = 0;

    let edgeIndex = 2;
    let vA = vertices[2];
    for (let index = 0; index < 3; ++index) {
        const vB = vertices[index];
        const triangleEdge = vec3.sub(vB, vA);
        const separation = edgeEdgeSeparation(
            p1,
            capsuleEdge,
            capsuleCenter,
            vA,
            triangleEdge,
            triangleCenter,
        );
        if (separation > maxSeparation) {
            maxSeparation = separation;
            maxIndex1 = edgeIndex;
        }
        vA = vB;
        edgeIndex = index;
    }

    return { separation: maxSeparation, indexA: maxIndex1, indexB: maxIndex2 };
}

function buildTriangleAndCapsuleFaceContact(
    manifold: LocalManifold,
    triangle: Vec3[],
    pl: Plane,
    capsule: Capsule,
): void {
    const segment: ClipVertex[] = [
        {
            position: capsule.center1,
            separation: 0,
            pair: makeFeaturePair(FeatureOwner.ShapeA, 0, FeatureOwner.ShapeA, 0),
        },
        {
            position: capsule.center2,
            separation: 0,
            pair: makeFeaturePair(FeatureOwner.ShapeA, 1, FeatureOwner.ShapeA, 1),
        },
    ];

    const havePoints = clipSegmentToTriangleFace(segment, triangle, pl);
    if (havePoints === false) {
        return;
    }

    const radius = capsule.radius;
    const distance1 = plane.separation(pl, segment[0].position);
    const distance2 = plane.separation(pl, segment[1].position);

    if (
        distance1 > f32(SPECULATIVE_DISTANCE + radius) &&
        distance2 > f32(SPECULATIVE_DISTANCE + radius)
    ) {
        return;
    }

    const point1 = vec3.mulSub(segment[0].position, f32(0.5 * f32(distance1 + radius)), pl.normal);
    const point2 = vec3.mulSub(segment[1].position, f32(0.5 * f32(distance2 + radius)), pl.normal);

    manifold.normal = pl.normal;
    manifold.feature = TriangleFeature.TriangleFace;
    manifold.pointCount = 2;

    const pt0 = manifold.points[0];
    pt0.point = point1;
    pt0.separation = f32(distance1 - radius);
    pt0.pair = segment[0].pair;

    const pt1 = manifold.points[1];
    pt1.point = point2;
    pt1.separation = f32(distance2 - radius);
    pt1.pair = segment[1].pair;
}

const EDGE_FEATURES = [TriangleFeature.Edge1, TriangleFeature.Edge2, TriangleFeature.Edge3];

function buildTriangleAndCapsuleEdgeContact(
    manifold: LocalManifold,
    triangle: Vec3[],
    capsule: Capsule,
    query: EdgeQuery,
): void {
    const p1 = capsule.center1;
    const p2 = capsule.center2;
    const capsuleEdge = vec3.sub(p2, p1);

    const triangleCenter = vec3.scale(
        f32(1 / 3),
        vec3.add(triangle[0], vec3.add(triangle[1], triangle[2])),
    );
    const vA = triangle[query.indexA];
    const vB = triangle[(query.indexA + 1) % 3];
    const triangleEdge = vec3.sub(vB, vA);

    let normal = vec3.normalize(vec3.cross(capsuleEdge, triangleEdge));

    // Normal should point away from triangle center
    if (vec3.dot(normal, vec3.sub(vA, triangleCenter)) < 0) {
        normal = vec3.neg(normal);
    }

    const result = lineDistance(vA, triangleEdge, p1, capsuleEdge);

    if (
        result.fraction1 < 0 ||
        1 < result.fraction1 ||
        result.fraction2 < 0 ||
        1 < result.fraction2
    ) {
        // closest point beyond end points
        return;
    }

    const point = vec3.lerp(
        vec3.mulSub(result.point1, capsule.radius, normal),
        result.point2,
        f32(0.5),
    );

    const separation = vec3.dot(normal, vec3.sub(result.point2, result.point1));

    manifold.normal = normal;
    manifold.pointCount = 1;
    manifold.feature = EDGE_FEATURES[query.indexA];

    const pt = manifold.points[0];
    pt.point = point;
    pt.separation = f32(separation - capsule.radius);
    pt.pair = makeFeaturePair(FeatureOwner.ShapeA, query.indexA, FeatureOwner.ShapeB, query.indexB);
}

/** b3CollideCapsuleAndTriangle — capsule A against triangle B, manifold in A's frame. */
export function collideCapsuleAndTriangle(
    manifold: LocalManifold,
    capacity: number,
    capsuleA: Capsule,
    v1: Vec3,
    v2: Vec3,
    v3: Vec3,
    cache: SimplexCache,
): void {
    manifold.pointCount = 0;

    if (capacity < 2) {
        return;
    }

    const triangleB = [v1, v2, v3];
    const pl = plane.fromPoints(v1, v2, v3);
    const capsuleCenter = vec3.lerp(capsuleA.center1, capsuleA.center2, f32(0.5));

    const offset = plane.separation(pl, capsuleCenter);
    if (offset < 0) {
        // Cull back side collision
        return;
    }

    const distanceInput: DistanceInput = {
        proxyA: { points: triangleB, count: 3, radius: 0 },
        proxyB: { points: [capsuleA.center1, capsuleA.center2], count: 2, radius: 0 },
        transform: xf.identity(),
        useRadii: false,
    };

    const distanceOutput = shapeDistance(distanceInput, cache);

    const radius = capsuleA.radius;
    if (distanceOutput.distance > f32(radius + SPECULATIVE_DISTANCE)) {
        // Shapes are separated, persist the cache
        return;
    }

    if (distanceOutput.distance > f32(100 * FLT_EPSILON)) {
        // Shallow penetration
        const delta = vec3.normalize(vec3.sub(distanceOutput.pointB, distanceOutput.pointA));

        // Try to create two contact points if closest points difference is nearly parallel to face normal
        const kTolerance = f32(0.2);
        const cosAngle = absf(vec3.dot(pl.normal, delta));
        if (cosAngle > kTolerance) {
            const segment: ClipVertex[] = [
                {
                    position: capsuleA.center1,
                    separation: 0,
                    pair: makeFeaturePair(FeatureOwner.ShapeA, 0, FeatureOwner.ShapeA, 0),
                },
                {
                    position: capsuleA.center2,
                    separation: 0,
                    pair: makeFeaturePair(FeatureOwner.ShapeA, 1, FeatureOwner.ShapeA, 1),
                },
            ];

            const havePoints = clipSegmentToTriangleFace(segment, triangleB, pl);

            if (havePoints === true) {
                const distance1 = plane.separation(pl, segment[0].position);
                const distance2 = plane.separation(pl, segment[1].position);

                const normal = pl.normal;
                const point1 = vec3.mulSub(
                    segment[0].position,
                    f32(0.5 * f32(radius + distance1)),
                    normal,
                );
                const point2 = vec3.mulSub(
                    segment[1].position,
                    f32(0.5 * f32(radius + distance2)),
                    normal,
                );

                manifold.normal = normal;
                manifold.feature = TriangleFeature.TriangleFace;
                manifold.pointCount = 2;

                const mp0 = manifold.points[0];
                mp0.point = point1;
                mp0.separation = f32(distance1 - radius);
                mp0.pair = segment[0].pair;

                const mp1 = manifold.points[1];
                mp1.point = point2;
                mp1.separation = f32(distance2 - radius);
                mp1.pair = segment[1].pair;

                return;
            }
        }

        // Create contact from closest points
        const point = vec3.scale(
            f32(0.5),
            vec3.add(vec3.mulSub(distanceOutput.pointA, radius, delta), distanceOutput.pointB),
        );

        manifold.normal = delta;
        manifold.pointCount = 1;
        manifold.feature = getTriangleFeature(cache);

        const mp = manifold.points[0];
        mp.point = point;
        mp.separation = f32(distanceOutput.distance - radius);
        mp.pair = singlePair();

        return;
    }

    // Deep penetration

    const faceQuery = queryTriangleFaceAndCapsule(pl, capsuleA);
    if (faceQuery.separation > radius) {
        return;
    }

    const edgeQuery = queryTriangleAndCapsuleEdges(triangleB, capsuleA);
    if (edgeQuery.separation > radius) {
        return;
    }

    // Create face contact
    let faceSeparation = f32(faceQuery.separation - radius);
    buildTriangleAndCapsuleFaceContact(manifold, triangleB, pl, capsuleA);
    if (manifold.pointCount === 2) {
        faceSeparation = minf(manifold.points[0].separation, manifold.points[1].separation);
    }

    // Face contact can be empty if it does not realize the axis of minimum penetration.
    // Create edge contact if face contact fails or edge contact is significantly better.
    const kRelEdgeTolerance = f32(0.5);
    const kAbsTolerance = f32(1 * LINEAR_SLOP);
    const edgeSeparation = f32(edgeQuery.separation - radius);
    if (
        manifold.pointCount === 0 ||
        edgeSeparation > f32(f32(kRelEdgeTolerance * faceSeparation) + kAbsTolerance)
    ) {
        buildTriangleAndCapsuleEdgeContact(manifold, triangleB, capsuleA, edgeQuery);
    }
}

// --- hull vs triangle --------------------------------------------------------------------------

function getTriangleSupport(points: Vec3[], direction: Vec3): number {
    let index = 0;
    let distance = vec3.dot(points[0], direction);

    const d1 = vec3.dot(points[1], direction);
    if (d1 > distance) {
        distance = d1;
        index = 1;
    }

    const d2 = vec3.dot(points[2], direction);
    if (d2 > distance) {
        return 2;
    }

    return index;
}

function queryTriangleFace(triangle: TriangleData, hull: HullData): FaceQuery {
    const pl = triangle.plane;
    const vertexIndex = findHullSupportVertex(hull, vec3.neg(pl.normal));
    const support = hull.points[vertexIndex];
    const separation = plane.separation(pl, support);
    return { separation, faceIndex: 0, vertexIndex };
}

function queryHullFace(triangle: TriangleData, hull: HullData): FaceQuery {
    const trianglePoints = [triangle.v1, triangle.v2, triangle.v3];
    const faceCount = hull.faceCount;

    let maxFaceIndex = -1;
    let maxVertexIndex = -1;
    let maxFaceSeparation = -FLT_MAX;

    for (let faceIndex = 0; faceIndex < faceCount; ++faceIndex) {
        const pl = hull.planes[faceIndex];
        const vertexIndex = getTriangleSupport(trianglePoints, vec3.neg(pl.normal));
        const support = trianglePoints[vertexIndex];
        const separation = plane.separation(pl, support);
        if (separation > maxFaceSeparation) {
            maxFaceIndex = faceIndex;
            maxVertexIndex = vertexIndex;
            maxFaceSeparation = separation;
        }
    }

    return { separation: maxFaceSeparation, faceIndex: maxFaceIndex, vertexIndex: maxVertexIndex };
}

function testEdgePairs(triangle: TriangleData, hull: HullData): EdgeQuery {
    let separation = -FLT_MAX;
    let indexA = NULL_INDEX;
    let indexB = NULL_INDEX;

    const trianglePoints = [triangle.v1, triangle.v2, triangle.v3];
    const triangleEdges = [triangle.e1, triangle.e2, triangle.e3];
    const triNormal = triangle.plane.normal;

    const hullEdges = hull.edges;
    const hullPoints = hull.points;
    const hullPlanes = hull.planes;
    const edgeCount = hull.edgeCount;

    for (let i = 0; i < edgeCount; i += 2) {
        const edge = hullEdges[i];
        const twin = hullEdges[i + 1];

        const hullPoint = hullPoints[edge.origin];
        const hullEdge = vec3.sub(hullPoints[twin.origin], hullPoint);

        const hullNormal1 = hullPlanes[edge.face].normal;
        const hullNormal2 = hullPlanes[twin.face].normal;

        for (let j = 0; j < 3; ++j) {
            const triEdge = triangleEdges[j];

            const cab = vec3.dot(hullNormal1, triEdge);
            const dab = vec3.dot(hullNormal2, triEdge);
            const bcd = vec3.dot(triNormal, hullEdge);
            if (f32(cab * dab) >= 0 || f32(cab * bcd) <= 0) {
                continue;
            }

            const triPoint = trianglePoints[j];
            const sep = edgeEdgeSeparation(
                triPoint,
                triEdge,
                triangle.center,
                hullPoint,
                hullEdge,
                hull.center,
            );

            if (sep > separation) {
                separation = sep;
                indexA = j;
                indexB = i;
            }
        }
    }

    return { separation, indexA, indexB };
}

// Reference face is the hull face; incident face is the triangle. Returns min separation.
function collideHullFace(
    manifold: LocalManifold,
    pointCapacity: number,
    triangle: TriangleData,
    hull: HullData,
    query: FaceQuery,
    cache: SATCache,
): number {
    manifold.pointCount = 0;

    const refFace = query.faceIndex;
    const refPlane = hull.planes[refFace];

    let input: ClipVertex[] = [
        {
            position: triangle.v1,
            separation: plane.separation(refPlane, triangle.v1),
            pair: makeFeaturePair(FeatureOwner.ShapeB, 2, FeatureOwner.ShapeB, 0),
        },
        {
            position: triangle.v2,
            separation: plane.separation(refPlane, triangle.v2),
            pair: makeFeaturePair(FeatureOwner.ShapeB, 0, FeatureOwner.ShapeB, 1),
        },
        {
            position: triangle.v3,
            separation: plane.separation(refPlane, triangle.v3),
            pair: makeFeaturePair(FeatureOwner.ShapeB, 1, FeatureOwner.ShapeB, 2),
        },
    ];
    let pointCount = 3;

    const face = hull.faces[refFace];
    let edgeIndex = face.edge;

    do {
        const edge = hull.edges[edgeIndex];
        const nextEdgeIndex = edge.next;
        const next = hull.edges[nextEdgeIndex];
        const vertex1 = hull.points[edge.origin];
        const vertex2 = hull.points[next.origin];
        const tangent = vec3.normalize(vec3.sub(vertex2, vertex1));
        const binormal = vec3.cross(tangent, refPlane.normal);
        const clipPlane = plane.fromNormalAndPoint(binormal, vertex1);

        const out = otherClipBuffer(input);
        pointCount = clipPolygon(input, pointCount, clipPlane, edgeIndex, refPlane, out);
        input = out;

        if (pointCount < 3) {
            // Using a stale cache
            resetCache(cache);
            return query.separation;
        }

        edgeIndex = nextEdgeIndex;
    } while (edgeIndex !== face.edge);

    pointCount = minInt(pointCount, pointCapacity);
    let minSeparation = FLT_MAX;

    for (let i = 0; i < pointCount; ++i) {
        const clipPoint = input[i];
        // Move point onto hull face for improved culling
        const point = vec3.mulSub(clipPoint.position, clipPoint.separation, refPlane.normal);

        const pt = manifold.points[i];
        pt.point = point;
        pt.separation = clipPoint.separation;
        pt.pair = flipPair(clipPoint.pair);

        minSeparation = minf(minSeparation, clipPoint.separation);
    }

    if (minSeparation > SPECULATIVE_DISTANCE) {
        // This can occur with a stale SAT cache
        manifold.pointCount = 0;
        resetCache(cache);
        return minSeparation;
    }

    manifold.pointCount = pointCount;
    manifold.normal = vec3.neg(refPlane.normal);
    manifold.feature = TriangleFeature.HullFace;

    cache.separation = minSeparation;
    cache.type = SeparatingFeature.FaceAxisB;
    cache.indexA = query.vertexIndex & 0xff;
    cache.indexB = query.faceIndex & 0xff;
    return minSeparation;
}

// Reference face is the triangle; incident face is a hull face. Returns min separation.
function collideTriangleFace(
    manifold: LocalManifold,
    pointCapacity: number,
    triangle: TriangleData,
    hull: HullData,
    query: FaceQuery,
    cache: SATCache,
): number {
    const refPlane = triangle.plane;

    const incFace = findIncidentFace(hull, refPlane.normal, query.vertexIndex);

    let input: ClipVertex[] = [];
    const face = hull.faces[incFace];
    let hullEdgeIndex = face.edge;

    do {
        const edge = hull.edges[hullEdgeIndex];
        const nextEdgeIndex = edge.next;
        const next = hull.edges[nextEdgeIndex];
        const hullPoint = hull.points[next.origin];
        input.push({
            position: hullPoint,
            separation: plane.separation(refPlane, hullPoint),
            pair: makeFeaturePair(
                FeatureOwner.ShapeB,
                hullEdgeIndex,
                FeatureOwner.ShapeB,
                nextEdgeIndex,
            ),
        });
        hullEdgeIndex = nextEdgeIndex;
    } while (hullEdgeIndex !== face.edge && input.length < 2 * MAX_CLIP_POINTS);

    let pointCount = input.length;

    const trianglePoints = [triangle.v1, triangle.v2, triangle.v3];
    const triangleEdges = [triangle.e1, triangle.e2, triangle.e3];

    for (let i = 0; i < 3 && pointCount > 0; ++i) {
        const sideNormal = vec3.normalize(vec3.cross(triangleEdges[i], refPlane.normal));
        const clipPlane = plane.fromNormalAndPoint(sideNormal, trianglePoints[i]);

        const out = otherClipBuffer(input);
        pointCount = clipPolygon(input, pointCount, clipPlane, i, refPlane, out);
        input = out;
    }

    if (pointCount === 0) {
        // Triangle face clipped away. Invalidate cache.
        resetCache(cache);
        return FLT_MAX;
    }

    pointCount = minInt(pointCount, pointCapacity);

    let minSeparation = FLT_MAX;

    for (let i = 0; i < pointCount; ++i) {
        const clipPoint = input[i];
        // Point stays on the triangle surface (no projection). `input` is a shared clip scratch buffer,
        // so the manifold takes its own copy of the position and pair (they outlive the next clip pass).
        const pt = manifold.points[i];
        pt.point = { x: clipPoint.position.x, y: clipPoint.position.y, z: clipPoint.position.z };
        pt.separation = clipPoint.separation;
        pt.pair = {
            owner1: clipPoint.pair.owner1,
            index1: clipPoint.pair.index1,
            owner2: clipPoint.pair.owner2,
            index2: clipPoint.pair.index2,
        };

        minSeparation = minf(minSeparation, clipPoint.separation);
    }

    if (minSeparation >= SPECULATIVE_DISTANCE) {
        // This can happen when re-using a cached axis after the objects move apart.
        resetCache(cache);
        return minSeparation;
    }

    manifold.pointCount = pointCount;
    manifold.normal = refPlane.normal;
    manifold.feature = TriangleFeature.TriangleFace;

    cache.separation = minSeparation;
    cache.type = SeparatingFeature.FaceAxisA;
    cache.indexA = query.faceIndex & 0xff;
    cache.indexB = query.vertexIndex & 0xff;
    return minSeparation;
}

function collideHullAndTriangleEdges(
    manifold: LocalManifold,
    capacity: number,
    trianglePoint: Vec3,
    triangleEdge: Vec3,
    triangleCenter: Vec3,
    hull: HullData,
    query: EdgeQuery,
    cache: SATCache,
): void {
    const cA = triangleCenter;
    const pA = trianglePoint;
    const eA = triangleEdge;

    const edgesB = hull.edges;
    const pointsB = hull.points;
    const edgeB = edgesB[query.indexB];
    const twinB = edgesB[edgeB.twin];
    const pB = pointsB[edgeB.origin];
    const qB = pointsB[twinB.origin];
    const eB = vec3.sub(qB, pB);

    let normal = vec3.normalize(vec3.cross(eA, eB));

    // Ensure normal points outward from triangle center
    const outwardA = vec3.dot(normal, vec3.sub(pA, cA));
    // Ensure normal points towards hull center
    const outwardB = vec3.dot(normal, vec3.sub(hull.center, pB));

    if (absf(outwardA) > absf(outwardB)) {
        if (outwardA < 0) normal = vec3.neg(normal);
    } else {
        if (outwardB < 0) normal = vec3.neg(normal);
    }

    const result = lineDistance(pA, eA, pB, eB);

    if (
        capacity === 0 ||
        result.fraction1 < 0 ||
        1 < result.fraction1 ||
        result.fraction2 < 0 ||
        1 < result.fraction2
    ) {
        // Invalid edge pair, no points generated
        resetCache(cache);
        return;
    }

    const separation = vec3.dot(normal, vec3.sub(result.point2, result.point1));
    const point = vec3.scale(f32(0.5), vec3.add(result.point1, result.point2));

    const pt = manifold.points[0];
    pt.point = point;
    pt.separation = separation;
    pt.pair = makeFeaturePair(FeatureOwner.ShapeA, query.indexA, FeatureOwner.ShapeB, query.indexB);

    cache.separation = separation;
    cache.type = SeparatingFeature.EdgePairAxis;
    cache.indexA = query.indexA & 0xff;
    cache.indexB = query.indexB & 0xff;

    manifold.normal = normal;
    manifold.pointCount = 1;
    manifold.feature = EDGE_FEATURES[query.indexA];
}

function isTriangleMinkowskiFace(
    triNormal: Vec3,
    triEdge: Vec3,
    hullNormal1: Vec3,
    hullNormal2: Vec3,
    hullEdge: Vec3,
): boolean {
    const cab = vec3.dot(hullNormal1, triEdge);
    const dab = vec3.dot(hullNormal2, triEdge);
    const bcd = vec3.dot(triNormal, hullEdge);
    return f32(cab * dab) < 0 && f32(cab * bcd) > 0;
}

/**
 * b3CollideHullAndTriangle — hull A against triangle (v1,v2,v3), manifold in A's frame. Uses and
 * updates the SAT `cache` for temporal coherence. Falls back to GJK when SAT yields no points.
 */
export function collideHullAndTriangle(
    manifold: LocalManifold,
    capacity: number,
    hullA: HullData,
    v1: Vec3,
    v2: Vec3,
    v3: Vec3,
    cache: SATCache,
): void {
    manifold.pointCount = 0;
    manifold.feature = TriangleFeature.None;

    if (capacity < 4) {
        return;
    }

    const trianglePlane = plane.fromPoints(v1, v2, v3);
    const linearSlop = LINEAR_SLOP;

    const offset = plane.separation(trianglePlane, hullA.center);
    if (cache.type === SeparatingFeature.Backside) {
        // Use hysteresis to avoid jitter on wavy meshes
        if (absf(f32(cache.separation - offset)) < linearSlop) {
            return;
        }
        cache.type = SeparatingFeature.Invalid;
    }

    if (offset < -linearSlop) {
        // Cull back side collision. Cache offset to add hysteresis.
        cache.type = SeparatingFeature.Backside;
        cache.separation = offset;
        return;
    }

    const triangleCenter = vec3.scale(f32(1 / 3), vec3.add(v1, vec3.add(v2, v3)));
    const trianglePoints = [v1, v2, v3];
    const triangleEdges = [vec3.sub(v2, v1), vec3.sub(v3, v2), vec3.sub(v1, v3)];

    const triangle: TriangleData = {
        v1,
        v2,
        v3,
        e1: triangleEdges[0],
        e2: triangleEdges[1],
        e3: triangleEdges[2],
        center: triangleCenter,
        plane: trianglePlane,
    };

    const hullPlanes = hullA.planes;
    const hullPoints = hullA.points;
    const edges = hullA.edges;

    const speculativeDistance = SPECULATIVE_DISTANCE;
    cache.hit = 1;

    // Attempt to use the cache to speed up collision
    switch (cache.type) {
        case SeparatingFeature.FaceAxisA: {
            const vertexIndex = findHullSupportVertex(hullA, vec3.neg(trianglePlane.normal));
            const support = hullPoints[vertexIndex];
            const separation = plane.separation(trianglePlane, support);
            if (separation >= speculativeDistance) {
                return;
            }

            const faceQuery: FaceQuery = { separation, faceIndex: cache.indexA, vertexIndex };
            const localCache = copyCache(cache);
            const clippedSeparation = collideTriangleFace(
                manifold,
                capacity,
                triangle,
                hullA,
                faceQuery,
                localCache,
            );

            if (
                manifold.pointCount > 0 &&
                absf(f32(cache.separation - clippedSeparation)) < linearSlop
            ) {
                return;
            }

            manifold.pointCount = 0;
            resetCache(cache);
            break;
        }

        case SeparatingFeature.FaceAxisB: {
            const pl = hullPlanes[cache.indexB];

            let vertexIndex = 0;
            let distance = -vec3.dot(v1, pl.normal);
            for (let i = 1; i < 3; ++i) {
                const d = -vec3.dot(trianglePoints[i], pl.normal);
                if (d > distance) {
                    distance = d;
                    vertexIndex = i;
                }
            }

            const support = trianglePoints[vertexIndex];
            const separation = plane.separation(pl, support);
            if (separation >= speculativeDistance) {
                return;
            }

            const isDeep = separation < f32(-2 * linearSlop);
            if (isDeep === false) {
                const faceQuery: FaceQuery = { separation, faceIndex: cache.indexB, vertexIndex };
                const localCache = copyCache(cache);
                const clippedSeparation = collideHullFace(
                    manifold,
                    capacity,
                    triangle,
                    hullA,
                    faceQuery,
                    localCache,
                );

                if (
                    manifold.pointCount > 0 &&
                    absf(f32(cache.separation - clippedSeparation)) < linearSlop
                ) {
                    return;
                }
            }

            manifold.pointCount = 0;
            resetCache(cache);
            break;
        }

        case SeparatingFeature.EdgePairAxis: {
            const indexA = cache.indexA;
            const triPoint = trianglePoints[indexA];
            const triEdge = triangleEdges[indexA];

            const indexB = cache.indexB;
            const edge2 = edges[indexB];
            const twin2 = edges[indexB + 1];

            const hullPoint = hullPoints[edge2.origin];
            const hullEdge = vec3.sub(hullPoints[twin2.origin], hullPoint);
            const hullNormal1 = hullPlanes[edge2.face].normal;
            const hullNormal2 = hullPlanes[twin2.face].normal;

            const isMink = isTriangleMinkowskiFace(
                trianglePlane.normal,
                triEdge,
                hullNormal1,
                hullNormal2,
                hullEdge,
            );
            if (isMink) {
                const separation = edgeEdgeSeparation(
                    triPoint,
                    triEdge,
                    triangleCenter,
                    hullPoint,
                    hullEdge,
                    hullA.center,
                );
                if (separation > speculativeDistance) {
                    return;
                }

                if (absf(f32(cache.separation - separation)) < linearSlop) {
                    const edgeQuery: EdgeQuery = { indexA, indexB, separation };
                    const localCache = copyCache(cache);
                    collideHullAndTriangleEdges(
                        manifold,
                        capacity,
                        triPoint,
                        triEdge,
                        triangleCenter,
                        hullA,
                        edgeQuery,
                        localCache,
                    );

                    if (manifold.pointCount > 0) {
                        return;
                    }
                }
            }

            resetCache(cache);
            break;
        }

        case SeparatingFeature.ManualFaceAxisA: {
            const faceQueryA = queryTriangleFace(triangle, hullA);
            collideTriangleFace(manifold, capacity, triangle, hullA, faceQueryA, cache);
            return;
        }

        case SeparatingFeature.ManualFaceAxisB: {
            const faceQueryB = queryHullFace(triangle, hullA);
            collideHullFace(manifold, capacity, triangle, hullA, faceQueryB, cache);
            return;
        }

        case SeparatingFeature.ManualEdgePairAxis: {
            const edgeQuery = testEdgePairs(triangle, hullA);
            if (edgeQuery.indexA !== NULL_INDEX) {
                const trianglePoint = trianglePoints[edgeQuery.indexA];
                const triangleEdge = triangleEdges[edgeQuery.indexA];
                collideHullAndTriangleEdges(
                    manifold,
                    capacity,
                    trianglePoint,
                    triangleEdge,
                    triangleCenter,
                    hullA,
                    edgeQuery,
                    cache,
                );
            }
            return;
        }

        default:
            break;
    }

    // Cache miss
    cache.hit = 0;

    // Find axis of minimum penetration
    const faceQueryA = queryTriangleFace(triangle, hullA);
    if (faceQueryA.separation > speculativeDistance) {
        cache.separation = faceQueryA.separation;
        cache.type = SeparatingFeature.FaceAxisA;
        cache.indexA = 0;
        cache.indexB = 0xff;
        return;
    }

    const faceQueryB = queryHullFace(triangle, hullA);
    if (faceQueryB.separation > speculativeDistance) {
        cache.separation = faceQueryB.separation;
        cache.type = SeparatingFeature.FaceAxisB;
        cache.indexA = 0xff;
        cache.indexB = faceQueryB.faceIndex & 0xff;
        return;
    }

    const edgeQuery = testEdgePairs(triangle, hullA);
    if (edgeQuery.separation > speculativeDistance) {
        cache.separation = edgeQuery.separation;
        cache.type = SeparatingFeature.EdgePairAxis;
        cache.indexA = edgeQuery.indexA & 0xff;
        cache.indexB = edgeQuery.indexB & 0xff;
        return;
    }

    let clippedFaceSeparation: number;

    // Don't allow a hull face opposed to the triangle face.
    const hullNormal = hullPlanes[faceQueryB.faceIndex].normal;
    const pushingUp = vec3.dot(hullNormal, trianglePlane.normal) < 0;
    if (faceQueryB.separation > f32(faceQueryA.separation + linearSlop) && pushingUp) {
        clippedFaceSeparation = collideHullFace(
            manifold,
            capacity,
            triangle,
            hullA,
            faceQueryB,
            cache,
        );
    } else {
        clippedFaceSeparation = collideTriangleFace(
            manifold,
            capacity,
            triangle,
            hullA,
            faceQueryA,
            cache,
        );
    }

    // Does an edge axis exist?
    if (edgeQuery.indexA !== NULL_INDEX) {
        const maxFaceSeparation = maxf(faceQueryA.separation, faceQueryB.separation);

        if (
            (manifold.pointCount === 0 && edgeQuery.separation > maxFaceSeparation) ||
            (manifold.pointCount === 1 &&
                edgeQuery.separation > f32(clippedFaceSeparation + linearSlop))
        ) {
            const trianglePoint = trianglePoints[edgeQuery.indexA];
            const triangleEdge = triangleEdges[edgeQuery.indexA];
            manifold.pointCount = 0;
            collideHullAndTriangleEdges(
                manifold,
                capacity,
                trianglePoint,
                triangleEdge,
                triangleCenter,
                hullA,
                edgeQuery,
                cache,
            );
        }
    }

    // Fall back to GJK when SAT produced no points (prevents rare tunneling).
    if (manifold.pointCount === 0) {
        const input: DistanceInput = {
            proxyA: { points: [v1, v2, v3], count: 3, radius: 0 },
            proxyB: { points: hullPoints, count: hullA.vertexCount, radius: 0 },
            transform: xf.identity(),
            useRadii: false,
        };

        const simplexCache = emptyCache();
        const output = shapeDistance(input, simplexCache);

        if (output.distance > 0) {
            manifold.pointCount = 1;
            manifold.feature = getTriangleFeature(simplexCache);
            manifold.normal = output.normal;
            manifold.points[0].point = output.pointB;
            manifold.points[0].separation = output.distance;
            manifold.points[0].pair = singlePair();
        }
    }
}
