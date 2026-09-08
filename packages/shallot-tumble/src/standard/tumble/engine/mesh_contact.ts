// Mesh narrowphase — Box3D's b3ComputeMeshManifolds (mesh_contact.c, Erin Catto, MIT). A convex
// shape B collides against every triangle of a static mesh or height-field shape A that overlaps its
// AABB (the two share this driver, differing only in the triangle source + material indexing). Each
// triangle produces a local manifold; ghost-collision culling drops manifolds at internal edges and
// vertices already covered by an adjacent face, the survivors are clustered by normal into 1..N
// persistent manifolds, and warm-start impulses carry forward by normal + feature id. The port
// targets the scalar force-overflow build; every op is fround-wrapped (see the README).

import { NULL_INDEX, qsort } from "./array";
import type { Contact, ConvexContactCache } from "./contact";
import { LINEAR_SLOP, MAX_AABB_MARGIN, MESH_REST_OFFSET, SPECULATIVE_DISTANCE } from "./core";
import { emptyCache } from "./distance";
import type { Capsule, Sphere } from "./geometry";
import { getHeightFieldTriangle, type HeightFieldData, queryHeightField } from "./heightfield";
import type { HullData } from "./hull";
import {
    emptySATCache,
    type LocalManifold,
    type LocalManifoldPoint,
    makeFeatureId,
    makeLocalManifold,
    SeparatingFeature,
    TriangleFeature,
} from "./manifold";
import {
    type AABB,
    aabb,
    absf,
    clampInt,
    f32,
    invMulWorldTransforms,
    mat3,
    maxf,
    minf,
    quat,
    subPos,
    type Vec2,
    type Vec3,
    vec2,
    vec3,
    type WorldTransform,
    xf,
} from "./math";
import { getMeshTriangle, type Mesh, MeshEdgeFlags, queryMesh } from "./mesh";
import { getShapeMaterials, type Shape } from "./shape";
import {
    collideCapsuleAndTriangle,
    collideHullAndTriangle,
    collideSphereAndTriangle,
} from "./triangle_manifold";
import { ShapeType } from "./types";
import type { WorldState } from "./world";

// This guards against excessive memory usage and complex collision.
const MAX_MESH_CONTACT_TRIANGLES = 256;
const MAX_POINTS_PER_TRIANGLE = 32;
const MAX_EDGE_COUNT = 64;
const MAX_VERTEX_COUNT = 64;

const emptyConvexCache = (): ConvexContactCache => ({
    simplexCache: emptyCache(),
    satCache: emptySATCache(),
});

// b3MakeNormalFromPoints: unit normal of the triangle p1-p2-p3.
const makeNormalFromPoints = (p1: Vec3, p2: Vec3, p3: Vec3): Vec3 =>
    vec3.normalize(vec3.cross(vec3.sub(p2, p1), vec3.sub(p3, p1)));

// --- ghost-collision sets ---------------------------------------------------------------------
// Edges and vertices already covered by an accepted face manifold. A tentative manifold whose
// feature is an edge/vertex is dropped when that feature is already claimed by a neighbour.

type FoundEdges = { keys: string[]; count: number };
type FoundVertices = { keys: number[]; count: number };

// Edge key uses min/max vertex indices so the two triangles sharing an edge produce the same key.
// A string key is exact for any vertex count (the C uint64 `i1<<32|i2` would overflow JS numbers).
const edgeKey = (v1: number, v2: number): string => {
    const i1 = v1 < v2 ? v1 : v2;
    const i2 = v1 < v2 ? v2 : v1;
    return `${i1},${i2}`;
};

function addEdge(edges: FoundEdges, v1: number, v2: number): boolean {
    const key = edgeKey(v1, v2);
    for (let i = 0; i < edges.count; ++i) {
        if (edges.keys[i] === key) return false;
    }
    if (edges.count === MAX_EDGE_COUNT) {
        // Overflow: leads to a potential ghost collision, but reported as "already present".
        return true;
    }
    edges.keys[edges.count] = key;
    edges.count += 1;
    return true;
}

function findEdge(edges: FoundEdges, v1: number, v2: number): boolean {
    const key = edgeKey(v1, v2);
    for (let i = 0; i < edges.count; ++i) {
        if (edges.keys[i] === key) return true;
    }
    return false;
}

function addVertex(vertices: FoundVertices, vertex: number): boolean {
    for (let i = 0; i < vertices.count; ++i) {
        if (vertices.keys[i] === vertex) return false;
    }
    if (vertices.count === MAX_VERTEX_COUNT) {
        return true;
    }
    vertices.keys[vertices.count] = vertex;
    vertices.count += 1;
    return true;
}

// --- cluster point reduction ------------------------------------------------------------------

type Point2D = { p: Vec2; separation: number; originalIndex: number };

// b3IsBetterCullCandidate: prefer a larger score, breaking ties within scoreTol by deeper separation.
function isBetterCullCandidate(
    score: number,
    separation: number,
    bestScore: number,
    bestSeparation: number,
    scoreTol: number,
    separationTol: number,
): boolean {
    if (score > f32(bestScore + scoreTol)) {
        return true;
    }
    if (score < f32(bestScore - scoreTol)) {
        return false;
    }
    return separation < f32(bestSeparation - separationTol);
}

// b3CullPoints: reduce a 2D point cloud to at most 4 points that best preserve the contact patch —
// the two farthest points, then the two adding the most triangular/quad area. Rearranges `points`
// so the survivors occupy [0, count2) and returns count2.
function cullPoints(points: Point2D[], count: number): number {
    if (count <= 1) {
        return count;
    }

    const tol = f32(0.25 * LINEAR_SLOP);
    const tolSqr = f32(tol * tol);
    const separationTol = LINEAR_SLOP;

    const finalPoints: Point2D[] = new Array(4);
    let count1 = count;

    // Step 1: the two points with the largest distance, ties broken by deepest combined separation.
    let bestScore = 0;
    let bestSeparation = Number.POSITIVE_INFINITY;
    let bestIndex1 = NULL_INDEX;
    let bestIndex2 = NULL_INDEX;

    for (let i = 0; i < count1; ++i) {
        const p1 = points[i].p;
        for (let j = i + 1; j < count1; ++j) {
            const score = vec2.distanceSquared(p1, points[j].p);
            const separation = f32(points[i].separation + points[j].separation);
            if (
                isBetterCullCandidate(
                    score,
                    separation,
                    bestScore,
                    bestSeparation,
                    tolSqr,
                    separationTol,
                )
            ) {
                bestIndex1 = i;
                bestIndex2 = j;
                bestScore = score;
                bestSeparation = separation;
            }
        }
    }

    if (bestScore < tolSqr) {
        // Choose the single deepest point.
        let deepestIndex = 0;
        for (let i = 1; i < count1; ++i) {
            if (points[i].separation < points[deepestIndex].separation) {
                deepestIndex = i;
            }
        }
        if (deepestIndex !== 0) {
            points[0] = points[deepestIndex];
        }
        return 1;
    }

    finalPoints[0] = points[bestIndex1];
    finalPoints[1] = points[bestIndex2];

    // Cull the two chosen points by swapping the tail in.
    points[bestIndex2] = points[count1 - 1];
    points[bestIndex1] = points[count1 - 2];
    count1 -= 2;

    if (count1 === 0) {
        points[0] = finalPoints[0];
        points[1] = finalPoints[1];
        return 2;
    }

    const a = finalPoints[0].p;
    let b = finalPoints[1].p;
    let ba = vec2.sub(b, a);

    // Step 2: the point with the maximum triangular area, ties broken by deepest separation.
    bestScore = 0;
    bestSeparation = Number.POSITIVE_INFINITY;
    let bestIndex = NULL_INDEX;
    let bestSignedArea = 0;
    for (let i = 0; i < count1; ++i) {
        const p = points[i].p;
        const signedArea = vec2.cross(ba, vec2.sub(p, a));
        const score = absf(signedArea);
        if (
            isBetterCullCandidate(
                score,
                points[i].separation,
                bestScore,
                bestSeparation,
                tolSqr,
                separationTol,
            )
        ) {
            bestSignedArea = signedArea;
            bestScore = score;
            bestSeparation = points[i].separation;
            bestIndex = i;
        }
    }

    if (bestIndex === NULL_INDEX) {
        // All points collinear.
        points[0] = finalPoints[0];
        points[1] = finalPoints[1];
        return 2;
    }

    finalPoints[2] = points[bestIndex];

    if (count1 === 1) {
        points[0] = finalPoints[0];
        points[1] = finalPoints[1];
        points[2] = finalPoints[2];
        return 3;
    }

    // Cull the chosen point.
    points[bestIndex] = points[count1 - 1];
    count1 -= 1;

    // Step 3: the point that adds the most area outside the current triangle.
    let c = finalPoints[2].p;

    // Ensure CCW ordering.
    if (bestSignedArea < 0) {
        const tmp = b;
        b = c;
        c = tmp;
        ba = vec2.sub(b, a);
    }

    const cb = vec2.sub(c, b);
    const ac = vec2.sub(a, c);

    bestScore = 0;
    bestSeparation = Number.POSITIVE_INFINITY;
    bestIndex = NULL_INDEX;
    for (let i = 0; i < count1; ++i) {
        const p = points[i].p;
        const u1 = vec2.cross(vec2.sub(p, a), ba);
        const u2 = vec2.cross(vec2.sub(p, b), cb);
        const u3 = vec2.cross(vec2.sub(p, c), ac);
        const score = maxf(u1, maxf(u2, u3));
        if (
            isBetterCullCandidate(
                score,
                points[i].separation,
                bestScore,
                bestSeparation,
                tolSqr,
                separationTol,
            )
        ) {
            bestScore = score;
            bestSeparation = points[i].separation;
            bestIndex = i;
        }
    }

    if (bestIndex === NULL_INDEX) {
        // No additional area.
        points[0] = finalPoints[0];
        points[1] = finalPoints[1];
        points[2] = finalPoints[2];
        return 3;
    }

    finalPoints[3] = points[bestIndex];

    points[0] = finalPoints[0];
    points[1] = finalPoints[1];
    points[2] = finalPoints[2];
    points[3] = finalPoints[3];
    return 4;
}

// b3ReduceCluster: project a cluster's points into the triangle-normal plane, cull to <= 4, and
// rearrange `points` so the survivors occupy [0, count2). Returns count2.
function reduceCluster(points: LocalManifoldPoint[], count1: number, normal: Vec3): number {
    if (count1 <= 1) {
        return count1;
    }

    const pts: Point2D[] = new Array(count1);
    const u = vec3.perp(normal);
    const v = vec3.cross(normal, u);
    const origin = points[0].point;

    for (let i = 0; i < count1; ++i) {
        const d = vec3.sub(points[i].point, origin);
        pts[i] = {
            p: { x: vec3.dot(d, u), y: vec3.dot(d, v) },
            separation: points[i].separation,
            originalIndex: i,
        };
    }

    const count2 = cullPoints(pts, count1);

    const finalPoints: LocalManifoldPoint[] = new Array(count2);
    for (let i = 0; i < count2; ++i) {
        finalPoints[i] = points[pts[i].originalIndex];
    }
    for (let i = 0; i < count2; ++i) {
        points[i] = finalPoints[i];
    }
    return count2;
}

// --- broadphase triangle cache ----------------------------------------------------------------

// b3RefreshCache: re-query the mesh BVH for triangles overlapping shape B's fattened AABB, but only
// when B moved out of the previously cached query bounds. New per-triangle caches are matched to old
// ones by triangle index (both lists are sorted) so warm-start state survives.
function refreshCache(contact: Contact, shapeA: Shape, xfA: WorldTransform, bounds: AABB): void {
    const meshContact = contact.meshContact;

    // If the dynamic body stayed within the cached query bounds we are done.
    if (aabb.contains(meshContact.queryBounds, bounds)) {
        return;
    }

    // Enlarge to the query bounds to absorb small movement.
    const radius = f32(MAX_AABB_MARGIN + SPECULATIVE_DISTANCE);
    const extension: Vec3 = { x: radius, y: radius, z: radius };
    meshContact.queryBounds = {
        lowerBound: vec3.sub(bounds.lowerBound, extension),
        upperBound: vec3.add(bounds.upperBound, extension),
    };

    // Bounds are world space; convert to the local mesh frame. b3ToRelativeTransform(xfA, zero) is a
    // plain demotion of the world transform in float mode.
    const meshTransform = { q: xfA.q, p: { x: xfA.p.x, y: xfA.p.y, z: xfA.p.z } };
    const localBounds = aabb.transform(xf.invert(meshTransform), meshContact.queryBounds);

    const triangleIndices: number[] = [];
    // b3CollectTriangleIndicesCallback: append until capacity. queryMesh honours the false return and
    // stops; queryHeightField ignores it, so the `>= capacity` guard alone caps the height-field path.
    const collect = (triangleIndex: number): boolean => {
        if (triangleIndices.length === MAX_MESH_CONTACT_TRIANGLES) {
            return false;
        }
        triangleIndices.push(triangleIndex);
        return triangleIndices.length < MAX_MESH_CONTACT_TRIANGLES;
    };
    if (shapeA.type === ShapeType.Mesh) {
        queryMesh(shapeA.mesh as Mesh, localBounds, (_a, _b, _c, triangleIndex) =>
            collect(triangleIndex),
        );
    } else {
        queryHeightField(shapeA.heightField as HeightFieldData, localBounds, (_a, _b, _c, ti) => {
            collect(ti);
        });
    }

    // Triangle indices are sorted (BVH DFS order matches sorted triangle order) — match to old cache.
    const oldCache = meshContact.triangleCache;
    const triangleCount = triangleIndices.length;
    const contactCaches: ConvexContactCache[] = new Array(triangleCount);

    let index2 = 0;
    for (let index1 = 0; index1 < triangleCount; ++index1) {
        contactCaches[index1] = emptyConvexCache();
        while (
            index2 < oldCache.length &&
            oldCache[index2].triangleIndex < triangleIndices[index1]
        ) {
            index2 += 1;
        }
        if (
            index2 < oldCache.length &&
            oldCache[index2].triangleIndex === triangleIndices[index1]
        ) {
            contactCaches[index1] = oldCache[index2].cache;
        }
    }

    meshContact.triangleCache = triangleIndices.map((triangleIndex, i) => ({
        triangleIndex,
        cache: contactCaches[i],
    }));
}

type Cluster = {
    manifoldNormal: Vec3;
    triangleNormal: Vec3;
    points: LocalManifoldPoint[];
    pointCapacity: number;
    pointCount: number;
};

// A plain-object snapshot of the previous step's manifolds, taken before the column-resident block is
// reallocated (a same-size realloc aliases the pool). The warm-start match reads and claims it.
type OldPoint = { featureId: number; triangleIndex: number; normalImpulse: number };
type OldManifold = {
    normal: Vec3;
    frictionImpulse: Vec3;
    rollingImpulse: Vec3;
    twistImpulse: number;
    pointCount: number;
    points: OldPoint[];
};

type TentativeTriangle = { squaredDistance: number; index: number };

/**
 * Collide convex shape B against the overlapping triangles of static mesh/height-field shape A,
 * building the contact's persistent, clustered, warm-started manifolds (b3ComputeMeshManifolds).
 * @returns whether the contact touches; on false the contact's manifolds are cleared.
 */
export function computeMeshManifolds(
    world: WorldState,
    contact: Contact,
    shapeA: Shape,
    materialMap: number[] | null,
    xfA: WorldTransform,
    shapeB: Shape,
    xfB: WorldTransform,
    isFast: boolean,
): boolean {
    refreshCache(contact, shapeA, xfA, shapeB.aabb);

    const meshContact = contact.meshContact;
    const triangleCaches = meshContact.triangleCache;
    const triangleCount = triangleCaches.length;

    const acceptedManifolds: LocalManifold[] = [];
    const tentativeManifolds: LocalManifold[] = [];
    const tentativeTriangles: TentativeTriangle[] = [];

    const foundEdges: FoundEdges = { keys: [], count: 0 };
    const foundVertices: FoundVertices = { keys: [], count: 0 };

    // Converts from the mesh frame into shape B's frame.
    const transformAtoB = invMulWorldTransforms(xfB, xfA);
    const relativeMatrix = mat3.fromQuat(transformAtoB.q);
    const linearSlop = LINEAR_SLOP;
    const restOffset = MESH_REST_OFFSET;

    const pointBufferCapacity = MAX_POINTS_PER_TRIANGLE * triangleCount;
    let totalPointCount = 0;

    const isMesh = shapeA.type === ShapeType.Mesh;
    const mesh = isMesh ? (shapeA.mesh as Mesh) : null;
    const heightField = isMesh ? null : (shapeA.heightField as HeightFieldData);
    const hullB = shapeB.type === ShapeType.Hull ? (shapeB.hull as HullData) : null;

    for (
        let index = 0;
        index < triangleCount && totalPointCount + 3 < pointBufferCapacity;
        ++index
    ) {
        const triangleIndex = triangleCaches[index].triangleIndex;
        const triangle = isMesh
            ? getMeshTriangle(mesh as Mesh, triangleIndex)
            : getHeightFieldTriangle(heightField as HeightFieldData, triangleIndex);

        // Transform the triangle into shape B's frame.
        const v0 = vec3.add(mat3.mulV(relativeMatrix, triangle.vertices[0]), transformAtoB.p);
        const v1 = vec3.add(mat3.mulV(relativeMatrix, triangle.vertices[1]), transformAtoB.p);
        const v2 = vec3.add(mat3.mulV(relativeMatrix, triangle.vertices[2]), transformAtoB.p);

        const cache = triangleCaches[index].cache;
        const pointCapacity = pointBufferCapacity - totalPointCount;
        const manifold = makeLocalManifold(MAX_POINTS_PER_TRIANGLE);
        manifold.triangleFlags = triangle.flags;
        manifold.feature = TriangleFeature.None;

        switch (shapeB.type) {
            case ShapeType.Capsule:
                collideCapsuleAndTriangle(
                    manifold,
                    pointCapacity,
                    shapeB.capsule as Capsule,
                    v0,
                    v1,
                    v2,
                    cache.simplexCache,
                );
                break;
            case ShapeType.Hull:
                // A cached edge contact is dangerous at high speed: the hull can rotate around the
                // edge and tunnel through the triangle, so discard it when fast.
                if (isFast && cache.satCache.type === SeparatingFeature.EdgePairAxis) {
                    cache.satCache = emptySATCache();
                }
                collideHullAndTriangle(
                    manifold,
                    pointCapacity,
                    hullB as HullData,
                    v0,
                    v1,
                    v2,
                    cache.satCache,
                );
                break;
            case ShapeType.Sphere:
                collideSphereAndTriangle(
                    manifold,
                    pointCapacity,
                    shapeB.sphere as Sphere,
                    v0,
                    v1,
                    v2,
                );
                break;
            default:
                return false;
        }

        const manifoldPointCount = manifold.pointCount;
        if (manifoldPointCount === 0) {
            continue;
        }

        totalPointCount += manifoldPointCount;
        manifold.triangleIndex = triangleIndex;
        manifold.triangleNormal = makeNormalFromPoints(v0, v1, v2);
        manifold.i1 = triangle.i1;
        manifold.i2 = triangle.i2;
        manifold.i3 = triangle.i3;

        if (manifold.feature === TriangleFeature.TriangleFace) {
            addEdge(foundEdges, triangle.i1, triangle.i2);
            addEdge(foundEdges, triangle.i2, triangle.i3);
            addEdge(foundEdges, triangle.i3, triangle.i1);
            addVertex(foundVertices, triangle.i1);
            addVertex(foundVertices, triangle.i2);
            addVertex(foundVertices, triangle.i3);
            acceptedManifolds.push(manifold);
        } else if (manifold.feature === TriangleFeature.HullFace) {
            const cosNormalAngle = vec3.dot(manifold.triangleNormal, manifold.normal);
            if (cosNormalAngle > f32(0.5)) {
                addEdge(foundEdges, triangle.i1, triangle.i2);
                addEdge(foundEdges, triangle.i2, triangle.i3);
                addEdge(foundEdges, triangle.i3, triangle.i1);
                addVertex(foundVertices, triangle.i1);
                addVertex(foundVertices, triangle.i2);
                addVertex(foundVertices, triangle.i3);
                acceptedManifolds.push(manifold);
            } else {
                let minSeparation = manifold.points[0].separation;
                for (let i = 1; i < manifoldPointCount; ++i) {
                    minSeparation = minf(minSeparation, manifold.points[i].separation);
                }
                if (minSeparation < f32(-2.0 * linearSlop)) {
                    // Deep overlap: accept despite the shallow-angle face.
                    addEdge(foundEdges, triangle.i1, triangle.i2);
                    addEdge(foundEdges, triangle.i2, triangle.i3);
                    addEdge(foundEdges, triangle.i3, triangle.i1);
                    addVertex(foundVertices, triangle.i1);
                    addVertex(foundVertices, triangle.i2);
                    addVertex(foundVertices, triangle.i3);
                    acceptedManifolds.push(manifold);
                } else {
                    tentativeTriangles.push({
                        squaredDistance: manifold.squaredDistance,
                        index: tentativeManifolds.length,
                    });
                    tentativeManifolds.push(manifold);
                }
            }
        } else {
            tentativeTriangles.push({
                squaredDistance: manifold.squaredDistance,
                index: tentativeManifolds.length,
            });
            tentativeManifolds.push(manifold);
        }
    }

    if (shapeB.type === ShapeType.Sphere) {
        // Sort tentative triangles so the closest are processed first, then add each unless it would
        // generate a ghost collision at an edge/vertex already claimed by a neighbour.
        qsort(
            tentativeTriangles.length,
            (i, j) => tentativeTriangles[i].squaredDistance < tentativeTriangles[j].squaredDistance,
            (i, j) => {
                const t = tentativeTriangles[i];
                tentativeTriangles[i] = tentativeTriangles[j];
                tentativeTriangles[j] = t;
            },
        );

        for (let i = 0; i < tentativeTriangles.length; ++i) {
            const m = tentativeManifolds[tentativeTriangles[i].index];
            const addedEdge1 = addEdge(foundEdges, m.i1, m.i2);
            const addedEdge2 = addEdge(foundEdges, m.i2, m.i3);
            const addedEdge3 = addEdge(foundEdges, m.i3, m.i1);
            const addedVertex1 = addVertex(foundVertices, m.i1);
            const addedVertex2 = addVertex(foundVertices, m.i2);
            const addedVertex3 = addVertex(foundVertices, m.i3);

            let shouldCollide = false;
            switch (m.feature) {
                case TriangleFeature.Edge1:
                    shouldCollide = addedEdge1;
                    break;
                case TriangleFeature.Edge2:
                    shouldCollide = addedEdge2;
                    break;
                case TriangleFeature.Edge3:
                    shouldCollide = addedEdge3;
                    break;
                case TriangleFeature.Vertex1:
                    shouldCollide = addedVertex1;
                    break;
                case TriangleFeature.Vertex2:
                    shouldCollide = addedVertex2;
                    break;
                case TriangleFeature.Vertex3:
                    shouldCollide = addedVertex3;
                    break;
                default:
                    break;
            }

            if (shouldCollide) {
                acceptedManifolds.push(m);
            }
        }
    } else {
        // Hull/capsule can tunnel if the time of impact is at a concave edge, so only ignore flat
        // edges already covered by a neighbouring face.
        for (let i = 0; i < tentativeManifolds.length; ++i) {
            const m = tentativeManifolds[i];
            const triangleFlags = m.triangleFlags;

            if ((triangleFlags & MeshEdgeFlags.AllFlatEdges) === MeshEdgeFlags.AllFlatEdges) {
                continue;
            }
            if ((triangleFlags & MeshEdgeFlags.FlatEdge1) === MeshEdgeFlags.FlatEdge1) {
                if (findEdge(foundEdges, m.i1, m.i2)) continue;
            }
            if ((triangleFlags & MeshEdgeFlags.FlatEdge2) === MeshEdgeFlags.FlatEdge2) {
                if (findEdge(foundEdges, m.i2, m.i3)) continue;
            }
            if ((triangleFlags & MeshEdgeFlags.FlatEdge3) === MeshEdgeFlags.FlatEdge3) {
                if (findEdge(foundEdges, m.i3, m.i1)) continue;
            }

            acceptedManifolds.push(m);
        }
    }

    const acceptedManifoldCount = acceptedManifolds.length;
    if (acceptedManifoldCount === 0) {
        if (contact.manifoldCount > 0) {
            contact.manifolds = [];
            contact.manifoldCount = 0;
        }
        return false;
    }

    // Cluster accepted manifolds by (contact normal, triangle normal). The first cluster within the
    // tight tolerance is accepted.
    const clusters: Cluster[] = [];
    const clusterMemberships: number[] = new Array(acceptedManifoldCount);
    const clusterThreshold = f32(0.996);
    let clusterPointCount = 0;

    for (let i = 0; i < acceptedManifoldCount; ++i) {
        clusterMemberships[i] = NULL_INDEX;
        const manifold = acceptedManifolds[i];
        clusterPointCount += manifold.pointCount;

        const manifoldNormal = manifold.normal;
        const triangleNormal = manifold.triangleNormal;
        let clusterIndex = NULL_INDEX;
        for (let j = 0; j < clusters.length; ++j) {
            const cosManifoldAngle = vec3.dot(clusters[j].manifoldNormal, manifoldNormal);
            const cosTriangleAngle = vec3.dot(clusters[j].triangleNormal, triangleNormal);
            if (cosManifoldAngle <= clusterThreshold || cosTriangleAngle <= clusterThreshold) {
                continue;
            }
            clusterIndex = j;
            break;
        }

        if (clusterIndex !== NULL_INDEX) {
            clusterMemberships[i] = clusterIndex;
            clusters[clusterIndex].pointCapacity += manifold.pointCount;
        } else {
            clusters.push({
                manifoldNormal,
                triangleNormal,
                points: [],
                pointCapacity: manifold.pointCount,
                pointCount: 0,
            });
            clusterMemberships[i] = clusters.length - 1;
        }
    }

    if (clusterPointCount === 0) {
        return false;
    }

    const clusterCount = clusters.length;

    // Populate clusters.
    for (let i = 0; i < acceptedManifoldCount; ++i) {
        const clusterIndex = clusterMemberships[i];
        if (clusterIndex === NULL_INDEX) {
            continue;
        }
        const am = acceptedManifolds[i];
        const cm = clusters[clusterIndex];
        for (let j = 0; j < am.pointCount; ++j) {
            const ap = am.points[j];
            cm.points.push({
                point: ap.point,
                separation: ap.separation,
                pair: ap.pair,
                triangleIndex: am.triangleIndex,
            });
            cm.pointCount += 1;
        }
    }

    // Simplify clusters.
    for (let i = 0; i < clusterCount; ++i) {
        const cm = clusters[i];
        cm.pointCount = reduceCluster(cm.points, cm.pointCount, cm.triangleNormal);
    }

    // Snapshot the previous manifolds' warm-start state into plain scratch BEFORE reallocating the
    // column-resident block: a same-size-class realloc recycles the same pool block, so the fresh views
    // would alias (and clobber) the old data mid-rebuild. The matching below reads and claims the snapshot.
    const oldManifoldCount = contact.manifoldCount;
    const oldManifolds: OldManifold[] = new Array(oldManifoldCount);
    for (let j = 0; j < oldManifoldCount; ++j) {
        const om = contact.manifolds[j];
        const opc = om.pointCount;
        const opts: OldPoint[] = new Array(opc);
        for (let k = 0; k < opc; ++k) {
            const op = om.points[k];
            opts[k] = {
                featureId: op.featureId,
                triangleIndex: op.triangleIndex,
                normalImpulse: op.normalImpulse,
            };
        }
        oldManifolds[j] = {
            normal: om.normal,
            frictionImpulse: om.frictionImpulse,
            rollingImpulse: om.rollingImpulse,
            twistImpulse: om.twistImpulse,
            pointCount: opc,
            points: opts,
        };
    }
    const consumed: boolean[] = new Array(oldManifoldCount).fill(false);

    // Allocate this step's clusters as a fresh column-resident block (may recycle a freed block).
    contact.manifolds = world.manifoldStore.alloc(contact.contactId, clusterCount);
    contact.manifoldCount = clusterCount;

    const matrixB = mat3.fromQuat(xfB.q);
    const offsetA = subPos(xfB.p, xfA.p);
    const normalMatchTolerance = f32(0.995);

    for (let i = 0; i < clusterCount; ++i) {
        const cm = clusters[i];
        const pointCount = cm.pointCount;

        const manifold = contact.manifolds[i];
        manifold.pointCount = pointCount;
        const clusterNormal = mat3.mulV(matrixB, cm.manifoldNormal);
        manifold.normal = clusterNormal;

        // Match to the best-aligned unconsumed old manifold to carry friction/twist impulses.
        let bestDot = normalMatchTolerance;
        let bestIndex = NULL_INDEX;
        for (let j = 0; j < oldManifoldCount; ++j) {
            if (consumed[j]) {
                continue;
            }
            const dot = vec3.dot(oldManifolds[j].normal, clusterNormal);
            if (dot > bestDot) {
                bestIndex = j;
                bestDot = dot;
            }
        }

        // Carry the matched manifold's friction/rolling/twist, or zero them for a fresh manifold (the
        // recycled block may still hold a prior occupant's impulses).
        let matchedManifold: OldManifold | null = null;
        if (bestIndex !== NULL_INDEX) {
            matchedManifold = oldManifolds[bestIndex];
            manifold.frictionImpulse = matchedManifold.frictionImpulse;
            manifold.rollingImpulse = matchedManifold.rollingImpulse;
            manifold.twistImpulse = matchedManifold.twistImpulse;
            consumed[bestIndex] = true;
        } else {
            manifold.frictionImpulse = { x: 0, y: 0, z: 0 };
            manifold.rollingImpulse = { x: 0, y: 0, z: 0 };
            manifold.twistImpulse = 0;
        }

        for (let j = 0; j < pointCount; ++j) {
            const source = cm.points[j];
            const mp = manifold.points[j];
            // Contact points are computed in frame B.
            const anchorB = mat3.mulV(matrixB, source.point);
            mp.anchorB = anchorB;
            mp.anchorA = vec3.add(anchorB, offsetA);
            mp.separation = f32(source.separation - restOffset);
            mp.baseSeparation = 0;
            mp.normalImpulse = 0;
            mp.totalNormalImpulse = 0;
            mp.normalVelocity = 0;
            const featureId = makeFeatureId(source.pair);
            mp.featureId = featureId;
            mp.triangleIndex = source.triangleIndex;
            mp.persisted = false;

            // Preserve normal impulse from a matching old point (by feature id + triangle).
            if (matchedManifold !== null) {
                const oldPointCount = matchedManifold.pointCount;
                for (let k = 0; k < oldPointCount; ++k) {
                    const oldPt = matchedManifold.points[k];
                    if (
                        featureId === oldPt.featureId &&
                        source.triangleIndex === oldPt.triangleIndex
                    ) {
                        mp.normalImpulse = oldPt.normalImpulse;
                        mp.persisted = true;
                        oldPt.triangleIndex = NULL_INDEX; // claimed
                        break;
                    }
                }
            }
        }
    }

    // Friction / restitution / tangent velocity, averaged over the mesh's per-triangle materials.
    const materialsA = getShapeMaterials(shapeA);
    const materialB = getShapeMaterials(shapeB)[0];
    let tangentVelocityA: Vec3 = { x: 0, y: 0, z: 0 };

    if (shapeA.materialCount > 0) {
        let friction = 0;
        let restitution = 0;
        let sampleCount = 0;

        const materialIndices = isMesh
            ? (mesh as Mesh).data.materialIndices
            : (heightField as HeightFieldData).materialIndices;

        for (let i = 0; i < clusterCount; ++i) {
            const manifold = contact.manifolds[i];
            const pointCount = manifold.pointCount;
            for (let j = 0; j < pointCount; ++j) {
                const triangleIndex = manifold.points[j].triangleIndex;
                // Mesh: one material per triangle, remapped by an optional compound child map. Height
                // field: one material per cell (triangleIndex >> 1), no map (b3ComputeMeshManifolds).
                let materialIndex: number;
                if (isMesh) {
                    materialIndex = materialIndices[triangleIndex];
                    if (materialMap !== null) {
                        materialIndex = materialMap[materialIndex];
                    }
                } else {
                    materialIndex = materialIndices[triangleIndex >> 1];
                }
                materialIndex = clampInt(materialIndex, 0, shapeA.materialCount - 1);
                const material = materialsA[materialIndex];
                friction = f32(
                    friction +
                        world.frictionCallback(
                            material.friction,
                            material.userMaterialId,
                            materialB.friction,
                            materialB.userMaterialId,
                        ),
                );
                restitution = f32(
                    restitution +
                        world.restitutionCallback(
                            material.restitution,
                            material.userMaterialId,
                            materialB.restitution,
                            materialB.userMaterialId,
                        ),
                );
                tangentVelocityA = vec3.add(tangentVelocityA, material.tangentVelocity);
                sampleCount = f32(sampleCount + 1.0);
            }
        }

        if (sampleCount > 0) {
            const invCount = f32(1.0 / sampleCount);
            contact.friction = f32(invCount * friction);
            contact.restitution = f32(invCount * restitution);
            tangentVelocityA = vec3.scale(invCount, tangentVelocityA);
        }
    } else {
        contact.friction = world.frictionCallback(
            materialsA[0].friction,
            materialsA[0].userMaterialId,
            materialB.friction,
            materialB.userMaterialId,
        );
        contact.restitution = world.restitutionCallback(
            materialsA[0].restitution,
            materialsA[0].userMaterialId,
            materialB.restitution,
            materialB.userMaterialId,
        );
        tangentVelocityA = materialsA[0].tangentVelocity;
    }

    tangentVelocityA = quat.rotate(xfA.q, tangentVelocityA);

    let radiusB = 0;
    if (shapeB.type === ShapeType.Sphere) {
        radiusB = (shapeB.sphere as Sphere).radius;
    } else if (shapeB.type === ShapeType.Capsule) {
        radiusB = (shapeB.capsule as Capsule).radius;
    } else if (shapeB.type === ShapeType.Hull) {
        radiusB = (shapeB.hull as HullData).innerRadius;
    }

    contact.rollingResistance = f32(materialB.rollingResistance * radiusB);

    const tangentVelocityB = quat.rotate(xfB.q, materialB.tangentVelocity);
    contact.tangentVelocity = vec3.sub(tangentVelocityA, tangentVelocityB);

    return true;
}
