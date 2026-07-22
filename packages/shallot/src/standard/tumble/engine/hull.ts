// Convex hulls: quickhull builder, baking (weld/merge/reduction), mass properties, box/cylinder
// generators. Ported op-for-op from Box3D's hull.c (Erin Catto; Dirk Gregorius contributed, MIT).
// fround discipline per the README.
//
// The C builder carves all working memory from one block with pointer-based intrusive lists and
// free lists for edge/face recycling. In TS the pointers are object references and the free lists
// vanish into GC: NewEdge/NewFace allocate fresh objects, Retire* just unlink. Slot reuse never
// affected geometry (nodes are fully initialized on creation), so the output is identical; the
// list *iteration order* is what the baked arrays inherit, and that is preserved exactly.
//
// The C b3HullData is a single blob with byte offsets into trailing arrays; the port stores plain
// arrays. The content `hash` is a deterministic DJB2 over that content (non-zero, stable) but is
// NOT byte-identical to the C struct hash — the world-state hash never consumes it (it hashes only
// body transforms + velocities), so hull geometry is what must stay bit-exact, not this field.

import { NULL_INDEX } from "./array";
import { OVERLAP_SLOP } from "./core";
import {
    type CastOutput,
    type DistanceInput,
    emptyCache,
    emptyCastOutput,
    type RayCastInput,
    type ShapeCastInput,
    type ShapeCastPairInput,
    type ShapeProxy,
    shapeCast,
    shapeDistance,
} from "./distance";
import type { Capsule, MassData } from "./geometry";
import {
    type AABB,
    aabb,
    absf,
    boxInertia,
    clampInt,
    computeCosSin,
    cos,
    FLT_EPSILON,
    FLT_MAX,
    f32,
    getLengthAndNormalize,
    type Mat3,
    mat3,
    maxElementIndex,
    maxf,
    minf,
    PI,
    type Plane,
    plane as planeOps,
    rotateInertia,
    scalarTripleProduct,
    sin,
    steiner,
    type Transform,
    type Vec3,
    vec3,
    xf,
} from "./math";
import type { PlaneResult } from "./mover";

// Final hull indices are uint8, so vertex/edge/face counts cap at 255.
const HULL_LIMIT = 255;

const MARK_VISIBLE = 0;
const MARK_DELETE = 1;

const SQRT3 = f32(1.732050808);

// B3_LINEAR_SLOP = 0.005 * lengthUnitsPerMeter; length units default to 1 (no world here).
const LINEAR_SLOP = f32(0.005);

/** A hull vertex: index of one half-edge with this vertex as origin (b3HullVertex). */
export type HullVertex = { edge: number };

/** Half-edge: next (CCW), twin, origin vertex, and left face — all uint8 indices (b3HullHalfEdge). */
export type HullHalfEdge = { next: number; twin: number; origin: number; face: number };

/** A hull face, identified by one of its half-edges (b3HullFace). */
export type HullFace = { edge: number };

/**
 * A convex hull (b3HullData). Counts mirror the parallel array lengths; the object is immutable
 * after construction.
 */
export type HullData = {
    aabb: AABB;
    surfaceArea: number;
    volume: number;
    innerRadius: number;
    center: Vec3;
    centralInertia: Mat3;
    vertexCount: number;
    edgeCount: number;
    faceCount: number;
    points: Vec3[];
    vertices: HullVertex[];
    edges: HullHalfEdge[];
    faces: HullFace[];
    planes: Plane[];
    hash: number;
    /** Index of this hull's record in the kernel's static geometry columns (geocolumns.ts), or
     * `NULL_INDEX` until it is interned into the hull database and uploaded. Refreshed on every
     * geometry rebuild; the convex narrowphase passes it to the kernel to read the hull from wasm. */
    geoIndex: number;
};

// --- quickhull builder ----------------------------------------------------------------------

// Intrusive doubly-linked circular list link. null prev/next mean "not in a list".
interface QHLink {
    prev: QHLink | null;
    next: QHLink | null;
}

class Sentinel implements QHLink {
    prev: QHLink | null = null;
    next: QHLink | null = null;
}

class QHVertex implements QHLink {
    prev: QHLink | null = null;
    next: QHLink | null = null;
    conflictFace: QHFace | null = null;
    position: Vec3;
    finalIndex = NULL_INDEX;
    reachable = false;
    constructor(position: Vec3) {
        this.position = position;
    }
}

// Edge ring (CCW) around the owning face; prev/next are ring links, not list links.
class QHHalfEdge {
    prev!: QHHalfEdge;
    next!: QHHalfEdge;
    origin!: QHVertex;
    face!: QHFace;
    twin!: QHHalfEdge;
    finalIndex = NULL_INDEX;
}

class QHFace implements QHLink {
    prev: QHLink | null = null;
    next: QHLink | null = null;
    edge: QHHalfEdge | null = null;
    mark = MARK_VISIBLE;
    area = 0;
    plane: Plane = { normal: { x: 0, y: 0, z: 0 }, offset: 0 };
    centroid: Vec3 = { x: 0, y: 0, z: 0 };
    maxConflictDistance = 0;
    conflictHead = new Sentinel();
    maxConflict: QHVertex | null = null;
    finalIndex = NULL_INDEX;
    flipped = false;
}

function listInit(h: QHLink): void {
    h.prev = h;
    h.next = h;
}
function listContains(n: QHLink): boolean {
    return n.prev !== null && n.next !== null;
}
function listInsert(node: QHLink, where: QHLink): void {
    node.prev = where.prev;
    node.next = where;
    (node.prev as QHLink).next = node;
    (node.next as QHLink).prev = node;
}
function listRemove(node: QHLink): void {
    (node.prev as QHLink).next = node.next;
    (node.next as QHLink).prev = node.prev;
    node.prev = null;
    node.next = null;
}
function listPushBack(head: QHLink, node: QHLink): void {
    listInsert(node, head.prev as QHLink);
}

type HorizonFrame = {
    face: QHFace;
    startEdge: QHHalfEdge;
    edge: QHHalfEdge;
    started: boolean;
};

class HullBuilder {
    tolerance = 0;
    minRadius = 0;
    minOutside = 0;
    interiorPoint: Vec3 = { x: 0, y: 0, z: 0 };

    orphanedList = new Sentinel();
    vertexList = new Sentinel();
    faceList = new Sentinel();

    horizon: QHHalfEdge[] = [];
    cone: QHFace[] = [];
    mergedFaces: QHFace[] = [];
    horizonStack: HorizonFrame[] = [];

    finalVertexCount = 0;
    finalHalfEdgeCount = 0;
    finalFaceCount = 0;

    constructor() {
        listInit(this.orphanedList);
        listInit(this.vertexList);
        listInit(this.faceList);
    }

    newVertex(position: Vec3): QHVertex {
        return new QHVertex(position);
    }

    // Build a face from three vertices, with its plane, area, centroid, and CCW edge ring.
    newFace(v1: QHVertex, v2: QHVertex, v3: QHVertex): QHFace {
        const face = new QHFace();
        const edge1 = new QHHalfEdge();
        const edge2 = new QHHalfEdge();
        const edge3 = new QHHalfEdge();

        const p1 = v1.position;
        const p2 = v2.position;
        const p3 = v3.position;

        const cross = vec3.cross(vec3.sub(p2, p1), vec3.sub(p3, p1));
        const { v: normal, length } = getLengthAndNormalize(cross);
        const pl: Plane = { normal, offset: vec3.dot(normal, p1) };
        const area = f32(0.5 * length);

        face.edge = edge1;
        face.mark = MARK_VISIBLE;
        face.area = area;
        face.centroid = vec3.scale(f32(1 / 3), vec3.add(p1, vec3.add(p2, p3)));
        face.plane = pl;
        face.flipped = planeOps.separation(pl, this.interiorPoint) > 0;
        listInit(face.conflictHead);

        edge1.prev = edge3;
        edge1.next = edge2;
        edge1.origin = v1;
        edge1.face = face;

        edge2.prev = edge1;
        edge2.next = edge3;
        edge2.origin = v2;
        edge2.face = face;

        edge3.prev = edge2;
        edge3.next = edge1;
        edge3.origin = v3;
        edge3.face = face;

        return face;
    }

    retireFace(face: QHFace): void {
        if (listContains(face)) {
            listRemove(face);
        }
        face.edge = null;
    }
}

function buildBounds(vertices: Vec3[]): AABB {
    let lower: Vec3 = { x: FLT_MAX, y: FLT_MAX, z: FLT_MAX };
    let upper: Vec3 = { x: -FLT_MAX, y: -FLT_MAX, z: -FLT_MAX };
    for (const v of vertices) {
        lower = vec3.min(lower, v);
        upper = vec3.max(upper, v);
    }
    return { lowerBound: lower, upperBound: upper };
}

// Farthest-apart pair of input points along whichever cardinal axis spans the most.
function findFarthestPointsAlongCardinalAxes(
    tolerance: number,
    points: Vec3[],
): { index1: number; index2: number } {
    const v0 = points[0];
    const minPt: Vec3[] = [{ ...v0 }, { ...v0 }, { ...v0 }];
    const maxPt: Vec3[] = [{ ...v0 }, { ...v0 }, { ...v0 }];
    const minIndex = [0, 0, 0];
    const maxIndex = [0, 0, 0];

    for (let i = 1; i < points.length; ++i) {
        const v = points[i];
        if (v.x < minPt[0].x) {
            minPt[0] = v;
            minIndex[0] = i;
        } else if (v.x > maxPt[0].x) {
            maxPt[0] = v;
            maxIndex[0] = i;
        }
        if (v.y < minPt[1].y) {
            minPt[1] = v;
            minIndex[1] = i;
        } else if (v.y > maxPt[1].y) {
            maxPt[1] = v;
            maxIndex[1] = i;
        }
        if (v.z < minPt[2].z) {
            minPt[2] = v;
            minIndex[2] = i;
        } else if (v.z > maxPt[2].z) {
            maxPt[2] = v;
            maxIndex[2] = i;
        }
    }

    const distance: Vec3 = {
        x: f32(maxPt[0].x - minPt[0].x),
        y: f32(maxPt[1].y - minPt[1].y),
        z: f32(maxPt[2].z - minPt[2].z),
    };
    const distanceArray = [distance.x, distance.y, distance.z];
    const axis = maxElementIndex(distance);

    if (distanceArray[axis] > f32(2 * tolerance)) {
        return { index1: minIndex[axis], index2: maxIndex[axis] };
    }
    return { index1: NULL_INDEX, index2: NULL_INDEX };
}

function findFarthestPointFromLine(
    index1: number,
    index2: number,
    tolerance: number,
    points: Vec3[],
): number {
    const a = points[index1];
    const b = points[index2];
    const ab = vec3.sub(b, a);
    const abLengthSqr = vec3.dot(ab, ab);

    const invAbLengthSqr = f32(1 / abLengthSqr);
    let maxDistanceSqr = f32(f32(4 * tolerance) * tolerance);
    let maxIndex = NULL_INDEX;

    for (let i = 0; i < points.length; ++i) {
        if (i === index1 || i === index2) continue;
        const ap = vec3.sub(points[i], a);
        const cross = vec3.cross(ap, ab);
        const distanceSqr = f32(vec3.dot(cross, cross) * invAbLengthSqr);
        if (distanceSqr > maxDistanceSqr) {
            maxDistanceSqr = distanceSqr;
            maxIndex = i;
        }
    }
    return maxIndex;
}

function findFarthestPointFromPlane(
    index1: number,
    index2: number,
    index3: number,
    tolerance: number,
    points: Vec3[],
): number {
    const pl = planeOps.fromPoints(points[index1], points[index2], points[index3]);
    let maxDistance = f32(2 * tolerance);
    let maxIndex = NULL_INDEX;

    for (let i = 0; i < points.length; ++i) {
        if (i === index1 || i === index2 || i === index3) continue;
        const distance = absf(planeOps.separation(pl, points[i]));
        if (distance > maxDistance) {
            maxDistance = distance;
            maxIndex = i;
        }
    }
    return maxIndex;
}

function isEdgeConvex(edge: QHHalfEdge, tolerance: number): boolean {
    const distance = planeOps.separation(edge.face.plane, edge.twin.face.centroid);
    return distance < -tolerance;
}
function isEdgeConcave(edge: QHHalfEdge, tolerance: number): boolean {
    const distance = planeOps.separation(edge.face.plane, edge.twin.face.centroid);
    return distance > tolerance;
}

function vertexCountOfFace(face: QHFace): number {
    let count = 0;
    let edge = face.edge as QHHalfEdge;
    do {
        count++;
        edge = edge.next;
    } while (edge !== face.edge);
    return count;
}

function linkFace(face: QHFace, index: number, twin: QHHalfEdge): void {
    let edge = face.edge as QHHalfEdge;
    while (index-- > 0) {
        edge = edge.next;
    }
    edge.twin = twin;
    twin.twin = edge;
}

function linkFaces(face1: QHFace, index1: number, face2: QHFace, index2: number): void {
    let edge1 = face1.edge as QHHalfEdge;
    while (index1-- > 0) edge1 = edge1.next;
    let edge2 = face2.edge as QHHalfEdge;
    while (index2-- > 0) edge2 = edge2.next;
    edge1.twin = edge2;
    edge2.twin = edge1;
}

// Recompute a face's plane from its full edge ring (Newell's method), robust to non-triangles.
function newellPlane(face: QHFace): void {
    let count = 0;
    let centroid: Vec3 = { x: 0, y: 0, z: 0 };
    const normal: Vec3 = { x: 0, y: 0, z: 0 };

    let edge = face.edge as QHHalfEdge;
    const origin = edge.origin.position;

    do {
        const twin = edge.twin;
        const v1 = vec3.sub(edge.origin.position, origin);
        const v2 = vec3.sub(twin.origin.position, origin);

        count++;
        centroid = vec3.add(centroid, v1);
        normal.x = f32(normal.x + f32(f32(v1.y - v2.y) * f32(v1.z + v2.z)));
        normal.y = f32(normal.y + f32(f32(v1.z - v2.z) * f32(v1.x + v2.x)));
        normal.z = f32(normal.z + f32(f32(v1.x - v2.x) * f32(v1.y + v2.y)));

        edge = edge.next;
    } while (edge !== face.edge);

    centroid = vec3.scale(f32(1 / count), centroid);
    centroid = vec3.add(centroid, origin);

    const length = vec3.length(normal);
    const unitNormal = vec3.scale(f32(1 / length), normal);

    face.centroid = centroid;
    face.plane = planeOps.fromNormalAndPoint(unitNormal, centroid);
    face.area = f32(0.5 * length);
}

function computeTolerance(b: HullBuilder, points: Vec3[]): void {
    const bounds = buildBounds(points);
    const maxAbs = vec3.max(vec3.abs(bounds.lowerBound), vec3.abs(bounds.upperBound));

    const maxSum = f32(f32(maxAbs.x + maxAbs.y) + maxAbs.z);
    const maxCoord = maxf(maxAbs.x, maxf(maxAbs.y, maxAbs.z));
    const maxDistance = minf(f32(SQRT3 * maxCoord), maxSum);

    const tolerance = f32(f32(f32(f32(3 * maxDistance) * f32(1.01)) + maxCoord) * FLT_EPSILON);

    b.tolerance = tolerance;
    b.minRadius = f32(4 * b.tolerance);
    b.minOutside = f32(2 * b.minRadius);
}

function buildInitialHull(b: HullBuilder, points: Vec3[]): boolean {
    const { index1, index2 } = findFarthestPointsAlongCardinalAxes(b.tolerance, points);
    if (index1 < 0 || index2 < 0) return false;

    const index3 = findFarthestPointFromLine(index1, index2, b.tolerance, points);
    if (index3 < 0) return false;

    let idx2 = index2;
    let idx3 = index3;
    const index4 = findFarthestPointFromPlane(index1, idx2, idx3, b.tolerance, points);
    if (index4 < 0) return false;

    const v1 = vec3.sub(points[index1], points[index4]);
    const v2 = vec3.sub(points[idx2], points[index4]);
    const v3 = vec3.sub(points[idx3], points[index4]);

    if (scalarTripleProduct(v1, v2, v3) < 0) {
        const temp = idx2;
        idx2 = idx3;
        idx3 = temp;
    }

    let interior: Vec3 = { x: 0, y: 0, z: 0 };
    interior = vec3.add(interior, points[index1]);
    interior = vec3.add(interior, points[idx2]);
    interior = vec3.add(interior, points[idx3]);
    interior = vec3.add(interior, points[index4]);
    b.interiorPoint = vec3.scale(0.25, interior);

    const vertex1 = b.newVertex(points[index1]);
    listPushBack(b.vertexList, vertex1);
    const vertex2 = b.newVertex(points[idx2]);
    listPushBack(b.vertexList, vertex2);
    const vertex3 = b.newVertex(points[idx3]);
    listPushBack(b.vertexList, vertex3);
    const vertex4 = b.newVertex(points[index4]);
    listPushBack(b.vertexList, vertex4);

    const face1 = b.newFace(vertex1, vertex2, vertex3);
    listPushBack(b.faceList, face1);
    const face2 = b.newFace(vertex4, vertex2, vertex1);
    listPushBack(b.faceList, face2);
    const face3 = b.newFace(vertex4, vertex3, vertex2);
    listPushBack(b.faceList, face3);
    const face4 = b.newFace(vertex4, vertex1, vertex3);
    listPushBack(b.faceList, face4);

    linkFaces(face1, 0, face2, 1);
    linkFaces(face1, 1, face3, 1);
    linkFaces(face1, 2, face4, 1);
    linkFaces(face2, 0, face3, 2);
    linkFaces(face3, 0, face4, 2);
    linkFaces(face4, 0, face2, 2);

    for (let index = 0; index < points.length; ++index) {
        if (index === index1 || index === idx2 || index === idx3 || index === index4) continue;

        const point = points[index];
        let maxDistance = b.minOutside;
        let maxFace: QHFace | null = null;

        for (let node = b.faceList.next; node !== b.faceList; node = (node as QHLink).next) {
            const face = node as QHFace;
            const distance = planeOps.separation(face.plane, point);
            if (distance > maxDistance) {
                maxDistance = distance;
                maxFace = face;
            }
        }

        if (maxFace !== null) {
            const vertex = b.newVertex(point);
            vertex.conflictFace = maxFace;
            listPushBack(maxFace.conflictHead, vertex);
            if (maxDistance > maxFace.maxConflictDistance) {
                maxFace.maxConflictDistance = maxDistance;
                maxFace.maxConflict = vertex;
            }
        }
    }

    return true;
}

function recacheConflicts(face: QHFace, minOutside: number): void {
    let maxVertex: QHVertex | null = null;
    let maxDistance = minOutside;

    for (
        let node = face.conflictHead.next;
        node !== face.conflictHead;
        node = (node as QHLink).next
    ) {
        const vertex = node as QHVertex;
        const distance = planeOps.separation(face.plane, vertex.position);
        if (distance > maxDistance) {
            maxDistance = distance;
            maxVertex = vertex;
        }
    }

    face.maxConflict = maxVertex;
    face.maxConflictDistance = maxDistance;
}

function nextConflictVertex(b: HullBuilder): QHVertex | null {
    let maxVertex: QHVertex | null = null;
    let maxDistance = b.minOutside;

    for (let node = b.faceList.next; node !== b.faceList; node = (node as QHLink).next) {
        const face = node as QHFace;
        if (face.maxConflict !== null && face.maxConflictDistance > maxDistance) {
            maxDistance = face.maxConflictDistance;
            maxVertex = face.maxConflict;
        }
    }
    return maxVertex;
}

function drainConflictList(b: HullBuilder, face: QHFace): void {
    let node = face.conflictHead.next;
    while (node !== face.conflictHead) {
        const orphan = node as QHVertex;
        node = (node as QHLink).next;
        orphan.conflictFace = null;
        listRemove(orphan);
        listPushBack(b.orphanedList, orphan);
    }
}

function enterHorizonFace(
    b: HullBuilder,
    face: QHFace,
    entryEdge: QHHalfEdge | null,
): HorizonFrame {
    face.mark = MARK_DELETE;
    drainConflictList(b, face);

    if (entryEdge !== null) {
        return { face, started: false, startEdge: entryEdge, edge: entryEdge.next };
    }
    return {
        face,
        started: false,
        startEdge: face.edge as QHHalfEdge,
        edge: face.edge as QHHalfEdge,
    };
}

function buildHorizon(b: HullBuilder, apex: QHVertex, seed: QHFace): void {
    const stack = b.horizonStack;
    stack.length = 0;
    stack.push(enterHorizonFace(b, seed, null));

    while (stack.length > 0) {
        const frame = stack[stack.length - 1];

        if (frame.started && frame.edge === frame.startEdge) {
            stack.pop();
            continue;
        }
        frame.started = true;

        const edge = frame.edge;
        const twin = edge.twin;
        frame.edge = edge.next;

        if (twin.face.mark !== MARK_VISIBLE) {
            continue;
        }

        const distance = planeOps.separation(twin.face.plane, apex.position);
        if (distance > b.minRadius) {
            stack.push(enterHorizonFace(b, twin.face, twin));
        } else {
            b.horizon.push(edge);
        }
    }
}

function buildCone(b: HullBuilder, apex: QHVertex): void {
    for (let i = 0; i < b.horizon.length; ++i) {
        const edge = b.horizon[i];
        const face = b.newFace(apex, edge.origin, edge.twin.origin);
        b.cone.push(face);
        linkFace(face, 1, edge.twin);
    }

    let face1 = b.cone[b.cone.length - 1];
    for (let i = 0; i < b.cone.length; ++i) {
        const face2 = b.cone[i];
        linkFaces(face1, 2, face2, 0);
        face1 = face2;
    }
}

function connectEdges(b: HullBuilder, prev: QHHalfEdge, next: QHHalfEdge): void {
    // If prev and next share the same opposing face, that face becomes redundant and is merged out.
    if (prev.twin.face === next.twin.face) {
        if (next.face.edge === next) {
            next.face.edge = prev;
        }

        let twin: QHHalfEdge;
        if (vertexCountOfFace(prev.twin.face) === 3) {
            const opposingFace = prev.twin.face;
            twin = next.twin.prev.twin;

            opposingFace.mark = MARK_DELETE;
            b.mergedFaces.push(opposingFace);

            prev.next = next.next;
            prev.next.prev = prev;

            prev.twin = twin;
            twin.twin = prev;

            listRemove(next.origin);
        } else {
            twin = next.twin;

            if (twin.face.edge === prev.twin) {
                twin.face.edge = twin;
            }

            twin.next = prev.twin.next;
            twin.next.prev = twin;

            prev.next = next.next;
            prev.next.prev = prev;

            prev.twin = twin;
            twin.twin = prev;

            listRemove(next.origin);
        }

        newellPlane(twin.face);
        recacheConflicts(twin.face, b.minOutside);
    } else {
        prev.next = next;
        next.prev = prev;
    }
}

function absorbFaces(b: HullBuilder, face: QHFace): void {
    for (let i = 0; i < b.mergedFaces.length; ++i) {
        const head = b.mergedFaces[i].conflictHead;

        let node = head.next;
        while (node !== head) {
            const vertex = node as QHVertex;
            node = (node as QHLink).next;

            listRemove(vertex);

            const distance = planeOps.separation(face.plane, vertex.position);
            if (distance > b.minOutside) {
                listPushBack(face.conflictHead, vertex);
                vertex.conflictFace = face;
                if (distance > face.maxConflictDistance) {
                    face.maxConflictDistance = distance;
                    face.maxConflict = vertex;
                }
            } else {
                listPushBack(b.orphanedList, vertex);
                vertex.conflictFace = null;
            }
        }

        b.retireFace(b.mergedFaces[i]);
    }
}

function connectFaces(b: HullBuilder, edge: QHHalfEdge): void {
    const face = edge.face;
    const twin = edge.twin;

    let edgePrev = edge.prev;
    let edgeNext = edge.next;
    let twinPrev = twin.prev;
    let twinNext = twin.next;

    while (edgePrev.twin.face === twin.face) {
        edgePrev = edgePrev.prev;
        twinNext = twinNext.next;
    }

    while (edgeNext.twin.face === twin.face) {
        edgeNext = edgeNext.next;
        twinPrev = twinPrev.prev;
    }

    face.edge = edgePrev;

    // ConnectFaces does not nest; mergedFaces is single-buffered.
    b.mergedFaces.length = 0;
    b.mergedFaces.push(twin.face);
    twin.face.mark = MARK_DELETE;
    twin.face.edge = null;

    for (let absorbed = twinNext; absorbed !== twinPrev.next; absorbed = absorbed.next) {
        absorbed.face = face;
    }

    // C retires the two half-edge ranges [edgePrev.next, edgeNext) and [twinPrev.next, twinNext)
    // to its free list here; in TS the rewire below drops them and GC reclaims them.
    connectEdges(b, edgePrev, twinNext);
    connectEdges(b, twinPrev, edgeNext);

    newellPlane(face);
    recacheConflicts(face, b.minOutside);

    absorbFaces(b, face);
}

function mergeConcave(b: HullBuilder, face: QHFace): boolean {
    let edge = face.edge as QHHalfEdge;
    do {
        const twin = edge.twin;
        if (isEdgeConcave(edge, b.minRadius) || isEdgeConcave(twin, b.minRadius)) {
            connectFaces(b, edge);
            return true;
        }
        edge = edge.next;
    } while (edge !== face.edge);
    return false;
}

function mergeCoplanar(b: HullBuilder, face: QHFace): boolean {
    let edge = face.edge as QHHalfEdge;
    do {
        const twin = edge.twin;
        if (!isEdgeConvex(edge, b.minRadius) || !isEdgeConvex(twin, b.minRadius)) {
            connectFaces(b, edge);
            return true;
        }
        edge = edge.next;
    } while (edge !== face.edge);
    return false;
}

function mergeFaces(b: HullBuilder): void {
    for (let i = 0; i < b.cone.length; ++i) {
        const face = b.cone[i];
        if (face.mark === MARK_VISIBLE && face.flipped) {
            face.flipped = false;

            let bestArea = 0;
            let bestEdge: QHHalfEdge | null = null;

            let edge = face.edge as QHHalfEdge;
            do {
                const twin = edge.twin;
                const area = twin.face.area;
                if (area > bestArea) {
                    bestArea = area;
                    bestEdge = edge;
                }
                edge = edge.next;
            } while (edge !== face.edge);

            connectFaces(b, bestEdge as QHHalfEdge);
        }
    }

    for (let i = 0; i < b.cone.length; ++i) {
        const face = b.cone[i];
        if (face.mark === MARK_VISIBLE) {
            while (mergeConcave(b, face)) {}
        }
    }

    for (let i = 0; i < b.cone.length; ++i) {
        const face = b.cone[i];
        if (face.mark === MARK_VISIBLE) {
            while (mergeCoplanar(b, face)) {}
        }
    }
}

function resolveVertices(b: HullBuilder): void {
    let node = b.orphanedList.next;
    while (node !== b.orphanedList) {
        const vertex = node as QHVertex;
        node = (node as QHLink).next;
        listRemove(vertex);

        let maxDistance = b.minOutside;
        let maxFace: QHFace | null = null;

        for (let i = 0; i < b.cone.length; ++i) {
            if (b.cone[i].mark === MARK_VISIBLE) {
                const distance = planeOps.separation(b.cone[i].plane, vertex.position);
                if (distance > maxDistance) {
                    maxDistance = distance;
                    maxFace = b.cone[i];
                }
            }
        }

        if (maxFace !== null) {
            listPushBack(maxFace.conflictHead, vertex);
            vertex.conflictFace = maxFace;
            if (maxDistance > maxFace.maxConflictDistance) {
                maxFace.maxConflictDistance = maxDistance;
                maxFace.maxConflict = vertex;
            }
        }
    }
}

function resolveFaces(b: HullBuilder): void {
    let node = b.faceList.next;
    while (node !== b.faceList) {
        const face = node as QHFace;
        node = (node as QHLink).next;
        if (face.mark === MARK_DELETE && listContains(face)) {
            listRemove(face);
        }
    }

    for (let i = 0; i < b.cone.length; ++i) {
        const face = b.cone[i];
        if (face.mark === MARK_DELETE) continue;
        listPushBack(b.faceList, face);
    }
}

function addVertexToHull(b: HullBuilder, vertex: QHVertex): void {
    const face = vertex.conflictFace as QHFace;
    vertex.conflictFace = null;
    listRemove(vertex);
    listPushBack(b.vertexList, vertex);

    b.horizon.length = 0;
    buildHorizon(b, vertex, face);

    b.cone.length = 0;
    buildCone(b, vertex);

    mergeFaces(b);
    resolveVertices(b);
    resolveFaces(b);
}

function cleanHull(b: HullBuilder, origin: Vec3): void {
    let faceCount = 0;
    let halfEdgeCount = 0;

    for (
        let faceNode = b.faceList.next;
        faceNode !== b.faceList;
        faceNode = (faceNode as QHLink).next
    ) {
        const face = faceNode as QHFace;
        let edge = face.edge as QHHalfEdge;
        do {
            edge.origin.reachable = true;
            edge = edge.next;
            halfEdgeCount++;
        } while (edge !== face.edge);

        face.plane.offset = f32(face.plane.offset + vec3.dot(face.plane.normal, origin));
        face.centroid = vec3.add(face.centroid, origin);
        faceCount++;
    }

    let vertexCount = 0;
    let node = b.vertexList.next;
    while (node !== b.vertexList) {
        const vertex = node as QHVertex;
        node = (node as QHLink).next;
        if (!vertex.reachable) {
            listRemove(vertex);
        } else {
            vertex.position = vec3.add(vertex.position, origin);
            vertexCount++;
        }
    }

    b.interiorPoint = vec3.add(b.interiorPoint, origin);

    b.finalVertexCount = vertexCount;
    b.finalHalfEdgeCount = halfEdgeCount;
    b.finalFaceCount = faceCount;
}

function hasHull(b: HullBuilder): boolean {
    const v = b.finalVertexCount;
    const e = (b.finalHalfEdgeCount / 2) | 0;
    const f = b.finalFaceCount;
    return v - e + f === 2 && f >= 4;
}

function construct(b: HullBuilder, points: Vec3[], maxVertexCount: number, origin: Vec3): boolean {
    if (points.length < 4) return false;

    const shifted = points.map((p) => vec3.sub(p, origin));

    computeTolerance(b, shifted);
    if (!buildInitialHull(b, shifted)) return false;

    let budget = clampInt(maxVertexCount - 4, 0, HULL_LIMIT - 4);

    let vertex = nextConflictVertex(b);
    while (vertex && budget > 0) {
        addVertexToHull(b, vertex);
        vertex = nextConflictVertex(b);
        budget -= 1;
    }

    cleanHull(b, origin);
    return hasHull(b);
}

// --- baking ---------------------------------------------------------------------------------

function updateHullBounds(hull: HullData): void {
    let lower = hull.points[0];
    let upper = hull.points[0];
    for (let i = 1; i < hull.vertexCount; ++i) {
        const p = hull.points[i];
        lower = vec3.min(lower, p);
        upper = vec3.max(upper, p);
    }
    hull.aabb = { lowerBound: lower, upperBound: upper };
}

// M. Kallay — "Computing the Moment of Inertia of a Solid Defined by a Triangle Mesh". Fills
// volume, surfaceArea, innerRadius, center, and centralInertia. Returns false on a degenerate hull.
function updateHullBulkProperties(hull: HullData): boolean {
    const points = hull.points;
    const faces = hull.faces;
    const edges = hull.edges;
    const planes = hull.planes;

    let area = 0;
    let volume = 0;
    let center: Vec3 = { x: 0, y: 0, z: 0 };

    const origin = points[0];

    let xx = 0;
    let xy = 0;
    let yy = 0;
    let xz = 0;
    let zz = 0;
    let yz = 0;

    for (let faceIndex = 0; faceIndex < hull.faceCount; ++faceIndex) {
        const face = faces[faceIndex];
        const edge1 = edges[face.edge];
        let edge2 = edges[edge1.next];
        let edge3 = edges[edge2.next];

        const v1 = vec3.sub(points[edge1.origin], origin);

        do {
            const v2 = vec3.sub(points[edge2.origin], origin);
            const v3 = vec3.sub(points[edge3.origin], origin);

            area = f32(area + vec3.length(vec3.cross(vec3.sub(v2, v1), vec3.sub(v3, v1))));

            const det = scalarTripleProduct(v1, v2, v3);
            volume = f32(volume + det);

            const v4 = vec3.add(v1, vec3.add(v2, v3));
            center = vec3.add(center, vec3.scale(det, v4));

            xx = f32(xx + f32(det * sumSquares(v1.x, v2.x, v3.x, v4.x)));
            yy = f32(yy + f32(det * sumSquares(v1.y, v2.y, v3.y, v4.y)));
            zz = f32(zz + f32(det * sumSquares(v1.z, v2.z, v3.z, v4.z)));
            xy = f32(xy + f32(det * sumProducts(v1.x, v1.y, v2.x, v2.y, v3.x, v3.y, v4.x, v4.y)));
            xz = f32(xz + f32(det * sumProducts(v1.x, v1.z, v2.x, v2.z, v3.x, v3.z, v4.x, v4.z)));
            yz = f32(yz + f32(det * sumProducts(v1.y, v1.z, v2.y, v2.z, v3.y, v3.z, v4.y, v4.z)));

            edge2 = edge3;
            edge3 = edges[edge3.next];
        } while (edge1 !== edge3);
    }

    const localCenter = volume > 0 ? vec3.scale(f32(0.25 / volume), center) : { x: 0, y: 0, z: 0 };
    center = vec3.add(localCenter, origin);

    let radius = FLT_MAX;
    for (let faceIndex = 0; faceIndex < hull.faceCount; ++faceIndex) {
        const distance = planeOps.separation(planes[faceIndex], center);
        radius = minf(radius, f32(-distance));
    }

    const inertia: Mat3 = {
        cx: { x: f32(yy + zz), y: f32(-xy), z: f32(-xz) },
        cy: { x: f32(-xy), y: f32(xx + zz), z: f32(-yz) },
        cz: { x: f32(-xz), y: f32(-yz), z: f32(xx + yy) },
    };

    const mass = f32(volume / 6);

    let centralInertia = mat3.scale(f32(1 / 120), inertia);
    centralInertia = mat3.sub(centralInertia, steiner(mass, localCenter));

    hull.center = center;
    hull.centralInertia = centralInertia;
    hull.volume = mass;
    hull.surfaceArea = f32(0.5 * area);
    hull.innerRadius = radius;

    return mass > 0 && volume > 0 && area > 0 && radius > 0;
}

// a*a + b*b + c*c + d*d, left-to-right (matches the C accumulation order).
function sumSquares(a: number, b: number, c: number, d: number): number {
    const aa = f32(a * a);
    const bb = f32(b * b);
    const cc = f32(c * c);
    const dd = f32(d * d);
    return f32(f32(f32(aa + bb) + cc) + dd);
}

// a1*a2 + b1*b2 + c1*c2 + d1*d2, left-to-right.
function sumProducts(
    a1: number,
    a2: number,
    b1: number,
    b2: number,
    c1: number,
    c2: number,
    d1: number,
    d2: number,
): number {
    const p1 = f32(a1 * a2);
    const p2 = f32(b1 * b2);
    const p3 = f32(c1 * c2);
    const p4 = f32(d1 * d2);
    return f32(f32(f32(p1 + p2) + p3) + p4);
}

const HASH_INIT = 5381;

// DJB2 over the hull's geometric content — deterministic and non-zero, mirroring b3Hash's mix.
// Not byte-identical to the C struct hash (see file header); the world hash never consumes it.
function hashHull(hull: HullData): number {
    let h = HASH_INIT >>> 0;
    const buf = new ArrayBuffer(4);
    const view = new DataView(buf);
    const bytes = new Uint8Array(buf);
    const mixFloat = (f: number) => {
        view.setFloat32(0, f, true);
        for (let i = 0; i < 4; ++i) h = ((h << 5) + h + bytes[i]) >>> 0;
    };
    const mixInt = (n: number) => {
        view.setInt32(0, n | 0, true);
        for (let i = 0; i < 4; ++i) h = ((h << 5) + h + bytes[i]) >>> 0;
    };

    mixInt(hull.vertexCount);
    mixInt(hull.edgeCount);
    mixInt(hull.faceCount);
    for (const p of hull.points) {
        mixFloat(p.x);
        mixFloat(p.y);
        mixFloat(p.z);
    }
    for (const e of hull.edges) {
        mixInt(e.next);
        mixInt(e.twin);
        mixInt(e.origin);
        mixInt(e.face);
    }
    for (const f of hull.faces) mixInt(f.edge);
    for (const pl of hull.planes) {
        mixFloat(pl.normal.x);
        mixFloat(pl.normal.y);
        mixFloat(pl.normal.z);
        mixFloat(pl.offset);
    }
    return h !== 0 ? h : 1;
}

// --- public hull construction ---------------------------------------------------------------

/** Build a convex hull from a point cloud, clamping the vertex budget to [4, 255] (b3CreateHull). */
export function createHull(points: Vec3[], maxVertexCount: number): HullData | null {
    if (points.length < 4) return null;

    const origin = points[0];
    const clampedMax = clampInt(maxVertexCount, 4, HULL_LIMIT);

    const b = new HullBuilder();
    if (!construct(b, points, clampedMax, origin)) return null;

    if (
        b.finalVertexCount >= HULL_LIMIT ||
        b.finalFaceCount >= HULL_LIMIT ||
        b.finalHalfEdgeCount >= HULL_LIMIT
    ) {
        return null;
    }

    // Walk the lists into arrays, stamping finalIndex so the resolution pass is O(E + F).
    const tempVertices: QHVertex[] = [];
    for (let node = b.vertexList.next; node !== b.vertexList; node = (node as QHLink).next) {
        const vertex = node as QHVertex;
        vertex.finalIndex = tempVertices.length;
        tempVertices.push(vertex);
    }

    const tempFaces: QHFace[] = [];
    const tempEdges: QHHalfEdge[] = [];
    for (
        let faceNode = b.faceList.next;
        faceNode !== b.faceList;
        faceNode = (faceNode as QHLink).next
    ) {
        const face = faceNode as QHFace;
        face.finalIndex = tempFaces.length;
        tempFaces.push(face);

        let edge = face.edge as QHHalfEdge;
        do {
            if (edge.finalIndex < 0) {
                edge.finalIndex = tempEdges.length;
                tempEdges.push(edge);
                edge.twin.finalIndex = tempEdges.length;
                tempEdges.push(edge.twin);
            }
            edge = edge.next;
        } while (edge !== face.edge);
    }

    const vertexCount = tempVertices.length;
    const edgeCount = tempEdges.length;
    const faceCount = tempFaces.length;

    const vertices: HullVertex[] = [];
    const finalPoints: Vec3[] = [];
    for (let index = 0; index < vertexCount; ++index) {
        vertices.push({ edge: 0 });
        finalPoints.push(tempVertices[index].position);
    }

    const edges: HullHalfEdge[] = [];
    for (let index = 0; index < edgeCount; ++index) {
        const edge = tempEdges[index];
        edges.push({
            next: edge.next.finalIndex,
            twin: edge.twin.finalIndex,
            face: edge.face.finalIndex,
            origin: edge.origin.finalIndex,
        });
        vertices[edge.origin.finalIndex].edge = index;
    }

    const faces: HullFace[] = [];
    const planes: Plane[] = [];
    for (let index = 0; index < faceCount; ++index) {
        const face = tempFaces[index];
        faces.push({ edge: (face.edge as QHHalfEdge).finalIndex });
        planes.push(face.plane);
    }

    const hull: HullData = {
        aabb: { lowerBound: { x: 0, y: 0, z: 0 }, upperBound: { x: 0, y: 0, z: 0 } },
        surfaceArea: 0,
        volume: 0,
        innerRadius: 0,
        center: { x: 0, y: 0, z: 0 },
        centralInertia: mat3.zero(),
        vertexCount,
        edgeCount,
        faceCount,
        points: finalPoints,
        vertices,
        edges,
        faces,
        planes,
        hash: 0,
        geoIndex: NULL_INDEX,
    };

    updateHullBounds(hull);
    if (!updateHullBulkProperties(hull)) return null;

    hull.hash = hashHull(hull);
    return hull;
}

/** Deep clone a hull (b3CloneHull). */
export function cloneHull(hull: HullData): HullData {
    return {
        aabb: {
            lowerBound: { ...hull.aabb.lowerBound },
            upperBound: { ...hull.aabb.upperBound },
        },
        surfaceArea: hull.surfaceArea,
        volume: hull.volume,
        innerRadius: hull.innerRadius,
        center: { ...hull.center },
        centralInertia: {
            cx: { ...hull.centralInertia.cx },
            cy: { ...hull.centralInertia.cy },
            cz: { ...hull.centralInertia.cz },
        },
        vertexCount: hull.vertexCount,
        edgeCount: hull.edgeCount,
        faceCount: hull.faceCount,
        points: hull.points.map((p) => ({ ...p })),
        vertices: hull.vertices.map((v) => ({ ...v })),
        edges: hull.edges.map((e) => ({ ...e })),
        faces: hull.faces.map((f) => ({ ...f })),
        planes: hull.planes.map((pl) => ({ normal: { ...pl.normal }, offset: pl.offset })),
        hash: hull.hash,
        geoIndex: NULL_INDEX,
    };
}

/** Tessellated n-gon prism as a hull, axis along y (b3CreateCylinder). */
export function createCylinder(
    height: number,
    radius: number,
    yOffset: number,
    sides: number,
): HullData {
    const pointCount = 2 * sides;
    const points: Vec3[] = new Array(pointCount);

    let alpha = 0;
    const deltaAlpha = f32(f32(2 * PI) / sides);

    for (let index = 0; index < sides; ++index) {
        const sinAlpha = sin(alpha);
        const cosAlpha = cos(alpha);
        points[2 * index + 0] = {
            x: f32(radius * cosAlpha),
            y: yOffset,
            z: f32(radius * sinAlpha),
        };
        points[2 * index + 1] = {
            x: f32(radius * cosAlpha),
            y: f32(yOffset + height),
            z: f32(radius * sinAlpha),
        };
        alpha = f32(alpha + deltaAlpha);
    }

    return createHull(points, pointCount) as HullData;
}

/** Tessellated truncated cone as a hull, axis along y (b3CreateCone). */
export function createCone(
    height: number,
    radius1: number,
    radius2: number,
    slices: number,
): HullData {
    const pointCount = 2 * slices;
    const points: Vec3[] = new Array(pointCount);

    let alpha = 0;
    const deltaAlpha = f32(f32(2 * PI) / slices);

    for (let index = 0; index < slices; ++index) {
        const sinAlpha = sin(alpha);
        const cosAlpha = cos(alpha);
        points[2 * index + 0] = { x: f32(radius1 * cosAlpha), y: 0, z: f32(radius1 * sinAlpha) };
        points[2 * index + 1] = {
            x: f32(radius2 * cosAlpha),
            y: height,
            z: f32(radius2 * sinAlpha),
        };
        alpha = f32(alpha + deltaAlpha);
    }

    return createHull(points, pointCount) as HullData;
}

/** Fibonacci-lattice rock hull of 10 points (b3CreateRock). */
export function createRock(radius: number): HullData {
    const pointCount = 10;
    const phi = f32(f32(1 + f32(Math.sqrt(5))) / 2);
    const theta = f32(f32(2 * PI) / phi);

    const points: Vec3[] = new Array(pointCount);
    let cs = { cosine: 1, sine: 0 };
    const deltaCS = computeCosSin(theta);

    for (let i = 0; i < pointCount; ++i) {
        const z = f32(1 - f32(f32(f32(2 * i) + 1) / pointCount));
        const radiusXY = f32(Math.sqrt(f32(1 - f32(z * z))));

        points[i] = {
            x: f32(f32(radius * radiusXY) * cs.cosine),
            y: f32(f32(radius * radiusXY) * cs.sine),
            z: f32(radius * z),
        };

        const cs0 = cs;
        cs = {
            cosine: f32(f32(deltaCS.cosine * cs0.cosine) - f32(deltaCS.sine * cs0.sine)),
            sine: f32(f32(deltaCS.sine * cs0.cosine) + f32(deltaCS.cosine * cs0.sine)),
        };
    }

    return createHull(points, pointCount) as HullData;
}

// --- box hull -------------------------------------------------------------------------------

// Constant box topology (b3BoxHull s_boxHull). makeTransformedBoxHull copies it and fills the
// runtime fields (points, planes, aabb, mass properties, hash).
const BOX_VERTEX_EDGE = [8, 1, 0, 9, 13, 3, 5, 11];
const BOX_EDGES: HullHalfEdge[] = [
    { next: 2, twin: 1, origin: 2, face: 0 },
    { next: 17, twin: 0, origin: 1, face: 5 },
    { next: 4, twin: 3, origin: 1, face: 0 },
    { next: 20, twin: 2, origin: 5, face: 3 },
    { next: 6, twin: 5, origin: 5, face: 0 },
    { next: 23, twin: 4, origin: 6, face: 4 },
    { next: 0, twin: 7, origin: 6, face: 0 },
    { next: 18, twin: 6, origin: 2, face: 2 },
    { next: 10, twin: 9, origin: 0, face: 1 },
    { next: 21, twin: 8, origin: 3, face: 5 },
    { next: 12, twin: 11, origin: 3, face: 1 },
    { next: 16, twin: 10, origin: 7, face: 2 },
    { next: 14, twin: 13, origin: 7, face: 1 },
    { next: 19, twin: 12, origin: 4, face: 4 },
    { next: 8, twin: 15, origin: 4, face: 1 },
    { next: 22, twin: 14, origin: 0, face: 3 },
    { next: 7, twin: 17, origin: 3, face: 2 },
    { next: 9, twin: 16, origin: 2, face: 5 },
    { next: 11, twin: 19, origin: 6, face: 2 },
    { next: 5, twin: 18, origin: 7, face: 4 },
    { next: 15, twin: 21, origin: 1, face: 3 },
    { next: 1, twin: 20, origin: 0, face: 5 },
    { next: 3, twin: 23, origin: 4, face: 3 },
    { next: 13, twin: 22, origin: 5, face: 4 },
];
const BOX_FACE_EDGE = [0, 8, 16, 20, 19, 21];

/** Box hull with the given half-widths under a local transform (b3MakeTransformedBoxHull). */
export function makeTransformedBoxHull(
    hx: number,
    hy: number,
    hz: number,
    transform: Transform,
): HullData {
    const minH = f32(0.2 * LINEAR_SLOP);
    // The C API takes f32 half-extents; fround the inputs so an f64 arg (e.g. 0.2) matches b3MakeBoxHull.
    const h = vec3.max({ x: minH, y: minH, z: minH }, { x: f32(hx), y: f32(hy), z: f32(hz) });

    const lower = vec3.neg(h);
    const upper = h;

    const hxy = f32(h.x * h.y);
    const hxz = f32(h.x * h.z);
    const hyz = f32(h.y * h.z);

    const volume = f32(f32(f32(8 * h.x) * h.y) * h.z);

    const planes: Plane[] = [
        planeOps.transform(transform, planeOps.fromNormalAndPoint(vec3.neg(vec3.axisX()), lower)),
        planeOps.transform(transform, planeOps.fromNormalAndPoint(vec3.axisX(), upper)),
        planeOps.transform(transform, planeOps.fromNormalAndPoint(vec3.neg(vec3.axisY()), lower)),
        planeOps.transform(transform, planeOps.fromNormalAndPoint(vec3.axisY(), upper)),
        planeOps.transform(transform, planeOps.fromNormalAndPoint(vec3.neg(vec3.axisZ()), lower)),
        planeOps.transform(transform, planeOps.fromNormalAndPoint(vec3.axisZ(), upper)),
    ];

    const points: Vec3[] = [
        xf.point(transform, { x: h.x, y: h.y, z: h.z }),
        xf.point(transform, { x: -h.x, y: h.y, z: h.z }),
        xf.point(transform, { x: -h.x, y: -h.y, z: h.z }),
        xf.point(transform, { x: h.x, y: -h.y, z: h.z }),
        xf.point(transform, { x: h.x, y: h.y, z: -h.z }),
        xf.point(transform, { x: -h.x, y: h.y, z: -h.z }),
        xf.point(transform, { x: -h.x, y: -h.y, z: -h.z }),
        xf.point(transform, { x: h.x, y: -h.y, z: -h.z }),
    ];

    const hull: HullData = {
        aabb: aabb.transform(transform, { lowerBound: lower, upperBound: upper }),
        surfaceArea: f32(8 * f32(f32(hxy + hxz) + hyz)),
        volume,
        innerRadius: minf(h.x, minf(h.y, h.z)),
        center: transform.p,
        centralInertia: rotateInertia(transform.q, boxInertia(volume, lower, upper)),
        vertexCount: 8,
        edgeCount: 24,
        faceCount: 6,
        points,
        vertices: BOX_VERTEX_EDGE.map((edge) => ({ edge })),
        edges: BOX_EDGES.map((e) => ({ ...e })),
        faces: BOX_FACE_EDGE.map((edge) => ({ edge })),
        planes,
        hash: 0,
        geoIndex: NULL_INDEX,
    };

    hull.hash = hashHull(hull);
    return hull;
}

/** Axis-aligned box hull with the given half-widths (b3MakeBoxHull). */
export function makeBoxHull(hx: number, hy: number, hz: number): HullData {
    return makeTransformedBoxHull(hx, hy, hz, xf.identity());
}

/** Cube hull with the given half-width (b3MakeCubeHull). */
export function makeCubeHull(halfWidth: number): HullData {
    return makeBoxHull(halfWidth, halfWidth, halfWidth);
}

/** Box hull translated by `offset` (b3MakeOffsetBoxHull). */
export function makeOffsetBoxHull(hx: number, hy: number, hz: number, offset: Vec3): HullData {
    return makeTransformedBoxHull(hx, hy, hz, { p: offset, q: { v: { x: 0, y: 0, z: 0 }, s: 1 } });
}

// --- geometry queries -----------------------------------------------------------------------

/** Mass properties of a hull at the given density (b3ComputeHullMass). */
export function computeHullMass(shape: HullData, density: number): MassData {
    return {
        mass: f32(density * shape.volume),
        center: shape.center,
        inertia: mat3.scale(density, shape.centralInertia),
    };
}

/** Enclosing AABB of a transformed hull (b3ComputeHullAABB). */
export function computeHullAABB(shape: HullData, transform: Transform): AABB {
    return aabb.transform(transform, shape.aabb);
}

/** AABB enclosing a hull swept between two poses (b3ComputeSweptHullAABB). */
export function computeSweptHullAABB(shape: HullData, xf1: Transform, xf2: Transform): AABB {
    return aabb.union(aabb.transform(xf1, shape.aabb), aabb.transform(xf2, shape.aabb));
}

/** Min/max extent of a hull relative to `origin`, for sleeping bounds (b3ComputeHullExtent). */
export function computeHullExtent(
    hull: HullData,
    origin: Vec3,
): { minExtent: number; maxExtent: Vec3 } {
    let maxExtent: Vec3 = { x: 0, y: 0, z: 0 };
    for (let index = 0; index < hull.vertexCount; ++index) {
        maxExtent = vec3.max(maxExtent, vec3.abs(vec3.sub(hull.points[index], origin)));
    }
    return { minExtent: hull.innerRadius, maxExtent };
}

/** Index of the hull vertex farthest along `direction` (b3FindHullSupportVertex). */
export function findHullSupportVertex(hull: HullData, direction: Vec3): number {
    let bestIndex = NULL_INDEX;
    let bestDot = -FLT_MAX;
    for (let index = 0; index < hull.vertexCount; ++index) {
        const dot = vec3.dot(direction, hull.points[index]);
        if (dot > bestDot) {
            bestIndex = index;
            bestDot = dot;
        }
    }
    return bestIndex;
}

/** Index of the hull face whose normal is most aligned with `direction` (b3FindHullSupportFace). */
export function findHullSupportFace(hull: HullData, direction: Vec3): number {
    let bestIndex = NULL_INDEX;
    let bestDot = -FLT_MAX;
    for (let index = 0; index < hull.faceCount; ++index) {
        const dot = vec3.dot(hull.planes[index].normal, direction);
        if (dot > bestDot) {
            bestDot = dot;
            bestIndex = index;
        }
    }
    return bestIndex;
}

// --- queries --------------------------------------------------------------------------------

/** True if `proxy` (in identity frame) is within overlap slop of the hull (b3OverlapHull). */
export function overlapHull(
    shape: HullData,
    shapeTransform: Transform,
    proxy: ShapeProxy,
): boolean {
    const input: DistanceInput = {
        proxyA: { points: shape.points, count: shape.vertexCount, radius: 0 },
        proxyB: proxy,
        transform: xf.invMul(shapeTransform, xf.identity()),
        useRadii: true,
    };
    const output = shapeDistance(input, emptyCache());
    return output.distance < OVERLAP_SLOP;
}

/** Ray vs convex hull, in the hull's frame, via slab clipping (b3RayCastHull). */
export function rayCastHull(shape: HullData, input: RayCastInput): CastOutput {
    const output = emptyCastOutput();

    let lower = 0;
    let upper = input.maxFraction;
    let bestFace = NULL_INDEX;

    const planes = shape.planes;

    for (let faceIndex = 0; faceIndex < shape.faceCount; ++faceIndex) {
        const plane = planes[faceIndex];

        const distance = f32(plane.offset - vec3.dot(plane.normal, input.origin));
        const denominator = vec3.dot(plane.normal, input.translation);

        if (denominator === 0) {
            if (distance < 0) {
                return output;
            }
        } else {
            const fraction = f32(distance / denominator);

            if (denominator < 0) {
                if (fraction > lower) {
                    bestFace = faceIndex;
                    lower = fraction;
                }
            } else {
                if (fraction < upper) {
                    upper = fraction;
                }
            }

            if (upper < lower) {
                return output;
            }
        }
    }

    if (bestFace >= 0) {
        output.point = vec3.add(input.origin, vec3.scale(lower, input.translation));
        output.normal = planes[bestFace].normal;
        output.fraction = lower;
        output.hit = true;
    } else {
        output.point = input.origin;
        output.hit = true;
    }

    return output;
}

/** Shape cast against a convex hull (b3ShapeCastHull). */
export function shapeCastHull(shape: HullData, input: ShapeCastInput): CastOutput {
    const pairInput: ShapeCastPairInput = {
        proxyA: { points: shape.points, count: shape.vertexCount, radius: 0 },
        proxyB: input.proxy,
        transform: xf.identity(),
        translationB: input.translation,
        maxFraction: input.maxFraction,
        canEncroach: input.canEncroach,
    };
    return shapeCast(pairInput);
}

/**
 * Collision plane between a capsule mover and a convex hull (b3CollideMoverAndHull), in the hull's
 * frame, via GJK distance from the hull to the mover's core segment. Deep overlap is dropped rather
 * than resolved (GJK gives no normal there, matching the C's deliberate no-SAT choice for movers).
 * @returns the plane, or null when separated or overlapping.
 */
export function collideMoverAndHull(shape: HullData, mover: Capsule): PlaneResult | null {
    const input: DistanceInput = {
        proxyA: { points: shape.points, count: shape.vertexCount, radius: 0 },
        proxyB: { points: [mover.center1, mover.center2], count: 2, radius: mover.radius },
        transform: xf.identity(),
        useRadii: false,
    };

    const totalRadius = mover.radius;

    const output = shapeDistance(input, emptyCache());

    if (output.distance === 0) {
        // Deep overlap is intentionally dropped: there is no reasonable deep-overlap resolution for
        // meshes, so hulls behave the same to avoid a hull-vs-mesh discontinuity.
        return null;
    }

    if (output.distance <= totalRadius) {
        const plane: Plane = { normal: output.normal, offset: f32(totalRadius - output.distance) };
        return { plane, point: output.pointA };
    }

    return null;
}
