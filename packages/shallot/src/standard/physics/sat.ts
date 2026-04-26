import type { ConvexHull, ConvexHullFace } from "./hull";

export interface SatContact {
    pointA: Float32Array;
    pointB: Float32Array;
    normal: Float32Array;
    depth: number;
    featureKey: number;
}

export interface SatBody {
    pos: Float32Array;
    quat: Float32Array;
    halfExtents: Float32Array;
}

const EMPTY: SatContact[] = [];
const FACE_EDGE_BIAS = 0.95;
const FACE_EDGE_OFFSET = 0.01;
const DEGEN_TOL = 1e-6;
const CONTACT_TOL = 1e-5;
const MAX_CLIP_VERTS = 64;

function quatRotate(q: Float32Array, v: Float32Array): Float32Array {
    const qx = q[0],
        qy = q[1],
        qz = q[2],
        qw = q[3];
    const vx = v[0],
        vy = v[1],
        vz = v[2];
    const tx = 2 * (qy * vz - qz * vy);
    const ty = 2 * (qz * vx - qx * vz);
    const tz = 2 * (qx * vy - qy * vx);
    return new Float32Array([
        vx + qw * tx + qy * tz - qz * ty,
        vy + qw * ty + qz * tx - qx * tz,
        vz + qw * tz + qx * ty - qy * tx,
    ]);
}

function quatConj(q: Float32Array): Float32Array {
    return new Float32Array([-q[0], -q[1], -q[2], q[3]]);
}

function dot3(a: Float32Array, b: Float32Array): number {
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

function cross3(a: Float32Array, b: Float32Array): Float32Array {
    return new Float32Array([
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]);
}

function sub3(a: Float32Array, b: Float32Array): Float32Array {
    return new Float32Array([a[0] - b[0], a[1] - b[1], a[2] - b[2]]);
}

function add3(a: Float32Array, b: Float32Array): Float32Array {
    return new Float32Array([a[0] + b[0], a[1] + b[1], a[2] + b[2]]);
}

function scale3(v: Float32Array, s: number): Float32Array {
    return new Float32Array([v[0] * s, v[1] * s, v[2] * s]);
}

function neg3(v: Float32Array): Float32Array {
    return new Float32Array([-v[0], -v[1], -v[2]]);
}

function len3(v: Float32Array): number {
    return Math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
}

function normalize3(v: Float32Array): Float32Array {
    const l = len3(v);
    if (l < 1e-12) return new Float32Array(3);
    return new Float32Array([v[0] / l, v[1] / l, v[2] / l]);
}

function lerp3(a: Float32Array, b: Float32Array, t: number): Float32Array {
    return new Float32Array([
        a[0] + (b[0] - a[0]) * t,
        a[1] + (b[1] - a[1]) * t,
        a[2] + (b[2] - a[2]) * t,
    ]);
}

function transformVertex(pos: Float32Array, quat: Float32Array, local: Float32Array): Float32Array {
    return add3(pos, quatRotate(quat, local));
}

function projectHull(
    hull: ConvexHull,
    pos: Float32Array,
    quat: Float32Array,
    axis: Float32Array,
): [number, number] {
    let min = Infinity;
    let max = -Infinity;
    for (let i = 0; i < hull.numVertices; i++) {
        const lv = new Float32Array([
            hull.vertices[i * 3],
            hull.vertices[i * 3 + 1],
            hull.vertices[i * 3 + 2],
        ]);
        const wv = transformVertex(pos, quat, lv);
        const d = dot3(wv, axis);
        if (d < min) min = d;
        if (d > max) max = d;
    }
    return [min, max];
}

function getWorldFaceNormal(face: ConvexHullFace, quat: Float32Array): Float32Array {
    const ln = new Float32Array([face.plane[0], face.plane[1], face.plane[2]]);
    return quatRotate(quat, ln);
}

function getWorldFaceVertices(
    hull: ConvexHull,
    face: ConvexHullFace,
    pos: Float32Array,
    quat: Float32Array,
): Float32Array {
    const n = face.vertexIndices.length;
    const out = new Float32Array(n * 3);
    for (let i = 0; i < n; i++) {
        const vi = face.vertexIndices[i];
        const lv = new Float32Array([
            hull.vertices[vi * 3],
            hull.vertices[vi * 3 + 1],
            hull.vertices[vi * 3 + 2],
        ]);
        const wv = transformVertex(pos, quat, lv);
        out[i * 3] = wv[0];
        out[i * 3 + 1] = wv[1];
        out[i * 3 + 2] = wv[2];
    }
    return out;
}

function clipPolygonAgainstPlane(
    verts: Float32Array,
    count: number,
    planeN: Float32Array,
    planeD: number,
): { verts: Float32Array; count: number } {
    const out = new Float32Array(MAX_CLIP_VERTS * 3);
    let outCount = 0;

    let a = new Float32Array([
        verts[(count - 1) * 3],
        verts[(count - 1) * 3 + 1],
        verts[(count - 1) * 3 + 2],
    ]);
    let da = dot3(planeN, a) - planeD;

    for (let i = 0; i < count; i++) {
        const b = new Float32Array([verts[i * 3], verts[i * 3 + 1], verts[i * 3 + 2]]);
        const db = dot3(planeN, b) - planeD;
        const aInside = da <= CONTACT_TOL;
        const bInside = db <= CONTACT_TOL;

        if (aInside !== bInside) {
            const denom = da - db;
            let t = 0;
            if (Math.abs(denom) > DEGEN_TOL) t = Math.max(0, Math.min(1, da / denom));
            if (outCount < MAX_CLIP_VERTS) {
                const p = lerp3(a, b, t);
                out[outCount * 3] = p[0];
                out[outCount * 3 + 1] = p[1];
                out[outCount * 3 + 2] = p[2];
                outCount++;
            }
        }
        if (bInside && outCount < MAX_CLIP_VERTS) {
            out[outCount * 3] = b[0];
            out[outCount * 3 + 1] = b[1];
            out[outCount * 3 + 2] = b[2];
            outCount++;
        }
        a = b;
        da = db;
    }

    return { verts: out, count: outCount };
}

function reduceManifold(
    points: Float32Array,
    depths: number[],
    count: number,
    normal: Float32Array,
): number[] {
    if (count <= 4) {
        const indices: number[] = [];
        for (let i = 0; i < count; i++) indices.push(i);
        return indices;
    }

    let cx = 0,
        cy = 0,
        cz = 0;
    for (let i = 0; i < count; i++) {
        cx += points[i * 3];
        cy += points[i * 3 + 1];
        cz += points[i * 3 + 2];
    }
    cx /= count;
    cy /= count;
    cz /= count;
    const center = new Float32Array([cx, cy, cz]);

    let p0: Float32Array = new Float32Array([points[0] - cx, points[1] - cy, points[2] - cz]);
    const p0len = len3(p0);
    if (p0len < 1e-12) p0 = new Float32Array([1, 0, 0]);
    else p0 = scale3(p0, 1 / p0len);

    const u = normalize3(cross3(normal, p0));
    const v = cross3(normal, u);

    const selected = new Set<number>();
    const dirs = [u, neg3(u), v, neg3(v)];
    for (const dir of dirs) {
        let bestIdx = 0;
        let bestProj = -Infinity;
        for (let i = 0; i < count; i++) {
            const rel = sub3(
                new Float32Array([points[i * 3], points[i * 3 + 1], points[i * 3 + 2]]),
                center,
            );
            const proj = dot3(rel, dir);
            if (proj > bestProj) {
                bestProj = proj;
                bestIdx = i;
            }
        }
        selected.add(bestIdx);
    }

    let deepestIdx = 0;
    let deepestVal = depths[0];
    for (let i = 1; i < count; i++) {
        if (depths[i] < deepestVal) {
            deepestVal = depths[i];
            deepestIdx = i;
        }
    }

    if (!selected.has(deepestIdx)) {
        let shallowestInSet = -1;
        let shallowestVal = -Infinity;
        for (const idx of selected) {
            if (depths[idx] > shallowestVal) {
                shallowestVal = depths[idx];
                shallowestInSet = idx;
            }
        }
        if (shallowestInSet >= 0) {
            selected.delete(shallowestInSet);
            selected.add(deepestIdx);
        }
    }

    return Array.from(selected);
}

interface SatAxisResult {
    separated: boolean;
    minPen: number;
    bestAxis: Float32Array;
}

function testAxis(
    axis: Float32Array,
    d: Float32Array,
    hullA: ConvexHull,
    posA: Float32Array,
    quatA: Float32Array,
    hullB: ConvexHull,
    posB: Float32Array,
    quatB: Float32Array,
    state: SatAxisResult,
): void {
    const [minA, maxA] = projectHull(hullA, posA, quatA, axis);
    const [minB, maxB] = projectHull(hullB, posB, quatB, axis);
    const pen = Math.min(maxA - minB, maxB - minA);
    if (pen < 0) {
        state.separated = true;
        return;
    }
    const threshold = state.minPen * FACE_EDGE_BIAS - FACE_EDGE_OFFSET;
    if (pen < threshold) {
        state.minPen = pen;
        let n = axis;
        if (dot3(d, axis) < 0) n = neg3(axis);
        state.bestAxis = n;
    }
}

export function satHullHull(
    bodyA: SatBody,
    hullA: ConvexHull,
    bodyB: SatBody,
    hullB: ConvexHull,
): SatContact[] {
    const d = sub3(bodyB.pos, bodyA.pos);

    const state: SatAxisResult = {
        separated: false,
        minPen: 1e30,
        bestAxis: new Float32Array([0, 1, 0]),
    };

    for (let i = 0; i < hullA.numFaces; i++) {
        const axis = getWorldFaceNormal(hullA.faces[i], bodyA.quat);
        testAxis(axis, d, hullA, bodyA.pos, bodyA.quat, hullB, bodyB.pos, bodyB.quat, state);
        if (state.separated) return EMPTY;
    }

    for (let i = 0; i < hullB.numFaces; i++) {
        const axis = getWorldFaceNormal(hullB.faces[i], bodyB.quat);
        testAxis(axis, d, hullA, bodyA.pos, bodyA.quat, hullB, bodyB.pos, bodyB.quat, state);
        if (state.separated) return EMPTY;
    }

    for (let ea = 0; ea < hullA.numUniqueEdges; ea++) {
        const edgeA = quatRotate(
            bodyA.quat,
            new Float32Array([
                hullA.uniqueEdges[ea * 3],
                hullA.uniqueEdges[ea * 3 + 1],
                hullA.uniqueEdges[ea * 3 + 2],
            ]),
        );
        for (let eb = 0; eb < hullB.numUniqueEdges; eb++) {
            const edgeB = quatRotate(
                bodyB.quat,
                new Float32Array([
                    hullB.uniqueEdges[eb * 3],
                    hullB.uniqueEdges[eb * 3 + 1],
                    hullB.uniqueEdges[eb * 3 + 2],
                ]),
            );
            const axis = cross3(edgeA, edgeB);
            const axLen = len3(axis);
            if (axLen < DEGEN_TOL) continue;
            const normAxis = scale3(axis, 1 / axLen);
            testAxis(
                normAxis,
                d,
                hullA,
                bodyA.pos,
                bodyA.quat,
                hullB,
                bodyB.pos,
                bodyB.quat,
                state,
            );
            if (state.separated) return EMPTY;
        }
    }

    const normal = state.bestAxis;
    const contacts: SatContact[] = [];

    // Always use face clipping for contact generation (matching Bullet3).
    // The separating axis selects which faces to clip, regardless of whether
    // the axis came from a face normal or edge cross product.

    // Reference face: on hull A, most aligned with separating normal (face of A facing B).
    // Bullet3 uses min dot with its B→A normal, which is equivalent to max dot with our A→B normal.
    let refFaceIdx = 0;
    let refDmax = -Infinity;
    for (let i = 0; i < hullA.numFaces; i++) {
        const fn = getWorldFaceNormal(hullA.faces[i], bodyA.quat);
        const d = dot3(fn, normal);
        if (d > refDmax) {
            refDmax = d;
            refFaceIdx = i;
        }
    }

    const refFace = hullA.faces[refFaceIdx];
    const refNormal = getWorldFaceNormal(refFace, bodyA.quat);
    const refVerts = getWorldFaceVertices(hullA, refFace, bodyA.pos, bodyA.quat);
    const refVertCount = refFace.vertexIndices.length;

    // Incident face: on hull B, most anti-aligned with separating normal (face of B facing A).
    // Bullet3 uses max dot with its B→A normal, which is equivalent to min dot with our A→B normal.
    let incFaceIdx = 0;
    let incDmin = Infinity;
    for (let i = 0; i < hullB.numFaces; i++) {
        const fn = getWorldFaceNormal(hullB.faces[i], bodyB.quat);
        const d = dot3(fn, normal);
        if (d < incDmin) {
            incDmin = d;
            incFaceIdx = i;
        }
    }

    const incFace = hullB.faces[incFaceIdx];
    let clipVerts = getWorldFaceVertices(hullB, incFace, bodyB.pos, bodyB.quat);
    let clipCount = incFace.vertexIndices.length;

    // Clip incident face against reference face edge planes
    // Matches Bullet3 b3ClipFaceAgainstHull exactly:
    //   edge0 = a - b (note: reversed from a→b)
    //   planeNormalWS = -cross(WorldEdge0, worldPlaneAnormal1)
    //   planeEqWS = -dot(worldA1, planeNormalWS)
    // b3ClipFace keeps points where dot(N, p) + D < 0
    // Our clipPolygon keeps points where dot(N, p) - planeD <= tol
    // So our planeD = -D
    for (let i = 0; i < refVertCount; i++) {
        const vi = i;
        const vj = (i + 1) % refVertCount;
        const va = new Float32Array([refVerts[vi * 3], refVerts[vi * 3 + 1], refVerts[vi * 3 + 2]]);
        const vb = new Float32Array([refVerts[vj * 3], refVerts[vj * 3 + 1], refVerts[vj * 3 + 2]]);
        const edge0 = sub3(va, vb);
        const clipPlaneN = neg3(cross3(edge0, refNormal));
        const clipPlaneEq = -dot3(va, clipPlaneN);
        const result = clipPolygonAgainstPlane(clipVerts, clipCount, clipPlaneN, -clipPlaneEq);
        clipVerts = result.verts;
        clipCount = result.count;
    }

    // Keep only points behind the reference face
    // Bullet3: depth = dot(planeNormalWS, pVtxIn[i]) + planeEqWS
    //   localPlaneEq = face.m_plane.w
    //   planeEqWS = localPlaneEq - dot(planeNormalWS, posA)
    const localPlaneEq = refFace.plane[3];
    const worldPlaneEq = localPlaneEq - dot3(refNormal, bodyA.pos);

    const candidatePoints: Float32Array[] = [];
    const candidateDepths: number[] = [];
    const candidateTags: number[] = [];

    for (let i = 0; i < clipCount; i++) {
        const p = new Float32Array([clipVerts[i * 3], clipVerts[i * 3 + 1], clipVerts[i * 3 + 2]]);
        const depth = dot3(refNormal, p) + worldPlaneEq;
        if (depth <= 0) {
            candidatePoints.push(p);
            candidateDepths.push(depth);
            candidateTags.push(i);
        }
    }

    if (candidatePoints.length === 0) return EMPTY;

    const allPoints = new Float32Array(candidatePoints.length * 3);
    for (let i = 0; i < candidatePoints.length; i++) {
        allPoints[i * 3] = candidatePoints[i][0];
        allPoints[i * 3 + 1] = candidatePoints[i][1];
        allPoints[i * 3 + 2] = candidatePoints[i][2];
    }

    const selected = reduceManifold(allPoints, candidateDepths, candidatePoints.length, normal);

    for (const si of selected) {
        const pB = candidatePoints[si];
        const depth = candidateDepths[si];

        const featureKey =
            ((refFaceIdx & 0xff) << 16) | ((incFaceIdx & 0xff) << 8) | (candidateTags[si] & 0xff);

        contacts.push({
            pointA: sub3(pB, scale3(refNormal, depth)),
            pointB: new Float32Array(pB),
            normal: new Float32Array(normal),
            depth,
            featureKey,
        });
    }

    return contacts;
}

export function boxToHull(halfExtents: Float32Array): ConvexHull {
    const hx = halfExtents[0],
        hy = halfExtents[1],
        hz = halfExtents[2];
    const vertices = new Float32Array([
        -hx,
        -hy,
        -hz,
        hx,
        -hy,
        -hz,
        hx,
        hy,
        -hz,
        -hx,
        hy,
        -hz,
        -hx,
        -hy,
        hz,
        hx,
        -hy,
        hz,
        hx,
        hy,
        hz,
        -hx,
        hy,
        hz,
    ]);
    const faces: ConvexHullFace[] = [
        { plane: new Float32Array([1, 0, 0, -hx]), vertexIndices: new Uint32Array([1, 2, 6, 5]) },
        { plane: new Float32Array([-1, 0, 0, -hx]), vertexIndices: new Uint32Array([0, 4, 7, 3]) },
        { plane: new Float32Array([0, 1, 0, -hy]), vertexIndices: new Uint32Array([2, 3, 7, 6]) },
        { plane: new Float32Array([0, -1, 0, -hy]), vertexIndices: new Uint32Array([0, 1, 5, 4]) },
        { plane: new Float32Array([0, 0, 1, -hz]), vertexIndices: new Uint32Array([4, 5, 6, 7]) },
        { plane: new Float32Array([0, 0, -1, -hz]), vertexIndices: new Uint32Array([0, 3, 2, 1]) },
    ];
    const uniqueEdges = new Float32Array([1, 0, 0, 0, 1, 0, 0, 0, 1]);
    return {
        vertices,
        numVertices: 8,
        faces,
        numFaces: 6,
        uniqueEdges,
        numUniqueEdges: 3,
        localCenter: new Float32Array(3),
        extents: new Float32Array([hx, hy, hz]),
    };
}

export function satHullBox(bodyA: SatBody, hullA: ConvexHull, bodyB: SatBody): SatContact[] {
    const hullB = boxToHull(bodyB.halfExtents);
    return satHullHull(bodyA, hullA, bodyB, hullB);
}

function closestPointOnSegment(
    p: Float32Array,
    a: Float32Array,
    b: Float32Array,
): { point: Float32Array; t: number } {
    const ab = sub3(b, a);
    const ap = sub3(p, a);
    const abLen2 = dot3(ab, ab);
    if (abLen2 < 1e-12) return { point: new Float32Array(a), t: 0 };
    const t = Math.max(0, Math.min(1, dot3(ap, ab) / abLen2));
    return { point: add3(a, scale3(ab, t)), t };
}

function pointInConvexPolygon(
    point: Float32Array,
    faceVerts: Float32Array,
    vertCount: number,
    faceNormal: Float32Array,
): boolean {
    for (let i = 0; i < vertCount; i++) {
        const j = (i + 1) % vertCount;
        const va = new Float32Array([faceVerts[i * 3], faceVerts[i * 3 + 1], faceVerts[i * 3 + 2]]);
        const vb = new Float32Array([faceVerts[j * 3], faceVerts[j * 3 + 1], faceVerts[j * 3 + 2]]);
        const edge = sub3(vb, va);
        const toPoint = sub3(point, va);
        const c = cross3(edge, toPoint);
        if (dot3(c, faceNormal) < -CONTACT_TOL) return false;
    }
    return true;
}

export function satHullSphere(bodyA: SatBody, hullA: ConvexHull, bodyB: SatBody): SatContact[] {
    const radius = bodyB.halfExtents[0];
    const sphereCenter = bodyB.pos;
    const invQuatA = quatConj(bodyA.quat);
    const localCenter = quatRotate(invQuatA, sub3(sphereCenter, bodyA.pos));

    let closestDist = Infinity;
    let closestPoint: Float32Array = new Float32Array(3);
    let closestType = 0;

    for (let fi = 0; fi < hullA.numFaces; fi++) {
        const face = hullA.faces[fi];
        const fn = new Float32Array([face.plane[0], face.plane[1], face.plane[2]]);
        const fd = face.plane[3];
        const dist = dot3(fn, localCenter) + fd;

        if (dist < -radius) continue;

        const projected = sub3(localCenter, scale3(fn, dist));

        const vertCount = face.vertexIndices.length;
        const localVerts = new Float32Array(vertCount * 3);
        for (let i = 0; i < vertCount; i++) {
            const vi = face.vertexIndices[i];
            localVerts[i * 3] = hullA.vertices[vi * 3];
            localVerts[i * 3 + 1] = hullA.vertices[vi * 3 + 1];
            localVerts[i * 3 + 2] = hullA.vertices[vi * 3 + 2];
        }

        if (pointInConvexPolygon(projected, localVerts, vertCount, fn)) {
            const absDist = Math.abs(dist);
            if (absDist < closestDist) {
                closestDist = absDist;
                closestPoint = projected;
                closestType = 0;
            }
        }
    }

    for (const face of hullA.faces) {
        const idx = face.vertexIndices;
        for (let i = 0; i < idx.length; i++) {
            const a = idx[i],
                b = idx[(i + 1) % idx.length];
            const va = new Float32Array([
                hullA.vertices[a * 3],
                hullA.vertices[a * 3 + 1],
                hullA.vertices[a * 3 + 2],
            ]);
            const vb = new Float32Array([
                hullA.vertices[b * 3],
                hullA.vertices[b * 3 + 1],
                hullA.vertices[b * 3 + 2],
            ]);
            const { point } = closestPointOnSegment(localCenter, va, vb);
            const dist = len3(sub3(localCenter, point));
            if (dist < closestDist) {
                closestDist = dist;
                closestPoint = point;
                closestType = 1;
            }
        }
    }

    for (let i = 0; i < hullA.numVertices; i++) {
        const v = new Float32Array([
            hullA.vertices[i * 3],
            hullA.vertices[i * 3 + 1],
            hullA.vertices[i * 3 + 2],
        ]);
        const dist = len3(sub3(localCenter, v));
        if (dist < closestDist) {
            closestDist = dist;
            closestPoint = v;
            closestType = 2;
        }
    }

    const penetration = closestDist - radius;
    if (penetration > CONTACT_TOL) return EMPTY;

    const closestWorld = transformVertex(bodyA.pos, bodyA.quat, closestPoint);
    const diff = sub3(sphereCenter, closestWorld);
    const diffLen = len3(diff);
    let normal: Float32Array;
    if (diffLen < 1e-8) {
        if (closestType === 0) {
            for (const face of hullA.faces) {
                const fn = new Float32Array([face.plane[0], face.plane[1], face.plane[2]]);
                const projected = sub3(
                    localCenter,
                    scale3(fn, dot3(fn, localCenter) + face.plane[3]),
                );
                const localVerts = new Float32Array(face.vertexIndices.length * 3);
                for (let i = 0; i < face.vertexIndices.length; i++) {
                    const vi = face.vertexIndices[i];
                    localVerts[i * 3] = hullA.vertices[vi * 3];
                    localVerts[i * 3 + 1] = hullA.vertices[vi * 3 + 1];
                    localVerts[i * 3 + 2] = hullA.vertices[vi * 3 + 2];
                }
                if (pointInConvexPolygon(projected, localVerts, face.vertexIndices.length, fn)) {
                    normal = quatRotate(bodyA.quat, fn);
                    break;
                }
            }
            normal ??= new Float32Array([0, 1, 0]);
        } else {
            normal = new Float32Array([0, 1, 0]);
        }
    } else {
        normal = scale3(diff, 1 / diffLen);
    }

    const pointA = closestWorld;
    const pointB = sub3(sphereCenter, scale3(normal, radius));

    return [
        {
            pointA: new Float32Array(pointA),
            pointB: new Float32Array(pointB),
            normal: new Float32Array(normal),
            depth: penetration,
            featureKey: 8 << 24,
        },
    ];
}

export function satHullCapsule(bodyA: SatBody, hullA: ConvexHull, bodyB: SatBody): SatContact[] {
    const radius = bodyB.halfExtents[0];
    const halfHeight = bodyB.halfExtents[1];
    const up = quatRotate(bodyB.quat, new Float32Array([0, halfHeight, 0]));
    const ep0 = add3(bodyB.pos, up);
    const ep1 = sub3(bodyB.pos, up);

    const contacts: SatContact[] = [];

    for (let endpoint = 0; endpoint < 2; endpoint++) {
        const center = endpoint === 0 ? ep0 : ep1;
        const sphereBody: SatBody = {
            pos: new Float32Array(center),
            quat: new Float32Array([0, 0, 0, 1]),
            halfExtents: new Float32Array([radius, radius, radius]),
        };
        const result = satHullSphere(bodyA, hullA, sphereBody);
        for (const c of result) {
            contacts.push({
                pointA: c.pointA,
                pointB: c.pointB,
                normal: c.normal,
                depth: c.depth,
                featureKey: (9 << 24) | endpoint,
            });
        }
    }

    return contacts;
}
