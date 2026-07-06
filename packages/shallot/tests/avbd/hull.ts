// Convex-hull narrowphase (Phase 6.3 hull) — the polytope-family SAT generalized from box-box
// (collide.ts) to arbitrary face + edge sets. Nothing new in the solve: a hull contact has the same
// SatResult/Contact shape (feature key + local arms + a B→A basis), so the solver (manifold.ts) consumes
// it unchanged. hull-box and hull-hull are the box path with more candidate axes — face-A normals,
// face-B normals, edge-A × edge-B cross products — the same Sutherland-Hodgman reference-face clip and
// reduce-to-4 (reuses collide.ts reduceManifold).
//
// No AVBD reference covers non-box hulls (demo3d is boxes only), and webphysics has none, so the shape +
// algorithm are grounded in the two production engines that do: Jolt (GJK/EPA on CPU — not the GPU path)
// and Bullet's GPU SAT (b3FindSeparatingAxis + b3ClipHullAgainstHull + b3ReduceContacts). We take Bullet's
// SAT decomposition (no GJK/EPA — GPU-hostile), validated against the independent bullet3-sat-harness gold
// (hull.test.ts).
//
// f64; the GPU `collideHull` (collide.ts WGSL, Phase B) reproduces this. A box fed as a Hull reproduces
// the box-box oracle `collide` on face manifolds (hull.test.ts) — the strongest gate, the trusted
// avbd-demo3d box gold transitively validating the whole generalization.

import { type Cand, type Contact, reduceManifold, SPECULATIVE_DISTANCE } from "./collide";
import {
    add,
    clamp,
    cross,
    dot,
    length,
    lengthSq,
    type Mat3,
    neg,
    normalize,
    orthonormal,
    type Quat,
    qconj,
    rotate,
    scale,
    sub,
    type Vec3,
} from "./math";

const SAT_AXIS_EPSILON = 1e-9;
const PLANE_EPSILON = 1e-7;
const CONTACT_MERGE_DIST_SQ = 1e-10;
const NORMAL_TOL = 1e-9;

// hull feature-key tags (high byte) — distinct from the box AXIS_FACE_A/B/EDGE (0/1/2) and ROUND (3), so
// a hull pair's keys never collide with another path's. The face manifold packs refFace/incFace/clip
// ordinal into the lower bytes (a body-fixed clip ordinal, the same warmstart discipline as box-box —
// keyed by (a,b)+feature, never re-ordinal'd to post-reduction rank).
const HULL_FACE = 0x40;
const HULL_EDGE = 0x41;

/** one polygonal hull face: outward unit normal + plane offset (`dot(normal, v) = offset` on the face) + CCW vertex indices */
export interface HullFace {
    normal: Vec3;
    offset: number;
    verts: number[];
}

/** a convex hull: local vertices, polygonal faces, and the unique (deduplicated, canonicalized) edge directions for the SAT */
export interface Hull {
    verts: Vec3[];
    faces: HullFace[];
    edges: Vec3[];
}

// ── builders ─────────────────────────────────────────────────────────
// These mirror the bullet3-sat-harness hulls (box / tet / cone) so the oracle SAT validates against its
// gold. The geometric comparison is order-independent, so the vertex/face/edge ordering is free; outward
// face normals are forced (centroid test) regardless of the input winding.

/** canonicalize an edge direction to a single representative so `+d` and `−d` dedup (legacy collectUniqueEdges) */
function canonicalEdge(d: Vec3): Vec3 {
    const [x, y, z] = d;
    if (
        x < -1e-8 ||
        (Math.abs(x) < 1e-8 && y < -1e-8) ||
        (Math.abs(x) < 1e-8 && Math.abs(y) < 1e-8 && z < 0)
    )
        return neg(d);
    return d;
}

/** collect unique edge directions across all faces (dedup parallel edges), the SAT's edge-axis set */
function uniqueEdges(verts: Vec3[], faces: HullFace[]): Vec3[] {
    const dirs: Vec3[] = [];
    for (const f of faces) {
        for (let i = 0; i < f.verts.length; i++) {
            const a = verts[f.verts[i]];
            const b = verts[f.verts[(i + 1) % f.verts.length]];
            const e = sub(b, a);
            const len = length(e);
            if (len < 1e-12) continue;
            const d = canonicalEdge(scale(e, 1 / len));
            if (!dirs.some((p) => dot(p, d) > 1 - NORMAL_TOL)) dirs.push(d);
        }
    }
    return dirs;
}

/**
 * Build a Hull from explicit vertices + face vertex-index loops. Each face's normal is forced outward
 * (centroid test) AND its loop reversed when needed so the winding is CCW around that outward normal —
 * the clip side planes (`−cross(edge, normal)`) only point inward (keep the face interior) when the loop
 * and normal agree, so an inconsistent input winding would clip the whole incident polygon away.
 */
function hullOf(verts: Vec3[], faceLoops: number[][]): Hull {
    const centroid = scale(
        verts.reduce((s, v) => add(s, v), [0, 0, 0] as Vec3),
        1 / verts.length,
    );
    const faces: HullFace[] = faceLoops.map((loop) => {
        const faceCenter = scale(
            loop.reduce((s, i) => add(s, verts[i]), [0, 0, 0] as Vec3),
            1 / loop.length,
        );
        // Newell normal: robust for any polygon + tells the loop's winding sense
        let nrm: Vec3 = [0, 0, 0];
        for (let i = 0; i < loop.length; i++) {
            const a = sub(verts[loop[i]], faceCenter);
            const b = sub(verts[loop[(i + 1) % loop.length]], faceCenter);
            nrm = add(nrm, cross(a, b));
        }
        let n = normalize(nrm);
        let verts2 = loop;
        // force outward (away from the hull centroid); flip the loop to keep it CCW around the outward normal
        if (dot(n, sub(faceCenter, centroid)) < 0) {
            n = neg(n);
            verts2 = [...loop].reverse();
        }
        return { normal: n, offset: dot(n, verts[verts2[0]]), verts: verts2 };
    });
    return { verts, faces, edges: uniqueEdges(verts, faces) };
}

/** an axis-aligned box hull from full-width size — the box as a polytope, to prove the hull path reproduces box-box */
export function boxHull(size: Vec3): Hull {
    const [hx, hy, hz] = scale(size, 0.5);
    const verts: Vec3[] = [
        [-hx, -hy, -hz],
        [hx, -hy, -hz],
        [hx, hy, -hz],
        [-hx, hy, -hz],
        [-hx, -hy, hz],
        [hx, -hy, hz],
        [hx, hy, hz],
        [-hx, hy, hz],
    ];
    // +X, -X, +Y, -Y, +Z, -Z, CCW around the outward normal
    const loops = [
        [1, 2, 6, 5],
        [0, 4, 7, 3],
        [2, 3, 7, 6],
        [0, 1, 5, 4],
        [4, 5, 6, 7],
        [0, 3, 2, 1],
    ];
    return hullOf(verts, loops);
}

/** a regular tetrahedron hull (half-extent `s`), matching the harness makeTetrahedron */
export function tetHull(s: number): Hull {
    const verts: Vec3[] = [
        [s, s, s],
        [s, -s, -s],
        [-s, s, -s],
        [-s, -s, s],
    ];
    const loops = [
        [0, 1, 2],
        [0, 3, 1],
        [0, 2, 3],
        [1, 3, 2],
    ];
    return hullOf(verts, loops);
}

/** a cone approximation (apex + `segments`-gon base), matching the harness makeCone */
export function coneHull(radius: number, height: number, segments: number): Hull {
    const verts: Vec3[] = [[0, height / 2, 0]];
    for (let i = 0; i < segments; i++) {
        const a = (i / segments) * 2 * Math.PI;
        verts.push([radius * Math.cos(a), -height / 2, radius * Math.sin(a)]);
    }
    const loops: number[][] = [];
    for (let i = 0; i < segments; i++) loops.push([0, 1 + i, 1 + ((i + 1) % segments)]); // side tris
    const base: number[] = [];
    for (let i = segments; i > 0; i--) base.push(i); // base polygon, inward-then-forced-outward
    loops.push(base);
    return hullOf(verts, loops);
}

// ── closest point on a hull (rounded × hull) ─────────────────────────

/** is the in-plane point `p` inside face `f`'s polygon? (on the inner side of every CCW edge) */
function pointInFace(h: Hull, f: HullFace, p: Vec3): boolean {
    for (let i = 0; i < f.verts.length; i++) {
        const a = h.verts[f.verts[i]];
        const b = h.verts[f.verts[(i + 1) % f.verts.length]];
        if (dot(cross(sub(b, a), sub(p, a)), f.normal) < -PLANE_EPSILON) return false;
    }
    return true;
}

/** closest point on segment [a,b] to `p` (clamped projection) */
function closestOnSegment(p: Vec3, a: Vec3, b: Vec3): Vec3 {
    const ab = sub(b, a);
    const len2 = lengthSq(ab);
    if (len2 < SAT_AXIS_EPSILON) return a;
    return add(a, scale(ab, clamp(dot(sub(p, a), ab) / len2, 0, 1)));
}

/**
 * Closest point on a convex hull to a query point `q` (in the hull's LOCAL frame) — the rounded × polytope
 * primitive (a box is its `boxHull`, so this is the one closest-point routine for sphere/capsule vs box or
 * hull). Outside the hull: the nearest surface feature (face region → edge → vertex), normal pointing
 * outward toward `q`, `signedDist > 0`. Inside: push out along the least-penetrating face, `signedDist < 0`.
 * On a box-hull it reduces to the OBB clamp + interior face push-out. No GJK/EPA — analytic + exact.
 */
export function closestPointOnHull(
    h: Hull,
    q: Vec3,
): { point: Vec3; normal: Vec3; signedDist: number } {
    // the face the point is least behind (max signed plane distance) — inside iff all are ≤ 0
    let maxD = -Infinity;
    let maxFace = 0;
    for (let i = 0; i < h.faces.length; i++) {
        const d = dot(h.faces[i].normal, q) - h.faces[i].offset;
        if (d > maxD) {
            maxD = d;
            maxFace = i;
        }
    }
    if (maxD <= 0) {
        // inside: push out along the least-penetrating face
        const f = h.faces[maxFace];
        return { point: sub(q, scale(f.normal, maxD)), normal: f.normal, signedDist: maxD };
    }

    // outside: the nearest surface feature (face region, then edge, then vertex)
    let bestDist = Infinity;
    let bestPoint: Vec3 = q;
    for (const f of h.faces) {
        const d = dot(f.normal, q) - f.offset;
        if (d <= 0) continue; // behind this face → not its front region
        const proj = sub(q, scale(f.normal, d));
        if (pointInFace(h, f, proj) && d < bestDist) {
            bestDist = d;
            bestPoint = proj;
        }
    }
    for (const f of h.faces) {
        for (let i = 0; i < f.verts.length; i++) {
            const cp = closestOnSegment(
                q,
                h.verts[f.verts[i]],
                h.verts[f.verts[(i + 1) % f.verts.length]],
            );
            const dd = length(sub(q, cp));
            if (dd < bestDist) {
                bestDist = dd;
                bestPoint = cp;
            }
        }
    }
    for (const v of h.verts) {
        const dd = length(sub(q, v));
        if (dd < bestDist) {
            bestDist = dd;
            bestPoint = v;
        }
    }
    const diff = sub(q, bestPoint);
    const dist = length(diff);
    const normal = dist > NORMAL_TOL ? scale(diff, 1 / dist) : h.faces[maxFace].normal;
    return { point: bestPoint, normal, signedDist: dist };
}

// ── SAT ──────────────────────────────────────────────────────────────

const worldVert = (h: Hull, pos: Vec3, quat: Quat, i: number): Vec3 =>
    add(rotate(quat, h.verts[i]), pos);

/** the world support vertex of a hull in direction `dir` (the deepest point along `dir`) */
function supportHull(h: Hull, pos: Vec3, quat: Quat, dir: Vec3): Vec3 {
    let best = 0;
    let bestD = -Infinity;
    for (let i = 0; i < h.verts.length; i++) {
        const d = dot(worldVert(h, pos, quat, i), dir);
        if (d > bestD) {
            bestD = d;
            best = i;
        }
    }
    return worldVert(h, pos, quat, best);
}

/** [min, max] projection of a hull's world vertices onto `axis` */
function projectHull(h: Hull, pos: Vec3, quat: Quat, axis: Vec3): [number, number] {
    let mn = Infinity;
    let mx = -Infinity;
    for (let i = 0; i < h.verts.length; i++) {
        const d = dot(worldVert(h, pos, quat, i), axis);
        if (d < mn) mn = d;
        if (d > mx) mx = d;
    }
    return [mn, mx];
}

interface HullAxis {
    fromA: boolean; // a face axis: did it come from hull A (else B)? (edge axes set isEdge)
    isEdge: boolean;
    indexA: number;
    indexB: number;
    separation: number; // signed gap: + separated, − penetrating; we keep the MAX (the MTV)
    normalAB: Vec3; // oriented A → B (dot(n, posB − posA) ≥ 0), like collide.ts before the basis flip
    valid: boolean;
}

/**
 * Test one candidate axis. Projects both hulls, computes the signed separation (positive = gap), aborts
 * (returns false) only past the speculative/swept band — the same unilateral band as box-box (Phase
 * 4.8.3/4.8.4). Keeps the axis of maximum separation (least penetration = the MTV). `dRel = (vA−vB)·dt`.
 */
function testHullAxis(
    hullA: Hull,
    posA: Vec3,
    quatA: Quat,
    hullB: Hull,
    posB: Vec3,
    quatB: Quat,
    delta: Vec3,
    axis: Vec3,
    isEdge: boolean,
    fromA: boolean,
    ia: number,
    ib: number,
    best: HullAxis,
    dRel: Vec3,
): boolean {
    const lenSq = lengthSq(axis);
    if (lenSq < SAT_AXIS_EPSILON) return true;
    let n = scale(axis, 1 / Math.sqrt(lenSq));
    if (dot(n, delta) < 0) n = neg(n); // orient A → B

    const [aMin, aMax] = projectHull(hullA, posA, quatA, n);
    const [bMin, bMax] = projectHull(hullB, posB, quatB, n);
    // penetration = overlap of the two intervals (positive when overlapping); separation = −penetration
    const penetration = Math.min(aMax - bMin, bMax - aMin);
    const separation = -penetration;

    // unilateral speculative band along n (Phase 4.8.3/4.8.4) — abort only past max(static skin, closing
    // displacement this step). max, not + (a slow body's band stays the velocity-independent skin).
    if (separation > Math.max(SPECULATIVE_DISTANCE, Math.max(0, dot(dRel, n)))) return false;

    if (!best.valid || separation > best.separation) {
        best.valid = true;
        best.isEdge = isEdge;
        best.fromA = fromA;
        best.indexA = ia;
        best.indexB = ib;
        best.separation = separation;
        best.normalAB = n;
    }
    return true;
}

/** the reference hull's face most aligned with `outward` (a world direction); returns the face index */
function bestFace(h: Hull, quat: Quat, outward: Vec3): number {
    let idx = 0;
    let best = -Infinity;
    for (let i = 0; i < h.faces.length; i++) {
        const d = dot(rotate(quat, h.faces[i].normal), outward);
        if (d > best) {
            best = d;
            idx = i;
        }
    }
    return idx;
}

/** Sutherland-Hodgman: clip the world polygon `poly` to the back of one side plane `dot(nrm, x) ≤ d` */
function clipPlane(poly: Vec3[], nrm: Vec3, d: number): Vec3[] {
    const out: Vec3[] = [];
    const n = poly.length;
    if (n === 0) return out;
    let a = poly[n - 1];
    let da = dot(nrm, a) - d;
    for (let i = 0; i < n; i++) {
        const b = poly[i];
        const db = dot(nrm, b) - d;
        const aIn = da <= PLANE_EPSILON;
        const bIn = db <= PLANE_EPSILON;
        if (aIn !== bIn) {
            let t = 0;
            const denom = da - db;
            if (Math.abs(denom) > SAT_AXIS_EPSILON) t = clamp(da / denom, 0, 1);
            out.push(add(a, scale(sub(b, a), t)));
        }
        if (bIn) out.push(b);
        a = b;
        da = db;
    }
    return out;
}

/**
 * Convex-hull SAT (hull-box / hull-hull). Returns the contact manifold (0–4 points, the reduced spread
 * set) + the contact `basis` (normal in row 0, B → A). Empty = no overlap past the band. `dRel = (vA−vB)·dt`
 * is the velocity sweep; `[0,0,0]` recovers the static speculative SAT. The contact has the same shape the
 * solver reads, so a box fed as a Hull reproduces the box-box `collide` face manifold exactly.
 */
export function collideHull(
    hullA: Hull,
    posA: Vec3,
    quatA: Quat,
    hullB: Hull,
    posB: Vec3,
    quatB: Quat,
    dRel: Vec3 = [0, 0, 0],
): { contacts: Contact[]; basis: Mat3 } {
    const none = {
        contacts: [] as Contact[],
        basis: [
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
        ] as Mat3,
    };
    const delta = sub(posB, posA);

    const mk = (): HullAxis => ({
        fromA: false,
        isEdge: false,
        indexA: -1,
        indexB: -1,
        separation: -Infinity,
        normalAB: [0, 0, 0],
        valid: false,
    });
    const bestFaceAxis = mk();
    const bestEdgeAxis = mk();

    for (let i = 0; i < hullA.faces.length; i++) {
        const axis = rotate(quatA, hullA.faces[i].normal);
        if (
            !testHullAxis(
                hullA,
                posA,
                quatA,
                hullB,
                posB,
                quatB,
                delta,
                axis,
                false,
                true,
                i,
                -1,
                bestFaceAxis,
                dRel,
            )
        )
            return none;
    }
    for (let i = 0; i < hullB.faces.length; i++) {
        const axis = rotate(quatB, hullB.faces[i].normal);
        if (
            !testHullAxis(
                hullA,
                posA,
                quatA,
                hullB,
                posB,
                quatB,
                delta,
                axis,
                false,
                false,
                -1,
                i,
                bestFaceAxis,
                dRel,
            )
        )
            return none;
    }
    for (let i = 0; i < hullA.edges.length; i++) {
        const ea = rotate(quatA, hullA.edges[i]);
        for (let j = 0; j < hullB.edges.length; j++) {
            const eb = rotate(quatB, hullB.edges[j]);
            const axis = cross(ea, eb);
            if (
                !testHullAxis(
                    hullA,
                    posA,
                    quatA,
                    hullB,
                    posB,
                    quatB,
                    delta,
                    axis,
                    true,
                    false,
                    i,
                    j,
                    bestEdgeAxis,
                    dRel,
                )
            )
                return none;
        }
    }

    if (!bestFaceAxis.valid) return none;

    // edge/face bias — prefer the edge axis only when clearly deeper (collide.ts 0.95/0.01)
    let best = bestFaceAxis;
    if (bestEdgeAxis.valid && 0.95 * bestEdgeAxis.separation > bestFaceAxis.separation + 0.01)
        best = bestEdgeAxis;

    const normalBA = neg(best.normalAB); // basis row 0 (B → A)
    const basis = orthonormal(normalBA);
    const band = Math.max(SPECULATIVE_DISTANCE, Math.max(0, dot(dRel, best.normalAB)));

    // reference = the hull whose face axis won (an edge axis defaults to A);
    // incident = the other hull's face most aligned with the reference's outward normal's opposite.
    const referenceIsA = best.isEdge ? true : best.fromA;
    const refHull = referenceIsA ? hullA : hullB;
    const refPos = referenceIsA ? posA : posB;
    const refQuat = referenceIsA ? quatA : quatB;
    const incHull = referenceIsA ? hullB : hullA;
    const incPos = referenceIsA ? posB : posA;
    const incQuat = referenceIsA ? quatB : quatA;
    const refOutward = referenceIsA ? best.normalAB : normalBA; // reference face points toward the incident hull

    const refFaceIdx = bestFace(refHull, refQuat, refOutward);
    const refFace = refHull.faces[refFaceIdx];
    const refNormalW = rotate(refQuat, refFace.normal);
    const refPlaneW = dot(refNormalW, refPos) + refFace.offset;

    // the incident face is selected by the CONTACT normal (−refOutward = inc→ref), not the reference
    // face's normal — they differ when the MTV is an edge axis (refNormalW ≠ the contact normal), and the
    // contact-normal choice (Bullet's) is what keeps a slanted edge-MTV clip non-empty.
    const incFaceIdx = bestFace(incHull, incQuat, neg(refOutward));
    const incFace = incHull.faces[incFaceIdx];
    let poly = incFace.verts.map((vi) => worldVert(incHull, incPos, incQuat, vi));

    // clip the incident polygon against the reference face's edge side planes (inward-facing)
    const refLoop = refFace.verts;
    for (let e = 0; e < refLoop.length && poly.length > 0; e++) {
        const va = worldVert(refHull, refPos, refQuat, refLoop[e]);
        const vb = worldVert(refHull, refPos, refQuat, refLoop[(e + 1) % refLoop.length]);
        const planeN = neg(cross(sub(va, vb), refNormalW));
        poly = clipPlane(poly, planeN, dot(planeN, va));
    }
    if (poly.length === 0) return none;

    // keep clipped verts within the band behind the reference plane; project onto it for the reference anchor
    const featurePrefix =
        ((best.isEdge ? HULL_EDGE : HULL_FACE) << 24) |
        ((refFaceIdx & 0xff) << 16) |
        ((incFaceIdx & 0xff) << 8);
    const cands: Cand[] = [];
    const mids: Vec3[] = [];
    for (let i = 0; i < poly.length; i++) {
        const pInc = poly[i];
        const depth = dot(refNormalW, pInc) - refPlaneW;
        if (depth > band) continue;
        const pRef = sub(pInc, scale(refNormalW, depth));
        const xA = referenceIsA ? pRef : pInc;
        const xB = referenceIsA ? pInc : pRef;
        const mid = scale(add(xA, xB), 0.5);
        if (mids.some((m) => lengthSq(sub(mid, m)) < CONTACT_MERGE_DIST_SQ)) continue;
        cands.push({ feature: featurePrefix | (i & 0xff), xA, xB });
        mids.push(mid);
    }
    // no candidate survived the clip (a grazing edge MTV) — fall back to a single support-point contact,
    // the deepest vertex of each hull along the contact normal (mirrors box-box buildFaceManifold).
    if (cands.length === 0) {
        const xA = supportHull(hullA, posA, quatA, best.normalAB);
        const xB = supportHull(hullB, posB, quatB, neg(best.normalAB));
        return {
            contacts: [
                {
                    feature: featurePrefix | 0,
                    rA: rotate(qconj(quatA), sub(xA, posA)),
                    rB: rotate(qconj(quatB), sub(xB, posB)),
                },
            ],
            basis,
        };
    }

    const sel = cands.length > 4 ? reduceManifold(cands, best.normalAB, posA) : cands;
    const contacts: Contact[] = sel.map((c) => ({
        feature: c.feature | 0,
        rA: rotate(qconj(quatA), sub(c.xA, posA)),
        rB: rotate(qconj(quatB), sub(c.xB, posB)),
    }));
    return { contacts, basis };
}

// ── capsule × hull (segment-clip manifold) ───────────────────────────

/** one rounded-vs-hull contact in WORLD space: the round CORE point, the hull SURFACE point, and a stable ordinal (the solver reconstructs the gap from the arms + radius) */
export interface RoundHit {
    capCore: Vec3;
    hullSurf: Vec3;
    ordinal: number;
}

/**
 * Capsule (core segment [e0,e1] world, radius `radius`) vs convex hull — the decided §6.3 manifold: the
 * capsule core is a degenerate 2-vertex incident face, clipped against the polytope reference face,
 * emitting up to 2 contacts that SHARE the reference-face normal (Jolt GetSupportingFace →
 * ManifoldBetweenTwoFaces; Bullet does the same). This catches the mid-segment rest endpoint sampling
 * misses (a capsule longer than the face it lies on). Off-face / edge contacts fall back to per-endpoint
 * closest points with the deeper endpoint's normal shared. `dRel = (vCap − vHull)·dt`. Returns world-space
 * anchors + the shared B→A-agnostic normal (hull → capsule, outward); the caller orients A/B + local arms.
 */
export function capsuleHull(
    e0: Vec3,
    e1: Vec3,
    radius: number,
    h: Hull,
    pos: Vec3,
    quat: Quat,
    dRel: Vec3,
): { contacts: RoundHit[]; normal: Vec3 } {
    const dir = sub(e1, e0);
    const toCap = sub(scale(add(e0, e1), 0.5), pos); // hull centre → segment, to orient the normal hull→capsule

    // SAT between the CORE segment (a degenerate 2-vertex incident face) and the hull → the contact normal
    // = the axis of MAX separation (the MTV). This is what makes the reference-face pick robust to an
    // OVERHANGING capsule (its endpoints' closest points are off to the side, but the SAT still finds the
    // face it rests on). Candidate axes: hull face normals + segDir × hull edges (the box path's axis set).
    const segProj = (axis: Vec3): [number, number] => {
        const a = dot(e0, axis);
        const b = dot(e1, axis);
        return a < b ? [a, b] : [b, a];
    };
    let bestSep = -Infinity;
    let n: Vec3 = [0, 1, 0];
    const consider = (raw: Vec3) => {
        const l2 = lengthSq(raw);
        if (l2 < SAT_AXIS_EPSILON) return;
        let ax = scale(raw, 1 / Math.sqrt(l2));
        if (dot(ax, toCap) < 0) ax = neg(ax); // orient hull → capsule
        const [sMin, sMax] = segProj(ax);
        const [hMin, hMax] = projectHull(h, pos, quat, ax);
        const sep = -Math.min(sMax - hMin, hMax - sMin);
        if (sep > bestSep) {
            bestSep = sep;
            n = ax;
        }
    };
    for (const f of h.faces) consider(rotate(quat, f.normal));
    for (const e of h.edges) consider(cross(dir, rotate(quat, e)));

    // the core is separated by `bestSep`; the radius closes it. No contact past the (swept) band.
    if (bestSep - radius > Math.max(SPECULATIVE_DISTANCE, Math.max(0, -dot(dRel, n))))
        return { contacts: [], normal: n };

    // reference face = the hull face most aligned with the contact normal; the manifold SHARES its plane
    // normal (Jolt GetSupportingFace → ManifoldBetweenTwoFaces; Bullet does the same; roadmap §6.3)
    const refIdx = bestFace(h, quat, n);
    const refFace = h.faces[refIdx];
    const nf = rotate(quat, refFace.normal);
    const refCenter = worldVert(h, pos, quat, refFace.verts[0]);
    const bandF = Math.max(SPECULATIVE_DISTANCE, Math.max(0, -dot(dRel, nf)));

    // clip the segment's parameter range to the reference face's side-plane prism (extruded along nf)
    let t0 = 0;
    let t1 = 1;
    let outside = false;
    for (let i = 0; i < refFace.verts.length; i++) {
        const va = worldVert(h, pos, quat, refFace.verts[i]);
        const vb = worldVert(h, pos, quat, refFace.verts[(i + 1) % refFace.verts.length]);
        const planeN = neg(cross(sub(va, vb), nf)); // inward (keeps the face interior)
        const d0 = dot(planeN, e0) - dot(planeN, va);
        const dd = dot(planeN, dir);
        if (Math.abs(dd) < SAT_AXIS_EPSILON) {
            if (d0 > PLANE_EPSILON) {
                outside = true;
                break;
            }
            continue;
        }
        const tc = -d0 / dd;
        if (dd > 0) t1 = Math.min(t1, tc);
        else t0 = Math.max(t0, tc);
    }

    const contacts: RoundHit[] = [];
    if (!outside && t1 > t0 + 1e-9) {
        // the segment lies over the reference face → contacts at the clipped ends, shared normal nf
        for (let k = 0; k < 2; k++) {
            const sp = add(e0, scale(dir, k === 0 ? t0 : t1));
            const distPlane = dot(sub(sp, refCenter), nf);
            if (distPlane - radius > bandF) continue;
            contacts.push({ capCore: sp, hullSurf: sub(sp, scale(nf, distPlane)), ordinal: k });
        }
        if (contacts.length) return { contacts, normal: nf };
    }

    // fallback (segment off the reference face — an edge/vertex contact): per-endpoint closest points
    const qc = qconj(quat);
    const endpoint = (e: Vec3) => {
        const r = closestPointOnHull(h, rotate(qc, sub(e, pos)));
        return { surf: add(pos, rotate(quat, r.point)), gap: r.signedDist - radius };
    };
    const ends = [endpoint(e0), endpoint(e1)];
    for (let k = 0; k < 2; k++) {
        if (ends[k].gap > bandF) continue;
        contacts.push({ capCore: k === 0 ? e0 : e1, hullSurf: ends[k].surf, ordinal: k });
    }
    return { contacts, normal: nf };
}
