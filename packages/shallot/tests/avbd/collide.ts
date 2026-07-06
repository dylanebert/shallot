// Port of reference/avbd-demo3d/source/collide.cpp — box-box (OBB) SAT with
// Sutherland-Hodgman manifold clipping. 15 candidate axes (3 face-A, 3 face-B, 9
// edge-edge), edge/face bias 0.95/0.01, persistent feature keys for warmstart. This
// is the narrowphase the contact solver feeds on, and the SAT crux gated against
// gold vectors regenerated from this exact C++ (sat.test.ts). Faithful port; f64.
//
// Phase 4.8.1: a face clip can yield up to 8 candidates; the manifold is reduced to a ≤4-point
// spread set — a faithful port of Jolt's `PruneContactPoints` (reference/jolt ManifoldBetweenTwoFaces.cpp,
// the Dirk Gregorius GDC-2015 "Robust Contact Creation" recipe; reference/bullet3 b3ContactCache
// corroborates): project onto the contact plane, keep the deepest-by-torque×depth point, its farthest
// partner, then the furthest point on each side of that line to maximize the quad area. Gold-gated
// against a verbatim-Jolt extract (reduce.test.ts). Each kept contact keeps its ORIGINAL clip-ordinal
// feature key (stable under rotation — the Mode-2 SAT probe); we do NOT re-ordinal to storage rank the
// way webphysics does, because its warmstart is slot-indexed while ours scans by (a,b)+feature key — a
// rank renumber would give one key to two different physical contacts when the selected set shifts,
// a false warmstart match. avbd-demo3d's collide.cpp runs the identical reduction (the harness fork),
// so the oracle stays a faithful port of the C++.

import {
    add,
    clamp,
    cross,
    dot,
    lengthSq,
    type Mat3,
    neg,
    orthonormal,
    type Quat,
    qconj,
    rotate,
    scale,
    sub,
    type Vec3,
} from "./math";

// output manifold cap — the reduced 4-point spread set (= GPU CONTACTS_PER_PAIR). Was 8 (keep-all);
// Phase 4.8.1 reduces to 4, halving the GPU pairContacts store.
const MAX_CONTACTS = 4;
// candidate cap before reduction: a quad clipped by 4 ref-face half-planes gains ≤1 vertex per plane
// (4→8), the exact Sutherland-Hodgman bound. Kept in sync with collide.ts COLLIDE_WGSL, where the same
// bound is load-bearing on Metal.
const MAX_CANDIDATES = 8;
const MAX_POLY_VERTS = 8;
const SAT_AXIS_EPSILON = 1e-6;
const PLANE_EPSILON = 1e-5;
const CONTACT_MERGE_DIST_SQ = 1e-6;

const AXIS_FACE_A = 0;
const AXIS_FACE_B = 1;
const AXIS_EDGE = 2;

// Speculative contact band (Phase 4.8.3) — the SAT emits a contact while the boxes are still separated
// by up to this gap, carrying the true signed gap in c0 so the solver's repulsion-only normal constraint
// (`force.x = min(force.x, 0)`) limits the approach velocity: the body lands at contact, no penetration
// pop, no tunnel for a body within the band at frame start (Firth 2011 / Box2D's speculative solver). It
// is a contact-GENERATION change, not a solver change — the constraint is already the unilateral
// speculative form. Derived as Box2D's `b2_speculativeDistance = 4 · b2_linearSlop`: COLLISION_MARGIN
// (manifold.ts, 0.01) is our slop/skin, so 4× = 0.04. Kept DISTINCT from COLLISION_MARGIN (the
// equilibrium offset, not a generation band) — don't conflate the two magnitudes (Box2D keeps them
// separate; webphysics mashes them into one tuned literal). The broadphase pads its sphere/AABB by the
// same distance (solver.ts / GPU step.ts) so the speculative pair is found before contact.
//
// Phase 4.8.4 (velocity sweep) extends this STATIC band by the relative closing displacement along each
// SAT axis — `collide(a, b, dRel)`, dRel = (vA−vB)·dt — so a fast mover (`v·dt` ≫ the band) crossing the
// whole contact between frames is caught at frame start (no tunnel): the swept contact carries the +gap,
// the unilateral constraint limits the approach. This is the Box2D/Bullet/Firth speculative-CCD
// completion (a motivated deviation from webphysics, whose narrowphase keeps a fixed slop band). The
// broadphase also sweeps its AABB by `|vel|·dt` (matching webphysics's velocity-fattened tree). The band
// degenerates to this static value at vRel = 0.
export const SPECULATIVE_DISTANCE = 0.04;

/** the minimal box pose the SAT reads — center, orientation, full widths */
export interface Box {
    pos: Vec3;
    quat: Quat;
    size: Vec3;
}

/** one contact point: persistent feature key + arms in each body's local space */
export interface Contact {
    feature: number;
    rA: Vec3;
    rB: Vec3;
}

interface OBB {
    center: Vec3;
    half: Vec3;
    axis: [Vec3, Vec3, Vec3];
}

interface SatAxis {
    type: number;
    indexA: number;
    indexB: number;
    separation: number;
    normalAB: Vec3;
    valid: boolean;
}

interface FaceFrame {
    normal: Vec3;
    center: Vec3;
    u: Vec3;
    v: Vec3;
    extentU: number;
    extentV: number;
}

function obb(b: Box): OBB {
    return {
        center: b.pos,
        half: scale(b.size, 0.5),
        axis: [rotate(b.quat, [1, 0, 0]), rotate(b.quat, [0, 1, 0]), rotate(b.quat, [0, 0, 1])],
    };
}

const absDot = (a: Vec3, b: Vec3): number => Math.abs(dot(a, b));

function supportPoint(box: OBB, dir: Vec3): Vec3 {
    const sx = dot(dir, box.axis[0]) >= 0 ? 1 : -1;
    const sy = dot(dir, box.axis[1]) >= 0 ? 1 : -1;
    const sz = dot(dir, box.axis[2]) >= 0 ? 1 : -1;
    return add(
        add(box.center, scale(box.axis[0], box.half[0] * sx)),
        add(scale(box.axis[1], box.half[1] * sy), scale(box.axis[2], box.half[2] * sz)),
    );
}

function getFaceAxes(
    box: OBB,
    axisIndex: number,
): { u: Vec3; v: Vec3; extentU: number; extentV: number } {
    if (axisIndex === 0)
        return { u: box.axis[1], v: box.axis[2], extentU: box.half[1], extentV: box.half[2] };
    if (axisIndex === 1)
        return { u: box.axis[0], v: box.axis[2], extentU: box.half[0], extentV: box.half[2] };
    return { u: box.axis[0], v: box.axis[1], extentU: box.half[0], extentV: box.half[1] };
}

function buildFaceFrame(box: OBB, axisIndex: number, outwardNormal: Vec3): FaceFrame {
    const sign = dot(outwardNormal, box.axis[axisIndex]) >= 0 ? 1 : -1;
    const normal = scale(box.axis[axisIndex], sign);
    const center = add(box.center, scale(normal, box.half[axisIndex]));
    const { u, v, extentU, extentV } = getFaceAxes(box, axisIndex);
    return { normal, center, u, v, extentU, extentV };
}

function incidentAxis(box: OBB, referenceNormal: Vec3): number {
    let axis = 0;
    let best = -Infinity;
    for (let i = 0; i < 3; i++) {
        const d = absDot(box.axis[i], referenceNormal);
        if (d > best) {
            best = d;
            axis = i;
        }
    }
    return axis;
}

function buildIncidentFace(box: OBB, axisIndex: number, referenceNormal: Vec3): Vec3[] {
    const sign = dot(box.axis[axisIndex], referenceNormal) > 0 ? -1 : 1;
    const faceNormal = scale(box.axis[axisIndex], sign);
    const faceCenter = add(box.center, scale(faceNormal, box.half[axisIndex]));
    const { u, v, extentU, extentV } = getFaceAxes(box, axisIndex);
    return [
        add(add(faceCenter, scale(u, extentU)), scale(v, extentV)),
        add(add(faceCenter, scale(u, -extentU)), scale(v, extentV)),
        add(add(faceCenter, scale(u, -extentU)), scale(v, -extentV)),
        add(add(faceCenter, scale(u, extentU)), scale(v, -extentV)),
    ];
}

function clip(inVerts: Vec3[], planeNormal: Vec3, planeOffset: number): Vec3[] {
    const out: Vec3[] = [];
    const n = inVerts.length;
    if (n <= 0) return out;

    let a = inVerts[n - 1];
    let da = dot(planeNormal, a) - planeOffset;

    for (let i = 0; i < n; i++) {
        const b = inVerts[i];
        const db = dot(planeNormal, b) - planeOffset;

        const aInside = da <= PLANE_EPSILON;
        const bInside = db <= PLANE_EPSILON;

        if (aInside !== bInside) {
            let t = 0;
            const denom = da - db;
            if (Math.abs(denom) > SAT_AXIS_EPSILON) t = clamp(da / denom, 0, 1);
            if (out.length < MAX_POLY_VERTS) out.push(add(a, scale(sub(b, a), t)));
        }
        if (bInside && out.length < MAX_POLY_VERTS) out.push(b);

        a = b;
        da = db;
    }
    return out;
}

// append one contact (local arms in each body's frame). Face-manifold dedup happens earlier, in the
// candidate build; this only runs for the final reduced set + the edge/fallback single-point cases.
function addContact(
    a: Box,
    b: Box,
    contacts: Contact[],
    xA: Vec3,
    xB: Vec3,
    featureKey: number,
): void {
    if (contacts.length >= MAX_CONTACTS) return;
    contacts.push({
        feature: featureKey | 0,
        rA: rotate(qconj(a.quat), sub(xA, a.pos)),
        rB: rotate(qconj(b.quat), sub(xB, b.pos)),
    });
}

/** a clip-vertex candidate carried through reduction + sort: world anchors + the feature key */
export interface Cand {
    feature: number;
    xA: Vec3;
    xB: Vec3;
}

// neither distance² nor depth² should reach zero in the point1/point2 heuristic (Jolt clamps both)
const REDUCE_MIN_DIST_SQ = 1e-6;

/**
 * Reduce a candidate set (>4) to a ≤4-point spread manifold — a faithful port of Jolt's
 * `PruneContactPoints` (reference/jolt ManifoldBetweenTwoFaces.cpp). Project each A-anchor onto the
 * contact plane (⊥ `normal`, relative to `comA`); keep the point maximizing (planar dist)²·depth²
 * (torque leverage × penetration), its farthest plane-partner, then the furthest candidate on EACH
 * side of that line — the two-sided selection that maximizes the contact-quad area (vs a one-sided
 * pick that can collapse the quad). Returns 2-4 points; gold-gated against verbatim Jolt (reduce.test.ts).
 */
export function reduceManifold(cands: Cand[], normal: Vec3, comA: Vec3): Cand[] {
    const proj = cands.map((c) => {
        const v = sub(c.xA, comA);
        return sub(v, scale(normal, dot(v, normal)));
    });
    const depth2 = cands.map((c) => Math.max(REDUCE_MIN_DIST_SQ, lengthSq(sub(c.xB, c.xA))));

    let p1 = 0;
    let best = Math.max(REDUCE_MIN_DIST_SQ, lengthSq(proj[0])) * depth2[0];
    for (let i = 1; i < cands.length; i++) {
        const v = Math.max(REDUCE_MIN_DIST_SQ, lengthSq(proj[i])) * depth2[i];
        if (v > best) {
            best = v;
            p1 = i;
        }
    }

    let p2 = -1;
    best = -Infinity;
    for (let i = 0; i < cands.length; i++) {
        if (i === p1) continue;
        const v = Math.max(REDUCE_MIN_DIST_SQ, lengthSq(sub(proj[i], proj[p1]))) * depth2[i];
        if (v > best) {
            best = v;
            p2 = i;
        }
    }

    // furthest candidate on each side of the p1→p2 line (signed perpendicular) — maximizes the quad area
    const perp = cross(sub(proj[p2], proj[p1]), normal);
    let p3 = -1;
    let p4 = -1;
    let minV = 0;
    let maxV = 0;
    for (let i = 0; i < cands.length; i++) {
        if (i === p1 || i === p2) continue;
        const v = dot(perp, sub(proj[i], proj[p1]));
        if (v < minV) {
            minV = v;
            p3 = i;
        } else if (v > maxV) {
            maxV = v;
            p4 = i;
        }
    }

    // polygon order [p1, p3, p2, p4]; p3/p4 absent (all candidates on one side) ⇒ 2-3 points
    const sel = [cands[p1]];
    if (p3 >= 0) sel.push(cands[p3]);
    sel.push(cands[p2]);
    if (p4 >= 0) sel.push(cands[p4]);
    return sel;
}

function testAxis(
    boxA: OBB,
    boxB: OBB,
    delta: Vec3,
    axis: Vec3,
    type: number,
    indexA: number,
    indexB: number,
    best: SatAxis,
    dRel: Vec3,
): boolean {
    const lenSq = lengthSq(axis);
    if (lenSq < SAT_AXIS_EPSILON) return true;

    const invLen = 1 / Math.sqrt(lenSq);
    let n = scale(axis, invLen);
    if (dot(n, delta) < 0) n = neg(n);

    const distance = Math.abs(dot(delta, n));
    const rA =
        boxA.half[0] * absDot(n, boxA.axis[0]) +
        boxA.half[1] * absDot(n, boxA.axis[1]) +
        boxA.half[2] * absDot(n, boxA.axis[2]);
    const rB =
        boxB.half[0] * absDot(n, boxB.axis[0]) +
        boxB.half[1] * absDot(n, boxB.axis[1]) +
        boxB.half[2] * absDot(n, boxB.axis[2]);

    const separation = distance - (rA + rB);
    // a separating axis aborts the SAT only past the speculative band; within it the axis of maximum
    // separation (least overlap) is kept and a speculative manifold built off it, so a body within the
    // band lands at contact rather than tunnelling through it (Phase 4.8.3). Phase 4.8.4 (velocity sweep):
    // the band along n is the LARGER of the static skin and the closing displacement this step,
    // max(SPECULATIVE_DISTANCE, max(0, dot(dRel, n))) (dRel = (vA−vB)·dt) — so a fast mover that would
    // cross the contact this step is caught at frame start (Box2D/Bullet/Firth). `max`, not `+`: a slow
    // body's band is the velocity-INDEPENDENT static skin (closing ≪ skin), so a settling pile doesn't
    // feed its own residual velocity back into the contact set — the feedback that limit-cycles the
    // marginal 8:1 bridge. The argmax stays on raw separation (the geometric MTV); closing 0 recovers 4.8.3.
    if (separation > Math.max(SPECULATIVE_DISTANCE, Math.max(0, dot(dRel, n)))) return false;

    if (!best.valid || separation > best.separation) {
        best.valid = true;
        best.type = type;
        best.indexA = indexA;
        best.indexB = indexB;
        best.separation = separation;
        best.normalAB = n;
    }
    return true;
}

function supportEdge(box: OBB, axisIndex: number, dir: Vec3): { edgeA: Vec3; edgeB: Vec3 } {
    const axis1 = (axisIndex + 1) % 3;
    const axis2 = (axisIndex + 2) % 3;
    const sign1 = dot(dir, box.axis[axis1]) >= 0 ? 1 : -1;
    const sign2 = dot(dir, box.axis[axis2]) >= 0 ? 1 : -1;
    const edgeCenter = add(
        box.center,
        add(
            scale(box.axis[axis1], box.half[axis1] * sign1),
            scale(box.axis[axis2], box.half[axis2] * sign2),
        ),
    );
    return {
        edgeA: sub(edgeCenter, scale(box.axis[axisIndex], box.half[axisIndex])),
        edgeB: add(edgeCenter, scale(box.axis[axisIndex], box.half[axisIndex])),
    };
}

/**
 * Closest points between two segments [p0,p1] and [q0,q1] — the clamped Ericson RTCD §5.1.9 routine.
 * Shared by the box edge-edge contact and the rounded narrowphase (rounded.ts), where a sphere is a
 * zero-length segment (a point), so one routine handles every rounded pair.
 */
export function closestSegments(p0: Vec3, p1: Vec3, q0: Vec3, q1: Vec3): { c0: Vec3; c1: Vec3 } {
    const d1 = sub(p1, p0);
    const d2 = sub(q1, q0);
    const r = sub(p0, q0);
    const a = dot(d1, d1);
    const e = dot(d2, d2);
    const f = dot(d2, r);

    let s = 0;
    let t = 0;

    if (a <= SAT_AXIS_EPSILON && e <= SAT_AXIS_EPSILON) {
        return { c0: p0, c1: q0 };
    }
    if (a <= SAT_AXIS_EPSILON) {
        t = clamp(f / e, 0, 1);
    } else {
        const c = dot(d1, r);
        if (e <= SAT_AXIS_EPSILON) {
            s = clamp(-c / a, 0, 1);
        } else {
            const b = dot(d1, d2);
            const denom = a * e - b * b;
            if (Math.abs(denom) > SAT_AXIS_EPSILON) s = clamp((b * f - c * e) / denom, 0, 1);
            t = (b * s + f) / e;
            if (t < 0) {
                t = 0;
                s = clamp(-c / a, 0, 1);
            } else if (t > 1) {
                t = 1;
                s = clamp((b - c) / a, 0, 1);
            }
        }
    }
    return { c0: add(p0, scale(d1, s)), c1: add(q0, scale(d2, t)) };
}

function buildFaceManifold(
    bodyA: Box,
    bodyB: Box,
    boxA: OBB,
    boxB: OBB,
    referenceIsA: boolean,
    referenceAxis: number,
    normalAB: Vec3,
    band: number,
): Contact[] {
    const referenceBox = referenceIsA ? boxA : boxB;
    const incidentBox = referenceIsA ? boxB : boxA;
    const referenceOutward = referenceIsA ? normalAB : neg(normalAB);

    const ref = buildFaceFrame(referenceBox, referenceAxis, referenceOutward);
    const incAxis = incidentAxis(incidentBox, ref.normal);

    let poly = buildIncidentFace(incidentBox, incAxis, ref.normal);

    poly = clip(poly, ref.u, dot(ref.u, ref.center) + ref.extentU);
    if (!poly.length) return [];
    poly = clip(poly, neg(ref.u), dot(neg(ref.u), ref.center) + ref.extentU);
    if (!poly.length) return [];
    poly = clip(poly, ref.v, dot(ref.v, ref.center) + ref.extentV);
    if (!poly.length) return [];
    poly = clip(poly, neg(ref.v), dot(neg(ref.v), ref.center) + ref.extentV);
    if (!poly.length) return [];

    const contacts: Contact[] = [];
    let featurePrefix = (referenceIsA ? AXIS_FACE_A : AXIS_FACE_B) << 24;
    featurePrefix |= (referenceAxis & 0xff) << 16;
    featurePrefix |= (incAxis & 0xff) << 8;

    // build the clipped candidates (up to MAX_CANDIDATES), dedup by midpoint
    const cands: Cand[] = [];
    const mids: Vec3[] = [];
    for (let i = 0; i < poly.length && cands.length < MAX_CANDIDATES; i++) {
        const pIncident = poly[i];
        const distance = dot(sub(pIncident, ref.center), ref.normal);
        // a clip vertex up to the (swept) band beyond the reference face is kept: its projection onto the
        // face plane carries the +gap into c0, generating a separated contact early (Phase 4.8.3). The band
        // is SPECULATIVE_DISTANCE + the closing displacement along the contact normal (Phase 4.8.4); a
        // penetrating vertex (distance ≤ 0) is always kept, so a settled pile's manifold is unchanged.
        if (distance > band) continue;
        const pReference = sub(pIncident, scale(ref.normal, distance));
        const xA = referenceIsA ? pReference : pIncident;
        const xB = referenceIsA ? pIncident : pReference;
        const mid = scale(add(xA, xB), 0.5);
        if (mids.some((m) => lengthSq(sub(mid, m)) < CONTACT_MERGE_DIST_SQ)) continue;
        cands.push({ feature: featurePrefix | (i & 0xff), xA, xB });
        mids.push(mid);
    }

    if (!cands.length) {
        const xA = supportPoint(boxA, normalAB);
        const xB = supportPoint(boxB, neg(normalAB));
        addContact(bodyA, bodyB, contacts, xA, xB, featurePrefix);
        return contacts;
    }

    // reduce to the spread set when the clip over-produced; each kept contact keeps its original
    // clip-ordinal feature key (stable, scan-matched — see the header on why we don't re-ordinal).
    const sel = cands.length > MAX_CONTACTS ? reduceManifold(cands, normalAB, bodyA.pos) : cands;
    for (const c of sel) addContact(bodyA, bodyB, contacts, c.xA, c.xB, c.feature);
    return contacts;
}

function buildEdgeContact(
    bodyA: Box,
    bodyB: Box,
    boxA: OBB,
    boxB: OBB,
    axisA: number,
    axisB: number,
    normalAB: Vec3,
): Contact[] {
    const ea = supportEdge(boxA, axisA, normalAB);
    const eb = supportEdge(boxB, axisB, neg(normalAB));
    const { c0: xA, c1: xB } = closestSegments(ea.edgeA, ea.edgeB, eb.edgeA, eb.edgeB);

    const contacts: Contact[] = [];
    const featureKey = (AXIS_EDGE << 24) | ((axisA & 0xff) << 8) | (axisB & 0xff);
    addContact(bodyA, bodyB, contacts, xA, xB, featureKey);

    if (!contacts.length) {
        const sA = supportPoint(boxA, normalAB);
        const sB = supportPoint(boxB, neg(normalAB));
        addContact(bodyA, bodyB, contacts, sA, sB, featureKey);
    }
    return contacts;
}

/**
 * Box-box SAT. Returns the contact manifold (0–4 points, the reduced spread set) and the
 * contact `basis` (normal in row 0, pointing from B to A; tangents in rows 1-2). Empty array =
 * no overlap. Mirrors `Manifold::collide` + the 4.8.1 reduction/sort.
 *
 * `dRel` is the relative displacement over the step `(vA − vB)·dt` — the velocity sweep (Phase
 * 4.8.4): it widens the per-axis SAT band so a fast mover crossing the contact this step generates
 * its swept contact at frame start. The default `[0,0,0]` recovers the static 4.8.3 SAT exactly.
 */
export function collide(
    bodyA: Box,
    bodyB: Box,
    dRel: Vec3 = [0, 0, 0],
): { contacts: Contact[]; basis: Mat3; separation: number } {
    const boxA = obb(bodyA);
    const boxB = obb(bodyB);
    const delta = sub(boxB.center, boxA.center);

    const bestFace: SatAxis = {
        type: 0,
        indexA: -1,
        indexB: -1,
        separation: -Infinity,
        normalAB: [0, 0, 0],
        valid: false,
    };
    const bestEdge: SatAxis = {
        type: 0,
        indexA: -1,
        indexB: -1,
        separation: -Infinity,
        normalAB: [0, 0, 0],
        valid: false,
    };
    const none = {
        contacts: [] as Contact[],
        basis: [
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
        ] as Mat3,
        // separated past the band → no penetration (the overlap gate reads -separation as depth, clamped ≥ 0)
        separation: Number.POSITIVE_INFINITY,
    };

    for (let i = 0; i < 3; i++) {
        if (!testAxis(boxA, boxB, delta, boxA.axis[i], AXIS_FACE_A, i, -1, bestFace, dRel))
            return none;
    }
    for (let i = 0; i < 3; i++) {
        if (!testAxis(boxA, boxB, delta, boxB.axis[i], AXIS_FACE_B, -1, i, bestFace, dRel))
            return none;
    }
    for (let i = 0; i < 3; i++) {
        for (let j = 0; j < 3; j++) {
            const axis = cross(boxA.axis[i], boxB.axis[j]);
            if (!testAxis(boxA, boxB, delta, axis, AXIS_EDGE, i, j, bestEdge, dRel)) return none;
        }
    }

    if (!bestFace.valid) return none;

    let best = bestFace;
    if (bestEdge.valid) {
        const edgeRelTol = 0.95;
        const edgeAbsTol = 0.01;
        if (edgeRelTol * bestEdge.separation > bestFace.separation + edgeAbsTol) best = bestEdge;
    }

    const basis = orthonormal(neg(best.normalAB));

    // the swept clip band along the contact axis (Phase 4.8.4): the larger of the static skin and the
    // closing displacement onto the winning normal — same max(SPECULATIVE_DISTANCE, closing) form as the
    // testAxis abort, evaluated for the winning axis. A max(0, …) magnitude, reference-orientation safe.
    const band = Math.max(SPECULATIVE_DISTANCE, Math.max(0, dot(dRel, best.normalAB)));

    let contacts: Contact[];
    if (best.type === AXIS_EDGE) {
        contacts = buildEdgeContact(
            bodyA,
            bodyB,
            boxA,
            boxB,
            best.indexA,
            best.indexB,
            best.normalAB,
        );
    } else if (best.type === AXIS_FACE_A) {
        contacts = buildFaceManifold(
            bodyA,
            bodyB,
            boxA,
            boxB,
            true,
            best.indexA,
            best.normalAB,
            band,
        );
    } else {
        contacts = buildFaceManifold(
            bodyA,
            bodyB,
            boxA,
            boxB,
            false,
            best.indexB,
            best.normalAB,
            band,
        );
    }
    // best.separation is the SAT min-separation along the winning axis: < 0 ⇒ penetrating (depth = -sep),
    // ≥ 0 ⇒ a speculative/touching contact. The overlap gate reads it to flag severe interpenetration.
    return { contacts, basis, separation: best.separation };
}
