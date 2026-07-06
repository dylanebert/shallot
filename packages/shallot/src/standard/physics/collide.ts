// Box-box (OBB) SAT with Sutherland-Hodgman manifold clipping — the WGSL port of the CPU
// oracle (tests/avbd/collide.ts), which is itself a faithful port of
// reference/avbd-demo3d/source/collide.cpp. 15 candidate axes (3 face-A, 3 face-B, 9
// edge-edge), edge/face bias 0.95/0.01, persistent feature keys for warmstart. The
// narrowphase pass calls `collideBoxBox` per overlapping pair and atomic-appends the
// resulting contacts into the source-agnostic constraint buffer.
//
// This is the SAT crux (scratch.md "AVBD rebuild" → Distributed cruxes): the feature keys
// it emits feed warmstart, so they must be bit-identical to the C++. The CPU SAT is validated
// against the gold vectors in tests/avbd/sat.test.ts; the gym `pile` scenario is the
// canonical real-GPU home (the GPU SAT reproduces it, checked by the single-step gate).
//
// HULL_WGSL (Phase 6.3, below) ports the convex-hull oracle (tests/avbd/hull.ts + the rounded × polytope
// path in rounded.ts): the same SAT generalized to arbitrary faces/edges (`collideHull`), closest-point
// (`closestPointOnHull`), the capsule segment-clip (`capsulePoly`), and the unified rounded × polytope
// dispatch (`collideRoundedPolytope`). A `Convex` is scale-unified — a box is the built-in unit cube
// (`UNIT_CUBE_ID`) scaled by its half-extents — so box and hull collide through one branch-free path. It
// reads the `hullData` buffer packed by `./hull` (`packHulls`); concatenate it AFTER COLLIDE_WGSL (it reuses
// the helpers + structs here) and declare a `hullData: array<u32>` binding.

import { HULL_FACE_STRIDE, HULL_HEADER, UNIT_CUBE_ID } from "./hull";

/** output manifold cap — the reduced spread set (= CONTACTS_PER_PAIR); was 8, halved by 4.8.1 */
export const MAX_CONTACTS = 4;

/**
 * speculative contact band (Phase 4.8.3) — the SAT emits a contact while the boxes are separated by up to
 * this gap, carrying the true signed gap in c0 so the repulsion-only normal constraint limits the approach:
 * a body within the band at frame start lands at contact, no penetration pop, no tunnel (Firth 2011 /
 * Box2D's speculative solver). Derived as Box2D's `b2_speculativeDistance = 4 · b2_linearSlop`:
 * COLLISION_MARGIN (0.01) is our slop/skin, so 4× = 0.04. Kept DISTINCT from COLLISION_MARGIN (the
 * equilibrium offset, not a generation band). step.ts pads the broadphase AABB + sphere test by this; the
 * f64 oracle + C++ harness carry the same value, so GPU == oracle == C++.
 *
 * Phase 4.8.4 (velocity sweep) extends this STATIC band per SAT axis to max(this, closing displacement)
 * via `collideBoxBox(…, dRel)` (dRel = (vA−vB)·dt), so a fast mover crossing the whole contact in one step
 * (`v·dt` ≫ the band) is caught at frame start too. step.ts also sweeps the broadphase AABB by `|vel|·dt`
 * (matching webphysics's velocity-fattened tree) and the sphere test by `|vRel|·dt`.
 */
export const SPECULATIVE_DISTANCE = 0.04;

// The box+rounded narrowphase, split into separately-compilable chunks. `HELPERS_WGSL` (math + structs +
// the shared manifold primitives), `BOXBOX_WGSL` (the box-box SAT), and `ROUNDED_WGSL` (sphere/capsule)
// recompose into `COLLIDE_WGSL` (below) for the gym `sat`/`pile` kernel gates + the physics/core re-export. The
// production collide pass (step.ts) concatenates only the chunks each shape-pair pipeline needs, so a
// box-only pipeline never compiles the hull/rounded SATs — DXC compile is superlinear in kernel size
// (gpu.md "DXC shader compilation": dead code isn't free; pipeline splits). The `_`-prefixed fragments are
// the interleaved shared/box sections; HELPERS_WGSL / BOXBOX_WGSL recombine them. Row-major Mat3 to match
// the oracle (tests/avbd/math.ts). f32 throughout (the rebuild's f32-first decision).
const _MATH = /* wgsl */ `
struct Mat3 { r0: vec3<f32>, r1: vec3<f32>, r2: vec3<f32> };

// up to MAX_CONTACTS (4) reduced contact points + the contact basis (normal in row 0, B->A; tangents rows 1-2)
struct SatResult {
    count: u32,
    basis: Mat3,
    feat: array<u32, ${MAX_CONTACTS}>,
    rA: array<vec3<f32>, ${MAX_CONTACTS}>,
    rB: array<vec3<f32>, ${MAX_CONTACTS}>,
};

const SAT_AXIS_EPSILON: f32 = 1e-6;
const PLANE_EPSILON: f32 = 1e-5;
const CONTACT_MERGE_DIST_SQ: f32 = 1e-6;
const REDUCE_MIN_DIST_SQ: f32 = 1e-6;
// the speculative contact band (Phase 4.8.3) — see the SPECULATIVE_DISTANCE JS export header. Mirrors the
// f64 oracle + C++ harness exactly (= 4 · COLLISION_MARGIN); step.ts pads the broadphase by the same.
const SPECULATIVE_DISTANCE: f32 = ${SPECULATIVE_DISTANCE};
const AXIS_FACE_A: u32 = 0u;
const AXIS_FACE_B: u32 = 1u;
const AXIS_EDGE: u32 = 2u;
// a quad clipped by 4 reference-face half-planes gains at most one vertex per plane (4→8), the exact
// candidate bound before the 4.8.1 reduction. Also load-bearing on Apple/Metal-3: a larger candidate
// footprint pushes collideBoxBox's peak function-private footprint past the register-spill threshold,
// where Metal miscompiles the per-lane offset of the spilled SatResult under multi-lane execution and
// every face manifold collapses to count=1 (gpu.md "WebGPU-specific traps"). Don't regrow it.
const MAX_CANDIDATES: u32 = 8u;
const MAX_POLY_VERTS: u32 = 8u;
// the single rounded (sphere/capsule) contact's fixed feature key (kind 3, distinct from the box
// AXIS_FACE/EDGE keys) — one contact per pair, so a stable key means warmstart always matches. Mirrors
// tests/avbd/rounded.ts ROUND_FEATURE.
const ROUND_FEATURE: u32 = 0x03000000u;
// below this core-to-core distance the rounded contact normal is undefined (concentric cores) — fall
// back to world up. A degenerate deep-overlap guard.
const ROUND_NORMAL_EPS: f32 = 1e-9;

fn satQConj(q: vec4<f32>) -> vec4<f32> { return vec4<f32>(-q.xyz, q.w); }
fn satQRotate(q: vec4<f32>, v: vec3<f32>) -> vec3<f32> {
    let t = 2.0 * cross(q.xyz, v);
    return v + q.w * t + cross(q.xyz, t);
}

// orthonormal frame with n as row 0 (matches math.ts orthonormal / the gym orthoBasis)
fn orthoBasis(n: vec3<f32>) -> Mat3 {
    var t1: vec3<f32>;
    if (abs(n.x) > abs(n.y)) { t1 = vec3<f32>(-n.z, 0.0, n.x); } else { t1 = vec3<f32>(0.0, n.z, -n.y); }
    t1 = normalize(t1);
    return Mat3(n, t1, cross(t1, n));
}
`;

// box-box SAT, part A: the OBB + its face/edge clip primitives.
const _BOX_A = /* wgsl */ `
struct Obb { c: vec3<f32>, h: vec3<f32>, ax0: vec3<f32>, ax1: vec3<f32>, ax2: vec3<f32> };

fn obbOf(pos: vec3<f32>, quat: vec4<f32>, size: vec3<f32>) -> Obb {
    return Obb(pos, size * 0.5,
        satQRotate(quat, vec3<f32>(1.0, 0.0, 0.0)),
        satQRotate(quat, vec3<f32>(0.0, 1.0, 0.0)),
        satQRotate(quat, vec3<f32>(0.0, 0.0, 1.0)));
}

fn obbAxis(o: Obb, i: u32) -> vec3<f32> {
    if (i == 0u) { return o.ax0; }
    if (i == 1u) { return o.ax1; }
    return o.ax2;
}
fn obbHalf(o: Obb, i: u32) -> f32 {
    if (i == 0u) { return o.h.x; }
    if (i == 1u) { return o.h.y; }
    return o.h.z;
}

fn absDot(a: vec3<f32>, b: vec3<f32>) -> f32 { return abs(dot(a, b)); }

fn supportPoint(o: Obb, dir: vec3<f32>) -> vec3<f32> {
    let sx = select(-1.0, 1.0, dot(dir, o.ax0) >= 0.0);
    let sy = select(-1.0, 1.0, dot(dir, o.ax1) >= 0.0);
    let sz = select(-1.0, 1.0, dot(dir, o.ax2) >= 0.0);
    return o.c + o.ax0 * (o.h.x * sx) + o.ax1 * (o.h.y * sy) + o.ax2 * (o.h.z * sz);
}

// face tangent axes + their extents for a given face axis index (collide.ts getFaceAxes)
struct FaceAxes { u: vec3<f32>, v: vec3<f32>, eu: f32, ev: f32 };
fn getFaceAxes(o: Obb, ai: u32) -> FaceAxes {
    if (ai == 0u) { return FaceAxes(o.ax1, o.ax2, o.h.y, o.h.z); }
    if (ai == 1u) { return FaceAxes(o.ax0, o.ax2, o.h.x, o.h.z); }
    return FaceAxes(o.ax0, o.ax1, o.h.x, o.h.y);
}

fn incidentAxis(o: Obb, refN: vec3<f32>) -> u32 {
    var axis = 0u;
    var best = -1e30;
    for (var i = 0u; i < 3u; i = i + 1u) {
        let d = absDot(obbAxis(o, i), refN);
        if (d > best) { best = d; axis = i; }
    }
    return axis;
}

struct Poly { v: array<vec3<f32>, MAX_POLY_VERTS>, n: u32 };

fn incidentFace(o: Obb, ai: u32, refN: vec3<f32>) -> Poly {
    let sgn = select(1.0, -1.0, dot(obbAxis(o, ai), refN) > 0.0);
    let faceNormal = obbAxis(o, ai) * sgn;
    let faceCenter = o.c + faceNormal * obbHalf(o, ai);
    let fa = getFaceAxes(o, ai);
    var p: Poly;
    p.n = 4u;
    p.v[0] = faceCenter + fa.u * fa.eu + fa.v * fa.ev;
    p.v[1] = faceCenter - fa.u * fa.eu + fa.v * fa.ev;
    p.v[2] = faceCenter - fa.u * fa.eu - fa.v * fa.ev;
    p.v[3] = faceCenter + fa.u * fa.eu - fa.v * fa.ev;
    return p;
}

// Sutherland-Hodgman clip against one half-space (planeNormal . x <= planeOffset)
fn clip(inp: Poly, planeNormal: vec3<f32>, planeOffset: f32) -> Poly {
    var out: Poly;
    out.n = 0u;
    if (inp.n == 0u) { return out; }

    var a = inp.v[inp.n - 1u];
    var da = dot(planeNormal, a) - planeOffset;
    for (var i = 0u; i < inp.n; i = i + 1u) {
        let b = inp.v[i];
        let db = dot(planeNormal, b) - planeOffset;
        let aInside = da <= PLANE_EPSILON;
        let bInside = db <= PLANE_EPSILON;
        if (aInside != bInside) {
            var t = 0.0;
            let denom = da - db;
            if (abs(denom) > SAT_AXIS_EPSILON) { t = clamp(da / denom, 0.0, 1.0); }
            if (out.n < MAX_POLY_VERTS) { out.v[out.n] = a + (b - a) * t; out.n = out.n + 1u; }
        }
        if (bInside && out.n < MAX_POLY_VERTS) { out.v[out.n] = b; out.n = out.n + 1u; }
        a = b;
        da = db;
    }
    return out;
}

fn supportEdge(o: Obb, ai: u32, dir: vec3<f32>) -> array<vec3<f32>, 2> {
    let a1 = (ai + 1u) % 3u;
    let a2 = (ai + 2u) % 3u;
    let s1 = select(-1.0, 1.0, dot(dir, obbAxis(o, a1)) >= 0.0);
    let s2 = select(-1.0, 1.0, dot(dir, obbAxis(o, a2)) >= 0.0);
    let edgeCenter = o.c + obbAxis(o, a1) * (obbHalf(o, a1) * s1) + obbAxis(o, a2) * (obbHalf(o, a2) * s2);
    let half = obbAxis(o, ai) * obbHalf(o, ai);
    return array<vec3<f32>, 2>(edgeCenter - half, edgeCenter + half);
}
`;

// shared: segment-segment closest points (Ericson RTCD §5.1.9) — box edge-edge + rounded cores both use it.
const _SEG = /* wgsl */ `
fn closestSegments(p0: vec3<f32>, p1: vec3<f32>, q0: vec3<f32>, q1: vec3<f32>) -> array<vec3<f32>, 2> {
    let d1 = p1 - p0;
    let d2 = q1 - q0;
    let r = p0 - q0;
    let a = dot(d1, d1);
    let e = dot(d2, d2);
    let f = dot(d2, r);
    var s = 0.0;
    var t = 0.0;
    if (a <= SAT_AXIS_EPSILON && e <= SAT_AXIS_EPSILON) {
        return array<vec3<f32>, 2>(p0, q0);
    }
    if (a <= SAT_AXIS_EPSILON) {
        t = clamp(f / e, 0.0, 1.0);
    } else {
        let c = dot(d1, r);
        if (e <= SAT_AXIS_EPSILON) {
            s = clamp(-c / a, 0.0, 1.0);
        } else {
            let b = dot(d1, d2);
            let denom = a * e - b * b;
            if (abs(denom) > SAT_AXIS_EPSILON) { s = clamp((b * f - c * e) / denom, 0.0, 1.0); }
            t = (b * s + f) / e;
            if (t < 0.0) {
                t = 0.0;
                s = clamp(-c / a, 0.0, 1.0);
            } else if (t > 1.0) {
                t = 1.0;
                s = clamp((b - c) / a, 0.0, 1.0);
            }
        }
    }
    return array<vec3<f32>, 2>(p0 + d1 * s, q0 + d2 * t);
}
`;

// box-box SAT, part B: the per-axis separation test.
const _BOX_B = /* wgsl */ `
struct SatAxis { kind: u32, indexA: u32, indexB: u32, separation: f32, normalAB: vec3<f32>, valid: bool };

fn testAxis(boxA: Obb, boxB: Obb, delta: vec3<f32>, axis: vec3<f32>,
            kind: u32, ia: u32, ib: u32, best: ptr<function, SatAxis>, dRel: vec3<f32>) -> bool {
    let lenSq = dot(axis, axis);
    if (lenSq < SAT_AXIS_EPSILON) { return true; }
    var n = axis * (1.0 / sqrt(lenSq));
    if (dot(n, delta) < 0.0) { n = -n; }
    let distance = abs(dot(delta, n));
    let rA = boxA.h.x * absDot(n, boxA.ax0) + boxA.h.y * absDot(n, boxA.ax1) + boxA.h.z * absDot(n, boxA.ax2);
    let rB = boxB.h.x * absDot(n, boxB.ax0) + boxB.h.y * absDot(n, boxB.ax1) + boxB.h.z * absDot(n, boxB.ax2);
    let separation = distance - (rA + rB);
    // a separating axis aborts the SAT only past the speculative band; within it the axis of maximum
    // separation is kept and a speculative manifold built off it, so a body within the band lands at
    // contact rather than tunnelling through it (Phase 4.8.3). Phase 4.8.4 (velocity sweep): the band along
    // n is max(SPECULATIVE_DISTANCE, closing displacement max(0, dot(dRel, n))) (dRel = (vA−vB)·dt) — a
    // fast mover crossing the contact this step is caught at frame start. max (not +) keeps a slow body's
    // band velocity-independent (no settling feedback). Matches the f64 oracle + C++ harness; closing 0 = 4.8.3.
    if (separation > max(SPECULATIVE_DISTANCE, max(0.0, dot(dRel, n)))) { return false; }
    if (!(*best).valid || separation > (*best).separation) {
        (*best).valid = true;
        (*best).kind = kind;
        (*best).indexA = ia;
        (*best).indexB = ib;
        (*best).separation = separation;
        (*best).normalAB = n;
    }
    return true;
}
`;

// shared: contact append + the Jolt reduce-to-4 (box face manifolds + hull both use them).
const _CONTACT = /* wgsl */ `
// append one contact (local arms) into the result. Face-manifold dedup happens earlier, in the
// candidate build; this only runs for the final reduced set + the edge/fallback single-point cases.
// Mirrors collide.ts addContact.
fn addContact(res: ptr<function, SatResult>,
              posA: vec3<f32>, quatA: vec4<f32>, posB: vec3<f32>, quatB: vec4<f32>,
              xA: vec3<f32>, xB: vec3<f32>, feature: u32) {
    let cnt = (*res).count;
    if (cnt >= ${MAX_CONTACTS}u) { return; }
    (*res).feat[cnt] = feature;
    (*res).rA[cnt] = satQRotate(satQConj(quatA), xA - posA);
    (*res).rB[cnt] = satQRotate(satQConj(quatB), xB - posB);
    (*res).count = cnt + 1u;
}

// Reduce a > 4 candidate set to a <=4-point spread set — a verbatim port of Jolt's PruneContactPoints
// (reference/jolt ManifoldBetweenTwoFaces.cpp; Gregorius GDC-2015; reference/bullet3 b3ContactCache
// corroborates). Project each A-anchor onto the contact plane (perp axis, relative to comA), keep the
// point maximizing (planar dist)²·depth², its farthest plane-partner, then the furthest candidate on
// EACH side of that line (max quad area). Mirrors tests/avbd/collide.ts reduceManifold; gold-gated by
// reduce.test.ts. Writes the kept candidate indices to sel, returns the kept count (2-4).
fn pruneContacts(candXA: ptr<function, array<vec3<f32>, MAX_CANDIDATES>>,
                 candXB: ptr<function, array<vec3<f32>, MAX_CANDIDATES>>,
                 n: u32, axis: vec3<f32>, comA: vec3<f32>,
                 sel: ptr<function, array<u32, ${MAX_CONTACTS}>>) -> u32 {
    var projected: array<vec3<f32>, MAX_CANDIDATES>;
    var depthSq: array<f32, MAX_CANDIDATES>;
    for (var i = 0u; i < n; i = i + 1u) {
        let v1 = (*candXA)[i] - comA;
        projected[i] = v1 - axis * dot(v1, axis);
        depthSq[i] = max(REDUCE_MIN_DIST_SQ, dot((*candXB)[i] - (*candXA)[i], (*candXB)[i] - (*candXA)[i]));
    }

    var p1 = 0u;
    var val = max(REDUCE_MIN_DIST_SQ, dot(projected[0], projected[0])) * depthSq[0];
    for (var i = 0u; i < n; i = i + 1u) {
        let v = max(REDUCE_MIN_DIST_SQ, dot(projected[i], projected[i])) * depthSq[i];
        if (v > val) { val = v; p1 = i; }
    }
    let p1v = projected[p1];

    // n > 4 (the only call condition) guarantees a partner ≠ p1, so p2 is always found (matches the
    // reference, which keeps p1 + p2 unconditionally; only p3/p4 are side-of-line conditional).
    var p2 = 0u;
    val = -1e30;
    for (var i = 0u; i < n; i = i + 1u) {
        if (i == p1) { continue; }
        let d = projected[i] - p1v;
        let v = max(REDUCE_MIN_DIST_SQ, dot(d, d)) * depthSq[i];
        if (v > val) { val = v; p2 = i; }
    }
    let p2v = projected[p2];

    var p3 = 0u; var haveP3 = false;
    var p4 = 0u; var haveP4 = false;
    var minV = 0.0;
    var maxV = 0.0;
    let perp = cross(p2v - p1v, axis);
    for (var i = 0u; i < n; i = i + 1u) {
        if (i == p1 || i == p2) { continue; }
        let v = dot(perp, projected[i] - p1v);
        if (v < minV) { minV = v; p3 = i; haveP3 = true; }
        else if (v > maxV) { maxV = v; p4 = i; haveP4 = true; }
    }

    var c = 0u;
    (*sel)[c] = p1; c = c + 1u;
    if (haveP3) { (*sel)[c] = p3; c = c + 1u; }
    (*sel)[c] = p2; c = c + 1u;
    if (haveP4) { (*sel)[c] = p4; c = c + 1u; }
    return c;
}
`;

// box-box SAT, part C: the reference-face manifold + edge contact + collideBoxBox entry point.
const _BOX_C = /* wgsl */ `
fn faceManifold(res: ptr<function, SatResult>,
                boxA: Obb, boxB: Obb, posA: vec3<f32>, quatA: vec4<f32>, posB: vec3<f32>, quatB: vec4<f32>,
                referenceIsA: bool, refAxis: u32, normalAB: vec3<f32>, band: f32) {
    var refBox = boxB;
    var incBox = boxA;
    var refOutward = -normalAB;
    if (referenceIsA) { refBox = boxA; incBox = boxB; refOutward = normalAB; }

    // reference face frame
    let refSign = select(-1.0, 1.0, dot(refOutward, obbAxis(refBox, refAxis)) >= 0.0);
    let refNormal = obbAxis(refBox, refAxis) * refSign;
    let refCenter = refBox.c + refNormal * obbHalf(refBox, refAxis);
    let refFa = getFaceAxes(refBox, refAxis);

    let incAxis = incidentAxis(incBox, refNormal);
    var poly = incidentFace(incBox, incAxis, refNormal);

    poly = clip(poly, refFa.u, dot(refFa.u, refCenter) + refFa.eu);
    if (poly.n == 0u) { return; }
    poly = clip(poly, -refFa.u, dot(-refFa.u, refCenter) + refFa.eu);
    if (poly.n == 0u) { return; }
    poly = clip(poly, refFa.v, dot(refFa.v, refCenter) + refFa.ev);
    if (poly.n == 0u) { return; }
    poly = clip(poly, -refFa.v, dot(-refFa.v, refCenter) + refFa.ev);
    if (poly.n == 0u) { return; }

    var prefix = select(AXIS_FACE_B, AXIS_FACE_A, referenceIsA) << 24u;
    prefix = prefix | ((refAxis & 0xffu) << 16u);
    prefix = prefix | ((incAxis & 0xffu) << 8u);

    // build the clipped candidates (up to MAX_CANDIDATES), dedup by midpoint
    var candXA: array<vec3<f32>, MAX_CANDIDATES>;
    var candXB: array<vec3<f32>, MAX_CANDIDATES>;
    var candFeat: array<u32, MAX_CANDIDATES>;
    var candN = 0u;
    for (var i = 0u; i < poly.n && candN < MAX_CANDIDATES; i = i + 1u) {
        let pIncident = poly.v[i];
        let distance = dot(pIncident - refCenter, refNormal);
        // a clip vertex up to the (swept) band beyond the reference face is kept: its projection onto the
        // face plane carries the +gap into c0, generating a separated contact early (Phase 4.8.3). The band
        // is max(SPECULATIVE_DISTANCE, closing displacement along the contact normal) (Phase 4.8.4); a
        // penetrating vertex (distance <= 0) is always kept, so a settled pile's manifold is unchanged.
        if (distance > band) { continue; }
        let pReference = pIncident - refNormal * distance;
        var xA = pIncident;
        var xB = pReference;
        if (referenceIsA) { xA = pReference; xB = pIncident; }
        let mid = (xA + xB) * 0.5;
        var dup = false;
        for (var k = 0u; k < candN; k = k + 1u) {
            let d = mid - (candXA[k] + candXB[k]) * 0.5;
            if (dot(d, d) < CONTACT_MERGE_DIST_SQ) { dup = true; break; }
        }
        if (dup) { continue; }
        candXA[candN] = xA;
        candXB[candN] = xB;
        candFeat[candN] = prefix | (i & 0xffu);
        candN = candN + 1u;
    }

    if (candN == 0u) {
        let xA = supportPoint(boxA, normalAB);
        let xB = supportPoint(boxB, -normalAB);
        addContact(res, posA, quatA, posB, quatB, xA, xB, prefix);
        return;
    }

    // reduce to the spread set when the clip over-produced; each kept contact keeps its original
    // clip-ordinal feature key (stable, scan-matched — no re-ordinal; see collide.ts oracle header).
    var sel: array<u32, ${MAX_CONTACTS}>;
    var selN: u32;
    if (candN > ${MAX_CONTACTS}u) {
        selN = pruneContacts(&candXA, &candXB, candN, normalAB, posA, &sel);
    } else {
        selN = candN;
        for (var i = 0u; i < candN; i = i + 1u) { sel[i] = i; }
    }
    for (var i = 0u; i < selN; i = i + 1u) {
        let s = sel[i];
        addContact(res, posA, quatA, posB, quatB, candXA[s], candXB[s], candFeat[s]);
    }
}

fn edgeContact(res: ptr<function, SatResult>,
               boxA: Obb, boxB: Obb, posA: vec3<f32>, quatA: vec4<f32>, posB: vec3<f32>, quatB: vec4<f32>,
               axisA: u32, axisB: u32, normalAB: vec3<f32>) {
    let ea = supportEdge(boxA, axisA, normalAB);
    let eb = supportEdge(boxB, axisB, -normalAB);
    let cs = closestSegments(ea[0], ea[1], eb[0], eb[1]);
    let feature = (AXIS_EDGE << 24u) | ((axisA & 0xffu) << 8u) | (axisB & 0xffu);
    addContact(res, posA, quatA, posB, quatB, cs[0], cs[1], feature);
    if ((*res).count == 0u) {
        let sA = supportPoint(boxA, normalAB);
        let sB = supportPoint(boxB, -normalAB);
        addContact(res, posA, quatA, posB, quatB, sA, sB, feature);
    }
}

// box-box SAT. count == 0 means no overlap (separating axis found). Mirrors Manifold::collide. dRel is
// the relative displacement over the step (vA−vB)·dt — the velocity sweep (Phase 4.8.4), zero recovers the
// static 4.8.3 SAT.
fn collideBoxBox(posA: vec3<f32>, quatA: vec4<f32>, sizeA: vec3<f32>,
                 posB: vec3<f32>, quatB: vec4<f32>, sizeB: vec3<f32>, dRel: vec3<f32>) -> SatResult {
    var res: SatResult;
    res.count = 0u;
    res.basis = Mat3(vec3<f32>(0.0), vec3<f32>(0.0), vec3<f32>(0.0));

    let boxA = obbOf(posA, quatA, sizeA);
    let boxB = obbOf(posB, quatB, sizeB);
    let delta = boxB.c - boxA.c;

    var bestFace: SatAxis;
    bestFace.valid = false;
    bestFace.separation = -1e30;
    var bestEdge: SatAxis;
    bestEdge.valid = false;
    bestEdge.separation = -1e30;

    for (var i = 0u; i < 3u; i = i + 1u) {
        if (!testAxis(boxA, boxB, delta, obbAxis(boxA, i), AXIS_FACE_A, i, 0u, &bestFace, dRel)) { return res; }
    }
    for (var i = 0u; i < 3u; i = i + 1u) {
        if (!testAxis(boxA, boxB, delta, obbAxis(boxB, i), AXIS_FACE_B, 0u, i, &bestFace, dRel)) { return res; }
    }
    for (var i = 0u; i < 3u; i = i + 1u) {
        for (var j = 0u; j < 3u; j = j + 1u) {
            let axis = cross(obbAxis(boxA, i), obbAxis(boxB, j));
            if (!testAxis(boxA, boxB, delta, axis, AXIS_EDGE, i, j, &bestEdge, dRel)) { return res; }
        }
    }

    if (!bestFace.valid) { return res; }

    var best = bestFace;
    if (bestEdge.valid) {
        if (0.95 * bestEdge.separation > bestFace.separation + 0.01) { best = bestEdge; }
    }

    res.basis = orthoBasis(-best.normalAB);

    // the swept clip band along the contact axis: the same max(SPECULATIVE_DISTANCE, closing) form as the
    // testAxis abort, for the winning normal. A max(0, …) magnitude, reference-orientation safe.
    let band = max(SPECULATIVE_DISTANCE, max(0.0, dot(dRel, best.normalAB)));

    if (best.kind == AXIS_EDGE) {
        edgeContact(&res, boxA, boxB, posA, quatA, posB, quatB, best.indexA, best.indexB, best.normalAB);
    } else if (best.kind == AXIS_FACE_A) {
        faceManifold(&res, boxA, boxB, posA, quatA, posB, quatB, true, best.indexA, best.normalAB, band);
    } else {
        faceManifold(&res, boxA, boxB, posA, quatA, posB, quatB, false, best.indexB, best.normalAB, band);
    }
    return res;
}
`;

/**
 * Rounded-rounded narrowphase chunk (sphere/capsule pairs). Concatenate after HELPERS_WGSL; the rounded
 * collide pipeline (step.ts) is HELPERS_WGSL + this, so it never compiles the box or hull SAT.
 */
export const ROUNDED_WGSL = /* wgsl */ `
// Rounded-rounded narrowphase (sphere/capsule pairs, Phase 6.3) — the WGSL port of the CPU oracle
// (tests/avbd/rounded.ts collideRounded). A sphere is a zero-length segment (a point), a capsule a
// segment; one segment-segment closest-point query (closestSegments) handles every rounded pair, then
// subtract the radii into one contact. The contact has the same SatResult shape as the box SAT's, so the
// narrowphase pass + solver consume it unchanged. size is the core full-width (segment = pos +/- rotate(
// quat, size*0.5)); radius the rounding. dRel = (vA-vB)*dt is the velocity sweep; zero recovers the
// static speculative band.
fn collideRounded(posA: vec3<f32>, quatA: vec4<f32>, sizeA: vec3<f32>, radiusA: f32,
                  posB: vec3<f32>, quatB: vec4<f32>, sizeB: vec3<f32>, radiusB: f32,
                  dRel: vec3<f32>) -> SatResult {
    var res: SatResult;
    res.count = 0u;
    res.basis = Mat3(vec3<f32>(0.0), vec3<f32>(0.0), vec3<f32>(0.0));

    let hA = satQRotate(quatA, sizeA * 0.5);
    let hB = satQRotate(quatB, sizeB * 0.5);
    let cs = closestSegments(posA - hA, posA + hA, posB - hB, posB + hB);
    let cA = cs[0];
    let cB = cs[1];

    let delta = cA - cB; // core-to-core, points B -> A
    let dist = length(delta);
    var normal = vec3<f32>(0.0, 1.0, 0.0);
    if (dist > ROUND_NORMAL_EPS) { normal = delta / dist; }

    // generate while still separated by up to the speculative band (surfaces touch at gap 0): the static
    // skin, widened by the closing displacement this step (velocity sweep). max, not + (a slow body's band
    // stays the skin). The cores approach along -normal, so closing = -dot(dRel, normal).
    let closing = max(0.0, -dot(dRel, normal));
    let band = max(SPECULATIVE_DISTANCE, closing);

    let gap = dist - radiusA - radiusB;
    if (gap > band) { return res; }

    // store the CORE closest points (NOT the inflated surface) — the radius offset (the −r·normal that
    // reaches the surface) is geometric and must stay along the fixed normal, not rotate with the body's
    // spin. A sphere's contact point is not a material point; freezing the surface arm + rotating it
    // injects a spurious tangential lever into the normal Jacobian → the rotational-instability bug.
    // contactForce (step.ts) re-applies ±radius·normal at solve time. Mirrors the oracle. roadmap §6.3.
    res.basis = orthoBasis(normal);
    addContact(&res, posA, quatA, posB, quatB, cA, cB, ROUND_FEATURE);
    return res;
}
`;

// shared: closest point on an OBB to a query — the rounded × box shortcut, also pulled in by HULL_WGSL's
// rounded × polytope path (closestPointOnPoly). Folded into HELPERS_WGSL.
const _CLOSEST = /* wgsl */ `
// closest point on an OBB (box-local) to a query point — the rounded-vs-box analog of closestSegments.
// Clamping into [−half, half] gives the surface point when the query is outside; inside, the clamp is a
// no-op, so push out along the least-penetrating face. Returns the surface point, the outward normal, and
// the signed distance (+ outside, − inside). Mirrors tests/avbd/rounded.ts closestPointBox; no GJK/EPA.
struct BoxClosest { point: vec3<f32>, normal: vec3<f32>, signedDist: f32 };
fn closestPointBox(pl: vec3<f32>, half: vec3<f32>) -> BoxClosest {
    let d = clamp(pl, -half, half);
    let diff = pl - d;
    let distSq = dot(diff, diff);
    if (distSq > ROUND_NORMAL_EPS * ROUND_NORMAL_EPS) {
        let dist = sqrt(distSq);
        return BoxClosest(d, diff / dist, dist);
    }
    // inside: the nearest face is the axis with the least clearance to its slab boundary
    var axis = 0u;
    var least = half.x - abs(pl.x);
    let cy = half.y - abs(pl.y);
    let cz = half.z - abs(pl.z);
    if (cy < least) { least = cy; axis = 1u; }
    if (cz < least) { least = cz; axis = 2u; }
    var normal = vec3<f32>(0.0);
    var point = pl;
    if (axis == 0u) { let s = select(-1.0, 1.0, pl.x >= 0.0); normal.x = s; point.x = s * half.x; }
    else if (axis == 1u) { let s = select(-1.0, 1.0, pl.y >= 0.0); normal.y = s; point.y = s * half.y; }
    else { let s = select(-1.0, 1.0, pl.z >= 0.0); normal.z = s; point.z = s * half.z; }
    return BoxClosest(point, normal, -least);
}

`;

// Shared narrowphase substrate: math + structs + segment closest-point + contact append/reduce + the
// rounded × box closest-point. Concatenated FIRST in every collide pipeline (step.ts); BOXBOX_WGSL /
// ROUNDED_WGSL / HULL_WGSL append the per-shape SAT on top.
export const HELPERS_WGSL = _MATH + _SEG + _CONTACT + _CLOSEST;

/**
 * The box-box SAT chunk. Concatenate after HELPERS_WGSL; the box collide pipeline (step.ts) is
 * HELPERS_WGSL + this, so it never compiles the hull/rounded SAT.
 */
export const BOXBOX_WGSL = _BOX_A + _BOX_B + _BOX_C;

// the full box+rounded chunk (helpers + box SAT + rounded), recomposed for the gym kernel gates (`sat` /
// `pile` rounded) + the physics/core re-export — unchanged surface. The production collide pass uses the split
// chunks per pipeline. WGSL allows module-scope forward references, so the recomposed order is valid.
export const COLLIDE_WGSL = HELPERS_WGSL + BOXBOX_WGSL + ROUNDED_WGSL;

// ── Convex-hull narrowphase (Phase 6.3) ──────────────────────────────────────────────────────────────
// The WGSL port of the f64 oracle (tests/avbd/hull.ts + the rounded × polytope path in rounded.ts).
// Concatenate AFTER COLLIDE_WGSL (it reuses satQRotate/satQConj/orthoBasis/closestSegments/closestPointBox/
// addContact + the SatResult/Mat3/BoxClosest structs + the epsilons/ROUND_FEATURE there) and declare a
// `hullData: array<u32>` binding (the buffer packed by ./hull packHulls). A `Poly` is a box (generated
// inline from pos/quat/half — the box-as-boxHull the oracle feeds) OR a registered hull (read from
// hullData), so box × hull and hull × hull share ONE path, exactly like the oracle.
//
// Split into three separately-compilable chunks (the 4-way pipeline split, gpu.md "DXC shader compilation":
// compile is superlinear in kernel size). HULL_CORE_WGSL = the shared scale-unified polytope substrate
// (Convex + accessors + projectPoly + bestPolyFace); HULL_SAT_WGSL = collideHull (polytope×polytope);
// ROUNDED_POLY_WGSL = collideRoundedPolytope (rounded × polytope). step.ts builds CORE+SAT and
// CORE+ROUNDED_POLY as two pipelines, so collideHull's big SAT and the rounded segment-clip compile apart
// (the hull pipeline was the standing ~920 ms long pole); HULL_WGSL (below) recomposes all three for the gym
// kernel gates, which compile a single kernel calling both.
const _HULL_CORE = /* wgsl */ `
struct HullRef { vertBase: u32, vertCount: u32, faceBase: u32, faceCount: u32, edgeBase: u32, edgeCount: u32, faceIdxBase: u32 };
fn hullRef(id: u32) -> HullRef {
    let o = id * ${HULL_HEADER}u;
    return HullRef(hullData[o], hullData[o+1u], hullData[o+2u], hullData[o+3u], hullData[o+4u], hullData[o+5u], hullData[o+6u]);
}

// hull-direct local accessors (read raw geometry from hullData; NO isBox branch). The scale-unified poly
// accessors below build box AND hull on these — the per-access isBox branch was the collide-pass compile
// cost (gpu.md "DXC shader compilation": dead code isn't free, the box branch compiled at every site).
fn hVertL(h: HullRef, i: u32) -> vec3<f32> {
    let o = h.vertBase + i * 3u;
    return vec3<f32>(bitcast<f32>(hullData[o]), bitcast<f32>(hullData[o+1u]), bitcast<f32>(hullData[o+2u]));
}
fn hFaceNormalL(h: HullRef, f: u32) -> vec3<f32> {
    let o = h.faceBase + f * ${HULL_FACE_STRIDE}u;
    return vec3<f32>(bitcast<f32>(hullData[o]), bitcast<f32>(hullData[o+1u]), bitcast<f32>(hullData[o+2u]));
}
fn hFaceOffset(h: HullRef, f: u32) -> f32 { return bitcast<f32>(hullData[h.faceBase + f * ${HULL_FACE_STRIDE}u + 3u]); }
fn hFaceVertCount(h: HullRef, f: u32) -> u32 { return hullData[h.faceBase + f * ${HULL_FACE_STRIDE}u + 5u]; }
fn hFaceVert(h: HullRef, f: u32, j: u32) -> u32 {
    let lo = hullData[h.faceBase + f * ${HULL_FACE_STRIDE}u + 4u];
    return hullData[h.faceIdxBase + lo + j];
}
fn hEdgeL(h: HullRef, e: u32) -> vec3<f32> {
    let o = h.edgeBase + e * 3u;
    return vec3<f32>(bitcast<f32>(hullData[o]), bitcast<f32>(hullData[o+1u]), bitcast<f32>(hullData[o+2u]));
}

// A polytope: a registered hull (hr) at pos/quat, scaled by the scale vec. A box is the built-in UNIT_CUBE
// hull scaled by its half-extents — so box and hull collide through ONE branch-free accessor path (the
// legacy narrowphase's scale-unified shape). isBox survives ONLY to pick the sphere's closest-point
// shortcut (closestPointBox vs the general closestPointOnHull); the hot SAT/clip accessors never read it.
struct Convex { isBox: u32, pos: vec3<f32>, quat: vec4<f32>, scale: vec3<f32>, hr: HullRef };
fn boxPoly(pos: vec3<f32>, quat: vec4<f32>, size: vec3<f32>) -> Convex {
    return Convex(1u, pos, quat, size * 0.5, hullRef(${UNIT_CUBE_ID}u));
}
fn hullPoly(pos: vec3<f32>, quat: vec4<f32>, id: u32) -> Convex {
    return Convex(0u, pos, quat, vec3<f32>(1.0), hullRef(id));
}
fn polyMake(shape: u32, pos: vec3<f32>, quat: vec4<f32>, size: vec3<f32>, hullId: u32) -> Convex {
    if (shape == 3u) { return hullPoly(pos, quat, hullId); }
    return boxPoly(pos, quat, size);
}

// scale-unified local accessors. The scale vec shapes the unit geometry (1 for a real hull, half-extents
// for a box). A vertex/edge scales directly; a face normal transforms by 1/scale + renormalize (the
// plane's inverse-transpose); the offset is recovered as dot(normal, a scaled vertex on the face) since
// scale moves the plane. For a real hull (scale = 1) these reduce to the raw reads.
fn polyVertCount(p: Convex) -> u32 { return p.hr.vertCount; }
fn polyVertLocal(p: Convex, i: u32) -> vec3<f32> { return hVertL(p.hr, i) * p.scale; }
fn polyFaceCount(p: Convex) -> u32 { return p.hr.faceCount; }
fn polyFaceNormalLocal(p: Convex, f: u32) -> vec3<f32> { return normalize(hFaceNormalL(p.hr, f) / p.scale); }
fn polyFaceVertCount(p: Convex, f: u32) -> u32 { return hFaceVertCount(p.hr, f); }
fn polyFaceVert(p: Convex, f: u32, j: u32) -> u32 { return hFaceVert(p.hr, f, j); }
fn polyEdgeCount(p: Convex) -> u32 { return p.hr.edgeCount; }
fn polyEdgeLocal(p: Convex, e: u32) -> vec3<f32> { return hEdgeL(p.hr, e) * p.scale; }
fn polyVertW(p: Convex, i: u32) -> vec3<f32> { return p.pos + satQRotate(p.quat, polyVertLocal(p, i)); }
fn polyFaceNormalW(p: Convex, f: u32) -> vec3<f32> { return satQRotate(p.quat, polyFaceNormalLocal(p, f)); }
fn polyEdgeW(p: Convex, e: u32) -> vec3<f32> { return satQRotate(p.quat, polyEdgeLocal(p, e)); }

fn projectPoly(p: Convex, axis: vec3<f32>) -> vec2<f32> { // [min, max]
    // dot(pos + R·v, axis) = dot(pos, axis) + dot(v, Rᵀ·axis): rotate the axis into the convex's local frame
    // ONCE, then dot with local verts — no per-vertex quat rotation in the hot SAT loop (a math identity,
    // and far less inlined code per axis test → big collideHull compile + runtime win; gpu.md hot loops).
    let axisL = satQRotate(satQConj(p.quat), axis);
    let base = dot(p.pos, axis);
    var mn = 1e30; var mx = -1e30;
    let n = polyVertCount(p);
    for (var i = 0u; i < n; i = i + 1u) {
        let d = base + dot(polyVertLocal(p, i), axisL);
        mn = min(mn, d); mx = max(mx, d);
    }
    return vec2<f32>(mn, mx);
}
fn bestPolyFace(p: Convex, outward: vec3<f32>) -> u32 {
    var idx = 0u; var best = -1e30;
    let n = polyFaceCount(p);
    for (var f = 0u; f < n; f = f + 1u) {
        let d = dot(polyFaceNormalW(p, f), outward);
        if (d > best) { best = d; idx = f; }
    }
    return idx;
}
`;

// collideHull — the polytope×polytope SAT (box×hull, hull×hull), compiled as its OWN pipeline (step.ts),
// apart from the rounded × polytope path below. Concatenate after HELPERS_WGSL + HULL_CORE_WGSL.
const _HULL_SAT = /* wgsl */ `
// Function-private working sets stay small (gpu.md "keep function-private working sets small" — large
// dynamically-indexed private arrays blow up DXC register allocation + risk the Metal multi-lane miscompile
// the box SAT hit). The intermediate Sutherland-Hodgman polygon can reach ~12 verts (an 8-gon cone face
// clipped by a box); the post-band candidate set the reduce sees is ≤ MAX_CANDIDATES (the box path's 8 —
// measured max 8, the full cone octagon), so the candidate + reduce arrays REUSE the box path's
// MAX_CANDIDATES + pruneContacts rather than a wider duplicate.
const MAX_HULL_CLIP: u32 = 12u;   // the Sutherland-Hodgman ping-pong polygon (incident verts + ref side planes)
const HULL_FACE_TAG: u32 = 0x40u; // feature high-byte tags (distinct from the box AXIS_FACE/EDGE + ROUND)
const HULL_EDGE_TAG: u32 = 0x41u;

fn supportPoly(p: Convex, dir: vec3<f32>) -> vec3<f32> {
    var best = vec3<f32>(0.0); var bestD = -1e30;
    let n = polyVertCount(p);
    for (var i = 0u; i < n; i = i + 1u) {
        let w = polyVertW(p, i); let d = dot(w, dir);
        if (d > bestD) { bestD = d; best = w; }
    }
    return best;
}

// the winning SAT axis: a face axis (fromA) or an edge×edge axis (isEdge), oriented A → B, kept at max
// separation (the MTV) — the hull analog of collide.ts's SatAxis search.
struct HullAxis { isEdge: bool, fromA: bool, sep: f32, normalAB: vec3<f32>, valid: bool };
fn testPolyAxis(a: Convex, b: Convex, delta: vec3<f32>, axis: vec3<f32>, isEdge: bool, fromA: bool,
                best: ptr<function, HullAxis>, dRel: vec3<f32>) -> bool {
    let lenSq = dot(axis, axis);
    if (lenSq < SAT_AXIS_EPSILON) { return true; }
    var n = axis * (1.0 / sqrt(lenSq));
    if (dot(n, delta) < 0.0) { n = -n; }
    let pa = projectPoly(a, n); let pb = projectPoly(b, n);
    let separation = -min(pa.y - pb.x, pb.y - pa.x); // overlap of the intervals, negated
    // unilateral speculative/swept band (Phase 4.8.3/4.8.4): abort only past max(static skin, closing)
    if (separation > max(SPECULATIVE_DISTANCE, max(0.0, dot(dRel, n)))) { return false; }
    if (!(*best).valid || separation > (*best).sep) {
        (*best).valid = true; (*best).isEdge = isEdge; (*best).fromA = fromA;
        (*best).sep = separation; (*best).normalAB = n;
    }
    return true;
}

// Sutherland-Hodgman: clip inN verts of inV to the back of (dot(nrm,x) <= d) → outV; returns out count.
fn clipPolyPlane(inV: ptr<function, array<vec3<f32>, MAX_HULL_CLIP>>, inN: u32, nrm: vec3<f32>, d: f32,
                 outV: ptr<function, array<vec3<f32>, MAX_HULL_CLIP>>) -> u32 {
    var outN = 0u;
    if (inN == 0u) { return 0u; }
    var a = (*inV)[inN - 1u];
    var da = dot(nrm, a) - d;
    for (var i = 0u; i < inN; i = i + 1u) {
        let b = (*inV)[i];
        let db = dot(nrm, b) - d;
        let aIn = da <= PLANE_EPSILON;
        let bIn = db <= PLANE_EPSILON;
        if (aIn != bIn) {
            var t = 0.0; let denom = da - db;
            if (abs(denom) > SAT_AXIS_EPSILON) { t = clamp(da / denom, 0.0, 1.0); }
            if (outN < MAX_HULL_CLIP) { (*outV)[outN] = a + (b - a) * t; outN = outN + 1u; }
        }
        if (bIn && outN < MAX_HULL_CLIP) { (*outV)[outN] = b; outN = outN + 1u; }
        a = b; da = db;
    }
    return outN;
}

// convex-hull SAT (box × hull / hull × hull). count == 0 = no overlap past the band. A box fed as a Convex
// reproduces collideBoxBox's face manifold (the box-as-hull gate). Mirrors tests/avbd/hull.ts collideHull.
fn collideHull(a: Convex, b: Convex, dRel: vec3<f32>) -> SatResult {
    var res: SatResult; res.count = 0u;
    res.basis = Mat3(vec3<f32>(0.0), vec3<f32>(0.0), vec3<f32>(0.0));
    let delta = b.pos - a.pos;

    var bestFaceAxis: HullAxis; bestFaceAxis.valid = false; bestFaceAxis.sep = -1e30;
    var bestEdgeAxis: HullAxis; bestEdgeAxis.valid = false; bestEdgeAxis.sep = -1e30;

    let fa = polyFaceCount(a); let fb = polyFaceCount(b);
    for (var i = 0u; i < fa; i = i + 1u) {
        if (!testPolyAxis(a, b, delta, polyFaceNormalW(a, i), false, true, &bestFaceAxis, dRel)) { return res; }
    }
    for (var i = 0u; i < fb; i = i + 1u) {
        if (!testPolyAxis(a, b, delta, polyFaceNormalW(b, i), false, false, &bestFaceAxis, dRel)) { return res; }
    }
    let ea = polyEdgeCount(a); let eb = polyEdgeCount(b);
    for (var i = 0u; i < ea; i = i + 1u) {
        let eai = polyEdgeW(a, i);
        for (var j = 0u; j < eb; j = j + 1u) {
            let axis = cross(eai, polyEdgeW(b, j));
            if (!testPolyAxis(a, b, delta, axis, true, false, &bestEdgeAxis, dRel)) { return res; }
        }
    }
    if (!bestFaceAxis.valid) { return res; }

    var best = bestFaceAxis;
    if (bestEdgeAxis.valid && 0.95 * bestEdgeAxis.sep > bestFaceAxis.sep + 0.01) { best = bestEdgeAxis; }

    let normalBA = -best.normalAB;
    res.basis = orthoBasis(normalBA);
    let band = max(SPECULATIVE_DISTANCE, max(0.0, dot(dRel, best.normalAB)));

    // reference = the winning face's hull (an edge axis defaults to A, like the oracle)
    let referenceIsA = best.isEdge || best.fromA;
    var refP = b; var incP = a; var refOutward = normalBA;
    if (referenceIsA) { refP = a; incP = b; refOutward = best.normalAB; }

    let refFaceIdx = bestPolyFace(refP, refOutward);
    let refNormalW = polyFaceNormalW(refP, refFaceIdx);
    let refV0 = polyVertW(refP, polyFaceVert(refP, refFaceIdx, 0u));
    let refPlaneW = dot(refNormalW, refV0);

    // incident face selected by the CONTACT normal (−refOutward), not the reference face normal — they
    // diverge on an edge MTV; the contact-normal choice keeps a slanted edge-MTV clip non-empty (Bullet).
    let incFaceIdx = bestPolyFace(incP, -refOutward);
    var polyA: array<vec3<f32>, MAX_HULL_CLIP>;
    var polyB: array<vec3<f32>, MAX_HULL_CLIP>;
    let incN0 = polyFaceVertCount(incP, incFaceIdx);
    var n = min(incN0, MAX_HULL_CLIP);
    for (var j = 0u; j < n; j = j + 1u) {
        polyA[j] = polyVertW(incP, polyFaceVert(incP, incFaceIdx, j));
    }

    // clip against the reference face's side planes (inward), ping-ponging polyA <-> polyB
    let refLoopN = polyFaceVertCount(refP, refFaceIdx);
    var src = 0u; // 0 = polyA holds the live polygon, 1 = polyB holds it
    for (var e = 0u; e < refLoopN && n > 0u; e = e + 1u) {
        let va = polyVertW(refP, polyFaceVert(refP, refFaceIdx, e));
        let vb = polyVertW(refP, polyFaceVert(refP, refFaceIdx, (e + 1u) % refLoopN));
        let planeN = -cross(va - vb, refNormalW);
        let pd = dot(planeN, va);
        if (src == 0u) { n = clipPolyPlane(&polyA, n, planeN, pd, &polyB); src = 1u; }
        else { n = clipPolyPlane(&polyB, n, planeN, pd, &polyA); src = 0u; }
    }
    if (n == 0u) { return res; }

    let featurePrefix = (select(HULL_FACE_TAG, HULL_EDGE_TAG, best.isEdge) << 24u)
        | ((refFaceIdx & 0xffu) << 16u) | ((incFaceIdx & 0xffu) << 8u);
    // the within-band candidate set the reduce sees is <= MAX_CANDIDATES (8 — the measured max, a full cone
    // octagon), so it shares the box path's array bound + pruneContacts; the dedup midpoint is recomputed
    // from the kept candidates, no separate midpoint array (gpu.md small function-private sets).
    var candXA: array<vec3<f32>, MAX_CANDIDATES>;
    var candXB: array<vec3<f32>, MAX_CANDIDATES>;
    var candFeat: array<u32, MAX_CANDIDATES>;
    var candN = 0u;
    for (var i = 0u; i < n && candN < MAX_CANDIDATES; i = i + 1u) {
        let pInc = select(polyB[i], polyA[i], src == 0u);
        let depth = dot(refNormalW, pInc) - refPlaneW;
        if (depth > band) { continue; }
        let pRef = pInc - refNormalW * depth;
        var xA = pRef; var xB = pInc;
        if (!referenceIsA) { xA = pInc; xB = pRef; }
        let mid = (xA + xB) * 0.5;
        var dup = false;
        for (var k = 0u; k < MAX_CANDIDATES; k = k + 1u) {
            if (k >= candN) { break; }
            let dd = mid - (candXA[k] + candXB[k]) * 0.5;
            if (dot(dd, dd) < CONTACT_MERGE_DIST_SQ) { dup = true; break; }
        }
        if (dup) { continue; }
        candXA[candN] = xA; candXB[candN] = xB; candFeat[candN] = featurePrefix | (i & 0xffu);
        candN = candN + 1u;
    }

    // no candidate survived (a grazing edge MTV) — fall back to a single support-point contact
    if (candN == 0u) {
        let xA = supportPoly(a, best.normalAB);
        let xB = supportPoly(b, -best.normalAB);
        addContact(&res, a.pos, a.quat, b.pos, b.quat, xA, xB, featurePrefix);
        return res;
    }

    var sel: array<u32, ${MAX_CONTACTS}>; var selN: u32;
    if (candN > ${MAX_CONTACTS}u) { selN = pruneContacts(&candXA, &candXB, candN, best.normalAB, a.pos, &sel); }
    else { selN = candN; for (var i = 0u; i < candN; i = i + 1u) { sel[i] = i; } }
    for (var i = 0u; i < selN; i = i + 1u) {
        let s = sel[i];
        addContact(&res, a.pos, a.quat, b.pos, b.quat, candXA[s], candXB[s], candFeat[s]);
    }
    return res;
}
`;

// collideRoundedPolytope — sphere/capsule × box/hull, compiled as its OWN pipeline (step.ts), apart from
// collideHull above (the segment-clip + closest-point work is the other half of the old monolithic kernel).
// Concatenate after HELPERS_WGSL + HULL_CORE_WGSL.
const _ROUNDED_POLY = /* wgsl */ `
// closest point on a hull to a query in the hull's LOCAL frame — the general rounded × hull primitive
// (face region → edge → vertex outside; least-penetrating face inside). Mirrors hull.ts closestPointOnHull.
fn pointInFaceLocal(h: HullRef, f: u32, pt: vec3<f32>) -> bool {
    let nf = hFaceNormalL(h, f);
    let cnt = hFaceVertCount(h, f);
    for (var i = 0u; i < cnt; i = i + 1u) {
        let va = hVertL(h, hFaceVert(h, f, i));
        let vb = hVertL(h, hFaceVert(h, f, (i + 1u) % cnt));
        if (dot(cross(vb - va, pt - va), nf) < -PLANE_EPSILON) { return false; }
    }
    return true;
}
fn closestOnSeg(pt: vec3<f32>, a: vec3<f32>, b: vec3<f32>) -> vec3<f32> {
    let ab = b - a; let len2 = dot(ab, ab);
    if (len2 < SAT_AXIS_EPSILON) { return a; }
    return a + ab * clamp(dot(pt - a, ab) / len2, 0.0, 1.0);
}
// closest point on a hull to a query in the hull's LOCAL frame (q already de-rotated by the caller). Only
// called on hull Convexes (a box uses closestPointBox), so it reads hull-direct (branch-free).
fn closestPointOnHull(h: HullRef, q: vec3<f32>) -> BoxClosest {
    var maxD = -1e30; var maxFace = 0u;
    let fn0 = h.faceCount;
    for (var f = 0u; f < fn0; f = f + 1u) {
        let d = dot(hFaceNormalL(h, f), q) - hFaceOffset(h, f);
        if (d > maxD) { maxD = d; maxFace = f; }
    }
    if (maxD <= 0.0) { // inside — push out along the least-penetrating face
        let nf = hFaceNormalL(h, maxFace);
        return BoxClosest(q - nf * maxD, nf, maxD);
    }
    var bestDist = 1e30; var bestPoint = q;
    for (var f = 0u; f < fn0; f = f + 1u) {
        let d = dot(hFaceNormalL(h, f), q) - hFaceOffset(h, f);
        if (d <= 0.0) { continue; }
        let proj = q - hFaceNormalL(h, f) * d;
        if (pointInFaceLocal(h, f, proj) && d < bestDist) { bestDist = d; bestPoint = proj; }
    }
    for (var f = 0u; f < fn0; f = f + 1u) {
        let cnt = hFaceVertCount(h, f);
        for (var i = 0u; i < cnt; i = i + 1u) {
            let cp = closestOnSeg(q, hVertL(h, hFaceVert(h, f, i)), hVertL(h, hFaceVert(h, f, (i + 1u) % cnt)));
            let dd = length(q - cp);
            if (dd < bestDist) { bestDist = dd; bestPoint = cp; }
        }
    }
    for (var i = 0u; i < h.vertCount; i = i + 1u) {
        let v = hVertL(h, i); let dd = length(q - v);
        if (dd < bestDist) { bestDist = dd; bestPoint = v; }
    }
    let diff = q - bestPoint; let dist = length(diff);
    var normal = hFaceNormalL(h, maxFace);
    if (dist > ROUND_NORMAL_EPS) { normal = diff / dist; }
    return BoxClosest(bestPoint, normal, dist);
}
fn closestPointOnPoly(p: Convex, q: vec3<f32>) -> BoxClosest {
    if (p.isBox == 1u) { return closestPointBox(q, p.scale); } // box: scale = half-extents; the exact clamp shortcut
    return closestPointOnHull(p.hr, q);
}

// capsule (core segment [e0,e1] world, radius) × polytope — the §6.3 segment-clip manifold (up to 2
// contacts sharing the reference-face normal), the mid-segment case endpoint sampling misses. The contact
// normal is a mini-SAT between the core segment and the hull (robust to an overhanging capsule), NOT the
// endpoints' closest normals. Mirrors hull.ts capsuleHull. dRel = (vCap − vPoly)·dt.
struct CapHits {
    count: u32,
    normal: vec3<f32>,            // hull → capsule (world, outward)
    core: array<vec3<f32>, 2>,   // capsule core point (world)
    surf: array<vec3<f32>, 2>,   // hull surface point (world)
    ord: array<u32, 2>,
};
fn considerCapAxis(raw: vec3<f32>, e0: vec3<f32>, e1: vec3<f32>, toCap: vec3<f32>, p: Convex,
                   bestSep: ptr<function, f32>, bestN: ptr<function, vec3<f32>>) {
    let l2 = dot(raw, raw);
    if (l2 < SAT_AXIS_EPSILON) { return; }
    var ax = raw * (1.0 / sqrt(l2));
    if (dot(ax, toCap) < 0.0) { ax = -ax; } // orient hull → capsule
    let a = dot(e0, ax); let b = dot(e1, ax);
    let sMin = min(a, b); let sMax = max(a, b);
    let hp = projectPoly(p, ax);
    let sep = -min(sMax - hp.x, hp.y - sMin);
    if (sep > *bestSep) { *bestSep = sep; *bestN = ax; }
}
fn capsulePoly(e0: vec3<f32>, e1: vec3<f32>, radius: f32, p: Convex, dRel: vec3<f32>) -> CapHits {
    var out: CapHits; out.count = 0u; out.normal = vec3<f32>(0.0, 1.0, 0.0);
    let dir = e1 - e0;
    let toCap = (e0 + e1) * 0.5 - p.pos;

    var bestSep = -1e30; var n = vec3<f32>(0.0, 1.0, 0.0);
    let fn0 = polyFaceCount(p);
    for (var f = 0u; f < fn0; f = f + 1u) {
        considerCapAxis(polyFaceNormalW(p, f), e0, e1, toCap, p, &bestSep, &n);
    }
    let en0 = polyEdgeCount(p);
    for (var e = 0u; e < en0; e = e + 1u) {
        considerCapAxis(cross(dir, polyEdgeW(p, e)), e0, e1, toCap, p, &bestSep, &n);
    }

    if (bestSep - radius > max(SPECULATIVE_DISTANCE, max(0.0, -dot(dRel, n)))) { out.normal = n; return out; }

    let refIdx = bestPolyFace(p, n);
    let nf = polyFaceNormalW(p, refIdx);
    let refCenter = polyVertW(p, polyFaceVert(p, refIdx, 0u));
    let bandF = max(SPECULATIVE_DISTANCE, max(0.0, -dot(dRel, nf)));

    // clip the segment's parameter range to the reference face's side-plane prism (extruded along nf)
    var t0 = 0.0; var t1 = 1.0; var outside = false;
    let cnt = polyFaceVertCount(p, refIdx);
    for (var i = 0u; i < cnt; i = i + 1u) {
        let va = polyVertW(p, polyFaceVert(p, refIdx, i));
        let vb = polyVertW(p, polyFaceVert(p, refIdx, (i + 1u) % cnt));
        let planeN = -cross(va - vb, nf); // inward
        let d0 = dot(planeN, e0) - dot(planeN, va);
        let dd = dot(planeN, dir);
        if (abs(dd) < SAT_AXIS_EPSILON) {
            if (d0 > PLANE_EPSILON) { outside = true; break; }
            continue;
        }
        let tc = -d0 / dd;
        if (dd > 0.0) { t1 = min(t1, tc); } else { t0 = max(t0, tc); }
    }

    if (!outside && t1 > t0 + 1e-9) {
        for (var k = 0u; k < 2u; k = k + 1u) {
            let tt = select(t1, t0, k == 0u);
            let sp = e0 + dir * tt;
            let distPlane = dot(sp - refCenter, nf);
            if (distPlane - radius > bandF) { continue; }
            out.core[out.count] = sp; out.surf[out.count] = sp - nf * distPlane; out.ord[out.count] = k;
            out.count = out.count + 1u;
        }
        if (out.count > 0u) { out.normal = nf; return out; }
    }

    // fallback (segment off the reference face — an edge/vertex contact): per-endpoint closest points,
    // the reference-face normal shared
    out.normal = nf;
    let qc = satQConj(p.quat);
    for (var k = 0u; k < 2u; k = k + 1u) {
        let e = select(e1, e0, k == 0u);
        let r = closestPointOnPoly(p, satQRotate(qc, e - p.pos));
        if (r.signedDist - radius > bandF) { continue; }
        out.core[out.count] = e; out.surf[out.count] = p.pos + satQRotate(p.quat, r.point); out.ord[out.count] = k;
        out.count = out.count + 1u;
    }
    return out;
}

// rounded (sphere/capsule) × polytope with the round shape as A. Sphere = closest point on the polytope
// (1 contact); capsule = the segment-clip (up to 2). The stored arm anchors the round CORE + polytope
// SURFACE; the radius offset is re-applied along the fixed normal at solve time. Mirrors the oracle
// collideRoundedPolytope. The collide pass orients box/hull-as-A by swapping (collideRoundedPolytope).
fn collideRoundedPolyA(rPos: vec3<f32>, rQuat: vec4<f32>, rSize: vec3<f32>, rRad: f32, rShape: u32,
                       p: Convex, dRel: vec3<f32>) -> SatResult {
    var res: SatResult; res.count = 0u;
    res.basis = Mat3(vec3<f32>(0.0), vec3<f32>(0.0), vec3<f32>(0.0));
    if (rShape == 1u) { // sphere — sample its centre
        let qc = satQConj(p.quat);
        let cp = closestPointOnPoly(p, satQRotate(qc, rPos - p.pos));
        let normal = satQRotate(p.quat, cp.normal); // polytope → sphere (B → A)
        let band = max(SPECULATIVE_DISTANCE, max(0.0, -dot(dRel, normal)));
        if (cp.signedDist - rRad > band) { return res; }
        let surf = p.pos + satQRotate(p.quat, cp.point);
        res.basis = orthoBasis(normal);
        addContact(&res, rPos, rQuat, p.pos, p.quat, rPos, surf, ROUND_FEATURE); // sphere core = centre
        return res;
    }
    let hh = satQRotate(rQuat, rSize * 0.5);
    let hits = capsulePoly(rPos - hh, rPos + hh, rRad, p, dRel);
    if (hits.count == 0u) { return res; }
    res.basis = orthoBasis(hits.normal);
    for (var k = 0u; k < hits.count; k = k + 1u) {
        addContact(&res, rPos, rQuat, p.pos, p.quat, hits.core[k], hits.surf[k], ROUND_FEATURE | hits.ord[k]);
    }
    return res;
}

// mixed rounded × polytope, A/B oriented like collideBoxBox (A = ia, B = ib). If A is the round shape,
// collide directly; else (the polytope is A) collide round-as-A on the flipped sweep, then swap the arms +
// flip the normal back into our A/B convention. Mirrors the oracle narrowphase round/poly branches.
fn collideRoundedPolytope(aPos: vec3<f32>, aQuat: vec4<f32>, aSize: vec3<f32>, aRad: f32, aShape: u32, aHull: u32,
                          bPos: vec3<f32>, bQuat: vec4<f32>, bSize: vec3<f32>, bRad: f32, bShape: u32, bHull: u32,
                          dRel: vec3<f32>) -> SatResult {
    let roundedA = (aShape == 1u || aShape == 2u);
    if (roundedA) {
        let poly = polyMake(bShape, bPos, bQuat, bSize, bHull);
        return collideRoundedPolyA(aPos, aQuat, aSize, aRad, aShape, poly, dRel);
    }
    let poly = polyMake(aShape, aPos, aQuat, aSize, aHull);
    let r = collideRoundedPolyA(bPos, bQuat, bSize, bRad, bShape, poly, -dRel);
    var res: SatResult; res.count = r.count;
    res.basis = orthoBasis(-r.basis.r0); // B → A in our convention = −(poly → round)
    for (var k = 0u; k < r.count; k = k + 1u) {
        res.feat[k] = r.feat[k];
        res.rA[k] = r.rB[k]; // our A = poly = r's B
        res.rB[k] = r.rA[k];
    }
    return res;
}
`;

/** the shared scale-unified polytope substrate (Convex + hull accessors + projectPoly + bestPolyFace). Both
 *  the hull SAT pipeline and the rounded × polytope pipeline (step.ts) prepend HELPERS_WGSL + this. */
export const HULL_CORE_WGSL = _HULL_CORE;
/** collideHull — the polytope×polytope SAT (box×hull, hull×hull). Concatenate after HELPERS_WGSL + HULL_CORE_WGSL. */
export const HULL_SAT_WGSL = _HULL_SAT;
/** collideRoundedPolytope — sphere/capsule × box/hull. Concatenate after HELPERS_WGSL + HULL_CORE_WGSL. */
export const ROUNDED_POLY_WGSL = _ROUNDED_POLY;
/**
 * the full convex-hull narrowphase recomposed (HULL_CORE + collideHull + collideRoundedPolytope in ONE
 * chunk) — for the gym kernel gates (`sat` / `pile` rounded) that compile a single kernel calling both. The
 * production collide pass (step.ts) compiles CORE+SAT and CORE+ROUNDED_POLY as SEPARATE pipelines instead.
 */
export const HULL_WGSL = _HULL_CORE + _HULL_SAT + _ROUNDED_POLY;
