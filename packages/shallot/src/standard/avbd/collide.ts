// Box-box (OBB) SAT with Sutherland-Hodgman manifold clipping — the TGSL port of the CPU
// oracle (tests/avbd/collide.ts), which is itself a faithful port of
// reference/avbd-demo3d/source/collide.cpp. 15 candidate axes (3 face-A, 3 face-B, 9
// edge-edge), edge/face bias 0.95/0.01, persistent feature keys for warmstart. The
// narrowphase pass calls `collideBoxBox` per overlapping pair and writes the resulting
// contacts into the source-agnostic constraint buffer.
//
// This is the SAT crux (avbd.md "Reference-fidelity"): the feature keys it emits feed warmstart, so
// they must be bit-identical to the C++. The CPU SAT is validated against the gold vectors in
// tests/avbd/sat.oracle.ts; the gym `sat` kernel gate runs this WGSL against that oracle on the real
// GPU, and the gym `pile` scenario is the full-pipeline home.
//
// The narrowphase is SIX chunks, each a lazily-resolved thunk so a pipeline compiles only the SAT it
// dispatches (DXC compile is superlinear in kernel size; gpu.md "DXC shader compilation"):
// `helpersWgsl` (math + structs + the shared manifold primitives) is prepended by every collide
// pipeline; `boxBoxWgsl` / `roundedWgsl` / `hullCoreWgsl` + `hullSatWgsl` / `hullCoreWgsl` +
// `roundedPolyWgsl` add one shape-pair class each. `collideWgsl` / `hullWgsl` recompose them for the
// gym kernel gates. Every dependent chunk forces its base chunks first (the shared `spliceNs`
// emits a shared definition into whichever chunk resolves first).
//
// The chunks emit **no WGSL consts** — every constant folds to a literal — so a raw splice site that
// needs one interpolates the TS export ({@link SPECULATIVE_DISTANCE}) rather than naming a spliced
// const. The hull geometry readers are the one WGSL-bodied layer: they read `hullData`, a module-scope
// global the *consumer* declares by name (the relocatable contract, sear's shadow-sampler shape), so a
// consumer of a hull chunk declares `var<storage, read> hullData: array<u32>`.
//
// Every SAT is CPU-callable, pointers and all: a local passed to a `ptr<function, …>`
// param goes through `d.ref(x)`, which types it, emits the same `(&x)`, and gives the CPU arm the
// reference semantics `.$` needs. So `collideBoxBox` runs against the C++ gold vectors in
// tests/avbd/sat.oracle.ts, `collideRounded` against rounded.oracle.ts, and `collideHull` against the f64
// hull oracle. The seven hull-data readers are duals: WGSL reads the consumer-declared storage binding,
// while the scoped CPU arm reads the exact packed `Uint32Array`. Neither CPU arm can see f32
// reassociation (it runs on f64 JS numbers), so the emitted-WGSL differential stays the guard for op order.

import tgpu from "typegpu";
import * as d from "typegpu/data";
import * as std from "typegpu/std";
import { chunk, spliceNs } from "../../engine/utils/core";
import { UNIT_CUBE_ID } from "../physics/core";
import { HULL_FACE_STRIDE, HULL_HEADER } from "./hull";

/** output manifold cap: the reduced spread set (= CONTACTS_PER_PAIR); was 8, halved by 4.8.1 */
export const MAX_CONTACTS = 4;

/**
 * speculative contact band (Phase 4.8.3): the SAT emits a contact while the boxes are separated by up to
 * this gap, carrying the true signed gap in c0 so the repulsion-only normal constraint limits the approach:
 * a body within the band at frame start lands at contact, no penetration pop, no tunnel (Firth 2011 /
 * Box2D's speculative solver). Derived as Box2D's `b2_speculativeDistance = 4 · b2_linearSlop`:
 * COLLISION_MARGIN (0.01) is our slop/skin, so 4× = 0.04. Kept DISTINCT from COLLISION_MARGIN (the
 * equilibrium offset, not a generation band). step.ts pads the broadphase AABB + sphere test by this
 * (interpolating this export — the chunks emit no consts); the f64 oracle + C++ harness carry the same
 * value, so GPU == oracle == C++.
 *
 * Phase 4.8.4 (velocity sweep) extends this STATIC band per SAT axis to max(this, closing displacement)
 * via `collideBoxBox(…, dRel)` (dRel = (vA−vB)·dt), so a fast mover crossing the whole contact in one step
 * (`v·dt` ≫ the band) is caught at frame start too. step.ts also sweeps the broadphase AABB by `|vel|·dt`
 * (matching webphysics's velocity-fattened tree) and the sphere test by `|vRel|·dt`.
 */
export const SPECULATIVE_DISTANCE = 0.04;

const SAT_AXIS_EPSILON = 1e-6;
const PLANE_EPSILON = 1e-5;
const CONTACT_MERGE_DIST_SQ = 1e-6;
const REDUCE_MIN_DIST_SQ = 1e-6;
// Conservative f32 comparison band for the symmetric edge/face and manifold-reduction scores. This is a
// selection policy, not an output tolerance: a face wins the former tie and the persistent feature key
// orders the latter, making equivalent extrema stable across the f32 shader and f64 oracle.
const F32_TIE_REL = 16 * 2 ** -23;
const AXIS_FACE_A = 0;
const AXIS_FACE_B = 1;
const AXIS_EDGE = 2;
// a quad clipped by 4 reference-face half-planes gains at most one vertex per plane (4→8), the exact
// candidate bound before the 4.8.1 reduction. Also load-bearing on Apple/Metal-3: a larger candidate
// footprint pushes collideBoxBox's peak function-private footprint past the register-spill threshold,
// where Metal miscompiles the per-lane offset of the spilled SatResult under multi-lane execution and
// every face manifold collapses to count=1 (gpu.md "WebGPU-specific traps"). Don't regrow it.
const MAX_CANDIDATES = 8;
const MAX_POLY_VERTS = 8;
// the single rounded (sphere/capsule) contact's fixed feature key (kind 3, distinct from the box
// AXIS_FACE/EDGE keys) — one contact per pair, so a stable key means warmstart always matches. Mirrors
// tests/avbd/rounded.ts ROUND_FEATURE.
const ROUND_FEATURE = 0x03000000;
// below this core-to-core distance the rounded contact normal is undefined (concentric cores) — fall
// back to world up. A degenerate deep-overlap guard.
const ROUND_NORMAL_EPS = 1e-9;
// the Sutherland-Hodgman ping-pong polygon (incident verts + ref side planes) can reach ~12 verts (an
// 8-gon cone face clipped by a box)
const MAX_HULL_CLIP = 12;
// feature high-byte tags (distinct from the box AXIS_FACE/EDGE + ROUND)
const HULL_FACE_TAG = 0x40;
const HULL_EDGE_TAG = 0x41;
// the "no axis kept yet" separation sentinel, below any real separation
const NO_AXIS = -1e30;

const Vec3x2 = d.arrayOf(d.vec3f, 2);
const CandVerts = d.arrayOf(d.vec3f, MAX_CANDIDATES);
const CandFeats = d.arrayOf(d.u32, MAX_CANDIDATES);
const CandScalars = d.arrayOf(d.f32, MAX_CANDIDATES);
const SelIdx = d.arrayOf(d.u32, MAX_CONTACTS);
const ClipVerts = d.arrayOf(d.vec3f, MAX_HULL_CLIP);
const PolyVerts = d.arrayOf(d.vec3f, MAX_POLY_VERTS);

/** row-major 3×3, matching the oracle (tests/avbd/math.ts) */
const Mat3 = d.struct({ r0: d.vec3f, r1: d.vec3f, r2: d.vec3f }).$name("Mat3");

/** up to {@link MAX_CONTACTS} reduced contact points + the contact basis (normal in row 0, B→A;
 *  tangents rows 1-2), with each point's local arms and its warmstart feature key.
 *  @internal exported for the step.ts collide-wrapper kernels (3b-iii), which call the SAT fns
 *  directly rather than splicing their WGSL text — its shape is the wrapper's warmstart-merge input. */
export const SatResult = d
    .struct({
        count: d.u32,
        basis: Mat3,
        feat: d.arrayOf(d.u32, MAX_CONTACTS),
        rA: d.arrayOf(d.vec3f, MAX_CONTACTS),
        rB: d.arrayOf(d.vec3f, MAX_CONTACTS),
    })
    .$name("SatResult");

// an empty manifold with a zero basis — the `var res: SatResult;` + `res.count = 0u;` +
// `res.basis = Mat3(0)` preamble every entry point opens with, as one call (WGSL zero-inits a
// function `var`, so the reference's explicit zeroing was already redundant)
const satZero = tgpu
    .fn(
        [],
        SatResult,
    )(() => {
        "use gpu";
        return SatResult({
            count: 0,
            basis: Mat3({ r0: d.vec3f(), r1: d.vec3f(), r2: d.vec3f() }),
            feat: d.arrayOf(d.u32, MAX_CONTACTS)(),
            rA: d.arrayOf(d.vec3f, MAX_CONTACTS)(),
            rB: d.arrayOf(d.vec3f, MAX_CONTACTS)(),
        });
    })
    .$name("satZero");

/** quaternion conjugate — the inverse of a unit quat.
 *  @example const inv = satQConj(vec4f(0, 0, 0, 1)); */
const satQConj = tgpu
    .fn(
        [d.vec4f],
        d.vec4f,
    )((q) => {
        "use gpu";
        return d.vec4f(std.neg(q.xyz), q.w);
    })
    .$name("satQConj");

/** rotate `v` by the unit quat `q` (active rotation, shallot's `composeTransform` form).
 *  @example const world = satQRotate(quat, local); */
const satQRotate = tgpu
    .fn(
        [d.vec4f, d.vec3f],
        d.vec3f,
    )((q, v) => {
        "use gpu";
        const t = std.mul(2, std.cross(q.xyz, v));
        return std.add(std.add(v, std.mul(q.w, t)), std.cross(q.xyz, t));
    })
    .$name("satQRotate");

/** orthonormal frame with `n` as row 0 (matches math.ts orthonormal / the gym orthoBasis).
 *  @example const basis = orthoBasis(normal); */
const orthoBasis = tgpu
    .fn(
        [d.vec3f],
        Mat3,
    )((n) => {
        "use gpu";
        let t1 = d.vec3f();
        if (std.abs(n.x) > std.abs(n.y)) t1 = d.vec3f(-n.z, 0, n.x);
        else t1 = d.vec3f(0, n.z, -n.y);
        t1 = std.normalize(t1);
        return Mat3({ r0: n, r1: t1, r2: std.cross(t1, n) });
    })
    .$name("orthoBasis");

/** the two closest points between segments [p0,p1] and [q0,q1] (Ericson RTCD §5.1.9) — the box
 *  edge-edge contact and every rounded core pair both route through it. Index 0 is on the first
 *  segment.
 *  @example const cs = closestSegments(a0, a1, b0, b1); */
const closestSegments = tgpu
    .fn(
        [d.vec3f, d.vec3f, d.vec3f, d.vec3f],
        Vec3x2,
    )((p0, p1, q0, q1) => {
        "use gpu";
        const d1 = std.sub(p1, p0);
        const d2 = std.sub(q1, q0);
        const r = std.sub(p0, q0);
        const a = std.dot(d1, d1);
        const e = std.dot(d2, d2);
        const f = std.dot(d2, r);
        let s = d.f32(0);
        let t = d.f32(0);
        if (a <= SAT_AXIS_EPSILON && e <= SAT_AXIS_EPSILON) {
            return Vec3x2([p0, q0]);
        }
        if (a <= SAT_AXIS_EPSILON) {
            t = std.clamp(f / e, 0, 1);
        } else {
            const c = std.dot(d1, r);
            if (e <= SAT_AXIS_EPSILON) {
                s = std.clamp(-c / a, 0, 1);
            } else {
                const b = std.dot(d1, d2);
                const denom = a * e - b * b;
                if (std.abs(denom) > SAT_AXIS_EPSILON) s = std.clamp((b * f - c * e) / denom, 0, 1);
                t = (b * s + f) / e;
                if (t < 0) {
                    t = 0;
                    s = std.clamp(-c / a, 0, 1);
                } else if (t > 1) {
                    t = 1;
                    s = std.clamp((b - c) / a, 0, 1);
                }
            }
        }
        return Vec3x2([std.add(p0, std.mul(d1, s)), std.add(q0, std.mul(d2, t))]);
    })
    .$name("closestSegments");

// append one contact (local arms) into the result. Face-manifold dedup happens earlier, in the
// candidate build; this only runs for the final reduced set + the edge/fallback single-point cases.
// Mirrors collide.ts addContact.
const addContact = tgpu
    .fn([d.ptrFn(SatResult), d.vec3f, d.vec4f, d.vec3f, d.vec4f, d.vec3f, d.vec3f, d.u32])(
        (res, posA, quatA, posB, quatB, xA, xB, feature) => {
            "use gpu";
            const cnt = res.$.count;
            if (cnt >= MAX_CONTACTS) return;
            res.$.feat[cnt] = feature;
            res.$.rA[cnt] = satQRotate(satQConj(quatA), std.sub(xA, posA));
            res.$.rB[cnt] = satQRotate(satQConj(quatB), std.sub(xB, posB));
            res.$.count = cnt + 1;
        },
    )
    .$name("addContact");

const preferReduce = tgpu
    .fn(
        [d.f32, d.u32, d.f32, d.u32],
        d.bool,
    )((value, feature, bestValue, bestFeature) => {
        "use gpu";
        const scale = std.max(std.abs(value), std.abs(bestValue));
        const tied = std.abs(value - bestValue) <= scale * F32_TIE_REL;
        return (value > bestValue && !tied) || (tied && feature < bestFeature);
    })
    .$name("preferReduce");

// Reduce a > 4 candidate set to a <=4-point spread set — a verbatim port of Jolt's PruneContactPoints
// (reference/jolt ManifoldBetweenTwoFaces.cpp; Gregorius GDC-2015; reference/bullet3 b3ContactCache
// corroborates). Project each A-anchor onto the contact plane (perp axis, relative to comA), keep the
// point maximizing (planar dist)²·depth², its farthest plane-partner, then the furthest candidate on
// EACH side of that line (max quad area). Mirrors tests/avbd/collide.ts reduceManifold; gold-gated by
// reduce.oracle.ts. Writes the kept candidate indices to sel, returns the kept count (2-4).
const pruneContacts = tgpu
    .fn(
        [
            d.ptrFn(CandVerts),
            d.ptrFn(CandVerts),
            d.ptrFn(CandFeats),
            d.u32,
            d.vec3f,
            d.vec3f,
            d.ptrFn(SelIdx),
        ],
        d.u32,
    )((candXA, candXB, candFeat, n, axis, comA, sel) => {
        "use gpu";
        const projected = CandVerts();
        const depthSq = CandScalars();
        for (let i = d.u32(0); i < n; i++) {
            const v1 = std.sub(candXA.$[i], comA);
            projected[i] = std.sub(v1, std.mul(axis, std.dot(v1, axis)));
            depthSq[i] = std.max(
                REDUCE_MIN_DIST_SQ,
                std.dot(std.sub(candXB.$[i], candXA.$[i]), std.sub(candXB.$[i], candXA.$[i])),
            );
        }

        let p1 = d.u32(0);
        let val = std.max(REDUCE_MIN_DIST_SQ, std.dot(projected[0], projected[0])) * depthSq[0];
        for (let i = d.u32(0); i < n; i++) {
            const v = std.max(REDUCE_MIN_DIST_SQ, std.dot(projected[i], projected[i])) * depthSq[i];
            if (preferReduce(v, candFeat.$[i], val, candFeat.$[p1])) {
                val = v;
                p1 = i;
            }
        }
        // copy, not `projected[p1]` — typegpu emits a *pointer* for an unaliased element read
        const p1v = d.vec3f(projected[p1]);

        // n > 4 (the only call condition) guarantees a partner ≠ p1, so p2 is always found (matches the
        // reference, which keeps p1 + p2 unconditionally; only p3/p4 are side-of-line conditional).
        let p2 = d.u32(0);
        let haveP2 = false;
        val = d.f32(0);
        for (let i = d.u32(0); i < n; i++) {
            if (i === p1) continue;
            const dv = std.sub(projected[i], p1v);
            const v = std.max(REDUCE_MIN_DIST_SQ, std.dot(dv, dv)) * depthSq[i];
            if (!haveP2 || preferReduce(v, candFeat.$[i], val, candFeat.$[p2])) {
                haveP2 = true;
                val = v;
                p2 = i;
            }
        }
        const p2v = d.vec3f(projected[p2]);

        let p3 = d.u32(0);
        let haveP3 = false;
        let p4 = d.u32(0);
        let haveP4 = false;
        let minV = d.f32(0);
        let maxV = d.f32(0);
        const perp = std.cross(std.sub(p2v, p1v), axis);
        for (let i = d.u32(0); i < n; i++) {
            if (i === p1 || i === p2) continue;
            const v = std.dot(perp, std.sub(projected[i], p1v));
            if (v < 0) {
                if (!haveP3 || preferReduce(-v, candFeat.$[i], -minV, candFeat.$[p3])) {
                    minV = v;
                    p3 = i;
                    haveP3 = true;
                }
            } else if (v > 0) {
                if (!haveP4 || preferReduce(v, candFeat.$[i], maxV, candFeat.$[p4])) {
                    maxV = v;
                    p4 = i;
                    haveP4 = true;
                }
            }
        }

        let c = d.u32(0);
        sel.$[c] = p1;
        c = c + 1;
        if (haveP3) {
            sel.$[c] = p3;
            c = c + 1;
        }
        sel.$[c] = p2;
        c = c + 1;
        if (haveP4) {
            sel.$[c] = p4;
            c = c + 1;
        }
        return c;
    })
    .$name("pruneContacts");

const helpersChunk = chunk(
    "helpersWgsl",
    [
        Mat3,
        SatResult,
        satZero,
        orthoBasis,
        closestSegments,
        addContact,
        preferReduce,
        pruneContacts,
    ],
    spliceNs,
);

/**
 * the shared narrowphase substrate every collide pipeline prepends: `Mat3` / `SatResult`, the quat
 * helpers + `orthoBasis`, the segment closest-point, and the contact append + the Jolt reduce-to-4.
 * Splice this **first**, then one or more of {@link boxBoxWgsl} / {@link roundedWgsl} /
 * {@link hullCoreWgsl}. A definition lands in the lowest chunk that needs it, so the box/rounded
 * pipelines never compile the closest-point primitives only the polytope path calls.
 */
export const helpersWgsl = helpersChunk;

// ── box-box SAT ──────────────────────────────────────────────────────────────────────────────────

/** an oriented box: center, half-extents, and its three world axes */
const Obb = d
    .struct({ c: d.vec3f, h: d.vec3f, ax0: d.vec3f, ax1: d.vec3f, ax2: d.vec3f })
    .$name("Obb");

/** face tangent axes + their extents for a given face axis index (collide.ts getFaceAxes) */
const FaceAxes = d.struct({ u: d.vec3f, v: d.vec3f, eu: d.f32, ev: d.f32 }).$name("FaceAxes");

/** a clipped face polygon: up to {@link MAX_POLY_VERTS} world verts */
const Poly = d.struct({ v: PolyVerts, n: d.u32 }).$name("Poly");

/** the winning SAT axis: which candidate class, the two feature indices, the separation, and the
 *  A→B-oriented normal */
const SatAxis = d
    .struct({
        kind: d.u32,
        indexA: d.u32,
        indexB: d.u32,
        separation: d.f32,
        normalAB: d.vec3f,
        valid: d.bool,
    })
    .$name("SatAxis");

const obbOf = tgpu
    .fn(
        [d.vec3f, d.vec4f, d.vec3f],
        Obb,
    )((pos, quat, size) => {
        "use gpu";
        return Obb({
            c: pos,
            h: std.mul(size, 0.5),
            ax0: satQRotate(quat, d.vec3f(1, 0, 0)),
            ax1: satQRotate(quat, d.vec3f(0, 1, 0)),
            ax2: satQRotate(quat, d.vec3f(0, 0, 1)),
        });
    })
    .$name("obbOf");

const obbAxis = tgpu
    .fn(
        [Obb, d.u32],
        d.vec3f,
    )((o, i) => {
        "use gpu";
        if (i === 0) return d.vec3f(o.ax0);
        if (i === 1) return d.vec3f(o.ax1);
        return d.vec3f(o.ax2);
    })
    .$name("obbAxis");

const obbHalf = tgpu
    .fn(
        [Obb, d.u32],
        d.f32,
    )((o, i) => {
        "use gpu";
        if (i === 0) return o.h.x;
        if (i === 1) return o.h.y;
        return o.h.z;
    })
    .$name("obbHalf");

const absDot = tgpu
    .fn(
        [d.vec3f, d.vec3f],
        d.f32,
    )((a, b) => {
        "use gpu";
        return std.abs(std.dot(a, b));
    })
    .$name("absDot");

const supportPoint = tgpu
    .fn(
        [Obb, d.vec3f],
        d.vec3f,
    )((o, dir) => {
        "use gpu";
        const sx = std.select(d.f32(-1), d.f32(1), std.dot(dir, o.ax0) >= 0);
        const sy = std.select(d.f32(-1), d.f32(1), std.dot(dir, o.ax1) >= 0);
        const sz = std.select(d.f32(-1), d.f32(1), std.dot(dir, o.ax2) >= 0);
        return std.add(
            std.add(std.add(o.c, std.mul(o.ax0, o.h.x * sx)), std.mul(o.ax1, o.h.y * sy)),
            std.mul(o.ax2, o.h.z * sz),
        );
    })
    .$name("supportPoint");

const getFaceAxes = tgpu
    .fn(
        [Obb, d.u32],
        FaceAxes,
    )((o, ai) => {
        "use gpu";
        if (ai === 0) return FaceAxes({ u: o.ax1, v: o.ax2, eu: o.h.y, ev: o.h.z });
        if (ai === 1) return FaceAxes({ u: o.ax0, v: o.ax2, eu: o.h.x, ev: o.h.z });
        return FaceAxes({ u: o.ax0, v: o.ax1, eu: o.h.x, ev: o.h.y });
    })
    .$name("getFaceAxes");

const incidentAxis = tgpu
    .fn(
        [Obb, d.vec3f],
        d.u32,
    )((o, refN) => {
        "use gpu";
        let axis = d.u32(0);
        let best = d.f32(NO_AXIS);
        for (let i = d.u32(0); i < 3; i++) {
            const dd = absDot(obbAxis(o, i), refN);
            if (dd > best) {
                best = dd;
                axis = i;
            }
        }
        return axis;
    })
    .$name("incidentAxis");

const incidentFace = tgpu
    .fn(
        [Obb, d.u32, d.vec3f],
        Poly,
    )((o, ai, refN) => {
        "use gpu";
        const sgn = std.select(d.f32(1), d.f32(-1), std.dot(obbAxis(o, ai), refN) > 0);
        const faceNormal = std.mul(obbAxis(o, ai), sgn);
        const faceCenter = std.add(o.c, std.mul(faceNormal, obbHalf(o, ai)));
        const fa = getFaceAxes(o, ai);
        const p = Poly({ v: PolyVerts(), n: 0 });
        p.n = 4;
        p.v[0] = std.add(std.add(faceCenter, std.mul(fa.u, fa.eu)), std.mul(fa.v, fa.ev));
        p.v[1] = std.add(std.sub(faceCenter, std.mul(fa.u, fa.eu)), std.mul(fa.v, fa.ev));
        p.v[2] = std.sub(std.sub(faceCenter, std.mul(fa.u, fa.eu)), std.mul(fa.v, fa.ev));
        p.v[3] = std.sub(std.add(faceCenter, std.mul(fa.u, fa.eu)), std.mul(fa.v, fa.ev));
        return p;
    })
    .$name("incidentFace");

// Sutherland-Hodgman clip against one half-space (planeNormal . x <= planeOffset)
const clip = tgpu
    .fn(
        [Poly, d.vec3f, d.f32],
        Poly,
    )((inp, planeNormal, planeOffset) => {
        "use gpu";
        const out = Poly({ v: PolyVerts(), n: 0 });
        out.n = 0;
        if (inp.n === 0) return out;

        let a = d.vec3f(inp.v[inp.n - 1]);
        let da = std.dot(planeNormal, a) - planeOffset;
        for (let i = d.u32(0); i < inp.n; i++) {
            const b = inp.v[i];
            const db = std.dot(planeNormal, b) - planeOffset;
            const aInside = da <= PLANE_EPSILON;
            const bInside = db <= PLANE_EPSILON;
            if (aInside !== bInside) {
                let t = d.f32(0);
                const denom = da - db;
                if (std.abs(denom) > SAT_AXIS_EPSILON) t = std.clamp(da / denom, 0, 1);
                if (out.n < MAX_POLY_VERTS) {
                    out.v[out.n] = std.add(a, std.mul(std.sub(b, a), t));
                    out.n = out.n + 1;
                }
            }
            if (bInside && out.n < MAX_POLY_VERTS) {
                out.v[out.n] = d.vec3f(b);
                out.n = out.n + 1;
            }
            a = d.vec3f(b);
            da = db;
        }
        return out;
    })
    .$name("clip");

const supportEdge = tgpu
    .fn(
        [Obb, d.u32, d.vec3f],
        Vec3x2,
    )((o, ai, dir) => {
        "use gpu";
        const a1 = (ai + 1) % 3;
        const a2 = (ai + 2) % 3;
        const s1 = std.select(d.f32(-1), d.f32(1), std.dot(dir, obbAxis(o, a1)) >= 0);
        const s2 = std.select(d.f32(-1), d.f32(1), std.dot(dir, obbAxis(o, a2)) >= 0);
        const edgeCenter = std.add(
            std.add(o.c, std.mul(obbAxis(o, a1), obbHalf(o, a1) * s1)),
            std.mul(obbAxis(o, a2), obbHalf(o, a2) * s2),
        );
        const half = std.mul(obbAxis(o, ai), obbHalf(o, ai));
        return Vec3x2([std.sub(edgeCenter, half), std.add(edgeCenter, half)]);
    })
    .$name("supportEdge");

const testAxis = tgpu
    .fn(
        [Obb, Obb, d.vec3f, d.vec3f, d.u32, d.u32, d.u32, d.ptrFn(SatAxis), d.vec3f],
        d.bool,
    )((boxA, boxB, delta, axis, kind, ia, ib, best, dRel) => {
        "use gpu";
        const lenSq = std.dot(axis, axis);
        if (lenSq < SAT_AXIS_EPSILON) return true;
        let n = std.mul(axis, 1 / std.sqrt(lenSq));
        if (std.dot(n, delta) < 0) n = std.neg(n);
        const gapDist = std.abs(std.dot(delta, n));
        const rA =
            boxA.h.x * absDot(n, boxA.ax0) +
            boxA.h.y * absDot(n, boxA.ax1) +
            boxA.h.z * absDot(n, boxA.ax2);
        const rB =
            boxB.h.x * absDot(n, boxB.ax0) +
            boxB.h.y * absDot(n, boxB.ax1) +
            boxB.h.z * absDot(n, boxB.ax2);
        const separation = gapDist - (rA + rB);
        // a separating axis aborts the SAT only past the speculative band; within it the axis of maximum
        // separation is kept and a speculative manifold built off it, so a body within the band lands at
        // contact rather than tunnelling through it (Phase 4.8.3). Phase 4.8.4 (velocity sweep): the band along
        // n is max(SPECULATIVE_DISTANCE, closing displacement max(0, dot(dRel, n))) (dRel = (vA−vB)·dt) — a
        // fast mover crossing the contact this step is caught at frame start. max (not +) keeps a slow body's
        // band velocity-independent (no settling feedback). Matches the f64 oracle + C++ harness; closing 0 = 4.8.3.
        if (separation > std.max(SPECULATIVE_DISTANCE, std.max(0, std.dot(dRel, n)))) return false;
        if (!best.$.valid || separation > best.$.separation) {
            best.$.valid = true;
            best.$.kind = kind;
            best.$.indexA = ia;
            best.$.indexB = ib;
            best.$.separation = separation;
            best.$.normalAB = d.vec3f(n);
        }
        return true;
    })
    .$name("testAxis");

const faceManifold = tgpu
    .fn([
        d.ptrFn(SatResult),
        Obb,
        Obb,
        d.vec3f,
        d.vec4f,
        d.vec3f,
        d.vec4f,
        d.bool,
        d.u32,
        d.vec3f,
        d.f32,
    ])((res, boxA, boxB, posA, quatA, posB, quatB, referenceIsA, refAxis, normalAB, band) => {
        "use gpu";
        let refBox = Obb(boxB);
        let incBox = Obb(boxA);
        let refOutward = std.neg(normalAB);
        if (referenceIsA) {
            refBox = Obb(boxA);
            incBox = Obb(boxB);
            refOutward = d.vec3f(normalAB);
        }

        // reference face frame
        const refSign = std.select(
            d.f32(-1),
            d.f32(1),
            std.dot(refOutward, obbAxis(refBox, refAxis)) >= 0,
        );
        const refNormal = std.mul(obbAxis(refBox, refAxis), refSign);
        const refCenter = std.add(refBox.c, std.mul(refNormal, obbHalf(refBox, refAxis)));
        const refFa = getFaceAxes(refBox, refAxis);

        const incAxis = incidentAxis(incBox, refNormal);
        let poly = incidentFace(incBox, incAxis, refNormal);

        poly = clip(poly, refFa.u, std.dot(refFa.u, refCenter) + refFa.eu);
        if (poly.n === 0) return;
        poly = clip(poly, std.neg(refFa.u), std.dot(std.neg(refFa.u), refCenter) + refFa.eu);
        if (poly.n === 0) return;
        poly = clip(poly, refFa.v, std.dot(refFa.v, refCenter) + refFa.ev);
        if (poly.n === 0) return;
        poly = clip(poly, std.neg(refFa.v), std.dot(std.neg(refFa.v), refCenter) + refFa.ev);
        if (poly.n === 0) return;

        let prefix = std.select(d.u32(AXIS_FACE_B), d.u32(AXIS_FACE_A), referenceIsA) << d.u32(24);
        prefix = prefix | ((refAxis & d.u32(0xff)) << d.u32(16));
        prefix = prefix | ((incAxis & d.u32(0xff)) << d.u32(8));

        // build the clipped candidates (up to MAX_CANDIDATES), dedup by midpoint
        const candXA = CandVerts();
        const candXB = CandVerts();
        const candFeat = CandFeats();
        let candN = d.u32(0);
        for (let i = d.u32(0); i < poly.n && candN < MAX_CANDIDATES; i++) {
            const pIncident = d.vec3f(poly.v[i]); // copy — an unwrapped element read emits a pointer
            const depth = std.dot(std.sub(pIncident, refCenter), refNormal);
            // a clip vertex up to the (swept) band beyond the reference face is kept: its projection onto the
            // face plane carries the +gap into c0, generating a separated contact early (Phase 4.8.3). The band
            // is max(SPECULATIVE_DISTANCE, closing displacement along the contact normal) (Phase 4.8.4); a
            // penetrating vertex (depth <= 0) is always kept, so a settled pile's manifold is unchanged.
            if (depth > band) continue;
            const pReference = std.sub(pIncident, std.mul(refNormal, depth));
            let xA = d.vec3f(pIncident);
            let xB = d.vec3f(pReference);
            if (referenceIsA) {
                xA = d.vec3f(pReference);
                xB = d.vec3f(pIncident);
            }
            const mid = std.mul(std.add(xA, xB), 0.5);
            let dup = false;
            for (let k = d.u32(0); k < candN; k++) {
                const dm = std.sub(mid, std.mul(std.add(candXA[k], candXB[k]), 0.5));
                if (std.dot(dm, dm) < CONTACT_MERGE_DIST_SQ) {
                    dup = true;
                    break;
                }
            }
            if (dup) continue;
            candXA[candN] = d.vec3f(xA);
            candXB[candN] = d.vec3f(xB);
            candFeat[candN] = prefix | (i & d.u32(0xff));
            candN = candN + 1;
        }

        if (candN === 0) {
            const xA = supportPoint(boxA, normalAB);
            const xB = supportPoint(boxB, std.neg(normalAB));
            addContact(res, posA, quatA, posB, quatB, xA, xB, prefix);
            return;
        }

        // reduce to the spread set when the clip over-produced; each kept contact keeps its original
        // clip-ordinal feature key (stable, scan-matched — no re-ordinal; see collide.ts oracle header).
        const sel = SelIdx();
        let selN = d.u32(0);
        if (candN > MAX_CONTACTS) {
            selN = pruneContacts(
                d.ref(candXA),
                d.ref(candXB),
                d.ref(candFeat),
                candN,
                normalAB,
                posA,
                d.ref(sel),
            );
        } else {
            selN = candN;
            for (let i = d.u32(0); i < candN; i++) sel[i] = i;
        }
        for (let i = d.u32(0); i < selN; i++) {
            const s = sel[i];
            addContact(res, posA, quatA, posB, quatB, candXA[s], candXB[s], candFeat[s]);
        }
    })
    .$name("faceManifold");

const edgeContact = tgpu
    .fn([d.ptrFn(SatResult), Obb, Obb, d.vec3f, d.vec4f, d.vec3f, d.vec4f, d.u32, d.u32, d.vec3f])(
        (res, boxA, boxB, posA, quatA, posB, quatB, axisA, axisB, normalAB) => {
            "use gpu";
            const ea = supportEdge(boxA, axisA, normalAB);
            const eb = supportEdge(boxB, axisB, std.neg(normalAB));
            const cs = closestSegments(ea[0], ea[1], eb[0], eb[1]);
            const feature =
                (d.u32(AXIS_EDGE) << d.u32(24)) |
                ((axisA & d.u32(0xff)) << d.u32(8)) |
                (axisB & d.u32(0xff));
            addContact(res, posA, quatA, posB, quatB, cs[0], cs[1], feature);
            if (res.$.count === 0) {
                const sA = supportPoint(boxA, normalAB);
                const sB = supportPoint(boxB, std.neg(normalAB));
                addContact(res, posA, quatA, posB, quatB, sA, sB, feature);
            }
        },
    )
    .$name("edgeContact");

/**
 * box-box SAT. `count == 0` means no overlap (a separating axis past the band was found). Mirrors
 * Manifold::collide. `dRel` is the relative displacement over the step (vA−vB)·dt — the velocity sweep
 * (Phase 4.8.4); zero recovers the static 4.8.3 SAT. `size` is the full box width (2 × half-extents).
 *
 * @internal Exported for the CPU differential (`tests/avbd/sat.oracle.ts` runs it against the C++ gold
 * vectors), not for a consumer: the WGSL a consumer wants is {@link boxBoxWgsl}.
 */
export const collideBoxBox = tgpu
    .fn(
        [d.vec3f, d.vec4f, d.vec3f, d.vec3f, d.vec4f, d.vec3f, d.vec3f],
        SatResult,
    )((posA, quatA, sizeA, posB, quatB, sizeB, dRel) => {
        "use gpu";
        const res = satZero();

        const boxA = obbOf(posA, quatA, sizeA);
        const boxB = obbOf(posB, quatB, sizeB);
        const delta = std.sub(boxB.c, boxA.c);

        const bestFace = SatAxis({
            kind: 0,
            indexA: 0,
            indexB: 0,
            separation: NO_AXIS,
            normalAB: d.vec3f(),
            valid: false,
        });
        bestFace.valid = false;
        bestFace.separation = NO_AXIS;
        const bestEdge = SatAxis({
            kind: 0,
            indexA: 0,
            indexB: 0,
            separation: NO_AXIS,
            normalAB: d.vec3f(),
            valid: false,
        });
        bestEdge.valid = false;
        bestEdge.separation = NO_AXIS;

        for (let i = d.u32(0); i < 3; i++) {
            if (
                !testAxis(
                    boxA,
                    boxB,
                    delta,
                    obbAxis(boxA, i),
                    AXIS_FACE_A,
                    i,
                    0,
                    d.ref(bestFace),
                    dRel,
                )
            )
                return res;
        }
        for (let i = d.u32(0); i < 3; i++) {
            if (
                !testAxis(
                    boxA,
                    boxB,
                    delta,
                    obbAxis(boxB, i),
                    AXIS_FACE_B,
                    0,
                    i,
                    d.ref(bestFace),
                    dRel,
                )
            )
                return res;
        }
        for (let i = d.u32(0); i < 3; i++) {
            for (let j = d.u32(0); j < 3; j++) {
                const axis = std.cross(obbAxis(boxA, i), obbAxis(boxB, j));
                if (!testAxis(boxA, boxB, delta, axis, AXIS_EDGE, i, j, d.ref(bestEdge), dRel))
                    return res;
            }
        }

        if (!bestFace.valid) return res;

        let best = SatAxis(bestFace);
        if (bestEdge.valid) {
            if (d.f32(0.95) * bestEdge.separation > bestFace.separation + d.f32(0.01))
                best = SatAxis(bestEdge);
        }

        res.basis = orthoBasis(std.neg(best.normalAB));

        // the swept clip band along the contact axis: the same max(SPECULATIVE_DISTANCE, closing) form as the
        // testAxis abort, for the winning normal. A max(0, …) magnitude, reference-orientation safe.
        const band = std.max(SPECULATIVE_DISTANCE, std.max(0, std.dot(dRel, best.normalAB)));

        if (best.kind === AXIS_EDGE) {
            edgeContact(
                d.ref(res),
                boxA,
                boxB,
                posA,
                quatA,
                posB,
                quatB,
                best.indexA,
                best.indexB,
                best.normalAB,
            );
        } else if (best.kind === AXIS_FACE_A) {
            faceManifold(
                d.ref(res),
                boxA,
                boxB,
                posA,
                quatA,
                posB,
                quatB,
                true,
                best.indexA,
                best.normalAB,
                band,
            );
        } else {
            faceManifold(
                d.ref(res),
                boxA,
                boxB,
                posA,
                quatA,
                posB,
                quatB,
                false,
                best.indexB,
                best.normalAB,
                band,
            );
        }
        return res;
    })
    .$name("collideBoxBox");

const boxBoxChunk = chunk("boxBoxWgsl", [collideBoxBox], spliceNs);

/**
 * the box-box SAT chunk (`collideBoxBox`). Splice after {@link helpersWgsl}; the box collide pipeline
 * (step.ts) is those two, so it never compiles the hull/rounded SAT.
 */
export function boxBoxWgsl(): string {
    helpersWgsl(); // force the base chunk first — the structs + manifold primitives live there
    return boxBoxChunk();
}

// ── rounded × rounded (sphere / capsule) ─────────────────────────────────────────────────────────

/**
 * Rounded-rounded narrowphase (sphere/capsule pairs, Phase 6.3) — the port of the CPU oracle
 * (tests/avbd/rounded.ts collideRounded). A sphere is a zero-length segment (a point), a capsule a
 * segment; one segment-segment closest-point query ({@link closestSegments}) handles every rounded
 * pair, then subtract the radii into one contact. `size` is the core full-width (segment =
 * pos ± rotate(quat, size·0.5)); `radius` the rounding. `dRel` = (vA−vB)·dt is the velocity sweep.
 *
 * @internal Exported for the CPU differential (`tests/avbd/rounded.oracle.ts`), like
 * {@link collideBoxBox}; a consumer splices {@link roundedWgsl}.
 */
export const collideRounded = tgpu
    .fn(
        [d.vec3f, d.vec4f, d.vec3f, d.f32, d.vec3f, d.vec4f, d.vec3f, d.f32, d.vec3f],
        SatResult,
    )((posA, quatA, sizeA, radiusA, posB, quatB, sizeB, radiusB, dRel) => {
        "use gpu";
        const res = satZero();

        const hA = satQRotate(quatA, std.mul(sizeA, 0.5));
        const hB = satQRotate(quatB, std.mul(sizeB, 0.5));
        const cs = closestSegments(
            std.sub(posA, hA),
            std.add(posA, hA),
            std.sub(posB, hB),
            std.add(posB, hB),
        );
        const cA = d.vec3f(cs[0]);
        const cB = d.vec3f(cs[1]);

        const delta = std.sub(cA, cB); // core-to-core, points B -> A
        const dist = std.length(delta);
        let normal = d.vec3f(0, 1, 0);
        if (dist > ROUND_NORMAL_EPS) normal = std.div(delta, dist);

        // generate while still separated by up to the speculative band (surfaces touch at gap 0): the static
        // skin, widened by the closing displacement this step (velocity sweep). max, not + (a slow body's band
        // stays the skin). The cores approach along -normal, so closing = -dot(dRel, normal).
        const closing = std.max(0, -std.dot(dRel, normal));
        const band = std.max(SPECULATIVE_DISTANCE, closing);

        const gap = dist - radiusA - radiusB;
        if (gap > band) return res;

        // store the CORE closest points (NOT the inflated surface) — the radius offset (the −r·normal that
        // reaches the surface) is geometric and must stay along the fixed normal, not rotate with the body's
        // spin. A sphere's contact point is not a material point; freezing the surface arm + rotating it
        // injects a spurious tangential lever into the normal Jacobian → the rotational-instability bug.
        // contactForce (step.ts) re-applies ±radius·normal at solve time. Mirrors the oracle. roadmap §6.3.
        res.basis = orthoBasis(normal);
        addContact(d.ref(res), posA, quatA, posB, quatB, cA, cB, ROUND_FEATURE);
        return res;
    })
    .$name("collideRounded");

const roundedChunk = chunk("roundedWgsl", [collideRounded], spliceNs);

/**
 * the rounded-rounded chunk (`collideRounded`, sphere/capsule pairs). Splice after
 * {@link helpersWgsl}; the rounded collide pipeline (step.ts) is those two, so it never compiles the
 * box or hull SAT.
 */
export function roundedWgsl(): string {
    helpersWgsl();
    return roundedChunk();
}

/** the closest surface point on an OBB, its outward normal, and the signed distance (+ outside, − inside) */
const BoxClosest = d
    .struct({ point: d.vec3f, normal: d.vec3f, signedDist: d.f32 })
    .$name("BoxClosest");

/** closest point on an OBB (box-local) to a query point — the rounded-vs-box analog of
 *  {@link closestSegments}. Clamping into [−half, half] gives the surface point when the query is
 *  outside; inside, the clamp is a no-op, so push out along the least-penetrating face. Mirrors
 *  tests/avbd/rounded.ts closestPointBox; no GJK/EPA.
 *  @example const cp = closestPointBox(localQuery, halfExtents); */
const closestPointBox = tgpu
    .fn(
        [d.vec3f, d.vec3f],
        BoxClosest,
    )((pl, half) => {
        "use gpu";
        const dc = std.clamp(pl, std.neg(half), half);
        const diff = std.sub(pl, dc);
        const distSq = std.dot(diff, diff);
        // the eps² folds at f32 (the seeds) — a bare literal product folds at f64 and can land a ulp off
        if (distSq > d.f32(ROUND_NORMAL_EPS) * d.f32(ROUND_NORMAL_EPS)) {
            const dist = std.sqrt(distSq);
            return BoxClosest({ point: dc, normal: std.div(diff, dist), signedDist: dist });
        }
        // inside: the nearest face is the axis with the least clearance to its slab boundary
        let axis = d.u32(0);
        let least = half.x - std.abs(pl.x);
        const cy = half.y - std.abs(pl.y);
        const cz = half.z - std.abs(pl.z);
        if (cy < least) {
            least = cy;
            axis = 1;
        }
        if (cz < least) {
            least = cz;
            axis = 2;
        }
        const normal = d.vec3f();
        const point = d.vec3f(pl);
        if (axis === 0) {
            const s = std.select(d.f32(-1), d.f32(1), pl.x >= 0);
            normal.x = s;
            point.x = s * half.x;
        } else if (axis === 1) {
            const s = std.select(d.f32(-1), d.f32(1), pl.y >= 0);
            normal.y = s;
            point.y = s * half.y;
        } else {
            const s = std.select(d.f32(-1), d.f32(1), pl.z >= 0);
            normal.z = s;
            point.z = s * half.z;
        }
        return BoxClosest({ point: point, normal: normal, signedDist: -least });
    })
    .$name("closestPointBox");

// ── the scale-unified polytope substrate (box OR registered hull) ────────────────────────────────
// The hull geometry readers are duals over `hullData`, the flat `array<u32>` packed by ./hull. Their
// WGSL leaves read the consumer-declared binding; their CPU arms read one synchronously-scoped packed
// array. Everything above them is the same TGSL graph on both paths.

/** one registered hull's slice of the packed `hullData` buffer */
const HullRef = d
    .struct({
        vertBase: d.u32,
        vertCount: d.u32,
        faceBase: d.u32,
        faceCount: d.u32,
        edgeBase: d.u32,
        edgeCount: d.u32,
        faceIdxBase: d.u32,
    })
    .$name("HullRef");

let cpuHullData: Uint32Array | undefined;

function cpuHullWord(i: number): number {
    const data = cpuHullData;
    if (!data)
        throw new Error("collideHull CPU execution requires withCpuHullData(packHulls(), fn)");
    const value = data[i];
    if (value === undefined)
        throw new Error(`collideHull CPU hullData index ${i} is out of bounds`);
    return value;
}

/** Run the hull TGSL graph on the CPU against the exact packed storage words a GPU dispatch reads.
 *  The scope is synchronous because TypeGPU CPU functions are synchronous; nesting restores the outer
 *  binding. @internal */
export function withCpuHullData<T>(data: Uint32Array, fn: () => T): T {
    const previous = cpuHullData;
    cpuHullData = data;
    try {
        const result = fn();
        if (
            result !== null &&
            (typeof result === "object" || typeof result === "function") &&
            "then" in result &&
            typeof result.then === "function"
        )
            throw new Error("withCpuHullData callback must be synchronous");
        return result;
    } finally {
        cpuHullData = previous;
    }
}

const hullRefWgsl = tgpu
    .fn(
        [d.u32],
        HullRef,
    )(/* wgsl */ `(id: u32) -> HullRef {
    let o = id * ${HULL_HEADER}u;
    return HullRef(hullData[o], hullData[o+1u], hullData[o+2u], hullData[o+3u], hullData[o+4u], hullData[o+5u], hullData[o+6u]);
}`)
    .$uses({ HullRef })
    .$name("hullRefWgsl");

const hullRef = tgpu
    .fn(
        [d.u32],
        HullRef,
    )((id) => {
        "use gpu";
        const o = id * HULL_HEADER;
        return std.isBeingTranspiled()
            ? hullRefWgsl(id)
            : HullRef({
                  vertBase: cpuHullWord(o),
                  vertCount: cpuHullWord(o + 1),
                  faceBase: cpuHullWord(o + 2),
                  faceCount: cpuHullWord(o + 3),
                  edgeBase: cpuHullWord(o + 4),
                  edgeCount: cpuHullWord(o + 5),
                  faceIdxBase: cpuHullWord(o + 6),
              });
    })
    .$name("hullRef");

// hull-direct local accessors (read raw geometry from hullData; NO isBox branch). The scale-unified poly
// accessors below build box AND hull on these — the per-access isBox branch was the collide-pass compile
// cost (gpu.md "DXC shader compilation": dead code isn't free, the box branch compiled at every site).
const hVertLWgsl = tgpu
    .fn(
        [HullRef, d.u32],
        d.vec3f,
    )(/* wgsl */ `(h: HullRef, i: u32) -> vec3f {
    let o = h.vertBase + i * 3u;
    return vec3f(bitcast<f32>(hullData[o]), bitcast<f32>(hullData[o+1u]), bitcast<f32>(hullData[o+2u]));
}`)
    .$uses({ HullRef })
    .$name("hVertLWgsl");

const hVertL = tgpu
    .fn(
        [HullRef, d.u32],
        d.vec3f,
    )((h, i) => {
        "use gpu";
        const o = h.vertBase + i * 3;
        return std.isBeingTranspiled()
            ? hVertLWgsl(h, i)
            : d.vec3f(
                  std.bitcastU32toF32(cpuHullWord(o)),
                  std.bitcastU32toF32(cpuHullWord(o + 1)),
                  std.bitcastU32toF32(cpuHullWord(o + 2)),
              );
    })
    .$name("hVertL");

const hFaceNormalLWgsl = tgpu
    .fn(
        [HullRef, d.u32],
        d.vec3f,
    )(/* wgsl */ `(h: HullRef, f: u32) -> vec3f {
    let o = h.faceBase + f * ${HULL_FACE_STRIDE}u;
    return vec3f(bitcast<f32>(hullData[o]), bitcast<f32>(hullData[o+1u]), bitcast<f32>(hullData[o+2u]));
}`)
    .$uses({ HullRef })
    .$name("hFaceNormalLWgsl");

const hFaceNormalL = tgpu
    .fn(
        [HullRef, d.u32],
        d.vec3f,
    )((h, f) => {
        "use gpu";
        const o = h.faceBase + f * HULL_FACE_STRIDE;
        return std.isBeingTranspiled()
            ? hFaceNormalLWgsl(h, f)
            : d.vec3f(
                  std.bitcastU32toF32(cpuHullWord(o)),
                  std.bitcastU32toF32(cpuHullWord(o + 1)),
                  std.bitcastU32toF32(cpuHullWord(o + 2)),
              );
    })
    .$name("hFaceNormalL");

const hFaceOffsetWgsl = tgpu
    .fn(
        [HullRef, d.u32],
        d.f32,
    )(
        /* wgsl */ `(h: HullRef, f: u32) -> f32 { return bitcast<f32>(hullData[h.faceBase + f * ${HULL_FACE_STRIDE}u + 3u]); }`,
    )
    .$uses({ HullRef })
    .$name("hFaceOffsetWgsl");

const hFaceOffset = tgpu
    .fn(
        [HullRef, d.u32],
        d.f32,
    )((h, f) => {
        "use gpu";
        return std.isBeingTranspiled()
            ? hFaceOffsetWgsl(h, f)
            : std.bitcastU32toF32(cpuHullWord(h.faceBase + f * HULL_FACE_STRIDE + 3));
    })
    .$name("hFaceOffset");

const hFaceVertCountWgsl = tgpu
    .fn(
        [HullRef, d.u32],
        d.u32,
    )(
        /* wgsl */ `(h: HullRef, f: u32) -> u32 { return hullData[h.faceBase + f * ${HULL_FACE_STRIDE}u + 5u]; }`,
    )
    .$uses({ HullRef })
    .$name("hFaceVertCountWgsl");

const hFaceVertCount = tgpu
    .fn(
        [HullRef, d.u32],
        d.u32,
    )((h, f) => {
        "use gpu";
        return std.isBeingTranspiled()
            ? hFaceVertCountWgsl(h, f)
            : cpuHullWord(h.faceBase + f * HULL_FACE_STRIDE + 5);
    })
    .$name("hFaceVertCount");

const hFaceVertWgsl = tgpu
    .fn(
        [HullRef, d.u32, d.u32],
        d.u32,
    )(/* wgsl */ `(h: HullRef, f: u32, j: u32) -> u32 {
    let lo = hullData[h.faceBase + f * ${HULL_FACE_STRIDE}u + 4u];
    return hullData[h.faceIdxBase + lo + j];
}`)
    .$uses({ HullRef })
    .$name("hFaceVertWgsl");

const hFaceVert = tgpu
    .fn(
        [HullRef, d.u32, d.u32],
        d.u32,
    )((h, f, j) => {
        "use gpu";
        return std.isBeingTranspiled()
            ? hFaceVertWgsl(h, f, j)
            : cpuHullWord(h.faceIdxBase + cpuHullWord(h.faceBase + f * HULL_FACE_STRIDE + 4) + j);
    })
    .$name("hFaceVert");

const hEdgeLWgsl = tgpu
    .fn(
        [HullRef, d.u32],
        d.vec3f,
    )(/* wgsl */ `(h: HullRef, e: u32) -> vec3f {
    let o = h.edgeBase + e * 3u;
    return vec3f(bitcast<f32>(hullData[o]), bitcast<f32>(hullData[o+1u]), bitcast<f32>(hullData[o+2u]));
}`)
    .$uses({ HullRef })
    .$name("hEdgeLWgsl");

const hEdgeL = tgpu
    .fn(
        [HullRef, d.u32],
        d.vec3f,
    )((h, e) => {
        "use gpu";
        const o = h.edgeBase + e * 3;
        return std.isBeingTranspiled()
            ? hEdgeLWgsl(h, e)
            : d.vec3f(
                  std.bitcastU32toF32(cpuHullWord(o)),
                  std.bitcastU32toF32(cpuHullWord(o + 1)),
                  std.bitcastU32toF32(cpuHullWord(o + 2)),
              );
    })
    .$name("hEdgeL");

/**
 * A polytope: a registered hull (`hr`) at pos/quat, scaled by `scale`. A box is the built-in UNIT_CUBE
 * hull scaled by its half-extents — so box and hull collide through ONE branch-free accessor path.
 * `isBox` survives ONLY to pick the sphere's closest-point shortcut (`closestPointBox` vs the general
 * `closestPointOnHull`); the hot SAT/clip accessors never read it.
 */
const Convex = d
    .struct({ isBox: d.u32, pos: d.vec3f, quat: d.vec4f, scale: d.vec3f, hr: HullRef })
    .$name("Convex");

const boxPoly = tgpu
    .fn(
        [d.vec3f, d.vec4f, d.vec3f],
        Convex,
    )((pos, quat, size) => {
        "use gpu";
        return Convex({
            isBox: 1,
            pos: pos,
            quat: quat,
            scale: std.mul(size, 0.5),
            hr: hullRef(UNIT_CUBE_ID),
        });
    })
    .$name("boxPoly");

const hullPoly = tgpu
    .fn(
        [d.vec3f, d.vec4f, d.u32],
        Convex,
    )((pos, quat, id) => {
        "use gpu";
        return Convex({
            isBox: 0,
            pos: pos,
            quat: quat,
            scale: d.vec3f(1, 1, 1),
            hr: hullRef(id),
        });
    })
    .$name("hullPoly");

/** the shape-tagged constructor the collide pass calls: shape 3 = a registered hull, else a box.
 *  @internal exported for the step.ts hull/rounded-poly collide-wrapper kernels (3b-iii), which
 *  build a {@link Convex} directly rather than splicing this by name into raw WGSL. */
export const polyMake = tgpu
    .fn(
        [d.u32, d.vec3f, d.vec4f, d.vec3f, d.u32],
        Convex,
    )((shape, pos, quat, size, hullId) => {
        "use gpu";
        if (shape === 3) return hullPoly(pos, quat, hullId);
        return boxPoly(pos, quat, size);
    })
    .$name("polyMake");

// scale-unified local accessors. The scale vec shapes the unit geometry (1 for a real hull, half-extents
// for a box). A vertex/edge scales directly; a face normal transforms by 1/scale + renormalize (the
// plane's inverse-transpose); the offset is recovered as dot(normal, a scaled vertex on the face) since
// scale moves the plane. For a real hull (scale = 1) these reduce to the raw reads.
const polyVertCount = tgpu
    .fn(
        [Convex],
        d.u32,
    )((p) => {
        "use gpu";
        return p.hr.vertCount;
    })
    .$name("polyVertCount");

const polyVertLocal = tgpu
    .fn(
        [Convex, d.u32],
        d.vec3f,
    )((p, i) => {
        "use gpu";
        return std.mul(hVertL(p.hr, i), p.scale);
    })
    .$name("polyVertLocal");

const polyFaceCount = tgpu
    .fn(
        [Convex],
        d.u32,
    )((p) => {
        "use gpu";
        return p.hr.faceCount;
    })
    .$name("polyFaceCount");

const polyFaceNormalLocal = tgpu
    .fn(
        [Convex, d.u32],
        d.vec3f,
    )((p, f) => {
        "use gpu";
        return std.normalize(std.div(hFaceNormalL(p.hr, f), p.scale));
    })
    .$name("polyFaceNormalLocal");

const polyFaceVertCount = tgpu
    .fn(
        [Convex, d.u32],
        d.u32,
    )((p, f) => {
        "use gpu";
        return hFaceVertCount(p.hr, f);
    })
    .$name("polyFaceVertCount");

const polyFaceVert = tgpu
    .fn(
        [Convex, d.u32, d.u32],
        d.u32,
    )((p, f, j) => {
        "use gpu";
        return hFaceVert(p.hr, f, j);
    })
    .$name("polyFaceVert");

const polyEdgeCount = tgpu
    .fn(
        [Convex],
        d.u32,
    )((p) => {
        "use gpu";
        return p.hr.edgeCount;
    })
    .$name("polyEdgeCount");

const polyEdgeLocal = tgpu
    .fn(
        [Convex, d.u32],
        d.vec3f,
    )((p, e) => {
        "use gpu";
        return std.mul(hEdgeL(p.hr, e), p.scale);
    })
    .$name("polyEdgeLocal");

const polyVertW = tgpu
    .fn(
        [Convex, d.u32],
        d.vec3f,
    )((p, i) => {
        "use gpu";
        return std.add(p.pos, satQRotate(p.quat, polyVertLocal(p, i)));
    })
    .$name("polyVertW");

const polyFaceNormalW = tgpu
    .fn(
        [Convex, d.u32],
        d.vec3f,
    )((p, f) => {
        "use gpu";
        return satQRotate(p.quat, polyFaceNormalLocal(p, f));
    })
    .$name("polyFaceNormalW");

const polyEdgeW = tgpu
    .fn(
        [Convex, d.u32],
        d.vec3f,
    )((p, e) => {
        "use gpu";
        return satQRotate(p.quat, polyEdgeLocal(p, e));
    })
    .$name("polyEdgeW");

/** the polytope's [min, max] extent along `axis` */
const projectPoly = tgpu
    .fn(
        [Convex, d.vec3f],
        d.vec2f,
    )((p, axis) => {
        "use gpu";
        // dot(pos + R·v, axis) = dot(pos, axis) + dot(v, Rᵀ·axis): rotate the axis into the convex's local frame
        // ONCE, then dot with local verts — no per-vertex quat rotation in the hot SAT loop (a math identity,
        // and far less inlined code per axis test → big collideHull compile + runtime win; gpu.md hot loops).
        const axisL = satQRotate(satQConj(p.quat), axis);
        const base = std.dot(p.pos, axis);
        let mn = d.f32(1e30);
        let mx = d.f32(NO_AXIS);
        const n = polyVertCount(p);
        for (let i = d.u32(0); i < n; i++) {
            const dd = base + std.dot(polyVertLocal(p, i), axisL);
            mn = std.min(mn, dd);
            mx = std.max(mx, dd);
        }
        return d.vec2f(mn, mx);
    })
    .$name("projectPoly");

/** the face whose world normal is most aligned with `outward` */
const bestPolyFace = tgpu
    .fn(
        [Convex, d.vec3f],
        d.u32,
    )((p, outward) => {
        "use gpu";
        let idx = d.u32(0);
        let best = d.f32(NO_AXIS);
        const n = polyFaceCount(p);
        for (let f = d.u32(0); f < n; f++) {
            const dd = std.dot(polyFaceNormalW(p, f), outward);
            if (dd > best) {
                best = dd;
                idx = f;
            }
        }
        return idx;
    })
    .$name("bestPolyFace");

const hullCoreChunk = chunk(
    "hullCoreWgsl",
    [
        // the rounded × polytope closest-point primitives: only this half of the narrowphase calls them,
        // so they compile out of the box + rounded pipelines (the DXC split's whole point)
        BoxClosest,
        closestPointBox,
        HullRef,
        hullRef,
        hVertL,
        hFaceNormalL,
        hFaceOffset,
        hFaceVertCount,
        hFaceVert,
        hEdgeL,
        Convex,
        polyMake,
        polyVertCount,
        polyVertLocal,
        polyFaceCount,
        polyFaceVertCount,
        polyFaceVert,
        polyEdgeCount,
        polyVertW,
        polyFaceNormalW,
        polyEdgeW,
        projectPoly,
        bestPolyFace,
    ],
    spliceNs,
);

/**
 * the shared scale-unified polytope substrate (`Convex` + the `hullData` readers + `polyMake` +
 * `projectPoly` + `bestPolyFace`). Splice after {@link helpersWgsl} and ahead of
 * {@link hullSatWgsl} / {@link roundedPolyWgsl}; the consumer declares
 * `var<storage, read> hullData: array<u32>` (packed by `packHulls`).
 */
export function hullCoreWgsl(): string {
    helpersWgsl();
    return hullCoreChunk();
}

// ── collideHull: the polytope × polytope SAT (box×hull, hull×hull) ───────────────────────────────
// Function-private working sets stay small (gpu.md "keep function-private working sets small" — large
// dynamically-indexed private arrays blow up DXC register allocation + risk the Metal multi-lane miscompile
// the box SAT hit). The intermediate Sutherland-Hodgman polygon can reach ~12 verts (an 8-gon cone face
// clipped by a box); the post-band candidate set the reduce sees is ≤ MAX_CANDIDATES (the box path's 8 —
// measured max 8, the full cone octagon), so the candidate + reduce arrays REUSE the box path's
// MAX_CANDIDATES + pruneContacts rather than a wider duplicate.

/** the winning hull SAT axis: a face axis (`fromA`) or an edge×edge axis (`isEdge`), oriented A → B,
 *  kept at max separation (the MTV) — the hull analog of {@link SatAxis} */
const HullAxis = d
    .struct({ isEdge: d.bool, fromA: d.bool, sep: d.f32, normalAB: d.vec3f, valid: d.bool })
    .$name("HullAxis");

const supportPoly = tgpu
    .fn(
        [Convex, d.vec3f],
        d.vec3f,
    )((p, dir) => {
        "use gpu";
        let best = d.vec3f();
        let bestD = d.f32(NO_AXIS);
        const n = polyVertCount(p);
        for (let i = d.u32(0); i < n; i++) {
            const w = polyVertW(p, i);
            const dd = std.dot(w, dir);
            if (dd > bestD) {
                bestD = dd;
                best = d.vec3f(w);
            }
        }
        return best;
    })
    .$name("supportPoly");

const testPolyAxis = tgpu
    .fn(
        [Convex, Convex, d.vec3f, d.vec3f, d.bool, d.bool, d.ptrFn(HullAxis), d.vec3f],
        d.bool,
    )((a, b, delta, axis, isEdge, fromA, best, dRel) => {
        "use gpu";
        const lenSq = std.dot(axis, axis);
        if (lenSq < SAT_AXIS_EPSILON) return true;
        let n = std.mul(axis, 1 / std.sqrt(lenSq));
        if (std.dot(n, delta) < 0) n = std.neg(n);
        const pa = projectPoly(a, n);
        const pb = projectPoly(b, n);
        const separation = -std.min(pa.y - pb.x, pb.y - pa.x); // overlap of the intervals, negated
        // unilateral speculative/swept band (Phase 4.8.3/4.8.4): abort only past max(static skin, closing)
        if (separation > std.max(SPECULATIVE_DISTANCE, std.max(0, std.dot(dRel, n)))) return false;
        if (!best.$.valid || separation > best.$.sep) {
            best.$.valid = true;
            best.$.isEdge = isEdge;
            best.$.fromA = fromA;
            best.$.sep = separation;
            best.$.normalAB = d.vec3f(n);
        }
        return true;
    })
    .$name("testPolyAxis");

// Sutherland-Hodgman: clip inN verts of inV to the back of (dot(nrm,x) <= d) → outV; returns out count.
const clipPolyPlane = tgpu
    .fn(
        [d.ptrFn(ClipVerts), d.u32, d.vec3f, d.f32, d.ptrFn(ClipVerts)],
        d.u32,
    )((inV, inN, nrm, dPlane, outV) => {
        "use gpu";
        let outN = d.u32(0);
        if (inN === 0) return 0;
        let a = d.vec3f(inV.$[inN - 1]);
        let da = std.dot(nrm, a) - dPlane;
        for (let i = d.u32(0); i < inN; i++) {
            const b = d.vec3f(inV.$[i]);
            const db = std.dot(nrm, b) - dPlane;
            const aIn = da <= PLANE_EPSILON;
            const bIn = db <= PLANE_EPSILON;
            if (aIn !== bIn) {
                let t = d.f32(0);
                const denom = da - db;
                if (std.abs(denom) > SAT_AXIS_EPSILON) t = std.clamp(da / denom, 0, 1);
                if (outN < MAX_HULL_CLIP) {
                    outV.$[outN] = std.add(a, std.mul(std.sub(b, a), t));
                    outN = outN + 1;
                }
            }
            if (bIn && outN < MAX_HULL_CLIP) {
                outV.$[outN] = d.vec3f(b);
                outN = outN + 1;
            }
            a = d.vec3f(b);
            da = db;
        }
        return outN;
    })
    .$name("clipPolyPlane");

/**
 * convex-hull SAT (box × hull / hull × hull). `count == 0` = no overlap past the band. A box fed as a
 * {@link Convex} reproduces `collideBoxBox`'s face manifold (the box-as-hull gate). Mirrors
 * tests/avbd/hull.ts collideHull.
 * @internal exported for the step.ts hull collide-wrapper kernel (3b-iii) — called directly, not
 * spliced by name; the hull-data readers underneath stay raw WGSL (2b-ii), so this stays GPU-only.
 */
export const collideHull = tgpu
    .fn(
        [Convex, Convex, d.vec3f],
        SatResult,
    )((a, b, dRel) => {
        "use gpu";
        const res = satZero();
        const delta = std.sub(b.pos, a.pos);

        const bestFaceAxis = HullAxis({
            isEdge: false,
            fromA: false,
            sep: NO_AXIS,
            normalAB: d.vec3f(),
            valid: false,
        });
        bestFaceAxis.valid = false;
        bestFaceAxis.sep = NO_AXIS;
        const bestEdgeAxis = HullAxis({
            isEdge: false,
            fromA: false,
            sep: NO_AXIS,
            normalAB: d.vec3f(),
            valid: false,
        });
        bestEdgeAxis.valid = false;
        bestEdgeAxis.sep = NO_AXIS;

        const fa = polyFaceCount(a);
        const fb = polyFaceCount(b);
        for (let i = d.u32(0); i < fa; i++) {
            if (
                !testPolyAxis(
                    a,
                    b,
                    delta,
                    polyFaceNormalW(a, i),
                    false,
                    true,
                    d.ref(bestFaceAxis),
                    dRel,
                )
            )
                return res;
        }
        for (let i = d.u32(0); i < fb; i++) {
            if (
                !testPolyAxis(
                    a,
                    b,
                    delta,
                    polyFaceNormalW(b, i),
                    false,
                    false,
                    d.ref(bestFaceAxis),
                    dRel,
                )
            )
                return res;
        }
        const ea = polyEdgeCount(a);
        const eb = polyEdgeCount(b);
        for (let i = d.u32(0); i < ea; i++) {
            const eai = polyEdgeW(a, i);
            for (let j = d.u32(0); j < eb; j++) {
                const axis = std.cross(eai, polyEdgeW(b, j));
                if (!testPolyAxis(a, b, delta, axis, true, false, d.ref(bestEdgeAxis), dRel))
                    return res;
            }
        }
        if (!bestFaceAxis.valid) return res;

        let best = HullAxis(bestFaceAxis);
        const edgeScore = d.f32(0.95) * bestEdgeAxis.sep;
        const faceScore = bestFaceAxis.sep + d.f32(0.01);
        const scoreScale = std.max(std.abs(edgeScore), std.abs(faceScore));
        if (bestEdgeAxis.valid && edgeScore > faceScore + scoreScale * F32_TIE_REL)
            best = HullAxis(bestEdgeAxis);

        const normalBA = std.neg(best.normalAB);
        res.basis = orthoBasis(normalBA);
        const band = std.max(SPECULATIVE_DISTANCE, std.max(0, std.dot(dRel, best.normalAB)));

        // reference = the winning face's hull (an edge axis defaults to A, like the oracle)
        const referenceIsA = best.isEdge || best.fromA;
        let refP = Convex(b);
        let incP = Convex(a);
        let refOutward = d.vec3f(normalBA);
        if (referenceIsA) {
            refP = Convex(a);
            incP = Convex(b);
            refOutward = d.vec3f(best.normalAB);
        }

        const refFaceIdx = bestPolyFace(refP, refOutward);
        const refNormalW = polyFaceNormalW(refP, refFaceIdx);
        const refV0 = polyVertW(refP, polyFaceVert(refP, refFaceIdx, 0));
        const refPlaneW = std.dot(refNormalW, refV0);

        // incident face selected by the CONTACT normal (−refOutward), not the reference face normal — they
        // diverge on an edge MTV; the contact-normal choice keeps a slanted edge-MTV clip non-empty (Bullet).
        const incFaceIdx = bestPolyFace(incP, std.neg(refOutward));
        const polyA = ClipVerts();
        const polyB = ClipVerts();
        const incN0 = polyFaceVertCount(incP, incFaceIdx);
        let n = std.min(incN0, MAX_HULL_CLIP);
        for (let j = d.u32(0); j < n; j++) {
            polyA[j] = polyVertW(incP, polyFaceVert(incP, incFaceIdx, j));
        }

        // clip against the reference face's side planes (inward), ping-ponging polyA <-> polyB
        const refLoopN = polyFaceVertCount(refP, refFaceIdx);
        let src = d.u32(0); // 0 = polyA holds the live polygon, 1 = polyB holds it
        for (let e = d.u32(0); e < refLoopN && n > 0; e++) {
            const va = polyVertW(refP, polyFaceVert(refP, refFaceIdx, e));
            const vb = polyVertW(refP, polyFaceVert(refP, refFaceIdx, (e + 1) % refLoopN));
            const planeN = std.neg(std.cross(std.sub(va, vb), refNormalW));
            const pd = std.dot(planeN, va);
            if (src === 0) {
                n = clipPolyPlane(d.ref(polyA), n, planeN, pd, d.ref(polyB));
                src = 1;
            } else {
                n = clipPolyPlane(d.ref(polyB), n, planeN, pd, d.ref(polyA));
                src = 0;
            }
        }
        if (n === 0) return res;

        const featurePrefix =
            (std.select(d.u32(HULL_FACE_TAG), d.u32(HULL_EDGE_TAG), best.isEdge) << d.u32(24)) |
            ((refFaceIdx & d.u32(0xff)) << d.u32(16)) |
            ((incFaceIdx & d.u32(0xff)) << d.u32(8));
        // the within-band candidate set the reduce sees is <= MAX_CANDIDATES (8 — the measured max, a full cone
        // octagon), so it shares the box path's array bound + pruneContacts; the dedup midpoint is recomputed
        // from the kept candidates, no separate midpoint array (gpu.md small function-private sets).
        const candXA = CandVerts();
        const candXB = CandVerts();
        const candFeat = CandFeats();
        let candN = d.u32(0);
        for (let i = d.u32(0); i < n && candN < MAX_CANDIDATES; i++) {
            const pInc = std.select(polyB[i], polyA[i], src === 0);
            const depth = std.dot(refNormalW, pInc) - refPlaneW;
            if (depth > band) continue;
            const pRef = std.sub(pInc, std.mul(refNormalW, depth));
            let xA = d.vec3f(pRef);
            let xB = d.vec3f(pInc);
            if (!referenceIsA) {
                xA = d.vec3f(pInc);
                xB = d.vec3f(pRef);
            }
            const mid = std.mul(std.add(xA, xB), 0.5);
            let dup = false;
            for (let k = d.u32(0); k < MAX_CANDIDATES; k++) {
                if (k >= candN) break;
                const dd = std.sub(mid, std.mul(std.add(candXA[k], candXB[k]), 0.5));
                if (std.dot(dd, dd) < CONTACT_MERGE_DIST_SQ) {
                    dup = true;
                    break;
                }
            }
            if (dup) continue;
            candXA[candN] = d.vec3f(xA);
            candXB[candN] = d.vec3f(xB);
            candFeat[candN] = featurePrefix | (i & d.u32(0xff));
            candN = candN + 1;
        }

        // no candidate survived (a grazing edge MTV) — fall back to a single support-point contact
        if (candN === 0) {
            const xA = supportPoly(a, best.normalAB);
            const xB = supportPoly(b, std.neg(best.normalAB));
            addContact(d.ref(res), a.pos, a.quat, b.pos, b.quat, xA, xB, featurePrefix);
            return res;
        }

        const sel = SelIdx();
        let selN = d.u32(0);
        if (candN > MAX_CONTACTS)
            selN = pruneContacts(
                d.ref(candXA),
                d.ref(candXB),
                d.ref(candFeat),
                candN,
                best.normalAB,
                a.pos,
                d.ref(sel),
            );
        else {
            selN = candN;
            for (let i = d.u32(0); i < candN; i++) sel[i] = i;
        }
        for (let i = d.u32(0); i < selN; i++) {
            const s = sel[i];
            addContact(d.ref(res), a.pos, a.quat, b.pos, b.quat, candXA[s], candXB[s], candFeat[s]);
        }
        return res;
    })
    .$name("collideHull");

const hullSatChunk = chunk("hullSatWgsl", [collideHull], spliceNs);

/**
 * `collideHull`, the polytope × polytope SAT (box × hull, hull × hull). Splice after
 * {@link helpersWgsl} + {@link hullCoreWgsl}; it compiles as its own pipeline so the big face/edge SAT
 * never pays the rounded segment-clip's compile cost.
 */
export function hullSatWgsl(): string {
    hullCoreWgsl();
    return hullSatChunk();
}

// ── collideRoundedPolytope: sphere / capsule × box / hull ────────────────────────────────────────

const pointInFaceLocal = tgpu
    .fn(
        [HullRef, d.u32, d.vec3f],
        d.bool,
    )((h, f, pt) => {
        "use gpu";
        const nf = hFaceNormalL(h, f);
        const cnt = hFaceVertCount(h, f);
        for (let i = d.u32(0); i < cnt; i++) {
            const va = hVertL(h, hFaceVert(h, f, i));
            const vb = hVertL(h, hFaceVert(h, f, (i + 1) % cnt));
            if (std.dot(std.cross(std.sub(vb, va), std.sub(pt, va)), nf) < -PLANE_EPSILON)
                return false;
        }
        return true;
    })
    .$name("pointInFaceLocal");

/** closest point on the segment [a,b] to `pt`.
 *  @example const cp = closestOnSeg(query, a, b); */
const closestOnSeg = tgpu
    .fn(
        [d.vec3f, d.vec3f, d.vec3f],
        d.vec3f,
    )((pt, a, b) => {
        "use gpu";
        const ab = std.sub(b, a);
        const len2 = std.dot(ab, ab);
        if (len2 < SAT_AXIS_EPSILON) return d.vec3f(a);
        return std.add(a, std.mul(ab, std.clamp(std.dot(std.sub(pt, a), ab) / len2, 0, 1)));
    })
    .$name("closestOnSeg");

// closest point on a hull to a query in the hull's LOCAL frame (q already de-rotated by the caller). Only
// called on hull Convexes (a box uses closestPointBox), so it reads hull-direct (branch-free).
const closestPointOnHull = tgpu
    .fn(
        [HullRef, d.vec3f],
        BoxClosest,
    )((h, q) => {
        "use gpu";
        let maxD = d.f32(NO_AXIS);
        let maxFace = d.u32(0);
        const fn0 = h.faceCount;
        for (let f = d.u32(0); f < fn0; f++) {
            const dd = std.dot(hFaceNormalL(h, f), q) - hFaceOffset(h, f);
            if (dd > maxD) {
                maxD = dd;
                maxFace = f;
            }
        }
        if (maxD <= 0) {
            // inside — push out along the least-penetrating face
            const nf = hFaceNormalL(h, maxFace);
            return BoxClosest({
                point: std.sub(q, std.mul(nf, maxD)),
                normal: nf,
                signedDist: maxD,
            });
        }
        let bestDist = d.f32(1e30);
        let bestPoint = d.vec3f(q);
        for (let f = d.u32(0); f < fn0; f++) {
            const dd = std.dot(hFaceNormalL(h, f), q) - hFaceOffset(h, f);
            if (dd <= 0) continue;
            const proj = std.sub(q, std.mul(hFaceNormalL(h, f), dd));
            if (pointInFaceLocal(h, f, proj) && dd < bestDist) {
                bestDist = dd;
                bestPoint = d.vec3f(proj);
            }
        }
        for (let f = d.u32(0); f < fn0; f++) {
            const cnt = hFaceVertCount(h, f);
            for (let i = d.u32(0); i < cnt; i++) {
                const cp = closestOnSeg(
                    q,
                    hVertL(h, hFaceVert(h, f, i)),
                    hVertL(h, hFaceVert(h, f, (i + 1) % cnt)),
                );
                const dd = std.length(std.sub(q, cp));
                if (dd < bestDist) {
                    bestDist = dd;
                    bestPoint = d.vec3f(cp);
                }
            }
        }
        for (let i = d.u32(0); i < h.vertCount; i++) {
            const v = hVertL(h, i);
            const dd = std.length(std.sub(q, v));
            if (dd < bestDist) {
                bestDist = dd;
                bestPoint = d.vec3f(v);
            }
        }
        const diff = std.sub(q, bestPoint);
        const dist = std.length(diff);
        let normal = hFaceNormalL(h, maxFace);
        if (dist > ROUND_NORMAL_EPS) normal = std.div(diff, dist);
        return BoxClosest({ point: bestPoint, normal: normal, signedDist: dist });
    })
    .$name("closestPointOnHull");

const closestPointOnPoly = tgpu
    .fn(
        [Convex, d.vec3f],
        BoxClosest,
    )((p, q) => {
        "use gpu";
        // box: scale = half-extents; the exact clamp shortcut
        if (p.isBox === 1) return closestPointBox(q, p.scale);
        return closestPointOnHull(p.hr, q);
    })
    .$name("closestPointOnPoly");

/** one capsule × polytope manifold: up to 2 contacts sharing the reference-face normal */
const CapHits = d
    .struct({
        count: d.u32,
        /** hull → capsule (world, outward) */
        normal: d.vec3f,
        /** capsule core point (world) */
        core: Vec3x2,
        /** hull surface point (world) */
        surf: Vec3x2,
        ord: d.arrayOf(d.u32, 2),
    })
    .$name("CapHits");

const considerCapAxisFn = tgpu
    .fn([d.vec3f, d.vec3f, d.vec3f, d.vec3f, Convex, d.ptrFn(d.f32), d.ptrFn(d.vec3f)])(
        (raw, e0, e1, toCap, p, bestSep, bestN) => {
            "use gpu";
            const l2 = std.dot(raw, raw);
            if (l2 < SAT_AXIS_EPSILON) return;
            let ax = std.mul(raw, 1 / std.sqrt(l2));
            if (std.dot(ax, toCap) < 0) ax = std.neg(ax); // orient hull → capsule
            const a = std.dot(e0, ax);
            const b = std.dot(e1, ax);
            const sMin = std.min(a, b);
            const sMax = std.max(a, b);
            const hp = projectPoly(p, ax);
            const sep = -std.min(sMax - hp.x, hp.y - sMin);
            if (sep > bestSep.$) {
                bestSep.$ = sep;
                bestN.$ = d.vec3f(ax);
            }
        },
    )
    .$name("considerCapAxis");

// the widened call signature is the `uniformLoad` / `compareExchange` typing gap (engine/utils/tgsl.ts):
// `d.ref` refuses a scalar ("cannot take a reference to a scalar value"), so the running-best separation
// can only be passed bare — and the transpiler has to see the `tgpu.fn` call at the site, so a JS
// forwarder can't narrow it either. Two pointers, matching the reference's `bestSep` / `bestN`.
const considerCapAxis = considerCapAxisFn as typeof considerCapAxisFn &
    ((
        raw: d.v3f,
        e0: d.v3f,
        e1: d.v3f,
        toCap: d.v3f,
        p: d.Infer<typeof Convex>,
        bestSep: number,
        bestN: d.v3f,
    ) => void);

// capsule (core segment [e0,e1] world, radius) × polytope — the §6.3 segment-clip manifold (up to 2
// contacts sharing the reference-face normal), the mid-segment case endpoint sampling misses. The contact
// normal is a mini-SAT between the core segment and the hull (robust to an overhanging capsule), NOT the
// endpoints' closest normals. Mirrors hull.ts capsuleHull. dRel = (vCap − vPoly)·dt.
const capsulePoly = tgpu
    .fn(
        [d.vec3f, d.vec3f, d.f32, Convex, d.vec3f],
        CapHits,
    )((e0, e1, radius, p, dRel) => {
        "use gpu";
        const out = CapHits({
            count: 0,
            normal: d.vec3f(0, 1, 0),
            core: Vec3x2(),
            surf: Vec3x2(),
            ord: d.arrayOf(d.u32, 2)(),
        });
        out.count = 0;
        out.normal = d.vec3f(0, 1, 0);
        const dir = std.sub(e1, e0);
        const toCap = std.sub(std.mul(std.add(e0, e1), 0.5), p.pos);

        let bestSep = d.f32(NO_AXIS);
        let n = d.vec3f(0, 1, 0);
        // typegpu declares a local `let` unless the body assigns it, and `&` on a `let` is invalid WGSL:
        // these two are written only through `considerCapAxis`'s pointers, and they can't go through
        // `d.ref` (it refuses a scalar), so the redundant self-assignments force `var`. Pinned by
        // `pointerDiscipline` (tests/wgsl.ts)
        bestSep = NO_AXIS;
        n = d.vec3f(0, 1, 0);
        const fn0 = polyFaceCount(p);
        for (let f = d.u32(0); f < fn0; f++) {
            considerCapAxis(polyFaceNormalW(p, f), e0, e1, toCap, p, bestSep, n);
        }
        const en0 = polyEdgeCount(p);
        for (let e = d.u32(0); e < en0; e++) {
            considerCapAxis(std.cross(dir, polyEdgeW(p, e)), e0, e1, toCap, p, bestSep, n);
        }

        if (bestSep - radius > std.max(SPECULATIVE_DISTANCE, std.max(0, -std.dot(dRel, n)))) {
            out.normal = d.vec3f(n);
            return out;
        }

        const refIdx = bestPolyFace(p, n);
        const nf = polyFaceNormalW(p, refIdx);
        const refCenter = polyVertW(p, polyFaceVert(p, refIdx, 0));
        const bandF = std.max(SPECULATIVE_DISTANCE, std.max(0, -std.dot(dRel, nf)));

        // clip the segment's parameter range to the reference face's side-plane prism (extruded along nf)
        let t0 = d.f32(0);
        let t1 = d.f32(1);
        let outside = false;
        const cnt = polyFaceVertCount(p, refIdx);
        for (let i = d.u32(0); i < cnt; i++) {
            const va = polyVertW(p, polyFaceVert(p, refIdx, i));
            const vb = polyVertW(p, polyFaceVert(p, refIdx, (i + 1) % cnt));
            const planeN = std.neg(std.cross(std.sub(va, vb), nf)); // inward
            const d0 = std.dot(planeN, e0) - std.dot(planeN, va);
            const dd = std.dot(planeN, dir);
            if (std.abs(dd) < SAT_AXIS_EPSILON) {
                if (d0 > PLANE_EPSILON) {
                    outside = true;
                    break;
                }
                continue;
            }
            const tc = -d0 / dd;
            if (dd > 0) t1 = std.min(t1, tc);
            else t0 = std.max(t0, tc);
        }

        if (!outside && t1 > t0 + 1e-9) {
            for (let k = d.u32(0); k < 2; k++) {
                const tt = std.select(t1, t0, k === 0);
                const sp = std.add(e0, std.mul(dir, tt));
                const distPlane = std.dot(std.sub(sp, refCenter), nf);
                if (distPlane - radius > bandF) continue;
                out.core[out.count] = d.vec3f(sp);
                out.surf[out.count] = std.sub(sp, std.mul(nf, distPlane));
                out.ord[out.count] = k;
                out.count = out.count + 1;
            }
            if (out.count > 0) {
                out.normal = d.vec3f(nf);
                return out;
            }
        }

        // fallback (segment off the reference face — an edge/vertex contact): per-endpoint closest points,
        // the reference-face normal shared
        out.normal = d.vec3f(nf);
        const qc = satQConj(p.quat);
        for (let k = d.u32(0); k < 2; k++) {
            const e = std.select(e1, e0, k === 0);
            const r = closestPointOnPoly(p, satQRotate(qc, std.sub(e, p.pos)));
            if (r.signedDist - radius > bandF) continue;
            out.core[out.count] = d.vec3f(e);
            out.surf[out.count] = std.add(p.pos, satQRotate(p.quat, r.point));
            out.ord[out.count] = k;
            out.count = out.count + 1;
        }
        return out;
    })
    .$name("capsulePoly");

// rounded (sphere/capsule) × polytope with the round shape as A. Sphere = closest point on the polytope
// (1 contact); capsule = the segment-clip (up to 2). The stored arm anchors the round CORE + polytope
// SURFACE; the radius offset is re-applied along the fixed normal at solve time. Mirrors the oracle
// collideRoundedPolytope. The collide pass orients box/hull-as-A by swapping (collideRoundedPolytope).
const collideRoundedPolyA = tgpu
    .fn(
        [d.vec3f, d.vec4f, d.vec3f, d.f32, d.u32, Convex, d.vec3f],
        SatResult,
    )((rPos, rQuat, rSize, rRad, rShape, p, dRel) => {
        "use gpu";
        const res = satZero();
        if (rShape === 1) {
            // sphere — sample its centre
            const qc = satQConj(p.quat);
            const cp = closestPointOnPoly(p, satQRotate(qc, std.sub(rPos, p.pos)));
            const normal = satQRotate(p.quat, cp.normal); // polytope → sphere (B → A)
            const band = std.max(SPECULATIVE_DISTANCE, std.max(0, -std.dot(dRel, normal)));
            if (cp.signedDist - rRad > band) return res;
            const surf = std.add(p.pos, satQRotate(p.quat, cp.point));
            res.basis = orthoBasis(normal);
            // sphere core = centre
            addContact(d.ref(res), rPos, rQuat, p.pos, p.quat, rPos, surf, ROUND_FEATURE);
            return res;
        }
        const hh = satQRotate(rQuat, std.mul(rSize, 0.5));
        const hits = capsulePoly(std.sub(rPos, hh), std.add(rPos, hh), rRad, p, dRel);
        if (hits.count === 0) return res;
        res.basis = orthoBasis(hits.normal);
        for (let k = d.u32(0); k < hits.count; k++) {
            addContact(
                d.ref(res),
                rPos,
                rQuat,
                p.pos,
                p.quat,
                hits.core[k],
                hits.surf[k],
                ROUND_FEATURE | hits.ord[k],
            );
        }
        return res;
    })
    .$name("collideRoundedPolyA");

/**
 * mixed rounded × polytope, A/B oriented like `collideBoxBox` (A = ia, B = ib). If A is the round
 * shape, collide directly; else (the polytope is A) collide round-as-A on the flipped sweep, then swap
 * the arms + flip the normal back into our A/B convention. Mirrors the oracle narrowphase branches.
 * @internal exported for the step.ts rounded-poly collide-wrapper kernel (3b-iii) — called directly.
 */
export const collideRoundedPolytope = tgpu
    .fn(
        [
            d.vec3f,
            d.vec4f,
            d.vec3f,
            d.f32,
            d.u32,
            d.u32,
            d.vec3f,
            d.vec4f,
            d.vec3f,
            d.f32,
            d.u32,
            d.u32,
            d.vec3f,
        ],
        SatResult,
    )((aPos, aQuat, aSize, aRad, aShape, aHull, bPos, bQuat, bSize, bRad, bShape, bHull, dRel) => {
        "use gpu";
        const roundedA = aShape === 1 || aShape === 2;
        if (roundedA) {
            const poly = polyMake(bShape, bPos, bQuat, bSize, bHull);
            return collideRoundedPolyA(aPos, aQuat, aSize, aRad, aShape, poly, dRel);
        }
        const poly = polyMake(aShape, aPos, aQuat, aSize, aHull);
        const r = collideRoundedPolyA(bPos, bQuat, bSize, bRad, bShape, poly, std.neg(dRel));
        const res = satZero();
        res.count = r.count;
        res.basis = orthoBasis(std.neg(r.basis.r0)); // B → A in our convention = −(poly → round)
        for (let k = d.u32(0); k < r.count; k++) {
            res.feat[k] = r.feat[k];
            res.rA[k] = d.vec3f(r.rB[k]); // our A = poly = r's B
            res.rB[k] = d.vec3f(r.rA[k]);
        }
        return res;
    })
    .$name("collideRoundedPolytope");

const roundedPolyChunk = chunk("roundedPolyWgsl", [collideRoundedPolytope], spliceNs);

/**
 * `collideRoundedPolytope`, sphere/capsule × box/hull (the capsule segment-clip + the closest-point
 * primitives). Splice after {@link helpersWgsl} + {@link hullCoreWgsl}; its own pipeline, apart from
 * {@link hullSatWgsl}.
 */
export function roundedPolyWgsl(): string {
    hullCoreWgsl();
    return roundedPolyChunk();
}

/**
 * the full box + rounded narrowphase (helpers + box SAT + rounded), recomposed for the gym kernel
 * gates (`sat` / `pile` rounded) + the `avbd/core` re-export. The production collide pass (step.ts)
 * splices the per-class chunks instead, so a box-only pipeline never compiles the rounded SAT.
 */
export function collideWgsl(): string {
    return helpersWgsl() + boxBoxWgsl() + roundedWgsl();
}

/**
 * the full convex-hull narrowphase recomposed (`hullCoreWgsl` + `collideHull` +
 * `collideRoundedPolytope`), for the gym kernel gates that compile a single kernel calling both. It
 * needs only {@link helpersWgsl} ahead of it — a hull-only consumer splices those two and never compiles
 * the box or rounded SAT — and declares `var<storage, read> hullData: array<u32>`.
 */
export function hullWgsl(): string {
    return hullCoreWgsl() + hullSatWgsl() + roundedPolyWgsl();
}
