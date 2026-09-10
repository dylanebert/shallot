// Rounded-shape narrowphase (Phase 6.3 step 1) — sphere + capsule as a core segment + radius.
//
// A sphere is a zero-length segment (a point), a capsule a segment; sphere-sphere / sphere-capsule /
// capsule-capsule all collapse to ONE segment-segment closest-point query (the `closestSegments` the box
// edge-edge path already uses — Ericson RTCD §5.1.9), then subtract the radii → a single contact. Curved
// surfaces meet at a point, so one contact is exact here, not an approximation (roadmap §6.3).
//
// No AVBD/webphysics reference covers this (demo3d is boxes only; webphysics has spheres, no capsule), so
// it's grounded in the production engines that do — Jolt's convex-radius / core-shape idea (a shape = a
// core polytope inflated by a radius; CapsuleShape). The contact it emits has the SAME shape as the box
// SAT's (feature key + local arms + a B→A basis), so the solver (manifold.ts) consumes it unchanged.
//
// The dispatch (`narrowphase`) is by shape pair: rounded × rounded → here; polytope × polytope → the
// box-box SAT (collide.ts) or the hull SAT (hull.ts); rounded × polytope → `collideRoundedPolytope` (sphere
// = closest point on the convex, capsule = the core segment clipped against the reference face, box fed as
// its `boxHull` so box + hull share one path). f64; the GPU twins (collide.ts WGSL) reproduce this.

import { type Box, type Contact, closestSegments, collide, SPECULATIVE_DISTANCE } from "./collide";
import { boxHull, capsuleHull, closestPointOnHull, collideHull, type Hull } from "./hull";
import {
    add,
    dot,
    length,
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
import { type Body, ShapeKind } from "./rigid";

// the rounded contact's feature key base — a fixed tag (kind 3, distinct from the box AXIS_FACE/EDGE
// keys) so warmstart always matches across frames. A rounded-rounded pair is one contact (`ROUND_FEATURE`);
// a capsule-box pair ORs the endpoint index (`ROUND_FEATURE | ep`) so its two contacts carry separately.
export const ROUND_FEATURE = 0x03000000;

// below this core-to-core distance the contact normal is undefined (concentric cores) — fall back to
// world up so the repulsion still separates them. A degenerate deep-overlap guard, not a hot path.
const ROUND_NORMAL_EPS = 1e-9;

const ZERO_BASIS: Mat3 = [
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0],
];

const boxOf = (b: Body): Box => ({ pos: b.posLin, quat: b.posAng, size: b.size });

// world endpoints of a body's core segment: centre ± rotate(quat, size/2). A sphere's size is 0, so both
// endpoints collapse to the centre (a point) — the same query then handles sphere / capsule uniformly.
function coreSegment(b: Body): { p0: Vec3; p1: Vec3 } {
    const h = rotate(b.posAng, scale(b.size, 0.5));
    return { p0: sub(b.posLin, h), p1: add(b.posLin, h) };
}

/**
 * Rounded-rounded contact (sphere/capsule pairs). Returns one contact (0 when separated past the band),
 * the basis with the normal (B→A) in row 0 — the box-SAT convention, so manifold.ts reads it identically.
 * `dRel = (vA−vB)·dt` is the velocity sweep (Phase 4.8.4); `[0,0,0]` recovers the static speculative band.
 */
export function collideRounded(
    a: Body,
    b: Body,
    dRel: Vec3 = [0, 0, 0],
): { contacts: Contact[]; basis: Mat3 } {
    const segA = coreSegment(a);
    const segB = coreSegment(b);
    const { c0: cA, c1: cB } = closestSegments(segA.p0, segA.p1, segB.p0, segB.p1);

    const delta = sub(cA, cB); // core-to-core, points B → A
    const dist = length(delta);
    const normal: Vec3 = dist > ROUND_NORMAL_EPS ? scale(delta, 1 / dist) : [0, 1, 0];

    // the contact is generated while the cores are still separated by up to the speculative band (the
    // surfaces touch at gap 0): the static skin SPECULATIVE_DISTANCE, widened by the closing displacement
    // this step (velocity sweep). `max`, not `+` — a slow body's band stays the velocity-independent skin.
    // The cores approach along −normal, so the closing displacement onto −normal is −dot(dRel, normal).
    const closing = Math.max(0, -dot(dRel, normal));
    const band = Math.max(SPECULATIVE_DISTANCE, closing);

    const gap = dist - a.roundRadius - b.roundRadius;
    if (gap > band) return { contacts: [], basis: ZERO_BASIS };

    // store the local arm to the CORE closest point, NOT the inflated surface point. The radius offset
    // (the −r·normal that reaches the surface) is GEOMETRIC: it must stay along the fixed contact normal,
    // not rotate with the body. A sphere's contact point is not a material point — freezing the surface
    // arm and rotating it by the body's spin injects a spurious tangential lever into the normal Jacobian
    // (jAng·n ≠ 0) → the rotational-instability bug (a spun sphere tunnels / spins up). The solver
    // (manifold.ts contactForce, GPU contactForce) re-applies ±radius·normal at solve time. roadmap §6.3.
    const contact: Contact = {
        feature: ROUND_FEATURE,
        rA: rotate(qconj(a.posAng), sub(cA, a.posLin)),
        rB: rotate(qconj(b.posAng), sub(cB, b.posLin)),
    };
    return { contacts: [contact], basis: orthonormal(normal) };
}

const isRounded = (b: Body): boolean =>
    b.shape === ShapeKind.Sphere || b.shape === ShapeKind.Capsule;

/** the polytope geometry for a non-rounded body: a hull carries its own, a box is its `boxHull` */
const polyOf = (b: Body): Hull => (b.shape === ShapeKind.Hull ? b.hull! : boxHull(b.size));

/**
 * Rounded × polytope (sphere/capsule vs box or hull) — the round shape is A, the polytope B, so the basis
 * (B→A = polytope→round) is already the manifold orientation. A sphere is the closest point on the hull to
 * its centre → 1 contact ({@link closestPointOnHull}); a capsule is the core segment clipped against the
 * reference face → up to 2 contacts sharing the reference-face normal ({@link capsuleHull}, the decided
 * §6.3 manifold). The stored arm anchors the round CORE (sphere centre / capsule segment point) and the
 * polytope SURFACE; the radius offset is re-applied along the fixed normal at solve time. A box is fed as
 * its `boxHull`, so box + hull share one path. `dRel = (vRound − vPoly)·dt` (the velocity sweep).
 */
function collideRoundedPolytope(
    round: Body,
    h: Hull,
    hp: Vec3,
    hq: Quat,
    dRel: Vec3,
): { contacts: Contact[]; basis: Mat3 } {
    const hqc = qconj(hq);
    if (round.shape === ShapeKind.Sphere) {
        const cp = closestPointOnHull(h, rotate(hqc, sub(round.posLin, hp)));
        const normal = rotate(hq, cp.normal); // polytope → sphere (world, outward = B→A)
        const band = Math.max(SPECULATIVE_DISTANCE, Math.max(0, -dot(dRel, normal)));
        if (cp.signedDist - round.roundRadius > band) return { contacts: [], basis: ZERO_BASIS };
        const surf = add(hp, rotate(hq, cp.point));
        // sphere core = its centre, so its local arm is 0; the polytope arm anchors the surface point
        const contact: Contact = {
            feature: ROUND_FEATURE,
            rA: [0, 0, 0],
            rB: rotate(hqc, sub(surf, hp)),
        };
        return { contacts: [contact], basis: orthonormal(normal) };
    }
    const { p0, p1 } = coreSegment(round);
    const { contacts: hits, normal } = capsuleHull(p0, p1, round.roundRadius, h, hp, hq, dRel);
    if (!hits.length) return { contacts: [], basis: ZERO_BASIS };
    const contacts: Contact[] = hits.map((hit) => ({
        feature: ROUND_FEATURE | hit.ordinal,
        rA: rotate(qconj(round.posAng), sub(hit.capCore, round.posLin)),
        rB: rotate(hqc, sub(hit.hullSurf, hp)),
    }));
    return { contacts, basis: orthonormal(normal) };
}

/**
 * The narrowphase the solver feeds on: dispatch by shape pair. rounded × rounded → {@link collideRounded};
 * polytope × polytope → the box-box SAT ({@link collide}) when both are boxes, else the hull SAT
 * ({@link collideHull}, box fed as its `boxHull`); rounded × polytope → {@link collideRoundedPolytope}.
 * `bodyA` is the higher creation index (the reference orientation), set by the caller (manifold.ts) — so
 * when the polytope is A the round-as-A result is swapped into A's convention (arms swapped, normal flipped).
 */
export function narrowphase(
    a: Body,
    b: Body,
    dRel: Vec3 = [0, 0, 0],
): { contacts: Contact[]; basis: Mat3 } {
    const roundedA = isRounded(a);
    const roundedB = isRounded(b);
    if (roundedA && roundedB) return collideRounded(a, b, dRel);

    if (!roundedA && !roundedB) {
        if (a.shape === ShapeKind.Box && b.shape === ShapeKind.Box)
            return collide(boxOf(a), boxOf(b), dRel);
        return collideHull(polyOf(a), a.posLin, a.posAng, polyOf(b), b.posLin, b.posAng, dRel);
    }

    if (roundedA) return collideRoundedPolytope(a, polyOf(b), b.posLin, b.posAng, dRel); // a=round=A
    // polytope is A: collide round-as-A (dRel flips to round−poly), then swap arms + flip the normal
    const r = collideRoundedPolytope(b, polyOf(a), a.posLin, a.posAng, neg(dRel));
    return {
        contacts: r.contacts.map((c) => ({ feature: c.feature, rA: c.rB, rB: c.rA })),
        basis: orthonormal(neg(r.basis[0])),
    };
}
