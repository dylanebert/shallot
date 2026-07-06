// Port of reference/avbd-demo3d/source/spring.cpp — the soft distance `Force`, the first
// non-contact constraint (roadmap "Phase 6.1 — Springs"). A spring is a single-row soft
// constraint between two body-local anchors: `C = ‖pA − pB‖ − rest`, force `f = stiffness·C`
// along the spring axis. Finite stiffness ⇒ no dual variable (λ forced to 0), so `updateDual`
// is a no-op and the penalty never ramps — the simplest member of the `Force` family the
// contact manifold (manifold.ts) defines (initialize / updatePrimal / updateDual). f64; the
// reference is the executable spec. Test scaffolding, kept out of the shipped src/.

import type { System } from "./manifold";
import {
    add,
    cross,
    length,
    type Mat3,
    neg,
    outer,
    rotate,
    scale,
    scaleM,
    sub,
    transform,
    type Vec3,
} from "./math";
import type { Body } from "./rigid";

export interface Spring {
    a: Body;
    b: Body;
    rA: Vec3; // anchor in a's local frame (relative to center)
    rB: Vec3; // anchor in b's local frame
    rest: number;
    stiffness: number;
}

const addMat = (a: Mat3, b: Mat3): Mat3 => [add(a[0], b[0]), add(a[1], b[1]), add(a[2], b[2])];

/**
 * A spring between two body-local anchors. `rest < 0` rests at the current anchor separation
 * (matching `Spring::Spring`), so a spring constructed at the scene pose holds that pose.
 */
export function spring(a: Body, b: Body, rA: Vec3, rB: Vec3, stiffness: number, rest = -1): Spring {
    let r = rest;
    if (r < 0) {
        const pA = transform(a.posLin, a.posAng, rA);
        const pB = transform(b.posLin, b.posAng, rB);
        r = length(sub(pA, pB));
    }
    return { a, b, rA, rB, rest: r, stiffness };
}

/**
 * Stamp the spring's force + Hessian into `body`'s 6×6 system (`Spring::updatePrimal`). The
 * Jacobian is the single spring axis `n` (linear) and `rWorld × n` (angular); the Hessian is
 * the stiffness-weighted outer products — the single-row form of the contact stamp.
 */
export function stampSpring(sp: Spring, body: Body, sys: System): void {
    const pA = transform(sp.a.posLin, sp.a.posAng, sp.rA);
    const pB = transform(sp.b.posLin, sp.b.posAng, sp.rB);
    const d = sub(pA, pB);
    const dLen = length(d);
    if (dLen <= 1e-6) return;

    const n = scale(d, 1 / dLen);
    const f = sp.stiffness * (dLen - sp.rest);

    const isA = body === sp.a;
    const rWorld = rotate(isA ? sp.a.posAng : sp.b.posAng, isA ? sp.rA : sp.rB);
    const jLin = isA ? n : neg(n);
    const jAng = cross(rWorld, jLin);

    sys.lhsLin = addMat(sys.lhsLin, scaleM(outer(jLin, jLin), sp.stiffness));
    sys.lhsAng = addMat(sys.lhsAng, scaleM(outer(jAng, jAng), sp.stiffness));
    sys.lhsCross = addMat(sys.lhsCross, scaleM(outer(jAng, jLin), sp.stiffness));
    sys.rhsLin = add(sys.rhsLin, scale(jLin, f));
    sys.rhsAng = add(sys.rhsAng, scale(jAng, f));
}
