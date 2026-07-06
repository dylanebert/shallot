// Port of reference/avbd-demo3d/source/manifold.cpp — the contact `Force`. A manifold holds up
// to 8 frictional contacts between two bodies; `initManifold` re-collides (via `narrowphase`, which
// dispatches box-SAT vs the rounded sphere/capsule path by shape — Phase 6.3) + merges persisted
// contacts by feature key + warmstarts, `stampPrimal` adds the contact's force +
// Hessian into a body's 6×6 system, `updateDual` advances λ + the penalty ramp. This is the
// `Force` interface (initialize / updatePrimal / updateDual) joints + springs reuse later.
//
// AL layer toggle (scratch.md "AVBD rebuild"): the same solver, three constraint laws.
//   penalty   — fixed stiffness, λ = 0, no ramp, no cross-frame state (phase 1, the mg/k gate)
//   dual      — λ accumulation + conditional penalty ramp, reset each frame (phase 2)
//   warmstart — dual + cross-frame persistence: merge by feature key, decay λ/k (phase 3)
// Only `initManifold` and whether the solver runs the dual pass differ by layer; the primal
// stamp + the dual force law are identical. f64; the reference is the executable spec.

import type { Contact } from "./collide";
import {
    add,
    clamp3,
    cross,
    diagonal,
    length2d,
    type Mat3,
    mulMM,
    mulMV,
    negM,
    qsub,
    rotate,
    scale,
    sub,
    transform,
    transpose,
    type Vec3,
} from "./math";
import { type Body, ShapeKind, solverStatic } from "./rigid";
import { narrowphase } from "./rounded";

export const PENALTY_MIN = 1.0;
export const PENALTY_MAX = 1e10;
export const COLLISION_MARGIN = 0.01;
export const STICK_THRESH = 1e-5;

/** which augmented-Lagrangian terms are live — the phase ladder as a toggle */
export type Layer = "penalty" | "dual" | "warmstart";

/** the parameters the contact law reads (the solver's full params extend this) */
export interface ContactParams {
    alpha: number;
    betaLin: number;
    gamma: number;
    layer: Layer;
    /** fixed penalty stiffness used in the `penalty` layer only */
    penaltyStiffness: number;
    /** timestep — the velocity sweep (Phase 4.8.4) scales the relative velocity by it for the SAT band */
    dt: number;
}

/** one persistent contact: feature key + local arms + the warmstartable dual state */
export interface ContactState {
    feature: number;
    rA: Vec3;
    rB: Vec3;
    c0: Vec3;
    penalty: Vec3;
    lambda: Vec3;
    stick: boolean;
}

export interface Manifold {
    a: Body; // bodyA — the higher creation index (reference linked-list orientation)
    b: Body; // bodyB
    basis: Mat3; // normal in row 0 (B→A), tangents in rows 1-2
    friction: number;
    contacts: ContactState[];
}

const freshContact = (c: Contact): ContactState => ({
    feature: c.feature,
    rA: c.rA,
    rB: c.rB,
    c0: [0, 0, 0],
    penalty: [0, 0, 0],
    lambda: [0, 0, 0],
    stick: false,
});

export function manifold(a: Body, b: Body): Manifold {
    return {
        a,
        b,
        basis: [
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
        ],
        friction: 0,
        contacts: [],
    };
}

/**
 * Re-collide, merge persisted contacts onto the new set by feature key, and warmstart.
 * Returns false when the boxes have separated (0 contacts) — the solver drops the manifold.
 * Mirrors `Manifold::initialize`; the merge + λ/k decay run only in the `warmstart` layer.
 */
export function initManifold(m: Manifold, p: ContactParams): boolean {
    m.friction = Math.sqrt(m.a.friction * m.b.friction);

    // the velocity sweep (Phase 4.8.4): the relative displacement over the step widens the SAT band so a
    // fast pair crossing the contact this frame is caught at frame start. velLin is last step's recovered
    // velocity (initManifold runs before the inertial reposition), matching the GPU collide pass.
    const dRel = scale(sub(m.a.velLin, m.b.velLin), p.dt);
    const { contacts: collided, basis } = narrowphase(m.a, m.b, dRel);
    m.basis = basis;
    const next = collided.map(freshContact);

    // a sticking contact keeps its frozen arms ONLY for polytope × polytope pairs (box/hull), where the
    // feature key pins a persistent vertex/edge. Any pair INVOLVING a rounded shape (sphere/capsule, even
    // vs a polytope) has a sliding closest point under a constant feature key, so freezing its arm anchors a
    // stale point: any rotation spins the frozen arm into a tangential offset → torque → runaway spin
    // (Phase 6.3). A rounded shape re-collides fresh arms vs a box/hull too (the surface anchor slides as
    // the sphere rolls along the face).
    const polytope = (s: number): boolean => s === ShapeKind.Box || s === ShapeKind.Hull;
    const polytopePair = polytope(m.a.shape) && polytope(m.b.shape);
    if (p.layer === "warmstart") {
        for (const c of next) {
            const old = m.contacts.find((o) => o.feature === c.feature);
            if (!old) continue;
            // adopt the persisted dual state; a sticking polytope pair keeps its old arms (else fresh ones)
            c.penalty = old.penalty;
            c.lambda = old.lambda;
            c.stick = old.stick;
            if (old.stick && polytopePair) {
                c.rA = old.rA;
                c.rB = old.rB;
            }
        }
    }

    m.contacts = next;

    // the stored arms anchor the CORE feature point (rounded.ts); the contact SURFACE is offset along the
    // fixed normal by each body's rounding radius (−r·n on A toward B, +r·n on B). Reconstruct the surface
    // points for c0 so the gap is the true surface gap. A box has radius 0, so this is the bare arm.
    const n = basis[0];
    for (const c of m.contacts) {
        const xA = sub(transform(m.a.posLin, m.a.posAng, c.rA), scale(n, m.a.roundRadius));
        const xB = add(transform(m.b.posLin, m.b.posAng, c.rB), scale(n, m.b.roundRadius));
        c.c0 = add(mulMV(basis, sub(xA, xB)), [COLLISION_MARGIN, 0, 0]);

        if (p.layer === "penalty") {
            c.penalty = [p.penaltyStiffness, p.penaltyStiffness, p.penaltyStiffness];
            c.lambda = [0, 0, 0];
        } else if (p.layer === "dual") {
            c.penalty = [PENALTY_MIN, PENALTY_MIN, PENALTY_MIN];
            c.lambda = [0, 0, 0];
        } else {
            // warmstart: decay the persisted dual state (Eq. 19); a fresh contact's 0 penalty
            // clamps up to PENALTY_MIN, matching dual-mode init
            c.lambda = scale(c.lambda, p.alpha * p.gamma);
            c.penalty = clamp3(scale(c.penalty, p.gamma), PENALTY_MIN, PENALTY_MAX);
        }
    }

    return m.contacts.length > 0;
}

/** per-manifold position deltas since x⁻, shared by every contact's Jacobian */
interface Deltas {
    aLin: Vec3;
    aAng: Vec3;
    bLin: Vec3;
    bAng: Vec3;
}

const deltas = (m: Manifold): Deltas => ({
    aLin: sub(m.a.posLin, m.a.initialLin),
    aAng: qsub(m.a.posAng, m.a.initialAng),
    bLin: sub(m.b.posLin, m.b.initialLin),
    bAng: qsub(m.b.posAng, m.b.initialAng),
});

interface ContactForce {
    constraint: Vec3;
    force: Vec3; // clamped: normal repulsion-only + friction cone
    jALin: Mat3;
    jBLin: Mat3;
    jAAng: Mat3;
    jBAng: Mat3;
    k: Mat3;
    // the cone test the dual ramp reads — the *pre-clamp* tangential magnitude vs the bound. The
    // reference (manifold.cpp:156,169) gates the friction-penalty ramp on this pre-clamp value, so a
    // sliding contact (|F_t| > bound) does NOT ramp; reading the post-clamp force would always satisfy
    // `<= bound` and ramp unboundedly, over-stiffening the tangential solve until kinetic friction fades.
    frictionScale: number;
    bounds: number;
}

// The Taylor constraint C, its Jacobians, and the cone-clamped force F for one contact —
// the shared core of updatePrimal + updateDual (the reference inlines it in both).
function contactForce(m: Manifold, c: ContactState, alpha: number, d: Deltas): ContactForce {
    // the stored arm anchors the CORE feature point; the contact surface is a fixed offset ±radius along
    // the contact normal (basis row 0). Applying it here — NOT in the stored arm — keeps the radius part
    // geometric: it never rotates with the body's spin, so a rounded contact's normal Jacobian stays
    // jAAng·n = cross(−r·n, n) = 0 (a sphere's normal force passes through its centre → no torque). A box
    // has radius 0, so rAWorld is the bare material arm, bit-identical to before. roadmap §6.3.
    const n = m.basis[0];
    const rAWorld = sub(rotate(m.a.posAng, c.rA), scale(n, m.a.roundRadius));
    const rBWorld = add(rotate(m.b.posAng, c.rB), scale(n, m.b.roundRadius));

    const jALin = m.basis;
    const jBLin = negM(m.basis);
    const jAAng: Mat3 = [
        cross(rAWorld, jALin[0]),
        cross(rAWorld, jALin[1]),
        cross(rAWorld, jALin[2]),
    ];
    const jBAng: Mat3 = [
        cross(rBWorld, jBLin[0]),
        cross(rBWorld, jBLin[1]),
        cross(rBWorld, jBLin[2]),
    ];

    const k = diagonal(c.penalty[0], c.penalty[1], c.penalty[2]);
    const constraint = add(
        add(scale(c.c0, 1 - alpha), mulMV(jALin, d.aLin)),
        add(add(mulMV(jBLin, d.bLin), mulMV(jAAng, d.aAng)), mulMV(jBAng, d.bAng)),
    );

    const force = add(mulMV(k, constraint), c.lambda);
    force[0] = Math.min(force[0], 0);

    const bounds = Math.abs(force[0]) * m.friction;
    const frictionScale = length2d(force[1], force[2]);
    if (frictionScale > bounds && frictionScale > 0) {
        force[1] *= bounds / frictionScale;
        force[2] *= bounds / frictionScale;
    }

    return { constraint, force, jALin, jBLin, jAAng, jBAng, k, frictionScale, bounds };
}

/** mutable 6×6 accumulator — the per-body system the primal solve fills */
export interface System {
    lhsLin: Mat3;
    lhsAng: Mat3;
    lhsCross: Mat3;
    rhsLin: Vec3;
    rhsAng: Vec3;
}

/** Stamp this manifold's contact forces + Hessians into `body`'s system (`Manifold::updatePrimal`). */
export function stampPrimal(m: Manifold, body: Body, alpha: number, sys: System): void {
    const d = deltas(m);
    const isA = body === m.a;
    for (const c of m.contacts) {
        const { force, jALin, jBLin, jAAng, jBAng, k } = contactForce(m, c, alpha, d);
        const jLin = isA ? jALin : jBLin;
        const jAng = isA ? jAAng : jBAng;
        const jLinT = transpose(jLin);
        const jAngT = transpose(jAng);
        const jAngTk = mulMM(jAngT, k);

        sys.lhsLin = addMat(sys.lhsLin, mulMM(mulMM(jLinT, k), jLin));
        sys.lhsAng = addMat(sys.lhsAng, mulMM(jAngTk, jAng));
        sys.lhsCross = addMat(sys.lhsCross, mulMM(jAngTk, jLin));
        sys.rhsLin = add(sys.rhsLin, mulMV(jLinT, force));
        sys.rhsAng = add(sys.rhsAng, mulMV(jAngT, force));
    }
}

const addMat = (a: Mat3, b: Mat3): Mat3 => [add(a[0], b[0]), add(a[1], b[1]), add(a[2], b[2])];

/** Advance λ + the conditional penalty ramp + the stick flag (`Manifold::updateDual`). */
export function updateDual(m: Manifold, alpha: number, betaLin: number): void {
    // The kinematic-pushing fix (roadmap §6.4): a contact NO dynamic body can resolve — both mass ≤ 0, a
    // kinematic character driven into a static wall — is never satisfiable by the primal (it skips both),
    // so its constraint stays violated frame after frame. The reference ramps every force's penalty
    // (solver.cpp:230 runs updateDual unconditionally), escalating this unsolvable contact's `k` unbounded
    // with nothing ever moving to relax it — the legacy escalating-force blow-up (physics.md "legacy
    // antipatterns"). The kinematic motion (the controller's collide-and-slide sweep) already resolves the
    // contact, so the dual must not stiffen it: skip the all-static manifold entirely (its λ decays to 0
    // via initManifold, its penalty holds at the seed). A contact with one dynamic body is unchanged.
    if (solverStatic(m.a) && solverStatic(m.b)) return;
    const d = deltas(m);
    for (const c of m.contacts) {
        const { constraint, force, frictionScale, bounds } = contactForce(m, c, alpha, d);
        c.lambda = force;

        // ramp gates read the *pre-clamp* friction magnitude (manifold.cpp:169, see ContactForce): a
        // sliding contact (frictionScale > bounds) skips the tangential ramp + stick update.
        if (force[0] < 0) {
            c.penalty[0] = Math.min(c.penalty[0] + betaLin * Math.abs(constraint[0]), PENALTY_MAX);
        }
        if (frictionScale <= bounds) {
            c.penalty[1] = Math.min(c.penalty[1] + betaLin * Math.abs(constraint[1]), PENALTY_MAX);
            c.penalty[2] = Math.min(c.penalty[2] + betaLin * Math.abs(constraint[2]), PENALTY_MAX);
            c.stick = length2d(constraint[1], constraint[2]) < STICK_THRESH;
        }
    }
}
