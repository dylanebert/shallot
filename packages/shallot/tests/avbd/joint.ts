// Port of reference/avbd-demo3d/source/joint.cpp — the hard `Force` (roadmap "Phase 6.2 — Joints").
// A joint is two stacked constraints between body-local anchors: a linear row triple pinning the anchor
// points together (`C = pA − pB`) and an angular row triple locking relative orientation
// (`C = (qA − qB)·torqueArm`). Unlike the spring it carries dual state — `λ` + a penalty that ramps per
// iteration — and the rigid (infinite-stiffness) form adds the explosive-error stabilization
// `C −= α·C₀`, both warmstarted across frames in the persistent record (Eq. 19). The two pin types are
// configs of the SAME force: spherical = `stiffnessAng = 0` (the angular rows never activate, so rotation is
// free), fixed = `stiffnessAng = ∞`. The third type is the **angular motor** (avbd-demo2d motor.cpp): a 1-DOF
// force-clamped drive (`C = deltaAngle − speed·dt`, force clamped to `±maxTorque`) carrying its own scalar
// λ/penalty, the bounded-`f_min`/`f_max` form, independent of the pin rows (a spherical joint still motors).
//
// `Force` interface parity with the contact manifold (manifold.ts): initialize / updatePrimal /
// updateDual. f64; the reference is the executable spec. Test scaffolding, kept out of the shipped src/.

import { COLLISION_MARGIN, PENALTY_MAX, PENALTY_MIN, type System } from "./manifold";
import {
    add,
    clamp,
    clamp3,
    diagonal,
    diagonalize,
    dot,
    length,
    lengthSq,
    type Mat3,
    mulMM,
    mulMV,
    neg,
    outer,
    type Quat,
    qsub,
    rotate,
    scale,
    scaleM,
    skew,
    sub,
    transform,
    transpose,
    type Vec3,
} from "./math";
import { type Body, solverStatic } from "./rigid";

/** a joint between two body-local anchors (`rA` on `a`, `rB` on `b`). `a = null` makes `rA` a WORLD-space
 *  point — the joint pins `b` to a fixed world anchor with no body A (avbd-demo3d's mouse-drag grab, joint.cpp
 *  `bodyA == null`): no anchor body means no anchor↔b contact, and `b` dangles freely from the world point. */
export interface Joint {
    a: Body | null;
    b: Body;
    rA: Vec3; // anchor in a's local frame — OR a world-space point when a is null
    rB: Vec3; // anchor in b's local frame
    stiffnessLin: number; // Infinity = rigid (adds the C −= α·C₀ stabilization)
    stiffnessAng: number; // 0 = spherical (rotation free); Infinity = fixed
    torqueArm: number; // ‖sizeA + sizeB‖² — scales the angular rows (joint.cpp)
    // warmstartable dual state, persisted across frames
    penaltyLin: Vec3;
    penaltyAng: Vec3;
    lambdaLin: Vec3;
    lambdaAng: Vec3;
    // C(x⁻) captured at the start of each step (initialize)
    c0Lin: Vec3;
    c0Ang: Vec3;
    // motor — a 1-DOF force-clamped angular drive about `motorAxis` (a port of avbd-demo2d motor.cpp).
    // Absent (`motorMaxTorque` undefined) ⇒ no motor, the joint behaves exactly as before. When present it
    // drives b's orientation RELATIVE TO a at `motorSpeed` rad/s about `motorAxis`, the angular force clamped
    // to ±`motorMaxTorque` (so it holds the target ω under a load up to that torque, and yields past it).
    // Independent of the spherical/fixed angular term above: it carries its own scalar λ + penalty, so a
    // spherical joint (`stiffnessAng = 0`, the angular rows off) still motors. World anchor (a = null): the
    // world is the static reference, so b spins at `motorSpeed` about `motorAxis`.
    motorAxis?: Vec3; // unit world axis the drive acts about
    motorSpeed?: number; // target rad/s of b relative to a about motorAxis
    motorMaxTorque?: number; // |angular force| clamp — the presence of this field activates the motor
    // warmstarted scalar dual state (meaningful only when the motor is active)
    motorLambda: number;
    motorPenalty: number;
    // the step's drive increment `speed·h`, captured in initialize (the only per-step capture the motor needs)
    motorDrive: number;
}

const addMat = (a: Mat3, b: Mat3): Mat3 => [add(a[0], b[0]), add(a[1], b[1]), add(a[2], b[2])];
const abs3 = (v: Vec3): Vec3 => [Math.abs(v[0]), Math.abs(v[1]), Math.abs(v[2])];
/** component-wise min of a vector with a scalar (the reference's `min(float3, float)`) */
const min3s = (v: Vec3, s: number): Vec3 => [
    Math.min(v[0], s),
    Math.min(v[1], s),
    Math.min(v[2], s),
];

const I: Mat3 = [
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1],
];
const NEG_I: Mat3 = [
    [-1, 0, 0],
    [0, -1, 0],
    [0, 0, -1],
];

// geometricStiffnessBallSocket(k, v) — the diagonal higher-order term: `-v[k]·I` with `v` added into
// column k (joint.cpp). Stamped into lhsAng scaled by the force component F[k].
const geomStiffness = (k: number, v: Vec3): Mat3 => {
    const m = diagonal(-v[k], -v[k], -v[k]);
    m[0][k] += v[0];
    m[1][k] += v[1];
    m[2][k] += v[2];
    return m;
};

/**
 * A joint pinning `b`'s anchor `rB` to `a`'s anchor `rA`. Defaults match `Joint::Joint`: rigid linear
 * (`stiffnessLin = ∞`) + free rotation (`stiffnessAng = 0`) = a spherical joint; pass
 * `stiffnessAng = Infinity` for a fixed joint.
 */
export function joint(
    a: Body | null,
    b: Body,
    rA: Vec3,
    rB: Vec3,
    stiffnessLin = Number.POSITIVE_INFINITY,
    stiffnessAng = 0,
    motor?: { axis: Vec3; speed: number; maxTorque: number },
): Joint {
    // Both-endpoints-non-dynamic guard (energy injection, roadmap §6). A joint between two `mass ≤ 0` bodies
    // (static/kinematic, OR the world anchor a = null) can never be satisfied by the primal — both skip it
    // (solver.ts) — so the dual ramps its penalty + λ unbounded with nothing ever moving to relax it (the
    // joint analog of the contact all-static dual guard, manifold.ts updateDual). Worse, if such an endpoint
    // is later made dynamic (released), the accumulated huge λ yanks it. Reject at construction; the GPU
    // jointInit carries the same guard (deactivate + a `counters[1]` bump, since the GPU can't throw). One
    // endpoint static (an anchor, or the world) is fine — the dynamic partner resolves the constraint.
    if ((a === null || a.mass <= 0) && b.mass <= 0) {
        throw new Error(
            `joint endpoints are both non-dynamic (mass ≤ 0): the primal can never satisfy it, so the dual ` +
                `penalty + λ ramp unbounded (energy injection). Joint one dynamic body to a static/kinematic/world anchor.`,
        );
    }

    // Energy-injection guard (the legacy rope explosion). A joint pins rA-on-A to rB-on-B (rA the world point
    // when a is null), so those world points must START coincident. A gross initial mismatch makes the rigid
    // constraint recover spurious velocity through BDF1 (v = Δx/dt) as it corrects — energy injected, the chain
    // explodes. The bound is geometric: anchors farther apart than the bodies' combined reach (radiusA +
    // radiusB) can't be pinned without a violent correction → a construction error. Measured (oracle.test.ts):
    // a 4.2 m mismatch injects +34% energy + 6.6 m/s; a ≤ 0.1 m offset is absorbed cleanly. Construct at a pose
    // where the anchors meet. NOTE: the GPU joint-authoring API (Phase 6.2 GPU) must carry this same guard.
    const pA = a ? transform(a.posLin, a.posAng, rA) : rA;
    const pB = transform(b.posLin, b.posAng, rB);
    const mismatch = length(sub(pA, pB));
    const reach = (a ? a.radius : 0) + b.radius + COLLISION_MARGIN;
    if (mismatch > reach) {
        throw new Error(
            `joint anchors must start coincident: rA and rB are ${mismatch.toFixed(3)} m apart, past the ` +
                `bodies' ${reach.toFixed(3)} m combined reach. A gross initial violation injects energy ` +
                `through the BDF1 velocity recovery (the rope explosion) — place the bodies so the anchors meet.`,
        );
    }

    return {
        a,
        b,
        rA,
        rB,
        stiffnessLin,
        stiffnessAng,
        torqueArm: lengthSq(add(a ? a.size : [0, 0, 0], b.size)),
        penaltyLin: [0, 0, 0],
        penaltyAng: [0, 0, 0],
        lambdaLin: [0, 0, 0],
        lambdaAng: [0, 0, 0],
        c0Lin: [0, 0, 0],
        c0Ang: [0, 0, 0],
        motorAxis: motor?.axis,
        motorSpeed: motor?.speed,
        motorMaxTorque: motor?.maxTorque,
        motorLambda: 0,
        motorPenalty: 0,
        motorDrive: 0,
    };
}

// a's anchor in world space (rA is already a world point when a is null — the world anchor)
const anchorWorld = (j: Joint): Vec3 => (j.a ? transform(j.a.posLin, j.a.posAng, j.rA) : j.rA);
// a's orientation (identity for the world anchor — joint.cpp's `bodyA ? bodyA->positionAng : {0,0,0,1}`)
const aQuat = (j: Joint): Quat => (j.a ? j.a.posAng : [0, 0, 0, 1]);

// the motor's scalar constraint C = deltaAngle − speed·h (motor.cpp). deltaAngle is each body's INCREMENTAL
// rotation about the axis SINCE STEP START (b's minus a's) — `dot(qsub(pose, initialPose), axis)`, the exact
// analog of the reference's `position.z − initial.z`. Measuring the increment (small per step) not an absolute
// `qsub(qB, qA)` against a fixed reference is load-bearing: the latter reads 2·sin(θ/2), nonlinear far from
// identity, so a continuously-spinning rotor would over-rotate to null it. `initialAng` is the BDF1 step-start
// pose (saved after inertial init, before warmstart), so this runs in the primal/dual, not in initJoint.
const motorC = (j: Joint): number => {
    const axis = j.motorAxis as Vec3;
    const dB = dot(qsub(j.b.posAng, j.b.initialAng), axis);
    const dA = j.a ? dot(qsub(j.a.posAng, j.a.initialAng), axis) : 0;
    return dB - dA - j.motorDrive;
};

/**
 * Cache `C(x⁻)`, decay the warmstarted dual state (Eq. 19), and clamp the penalty to the material
 * stiffness (`Joint::initialize`).
 */
export function initJoint(j: Joint, alpha: number, gamma: number, h: number): void {
    j.c0Lin = sub(anchorWorld(j), transform(j.b.posLin, j.b.posAng, j.rB));
    j.c0Ang = scale(qsub(aQuat(j), j.b.posAng), j.torqueArm);

    j.lambdaLin = scale(j.lambdaLin, alpha * gamma);
    j.lambdaAng = scale(j.lambdaAng, alpha * gamma);
    j.penaltyLin = clamp3(scale(j.penaltyLin, gamma), PENALTY_MIN, PENALTY_MAX);
    j.penaltyAng = clamp3(scale(j.penaltyAng, gamma), PENALTY_MIN, PENALTY_MAX);
    j.penaltyLin = min3s(j.penaltyLin, j.stiffnessLin);
    j.penaltyAng = min3s(j.penaltyAng, j.stiffnessAng);

    // motor: capture b's rotation about the axis (relative to a) at step start + this step's drive Δ
    // (`speed·h`, the reference's `deltaAngle − speed·dt` target), and warmstart its scalar λ/penalty.
    // The motor's stiffness is effectively ∞ (the force is bounded by the torque clamp, not the penalty),
    // so the penalty ramps toward PENALTY_MAX — unclamped by stiffnessAng, which the spherical case sets to 0.
    if (j.motorMaxTorque !== undefined) {
        j.motorDrive = (j.motorSpeed ?? 0) * h;
        j.motorLambda *= alpha * gamma;
        j.motorPenalty = clamp(j.motorPenalty * gamma, PENALTY_MIN, PENALTY_MAX);
    }
}

/** Stamp the joint's force + Hessian into `body`'s 6×6 system (`Joint::updatePrimal`). */
export function stampJoint(j: Joint, body: Body, alpha: number, sys: System): void {
    const isA = body === j.a;

    // linear constraint — the anchor-pinning rows
    if (lengthSq(j.penaltyLin) > 0) {
        const k = diagonal(j.penaltyLin[0], j.penaltyLin[1], j.penaltyLin[2]);
        let c = sub(anchorWorld(j), transform(j.b.posLin, j.b.posAng, j.rB));
        if (!Number.isFinite(j.stiffnessLin)) c = sub(c, scale(j.c0Lin, alpha));
        const f = add(mulMV(k, c), j.lambdaLin);

        const jLin = isA ? I : NEG_I;
        // isA ⟹ body === j.a ⟹ j.a is a real body (a world anchor has a = null, so isA is always false there)
        const jAng = isA
            ? skew(neg(rotate((j.a as Body).posAng, j.rA)))
            : skew(rotate(j.b.posAng, j.rB));
        const jLinT = transpose(jLin);
        const jAngT = transpose(jAng);
        const jAngTk = mulMM(jAngT, k);

        sys.lhsLin = addMat(sys.lhsLin, mulMM(mulMM(jLinT, k), jLin));
        sys.lhsAng = addMat(sys.lhsAng, mulMM(jAngTk, jAng));
        sys.lhsCross = addMat(sys.lhsCross, mulMM(jAngTk, jLin));

        // diagonal approximation of the higher-order (geometric-stiffness) term
        const r = isA ? rotate((j.a as Body).posAng, j.rA) : neg(rotate(j.b.posAng, j.rB));
        const h = addMat(
            addMat(scaleM(geomStiffness(0, r), f[0]), scaleM(geomStiffness(1, r), f[1])),
            scaleM(geomStiffness(2, r), f[2]),
        );
        sys.lhsAng = addMat(sys.lhsAng, diagonalize(h));

        sys.rhsLin = add(sys.rhsLin, mulMV(jLinT, f));
        sys.rhsAng = add(sys.rhsAng, mulMV(jAngT, f));
    }

    // angular constraint — the relative-orientation rows (skipped for a spherical joint, penalty 0)
    if (lengthSq(j.penaltyAng) > 0) {
        const k = diagonal(j.penaltyAng[0], j.penaltyAng[1], j.penaltyAng[2]);
        let c = scale(qsub(aQuat(j), j.b.posAng), j.torqueArm);
        if (!Number.isFinite(j.stiffnessAng)) c = sub(c, scale(j.c0Ang, alpha));
        const f = add(mulMV(k, c), j.lambdaAng);

        const s = isA ? j.torqueArm : -j.torqueArm;
        const jAng = diagonal(s, s, s); // (±I)·torqueArm
        const jAngT = transpose(jAng);
        sys.lhsAng = addMat(sys.lhsAng, mulMM(mulMM(jAngT, k), jAng));
        sys.rhsAng = add(sys.rhsAng, mulMV(jAngT, f));
    }

    // motor — a 1-DOF force-clamped angular drive about motorAxis (motor.cpp). The angular force competes
    // inside each iteration, clamped to ±motorMaxTorque, so a driven body holds its target ω under a load that
    // stalls a forced-velocity drive. Jacobian J = ±axis (unit, Hessian 0 — motor.cpp `computeDerivatives`);
    // independent of the spherical/fixed rows above, so a spherical joint still motors.
    if (j.motorMaxTorque !== undefined) {
        const axis = j.motorAxis as Vec3;
        const f = clamp(
            j.motorPenalty * motorC(j) + j.motorLambda,
            -j.motorMaxTorque,
            j.motorMaxTorque,
        );
        const s = isA ? -1 : 1; // ∂(qsub(qB,qA))/∂θ = +axis on b, −axis on a
        sys.lhsAng = addMat(sys.lhsAng, scaleM(outer(axis, axis), j.motorPenalty));
        sys.rhsAng = add(sys.rhsAng, scale(axis, s * f));
    }
}

/** Advance λ + the penalty ramp on both row triples (`Joint::updateDual`). */
export function updateJointDual(j: Joint, alpha: number, betaLin: number, betaAng: number): void {
    // All-static gate: skip a joint no dynamic body can resolve — both endpoints static, the world anchor
    // (a = null) counting as static. The primal can't satisfy it, so ramping its λ/penalty injects energy
    // (the joint analog of the contact all-static gate, manifold.ts updateDual). Construction already
    // rejects a both-non-dynamic joint, so this never fires for a valid joint.
    if (solverStatic(j.b) && (j.a === null || solverStatic(j.a))) return;
    if (lengthSq(j.penaltyLin) > 0) {
        const k = diagonal(j.penaltyLin[0], j.penaltyLin[1], j.penaltyLin[2]);
        let c = sub(anchorWorld(j), transform(j.b.posLin, j.b.posAng, j.rB));
        if (!Number.isFinite(j.stiffnessLin)) {
            c = sub(c, scale(j.c0Lin, alpha));
            j.lambdaLin = add(mulMV(k, c), j.lambdaLin);
        }
        j.penaltyLin = min3s(
            add(j.penaltyLin, scale(abs3(c), betaLin)),
            Math.min(j.stiffnessLin, PENALTY_MAX),
        );
    }

    if (lengthSq(j.penaltyAng) > 0) {
        const k = diagonal(j.penaltyAng[0], j.penaltyAng[1], j.penaltyAng[2]);
        let c = scale(qsub(aQuat(j), j.b.posAng), j.torqueArm);
        if (!Number.isFinite(j.stiffnessAng)) {
            c = sub(c, scale(j.c0Ang, alpha));
            j.lambdaAng = add(mulMV(k, c), j.lambdaAng);
        }
        j.penaltyAng = min3s(
            add(j.penaltyAng, scale(abs3(c), betaAng)),
            Math.min(j.stiffnessAng, PENALTY_MAX),
        );
    }

    // motor dual — λ accumulates and is CLAMPED to ±motorMaxTorque (the bounded-constraint update, the analog
    // of the contact normal's repulsion-only clamp). The penalty ramps toward PENALTY_MAX (stiffness ∞) ONLY
    // while λ is strictly inside the force bounds (solver.cpp's `if (lambda > fmin && lambda < fmax)`): a
    // saturated motor (λ at the clamp) must keep a small penalty so it stays a constant-torque drive (accel
    // maxTorque/I); ramping it there over-stiffens the Hessian and drags the spin-up below that rate. The rigid
    // angular lock above needs no such gate — its ±∞ bounds make λ always "inside".
    if (j.motorMaxTorque !== undefined) {
        const c = motorC(j);
        j.motorLambda = clamp(
            j.motorLambda + j.motorPenalty * c,
            -j.motorMaxTorque,
            j.motorMaxTorque,
        );
        if (j.motorLambda > -j.motorMaxTorque && j.motorLambda < j.motorMaxTorque) {
            j.motorPenalty = Math.min(j.motorPenalty + Math.abs(c) * betaAng, PENALTY_MAX);
        }
    }
}
