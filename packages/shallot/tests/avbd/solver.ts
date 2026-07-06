// Port of reference/avbd-demo3d/source/solver.cpp `step()` — the AVBD time step, and the
// executable spec the GPU solver validates against. One step: O(n²) sphere broadphase →
// force initialize/warmstart → body inertial init + adaptive warmstart → the main loop
// (primal solve + dual update) → BDF1 velocity recovery.
//
// Two axes the reference doesn't have, both required by scratch.md "AVBD rebuild":
//   • scheduler — `sequential` is the C++ exactly (Gauss-Seidel, reverse creation order, commit
//     in place); `colored` runs a supplied coloring (commit deferred within a color → a rare
//     same-color neighbor degrades to Jacobi, the paper's §Parallelization).
//   • AL layer — `penalty | dual | warmstart`, threaded through to the contact law (manifold.ts).
// f64 throughout — the oracle is the exact sequential reference; f32 is the GPU's to match.

import { SPECULATIVE_DISTANCE } from "./collide";
import { initJoint, type Joint, stampJoint, updateJointDual } from "./joint";
import {
    type ContactParams,
    initManifold,
    type Manifold,
    manifold,
    type System,
    stampPrimal,
    updateDual,
} from "./manifold";
import {
    add,
    clamp,
    diagonal,
    dot,
    length,
    mulMV,
    neg,
    type Quat,
    qadd,
    qsub,
    scale,
    scaleM,
    sign,
    solve,
    sub,
    type Vec3,
} from "./math";
import { type Body, solverStatic } from "./rigid";
import { type Spring, stampSpring } from "./spring";

export interface Params extends ContactParams {
    dt: number;
    gravity: number;
    iterations: number;
    betaAng: number; // angular penalty ramp — the joint angular constraint (Phase 6.2); contacts don't ramp it
    /**
     * sub-steps per fixed step (the "small steps" form, Macklin 2019): one `step()` runs `substeps`
     * complete AVBD sub-steps of `h = dt/substeps`. Smaller per-sub-step motion keeps the penalty ramp
     * (`k += β|C|`) bounded — a tall dense pile that can't converge in one big step otherwise runs `k`
     * away into a lateral ejection (scratch.md "Phase 4.9 robustness"). `1` is byte-identical to the
     * single-step reference, so the corpus + closed-form gates are unchanged.
     */
    substeps: number;
}

/** the reference's `defaultParams()` (canonical AVBD set), plus the oracle's layer toggle */
export const defaultParams = (): Params => ({
    dt: 1 / 60,
    gravity: -10,
    iterations: 10,
    alpha: 0.99,
    betaLin: 10000,
    betaAng: 100,
    gamma: 0.999,
    layer: "warmstart",
    penaltyStiffness: 1e5,
    substeps: 1,
});

/** `sequential` = the C++ Gauss-Seidel; `colored` runs the supplied per-body coloring */
export type Schedule = { kind: "sequential" } | { kind: "colored"; colors: number[] };

export interface Solver {
    bodies: Body[];
    manifolds: Map<number, Manifold>;
    /** authored constraints — persist across frames (no broadphase), stamped alongside contacts */
    springs: Spring[];
    /** authored hard constraints — persist + carry warmstartable dual state, like contacts */
    joints: Joint[];
    params: Params;
}

export function makeSolver(bodies: Body[], params: Partial<Params> = {}): Solver {
    return {
        bodies,
        manifolds: new Map(),
        springs: [],
        joints: [],
        params: { ...defaultParams(), ...params },
    };
}

const clone3 = (v: Vec3): Vec3 => [v[0], v[1], v[2]];
const clone4 = (q: Quat): Quat => [q[0], q[1], q[2], q[3]];

// Build the per-body system M/dt² + Σ contact Hessians, RHS −M/dt²·(x−y) − Σ Jᵀ·F, and
// solve the 6×6 SPD system for (dxLin, dxAng) via LDLᵀ (solver.cpp Eqs. 4-6).
function solveBody(
    bd: Body,
    forces: Manifold[],
    springs: Spring[],
    joints: Joint[],
    p: Params,
    h: number,
): { xLin: Vec3; xAng: Vec3 } {
    const invDt2 = 1 / (h * h);
    const mLin = scaleM(diagonal(bd.mass, bd.mass, bd.mass), invDt2);
    const mAng = scaleM(diagonal(bd.moment[0], bd.moment[1], bd.moment[2]), invDt2);
    const sys: System = {
        lhsLin: mLin,
        lhsAng: mAng,
        lhsCross: [
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
        ],
        rhsLin: mulMV(mLin, sub(bd.posLin, bd.inertialLin)),
        rhsAng: mulMV(mAng, qsub(bd.posAng, bd.inertialAng)),
    };
    // additive into one 6×6 system, so contact/spring/joint order is irrelevant (the reference iterates
    // one interleaved per-body force list; the system sum is the same)
    for (const m of forces) stampPrimal(m, bd, p.alpha, sys);
    for (const sp of springs) stampSpring(sp, bd, sys);
    for (const jt of joints) stampJoint(jt, bd, p.alpha, sys);
    return solve(sys.lhsLin, sys.lhsAng, sys.lhsCross, neg(sys.rhsLin), neg(sys.rhsAng));
}

// Sequential Gauss-Seidel in reverse creation order (the reference iterates its newest-first
// linked list), committing each body's update before the next reads it.
function primalSequential(
    s: Solver,
    forces: Manifold[][],
    springs: Spring[][],
    joints: Joint[][],
    h: number,
): void {
    for (let i = s.bodies.length - 1; i >= 0; i--) {
        const bd = s.bodies[i];
        if (solverStatic(bd)) continue;
        const { xLin, xAng } = solveBody(bd, forces[i], springs[i], joints[i], s.params, h);
        bd.posLin = add(bd.posLin, xLin);
        bd.posAng = qadd(bd.posAng, xAng);
    }
}

// Colored Gauss-Seidel: colors stepped in ascending order, same-color bodies solved from the
// color's start state and committed together (deferred → a same-color neighbor degrades to
// Jacobi for the step; a valid coloring has none, so it equals a per-color GS ordering).
function primalColored(
    s: Solver,
    forces: Manifold[][],
    springs: Spring[][],
    joints: Joint[][],
    colors: number[],
    h: number,
): void {
    let maxColor = 0;
    for (const c of colors) maxColor = Math.max(maxColor, c);
    for (let color = 0; color <= maxColor; color++) {
        const updates: { bd: Body; xLin: Vec3; xAng: Vec3 }[] = [];
        for (let i = 0; i < s.bodies.length; i++) {
            const bd = s.bodies[i];
            if (colors[i] !== color || solverStatic(bd)) continue;
            const { xLin, xAng } = solveBody(bd, forces[i], springs[i], joints[i], s.params, h);
            updates.push({ bd, xLin, xAng });
        }
        for (const u of updates) {
            u.bd.posLin = add(u.bd.posLin, u.xLin);
            u.bd.posAng = qadd(u.bd.posAng, u.xAng);
        }
    }
}

/**
 * Advance the simulation one fixed step = `substeps` sub-steps of `h = dt/substeps` (the "small steps"
 * form). `substeps = 1` runs exactly one sub-step at `h = dt`, byte-identical to `Solver::step`.
 */
export function step(s: Solver, schedule: Schedule = { kind: "sequential" }): void {
    const n = Math.max(1, Math.round(s.params.substeps));
    const h = s.params.dt / n;
    for (let sub = 0; sub < n; sub++) subStep(s, schedule, h);
}

// One sub-step at timestep `h` — the reference `Solver::step` body: broadphase → warmstart → inertial
// init → primal/dual loop → BDF1 velocity recovery, every dt-bearing term using `h`. The manifold map
// persists across sub-steps, so each sub-step warmstarts off the previous one exactly as it warmstarts
// across frames; re-colliding per sub-step keeps each contact's `c0` gap fresh as the bodies move.
function subStep(s: Solver, schedule: Schedule, h: number): void {
    const { bodies, manifolds, params: p } = s;
    const n = bodies.length;
    const g = p.gravity;
    // the contact law reads `dt` only for the velocity sweep (initManifold); feed it the sub-step `h`
    const ph: ContactParams = h === p.dt ? p : { ...p, dt: h };

    // 1. broadphase — sphere-radius overlap; bodyA = the higher creation index (reference orientation)
    for (let ia = n - 1; ia >= 1; ia--) {
        const A = bodies[ia];
        for (let ib = ia - 1; ib >= 0; ib--) {
            const B = bodies[ib];
            const dp = sub(A.posLin, B.posLin);
            // the speculative band pads the broadphase so a pair within it is found before contact and its
            // manifold survives a momentary separation (Phase 4.8.3); the velocity-sweep term `|vRel|·h`
            // (Phase 4.8.4) extends the pad so a fast approaching pair is found while still separated by the
            // sweep. Matches the GPU AABB/sphere pad (step.ts) and the C++ harness fork (GPU == oracle == C++).
            const r =
                A.radius + B.radius + SPECULATIVE_DISTANCE + length(sub(A.velLin, B.velLin)) * h;
            if (dot(dp, dp) <= r * r) {
                const key = ib * n + ia;
                if (!manifolds.has(key)) manifolds.set(key, manifold(A, B));
            }
        }
    }

    // 2. initialize + warmstart forces; drop any contacts that have separated (0 contacts)
    for (const [key, m] of manifolds) {
        if (!initManifold(m, ph)) manifolds.delete(key);
    }
    for (const jt of s.joints) initJoint(jt, p.alpha, p.gamma, h);

    // per-body force adjacency for the primal (rebuilt each sub-step — the manifold set churns;
    // springs are authored + persistent, but the adjacency is rebuilt the same way)
    const index = new Map<Body, number>();
    for (let i = 0; i < n; i++) index.set(bodies[i], i);
    const forces: Manifold[][] = Array.from({ length: n }, () => []);
    for (const m of manifolds.values()) {
        forces[index.get(m.a) as number].push(m);
        forces[index.get(m.b) as number].push(m);
    }
    const springs: Spring[][] = Array.from({ length: n }, () => []);
    for (const sp of s.springs) {
        springs[index.get(sp.a) as number].push(sp);
        springs[index.get(sp.b) as number].push(sp);
    }
    const joints: Joint[][] = Array.from({ length: n }, () => []);
    for (const jt of s.joints) {
        if (jt.a) joints[index.get(jt.a) as number].push(jt); // a null = the world anchor (no body to stamp)
        joints[index.get(jt.b) as number].push(jt);
    }

    // 3. inertial target (Eq. 2) + adaptive warmstart (VBD): the inertial pose uses full gravity,
    // the warmstart start position scales gravity by accelWeight — two different gravity terms.
    // A static / kinematic body is frozen here (inertial = initial = current pose, dq = 0).
    for (const bd of bodies) {
        bd.inertialLin = add(bd.posLin, scale(bd.velLin, h));
        if (!solverStatic(bd)) bd.inertialLin = add(bd.inertialLin, [0, g * h * h, 0]);
        bd.inertialAng = qadd(bd.posAng, scale(bd.velAng, h));

        const accel = scale(sub(bd.velLin, bd.prevVelLin), 1 / h);
        let accelWeight = clamp((accel[1] * sign(g)) / Math.abs(g), 0, 1);
        if (!Number.isFinite(accelWeight)) accelWeight = 0;

        bd.initialLin = clone3(bd.posLin);
        bd.initialAng = clone4(bd.posAng);
        if (!solverStatic(bd)) {
            bd.posLin = add(add(bd.posLin, scale(bd.velLin, h)), [0, g * accelWeight * h * h, 0]);
            bd.posAng = qadd(bd.posAng, scale(bd.velAng, h));
        }
    }

    // 4. main loop — primal solve then dual update (the dual is skipped in the penalty layer)
    for (let it = 0; it < p.iterations; it++) {
        if (schedule.kind === "colored") {
            primalColored(s, forces, springs, joints, schedule.colors, h);
        } else {
            primalSequential(s, forces, springs, joints, h);
        }
        if (p.layer !== "penalty") {
            for (const m of manifolds.values()) updateDual(m, p.alpha, p.betaLin);
            // joints carry dual state in every layer (a hard constraint's λ + ramp is intrinsic, not a
            // contact-only augmentation) — the penalty layer is a contact-build-up rung the joint skips.
            // updateJointDual itself skips an all-static joint (both ends static), like updateDual.
            for (const jt of s.joints) updateJointDual(jt, p.alpha, p.betaLin, p.betaAng);
        }
    }

    // 5. BDF1 velocity recovery (static / kinematic bodies keep their frozen velocity)
    for (const bd of bodies) {
        bd.prevVelLin = bd.velLin;
        if (!solverStatic(bd)) {
            bd.velLin = scale(sub(bd.posLin, bd.initialLin), 1 / h);
            bd.velAng = scale(qsub(bd.posAng, bd.initialAng), 1 / h);
        }
    }
}
