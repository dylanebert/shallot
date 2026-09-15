// The step's phase-timing seam — Box3D's b3Profile (physics_world.c / solver.c, Erin Catto, MIT). The
// step and solve mark and close named phases on the world's clock; the default clock does nothing, so
// the default step performs no timing work and reads no wall clock. A profiler installs a timing clock
// (the `profile` extra's) as a separate path, not a per-step branch. Diagnostic only — outside the
// bit-exact contract.
//
// Two fields stay zero by design: `solverSetup` is not timed separately, and `refit` is folded into
// finalize, so its cost lands in `transforms`.

/** One step's phase timings in milliseconds (b3Profile). Read via {@link World.getProfile}. */
export type Profile = {
    step: number;
    pairs: number;
    collide: number;
    solve: number;
    solverSetup: number;
    constraints: number;
    prepareConstraints: number;
    integrateVelocities: number;
    warmStart: number;
    solveImpulses: number;
    integratePositions: number;
    relaxImpulses: number;
    applyRestitution: number;
    storeImpulses: number;
    splitIslands: number;
    transforms: number;
    sensorHits: number;
    jointEvents: number;
    hitEvents: number;
    refit: number;
    bullets: number;
    sleepIslands: number;
    sensors: number;
};

/** The phase cursors the step and solve nest: the whole step, a step phase, the solve's constraint
 * region, its accumulating inner cursor, and a solve phase. A clock keeps one start per slot. */
export const STEP_SLOT = 0;
export const PHASE_SLOT = 1;
export const CONSTRAINTS_SLOT = 2;
export const CURSOR_SLOT = 3;
export const SOLVE_PHASE_SLOT = 4;
export const CLOCK_SLOTS = 5;

/** The step's timing seam. Every method is called at a phase boundary of one `step`. */
export interface StepClock {
    /** Zero the last step's phases and mark `slot` (the top of a step). */
    begin(slot: number): void;
    /** Mark `slot` as starting now. */
    mark(slot: number): void;
    /** Set `field` to the time since `slot` was marked. */
    span(field: keyof Profile, slot: number): void;
    /** Add the time since `slot` was marked to `field`, then re-mark `slot`. */
    lap(field: keyof Profile, slot: number): void;
    /** @returns a copy of the last step's phase timings. */
    read(): Profile;
}

/** A fresh zeroed profile (world->profile = (b3Profile){0}). */
export function zeroProfile(): Profile {
    return {
        step: 0,
        pairs: 0,
        collide: 0,
        solve: 0,
        solverSetup: 0,
        constraints: 0,
        prepareConstraints: 0,
        integrateVelocities: 0,
        warmStart: 0,
        solveImpulses: 0,
        integratePositions: 0,
        relaxImpulses: 0,
        applyRestitution: 0,
        storeImpulses: 0,
        splitIslands: 0,
        transforms: 0,
        sensorHits: 0,
        jointEvents: 0,
        hitEvents: 0,
        refit: 0,
        bullets: 0,
        sleepIslands: 0,
        sensors: 0,
    };
}

/** Timing clocks a profiler composed into a State, keyed by that State. `PhysicsPlugin` installs a
 * State's clock on the world it warms; a State without one keeps {@link NO_CLOCK}. */
export const stepClocks = new WeakMap<object, StepClock>();

/** The default clock: no timing work, and every phase reads zero. */
export const NO_CLOCK: StepClock = {
    begin() {},
    mark() {},
    span() {},
    lap() {},
    read: zeroProfile,
};
