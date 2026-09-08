// Per-phase step timing — Box3D's b3Profile (physics_world.c / solver.c, Erin Catto, MIT). Times are
// in milliseconds, covering one step(); the world's profile is zeroed at the top of each step and read
// back with world.getProfile(). Diagnostic only — outside the bit-exact contract, so timings keep f64
// precision (the C struct is float, a benign truncation the port doesn't reproduce).
//
// The serial force-overflow port records the fields worker 0 records in the C task schedule. Two fields
// have no serial equivalent and stay 0: `solverSetup` (the parallel graph-coloring/wide-constraint
// build the overflow path doesn't run) and `refit` (the port folds the broad-phase re-fit into finalize,
// so its cost lands in `transforms`).

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

/** A fresh zeroed profile (world->profile = (b3Profile){0}). */
export function newProfile(): Profile {
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

/** Zero every field in place (reused each step, no per-step allocation). */
export function resetProfile(p: Profile): void {
    p.step = 0;
    p.pairs = 0;
    p.collide = 0;
    p.solve = 0;
    p.solverSetup = 0;
    p.constraints = 0;
    p.prepareConstraints = 0;
    p.integrateVelocities = 0;
    p.warmStart = 0;
    p.solveImpulses = 0;
    p.integratePositions = 0;
    p.relaxImpulses = 0;
    p.applyRestitution = 0;
    p.storeImpulses = 0;
    p.splitIslands = 0;
    p.transforms = 0;
    p.sensorHits = 0;
    p.jointEvents = 0;
    p.hitEvents = 0;
    p.refit = 0;
    p.bullets = 0;
    p.sleepIslands = 0;
    p.sensors = 0;
}

/** Wall-clock cursor in ms (b3GetTicks). */
export function ticks(): number {
    return performance.now();
}

/** Milliseconds elapsed since a cursor from {@link ticks} (b3GetMilliseconds). */
export function elapsed(since: number): number {
    return performance.now() - since;
}

/** A resettable cursor for the accumulate-and-reset pattern (b3GetMillisecondsAndReset). */
export type Timer = { t: number };

/** Start a resettable cursor. */
export function makeTimer(): Timer {
    return { t: performance.now() };
}

/** Elapsed ms since the cursor's last mark, then advance the cursor to now (b3GetMillisecondsAndReset). */
export function reset(timer: Timer): number {
    const now = performance.now();
    const d = now - timer.t;
    timer.t = now;
    return d;
}
