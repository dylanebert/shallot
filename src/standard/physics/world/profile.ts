/**
 * One step's phase timings in milliseconds (Box3D b3Profile), diagnostic only.
 * Resolution is the host's `performance.now()`, coarsened in pages without cross-origin isolation.
 * Fields inside the kernel's whole-solve crossing remain zero when not timed separately.
 */
export type StepProfile = {
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

export const PROFILE_FIELDS = Object.keys(createStepProfile()) as (keyof StepProfile)[];

export function resetStepProfile(profile: StepProfile): void {
    for (let i = 0; i < PROFILE_FIELDS.length; i++) profile[PROFILE_FIELDS[i]] = 0;
}

export function createStepProfile(): StepProfile {
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
