import { kernel } from "../kernel/kernel";
import type { WorldState } from "./world";

/** One step's kernel-measured timings in milliseconds (Box3D b3Profile), diagnostic only.
 * Resolution follows the platform clock, coarsened in pages without cross-origin isolation. */
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

let timings = new Float32Array(0);
/** Read the kernel's profile into the reusable API observation. */
export function readStepProfile(world: WorldState): void {
    const k = kernel(world.ecsState);
    const ptr = k.stepProfilePtr(world.worldId);
    if (timings.buffer !== k.memory.buffer) timings = new Float32Array(k.memory.buffer);
    const start = ptr >>> 2;
    const p = world.profile;
    p.step = timings[start];
    p.pairs = timings[start + 1];
    p.collide = timings[start + 2];
    p.solve = timings[start + 3];
    p.solverSetup = timings[start + 4];
    p.constraints = timings[start + 5];
    p.prepareConstraints = timings[start + 6];
    p.integrateVelocities = timings[start + 7];
    p.warmStart = timings[start + 8];
    p.solveImpulses = timings[start + 9];
    p.integratePositions = timings[start + 10];
    p.relaxImpulses = timings[start + 11];
    p.applyRestitution = timings[start + 12];
    p.storeImpulses = timings[start + 13];
    p.splitIslands = timings[start + 14];
    p.transforms = timings[start + 15];
    p.sensorHits = timings[start + 16];
    p.jointEvents = timings[start + 17];
    p.hitEvents = timings[start + 18];
    p.refit = timings[start + 19];
    p.bullets = timings[start + 20];
    p.sleepIslands = timings[start + 21];
    p.sensors = timings[start + 22];
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
