import type { Plugin, System } from "../../engine";
import {
    CLOCK_SLOTS,
    PhysicsPlugin,
    type PhysicsWorld,
    type Profile,
    physicsWorld,
    type StepClock,
    StepPhysicsSystem,
    zeroProfile,
} from "../../transitional/physics";

/** A wall-clock physics step clock (b3GetTicks / b3GetMillisecondsAndReset), read back through
 * `World.getProfile`. Only the profiler composes it, so the default step does no timing work. */
export function timingClock(): StepClock {
    const starts = new Float64Array(CLOCK_SLOTS);
    const profile = zeroProfile();
    const fields = Object.keys(profile) as (keyof Profile)[];
    return {
        begin(slot) {
            for (const field of fields) profile[field] = 0;
            starts[slot] = performance.now();
        },
        mark(slot) {
            starts[slot] = performance.now();
        },
        span(field, slot) {
            profile[field] = performance.now() - starts[slot];
        },
        lap(field, slot) {
            const now = performance.now();
            profile[field] += now - starts[slot];
            starts[slot] = now;
        },
        read() {
            return { ...profile };
        },
    };
}

// the physics worlds already given a timing clock; a rebuild warms a new world, which gets its own
const timed = new WeakSet<PhysicsWorld>();

// installs the clock before the step, since plugin warms run concurrently and the world appears only once
// the physics warm finishes
const PhysicsClockSystem: System = {
    name: "physics-clock",
    group: "fixed",
    before: [StepPhysicsSystem],
    update(world) {
        const solverWorld = physicsWorld(world);
        if (!solverWorld || timed.has(solverWorld)) return;
        timed.add(solverWorld);
        solverWorld.setClock(timingClock());
    },
};

/**
 * physics phase timings: gives each physics world a {@link timingClock} before its first step, so
 * `physicsWorld(world).getProfile()` reads wall-clock milliseconds per step phase. Without it the step
 * does no timing work and every phase reads zero.
 * @example
 * const config = { plugins: [PhysicsPlugin, PhysicsProfilePlugin] };
 */
export const PhysicsProfilePlugin: Plugin = {
    name: "PhysicsProfile",
    dependencies: [PhysicsPlugin],
    systems: [PhysicsClockSystem],
};
