import {
    CLOCK_SLOTS,
    type Profile,
    type StepClock,
    zeroProfile,
} from "../../standard/physics/world/clock";

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
