import { expect, setDefaultTimeout, test } from "bun:test";
import { resolve } from "node:path";
import {
    allocationFailure,
    sampleAllocation,
} from "../../../../diagnostics/first-person-allocation/allocation";
import { CEILING } from "../../../../scripts/test-tiers";

setDefaultTimeout(CEILING.node);

// The contact gate's warms: a step's late first optimizations (flush near 3,000 steps, buildJointEvents
// near 6,600) fall between these windows, and a window holding one is refused, naming it.
for (const count of [64, 128]) {
    test(`a warm step of ${count} awake boxes resting inside one static sensor allocates no JavaScript heap in overlap tracking`, async () => {
        const sample = await sampleAllocation(
            resolve(import.meta.dir, "sensor-allocation.entry.ts"),
            {
                warm: count === 128 ? 1800 : 6000,
                frames: count === 128 ? 180 : 600,
                input: String(count),
            },
        );
        expect(sample.control.length).toBeGreaterThan(0);
        const failure = allocationFailure(sample);
        if (failure !== undefined) throw new Error(failure);
    });
}
