import { expect, setDefaultTimeout, test } from "bun:test";
import { resolve } from "node:path";
import {
    allocationFailure,
    sampleAllocation,
} from "../../../../diagnostics/first-person-allocation/allocation";
import { CEILING } from "../../../../scripts/test-tiers";

setDefaultTimeout(CEILING.node);

for (const kind of ["mesh", "compound"]) {
    test(`warm ${kind} contacts with custom material mixing allocate no JavaScript heap`, async () => {
        const sample = await sampleAllocation(
            resolve(import.meta.dir, "material-allocation.entry.ts"),
            { warm: 6000, frames: 600, input: kind },
        );
        expect(sample.control.length).toBeGreaterThan(0);
        const failure = allocationFailure(sample);
        if (failure !== undefined) throw new Error(failure);
    });
}

// 128 is the fewest boxes whose red names every site the 1,000-box scene named before the fix.
for (const count of [64, 128]) {
    test(`a warm step of ${count} awake boxes sliding on a kinematic platform allocates no JavaScript heap in the contact path`, async () => {
        const sample = await sampleAllocation(
            resolve(import.meta.dir, "contact-allocation.entry.ts"),
            {
                warm: 6000,
                frames: 600,
                input: String(count),
            },
        );
        expect(sample.control.length).toBeGreaterThan(0);
        const failure = allocationFailure(sample);
        if (failure !== undefined) throw new Error(failure);
    });
}
