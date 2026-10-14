import { expect, setDefaultTimeout, test } from "bun:test";
import { resolve } from "node:path";
import { TIMEOUT } from "@dylanebert/shallot/testing";
import {
    allocationFailure,
    sampleAllocation,
} from "../../../../diagnostics/first-person-allocation/allocation";

setDefaultTimeout(TIMEOUT.node);

const materialMixingScenes = [
    ...[0, 16, 32, 48].map((start) => ({
        label: `mesh contacts ${start + 1}-${start + 16}`,
        input: `mesh:${start}`,
        warm: 2400,
    })),
    { label: "compound contacts", input: "compound", warm: 3000 },
];

for (const { label, input, warm } of materialMixingScenes) {
    test(`warm ${label} with custom material mixing allocate no JavaScript heap`, async () => {
        const sample = await sampleAllocation(
            resolve(import.meta.dir, "material-allocation.entry.ts"),
            { warm, frames: 300, input },
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
                frames: 300,
                input: String(count),
            },
        );
        expect(sample.control.length).toBeGreaterThan(0);
        const failure = allocationFailure(sample);
        if (failure !== undefined) throw new Error(failure);
    });
}
