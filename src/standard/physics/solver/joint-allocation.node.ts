import { setDefaultTimeout, test } from "bun:test";
import { resolve } from "node:path";
import {
    allocationFailure,
    sampleAllocation,
    windowBytes,
} from "../../../../diagnostics/first-person-allocation/allocation";
import { CEILING } from "../../../../scripts/test-tiers";

setDefaultTimeout(CEILING.node);

for (const count of [0, 4]) {
    test(`a warm GPU-backed scene with every joint type allocates no JavaScript heap at ${count} requested threads`, async () => {
        const sample = await sampleAllocation(
            resolve(import.meta.dir, "joint-allocation.entry.ts"),
            { warm: 6000, frames: 600, input: String(count) },
        );
        if (sample.control.length === 0 || windowBytes({ sites: sample.control }) <= 0)
            throw new Error("the allocation control was not observed");
        const failure = allocationFailure(sample);
        if (failure !== undefined) throw new Error(failure);
    });
}
