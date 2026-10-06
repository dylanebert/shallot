import { expect, setDefaultTimeout, test } from "bun:test";
import { resolve } from "node:path";
import {
    allocationFailure,
    sampleAllocation,
} from "../../../../diagnostics/first-person-allocation/allocation";
import { CEILING } from "../../../../scripts/test-tiers";

setDefaultTimeout(CEILING.node);

test("steady contact touch-link churn inside an island and merging two islands allocates no JavaScript heap", async () => {
    const sample = await sampleAllocation(resolve(import.meta.dir, "island-allocation.entry.ts"), {
        warm: 600,
        frames: 600,
    });
    expect(sample.control.length).toBeGreaterThan(0);
    const failure = allocationFailure(sample);
    if (failure !== undefined) throw new Error(failure);
});
