import { expect, setDefaultTimeout, test } from "bun:test";
import { resolve } from "node:path";
import {
    allocationFailure,
    sampleAllocation,
} from "../../../../diagnostics/first-person-allocation/allocation";
import { CEILING } from "../../../../scripts/test-tiers";

setDefaultTimeout(CEILING.node);

test("warm internal mover and closest-ray queries over every collider kind allocate no JavaScript heap", async () => {
    // QueryColumns.prepare/stale and the chunk's inlined query glue tier late; the margin covers concurrent compilation.
    const sample = await sampleAllocation(resolve(import.meta.dir, "query-allocation.entry.ts"), {
        warm: 1800,
    });
    expect(sample.control.length).toBeGreaterThan(0);
    const failure = allocationFailure(sample);
    if (failure !== undefined) throw new Error(failure);
});
