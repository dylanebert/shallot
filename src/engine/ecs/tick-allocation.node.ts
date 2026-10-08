import { expect, setDefaultTimeout, test } from "bun:test";
import { resolve } from "node:path";
import {
    allocationFailure,
    sampleAllocation,
} from "../../../diagnostics/first-person-allocation/allocation";
import { CEILING } from "../../../scripts/test-tiers";

setDefaultTimeout(CEILING.node);
test("steady exact ticks allocate no JavaScript heap", async () => {
    const sample = await sampleAllocation(resolve(import.meta.dir, "tick-allocation.entry.ts"), {
        warm: 12000,
    });
    expect(sample.control.length).toBeGreaterThan(0);
    const failure = allocationFailure(sample);
    if (failure !== undefined) throw new Error(failure);
});
