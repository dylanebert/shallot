import { setDefaultTimeout, test } from "bun:test";
import { resolve } from "node:path";
import {
    allocationFailure,
    sampleAllocation,
    windowBytes,
} from "../../../../diagnostics/first-person-allocation/allocation";
import { CEILING } from "../../../../scripts/test-tiers";

setDefaultTimeout(CEILING.node);

test("a warm GPU-backed step moving kinematic proxies outside their fat AABBs allocates no JavaScript heap", async () => {
    const sample = await sampleAllocation(resolve(import.meta.dir, "tree-allocation.entry.ts"), {
        warm: 6000,
        frames: 600,
    });
    if (sample.control.length === 0 || windowBytes({ sites: sample.control }) <= 0)
        throw new Error("the allocation control was not observed");
    const failure = allocationFailure(sample);
    if (failure !== undefined) throw new Error(failure);
});
