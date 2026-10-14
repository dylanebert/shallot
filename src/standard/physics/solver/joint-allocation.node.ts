import { setDefaultTimeout, test } from "bun:test";
import { resolve } from "node:path";
import { TIMEOUT } from "@dylanebert/shallot/testing";
import {
    allocationFailure,
    sampleAllocation,
    windowBytes,
} from "../../../../diagnostics/first-person-allocation/allocation";

setDefaultTimeout(TIMEOUT.node);

// Two threads hold the threaded claim: the sampler measures only the main isolate, whose code is the
// same for any pool of two or more. Four oversubscribed a four-vCPU Windows runner beside the sampled
// child's own threads, and the spinning stage barriers stretched the run past its ceiling.
for (const count of [0, 2]) {
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
