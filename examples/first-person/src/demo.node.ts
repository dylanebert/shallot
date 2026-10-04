import { setDefaultTimeout, test } from "bun:test";

import { CEILING } from "../../../scripts/test-tiers";

setDefaultTimeout(CEILING.node);

import { resolve } from "node:path";
import {
    allocationFailure,
    sampleAllocation,
    windowBytes,
} from "../../../diagnostics/first-person-allocation/allocation";

test("a warm fixed step of the actual first-person GPU-backed gameplay composition allocates no JavaScript heap, so no periodic scavenge follows play", async () => {
    // Exercise the once-per-escape collider refit path before measuring; a compile still refuses a reading.
    const sample = await sampleAllocation(resolve(import.meta.dir, "allocation.entry.ts"), {
        warm: 6000,
        frames: 600,
    });
    // The entry's control literal, attributed as the windows are, proves the sampler sees subject
    // allocation; without it an empty site set proves nothing.
    const control = { label: "control", sites: sample.control };
    if (control.sites.length === 0 || windowBytes(control) <= 0)
        throw new Error(
            "inconclusive: the sampler attributed no site to the entry's control literal",
        );
    const failure = allocationFailure(sample);
    if (failure !== undefined) throw new Error(failure);
});
