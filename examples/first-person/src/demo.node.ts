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
    // 6,000 frames: the once-per-escape refit path (`commitRefit`, the fat-AABB write, the tree enlarge)
    // is called about once a frame, so it reaches TurboFan late; at 1,200 it runs Maglev code inside
    // every window and at 2,400 it can still tier inside the first. From 3,600 all three windows agree.
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
