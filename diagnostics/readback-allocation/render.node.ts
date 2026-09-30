import { setDefaultTimeout, test } from "bun:test";
import { resolve } from "node:path";
import { allocationFailure, sampleAllocation, windowBytes } from "../first-person-allocation/allocation";

import { CEILING } from "../../scripts/test-tiers";

setDefaultTimeout(CEILING.node);

test.todo("rendering-hardening: steady play renders with the default plugins without allocating JavaScript heap", async () => {
    const sample = await sampleAllocation(resolve(import.meta.dir, "render.entry.ts"), { warm: 6000, frames: 600 });
    if (windowBytes({ sites: sample.control }) === 0) throw new Error("allocation control was invisible");
    const failure = allocationFailure(sample);
    if (failure) throw new Error(failure);
});
