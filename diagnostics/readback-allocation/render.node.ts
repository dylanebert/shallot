import { setDefaultTimeout, test } from "bun:test";
import { resolve } from "node:path";
import { allocationFailure, sampleAllocation, windowBytes } from "../first-person-allocation/allocation";

setDefaultTimeout(20_000);

test("steady play renders with the default plugins without allocating JavaScript heap", async () => {
    const sample = await sampleAllocation(resolve(import.meta.dir, "render.entry.ts"), { warm: 1200, frames: 600 });
    if (windowBytes({ sites: sample.control }) === 0) throw new Error("allocation control was invisible");
    const failure = allocationFailure(sample);
    if (failure) throw new Error(failure);
}, 20_000);
