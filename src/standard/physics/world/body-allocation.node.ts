import { expect, setDefaultTimeout, test } from "bun:test";
import { resolve } from "node:path";
import { CEILING } from "@dylanebert/shallot/testing";
import {
    allocationFailure,
    sampleAllocation,
} from "../../../../diagnostics/first-person-allocation/allocation";

setDefaultTimeout(CEILING.node);

test("body velocity, transform, force, impulse, sleep, wake and contact churn allocate no steady JavaScript heap", async () => {
    const entry = resolve(import.meta.dir, "body-allocation.entry.ts");
    const sample = await sampleAllocation(entry, { warm: 600, frames: 600 });
    expect(sample.control.length).toBeGreaterThan(0);
    const failure = allocationFailure(sample);
    if (failure !== undefined) throw new Error(failure);
    const allocating = await sampleAllocation(entry, {
        warm: 600,
        frames: 600,
        input: "allocating",
    });
    expect(allocationFailure(allocating)).toContain("allocated JavaScript heap");
});
