import { expect, setDefaultTimeout, test } from "bun:test";
import { resolve } from "node:path";
import { CEILING } from "@dylanebert/shallot/testing";
import {
    allocationFailure,
    sampleAllocation,
} from "../../../../diagnostics/first-person-allocation/allocation";

setDefaultTimeout(CEILING.node);

test("warm kernel and ECS physics queries allocate no JavaScript heap", async () => {
    const sample = await sampleAllocation(resolve(import.meta.dir, "query-allocation.entry.ts"), {
        warm: 1800,
    });
    expect(sample.control.length).toBeGreaterThan(0);
    const failure = allocationFailure(sample);
    if (failure !== undefined) throw new Error(failure);
});
