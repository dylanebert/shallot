import { expect, setDefaultTimeout, test } from "bun:test";
import { TIMEOUT } from "@dylanebert/shallot/testing";
import {
    allocationFailure,
    sampleAllocation,
    windowBytes,
} from "../../../diagnostics/first-person-allocation/allocation";

setDefaultTimeout(TIMEOUT.node);

test("1,000 unchanged labels allocate no JavaScript heap while maintaining their layouts", async () => {
    const sample = await sampleAllocation(
        `${import.meta.dir}/fixtures/layout-allocation.entry.ts`,
        {
            warm: 6000,
            frames: 600,
        },
    );
    expect(windowBytes({ sites: sample.control })).toBeGreaterThan(0);
    const failure = allocationFailure(sample);
    if (failure) throw new Error(failure);
});
