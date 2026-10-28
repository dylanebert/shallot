import { expect, setDefaultTimeout, test } from "bun:test";
import { resolve } from "node:path";
import { TIMEOUT } from "@dylanebert/shallot/testing";
import {
    allocationFailure,
    sampleAllocation,
} from "../../../diagnostics/first-person-allocation/allocation";

setDefaultTimeout(TIMEOUT.node);
test("all nine warm manifold calls and caches allocate no JavaScript heap, with an allocating control", async () => {
    const sample = await sampleAllocation(
        resolve(import.meta.dir, "standalone-manifold-allocation.fixture.ts"),
        { warm: 2400 },
    );
    expect(sample.control.length).toBeGreaterThan(0);
    const failure = allocationFailure(sample);
    if (failure !== undefined) throw new Error(failure);
});
