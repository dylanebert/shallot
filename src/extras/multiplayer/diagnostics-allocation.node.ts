import { expect, setDefaultTimeout, test } from "bun:test";
import { resolve } from "node:path";
import { TIMEOUT } from "@dylanebert/shallot/testing";
import {
    allocationFailure,
    sampleAllocation,
} from "../../../diagnostics/first-person-allocation/allocation";

setDefaultTimeout(TIMEOUT.node);
test("steady network diagnostics and ping/pong recording allocate nothing", async () => {
    const sample = await sampleAllocation(
        resolve(import.meta.dir, "diagnostics-allocation.fixture.ts"),
        { warm: 24000, frames: 120 },
    );
    expect(sample.control.length).toBeGreaterThan(0);
    const failure = allocationFailure(sample);
    if (failure) throw new Error(failure);
});
