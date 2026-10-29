import { expect, setDefaultTimeout, test } from "bun:test";
import { resolve } from "node:path";
import { TIMEOUT } from "@dylanebert/shallot/testing";
import {
    allocationFailure,
    sampleAllocation,
} from "../../../diagnostics/first-person-allocation/allocation";

setDefaultTimeout(TIMEOUT.node);
const entry = resolve(import.meta.dir, "fixtures/body-query-allocation.entry.ts");
test("warmed body queries and borrowed transform and inertia reads allocate nothing", async () => {
    const sample = await sampleAllocation(entry, { warm: 12000 });
    expect(sample.control.length).toBeGreaterThan(0);
    const failure = allocationFailure(sample);
    if (failure !== undefined) throw new Error(failure);
});
test("body transform allocation gate rejects a spread result control", async () => {
    const sample = await sampleAllocation(entry, { warm: 12000, input: "transform-leak" });
    expect(sample.control.length).toBeGreaterThan(0);
    expect(allocationFailure(sample)).toContain("allocated JavaScript heap");
});
test("body query allocation gate rejects a spread result control", async () => {
    const sample = await sampleAllocation(entry, { warm: 12000, input: "leak" });
    expect(sample.control.length).toBeGreaterThan(0);
    expect(allocationFailure(sample)).toContain("allocated JavaScript heap");
});
