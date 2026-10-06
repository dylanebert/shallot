import { expect, setDefaultTimeout, test } from "bun:test";
import { resolve } from "node:path";
import {
    allocationFailure,
    sampleAllocation,
} from "../../../../diagnostics/first-person-allocation/allocation";
import { CEILING } from "../../../../scripts/test-tiers";

setDefaultTimeout(CEILING.node);
for (const input of ["walk", "buffer"]) {
    test(`warmed debug draw ${input} allocates nothing beyond fresh callback arguments`, async () => {
        const sample = await sampleAllocation(
            resolve(import.meta.dir, "draw-allocation.entry.ts"),
            { warm: 12000, input },
        );
        expect(sample.control.length).toBeGreaterThan(0);
        const failure = allocationFailure(sample);
        if (failure !== undefined) throw new Error(failure);
    });
}
test("draw allocation gate rejects an allocating Set control", async () => {
    const sample = await sampleAllocation(resolve(import.meta.dir, "draw-allocation.entry.ts"), {
        warm: 12000,
        input: "leak",
    });
    expect(sample.control.length).toBeGreaterThan(0);
    expect(allocationFailure(sample)).toContain("allocated JavaScript heap");
});
