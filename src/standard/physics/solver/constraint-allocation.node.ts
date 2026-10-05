import { expect, setDefaultTimeout, test } from "bun:test";
import { resolve } from "node:path";
import {
    allocationFailure,
    sampleAllocation,
} from "../../../../diagnostics/first-person-allocation/allocation";
import { CEILING } from "../../../../scripts/test-tiers";

setDefaultTimeout(CEILING.node);

const scenes = {
    authored: "64 authored distance and spherical joint pendulums",
    spherical: "64 spherical joints with a localFrameB on static anchors",
};

for (const [input, scene] of Object.entries(scenes)) {
    test(`a warm step of ${scene} allocates no JavaScript heap`, async () => {
        const sample = await sampleAllocation(
            resolve(import.meta.dir, "constraint-allocation.entry.ts"),
            { warm: 6000, frames: 600, input },
        );
        expect(sample.control.length).toBeGreaterThan(0);
        const failure = allocationFailure(sample);
        if (failure !== undefined) throw new Error(failure);
    });
}
