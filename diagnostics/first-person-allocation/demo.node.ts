import { setDefaultTimeout, test } from "bun:test";

import { TIMEOUT } from "@dylanebert/shallot/testing";

setDefaultTimeout(TIMEOUT.node);

import { resolve } from "node:path";
import { allocationFailure, sampleAllocation, windowBytes } from "./allocation";
import { isAdmittedAllocationSite } from "./allocation-admissions";

async function expectSteadyStep(input: string) {
    // Exercise the once-per-escape collider refit path before measuring; a compile still refuses a reading.
    const sample = await sampleAllocation(resolve(import.meta.dir, "allocation.entry.ts"), {
        warm: 6000,
        frames: 600,
        input,
    });
    // The entry's control literal, attributed as the windows are, proves the sampler sees subject
    // allocation; without it an empty site set proves nothing.
    const control = { label: "control", sites: sample.control };
    if (control.sites.length === 0 || windowBytes(control) <= 0)
        throw new Error(
            "inconclusive: the sampler attributed no site to the entry's control literal",
        );
    // Admit only the source-checked WebGPU wrappers; any other byte reds.
    const failure = allocationFailure({
        ...sample,
        windows: sample.windows.map((window) => ({
            ...window,
            sites: window.sites.filter((site) => !isAdmittedAllocationSite(site)),
        })),
    });
    if (failure !== undefined) throw new Error(failure);
}

test("a warm fixed step of the actual first-person GPU-backed gameplay composition allocates no JavaScript heap outside WebGPU calls", async () => {
    await expectSteadyStep("");
});

test("a warm fixed step of that composition with a world Text label, whose core pipeline submits a GPU frame every step, allocates no JavaScript heap outside WebGPU calls", async () => {
    await expectSteadyStep("text");
});
