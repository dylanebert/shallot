import { expect, setDefaultTimeout, test } from "bun:test";
import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { CEILING } from "@dylanebert/shallot/testing";
import {
    allocationFailure,
    sampleAllocation,
} from "../../../diagnostics/first-person-allocation/allocation";
import {
    assertAllocationWrappers,
    isAdmittedAllocationSite,
} from "../../../diagnostics/first-person-allocation/allocation-admissions";

setDefaultTimeout(CEILING.node);
test("runApp without a fence-wait reader allocates no fence-reaction heap per frame", async () => {
    assertAllocationWrappers(
        readFileSync(resolve(import.meta.dir, "../runtime/allocation-site.ts"), "utf8"),
    );
    // MAX_FRAMES_IN_FLIGHT admits only six frames per 60-frame sampler chunk; the remaining callbacks are skipped.
    const sample = await sampleAllocation(resolve(import.meta.dir, "run-allocation.entry.ts"), {
        warm: 12_000,
    });
    expect(sample.control.length).toBeGreaterThan(0);
    const harnessSites = sample.windows.flatMap((window) =>
        window.sites.filter((site) => site.site.includes("run-allocation.entry.ts")),
    );
    expect(harnessSites).toEqual([]);
    // Admit only source-checked WebGPU wrappers; the complete unadmitted sample still gates steady play.
    const failure = allocationFailure({
        ...sample,
        windows: sample.windows.map((window) => ({
            ...window,
            sites: window.sites.filter((site) => !isAdmittedAllocationSite(site)),
        })),
    });
    if (failure !== undefined) throw new Error(failure);
});
