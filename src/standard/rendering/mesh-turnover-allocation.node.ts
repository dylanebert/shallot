import { afterAll, beforeAll, expect, setDefaultTimeout, test } from "bun:test";
import { resolve } from "node:path";
import { TIMEOUT } from "@dylanebert/shallot/testing";
import {
    allocationFailure,
    type PreparedAllocationSampler,
    prepareAllocationSampler,
} from "../../../diagnostics/first-person-allocation/allocation";
import { isAdmittedAllocationSite } from "../../../diagnostics/first-person-allocation/allocation-admissions";

setDefaultTimeout(TIMEOUT.node);

let sampler: PreparedAllocationSampler;
beforeAll(async () => {
    sampler = await prepareAllocationSampler(
        resolve(import.meta.dir, "../../../diagnostics/turnover-allocation/mesh.entry.ts"),
    );
}, TIMEOUT.startup);
afterAll(() => sampler?.dispose());

test("warm mesh turnover allocates no JavaScript heap outside the admitted wrappers", async () => {
    // Early-returning steady functions first optimize as late as frame 11 800 (renderPointShadows), after
    // V8's wait following a feedback change; the entry's 2 400-frame session rewarms put the A/A window past it.
    const sample = await sampler.sample({ warm: 2_400, frames: 60 });
    expect(sample.control.length).toBeGreaterThan(0);
    const failure = allocationFailure({
        ...sample,
        windows: sample.windows.map((window) => ({
            ...window,
            sites: window.sites.filter((site) => !isAdmittedAllocationSite(site)),
        })),
    });
    if (failure !== undefined) throw new Error(failure);
});
