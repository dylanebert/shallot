import { expect, setDefaultTimeout, test } from "bun:test";
import { resolve } from "node:path";
import { TIMEOUT } from "@dylanebert/shallot/testing";
import {
    allocationFailure,
    sampleAllocation,
} from "../../../diagnostics/first-person-allocation/allocation";
import { isAdmittedAllocationSite } from "../../../diagnostics/first-person-allocation/allocation-admissions";

setDefaultTimeout(TIMEOUT.node);

test("warm mesh turnover allocates no JavaScript heap outside the admitted wrappers", async () => {
    // Bound fields and bind scans optimize between 1 200 and 2 400 warm frames.
    const sample = await sampleAllocation(
        resolve(import.meta.dir, "../../../diagnostics/turnover-allocation/mesh.entry.ts"),
        { warm: 2_400, frames: 60 },
    );
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
