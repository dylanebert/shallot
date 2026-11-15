import { expect, setDefaultTimeout, test } from "bun:test";
import { resolve } from "node:path";
import { TIMEOUT } from "@dylanebert/shallot/testing";
import {
    allocationFailure,
    sampleAllocation,
} from "../../../diagnostics/first-person-allocation/allocation";

setDefaultTimeout(TIMEOUT.node);
for (const input of ["unrestored", "restore"]) {
    test(`physics ticks allocate no heap ${input === "restore" ? "after every restore" : "without recovery"}`, async () => {
        const sample = await sampleAllocation(
            resolve(import.meta.dir, "recovery-allocation.fixture.ts"),
            { input, warm: 18000, frames: 120, retainOptimizedSites: true },
        );
        expect(sample.control.length).toBeGreaterThan(0);
        // Recovery copies the captured image; the following tick has no such admission.
        const windows = sample.windows.map((window) => ({
            ...window,
            sites: window.sites.filter(
                (site) => !site.stack?.some((frame) => frame.functionName === "restore"),
            ),
        }));
        const failure = allocationFailure({ ...sample, windows });
        if (failure) throw new Error(failure);
    });
}
