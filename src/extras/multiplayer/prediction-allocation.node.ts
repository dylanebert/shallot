import { expect, setDefaultTimeout, test } from "bun:test";
import { resolve } from "node:path";
import { TIMEOUT } from "@dylanebert/shallot/testing";
import {
    allocationFailure,
    sampleAllocation,
} from "../../../diagnostics/first-person-allocation/allocation";

setDefaultTimeout(TIMEOUT.node);

test("steady first-person replay ticks allocate nothing beyond snapshot capture", async () => {
    const sample = await sampleAllocation(
        resolve(import.meta.dir, "prediction-allocation.fixture.ts"),
        { warm: 24000, frames: 120 },
    );
    expect(sample.control.length).toBeGreaterThan(0);
    // Capture owns a new World image; restore is once per replay, not per replay tick.
    const windows = sample.windows.map((window) => ({
        ...window,
        sites: window.sites.filter(
            (site) =>
                !site.stack?.some(
                    (frame) =>
                        frame.functionName === "snapshot" || frame.functionName === "restore",
                ),
        ),
    }));
    const failure = allocationFailure({ ...sample, windows });
    if (failure) throw new Error(failure);
});
