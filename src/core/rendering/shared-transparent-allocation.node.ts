import { setDefaultTimeout, test } from "bun:test";
import { resolve } from "node:path";
import {
    allocationFailure,
    sampleAllocation,
    windowBytes,
} from "../../../diagnostics/first-person-allocation/allocation";
import { CEILING } from "../../../scripts/test-tiers";

setDefaultTimeout(CEILING.node);

test("steady transparent phase keying, sorting and line staging allocate no JavaScript heap", async () => {
    const entry = resolve(
        import.meta.dir,
        "../../../diagnostics/first-person-allocation/transparent-phase.entry.ts",
    );
    for (const input of ["phase", "segments", "stage"]) {
        const sample = await sampleAllocation(entry, { warm: 6000, frames: 600, input });
        const controlBytes = windowBytes({ sites: sample.control });
        if (controlBytes === 0) throw new Error(`${input} allocation control was invisible`);
        const failure = allocationFailure(sample);
        if (failure) throw new Error(`${input} allocation sample failed:\n${failure}`);
        console.info(`${input} steady heap profile`, {
            controlBytes,
            windows: sample.windows.map((window) => ({
                label: window.label,
                frames: window.frames,
                bytes: windowBytes(window),
                sites: window.sites.length,
            })),
        });
    }
});
