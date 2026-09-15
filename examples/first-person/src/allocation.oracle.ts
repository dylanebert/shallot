import { resolve } from "node:path";
import {
    type AllocationSite,
    allocatesNothing,
    samplePage,
    windowBytes,
} from "@dylanebert/shallot/harness/allocation";
import { check } from "@dylanebert/shallot/harness/check";

function table(label: string, sites: readonly AllocationSite[], frames: number): string {
    const bytes = windowBytes({ label, sites });
    const perFrame = (value: number) => (value / frames).toFixed(1);
    return [
        `${label}: ${bytes} bytes (${perFrame(bytes)}/f) at ${sites.length} sites`,
        ...sites.map(
            (row) =>
                `  ${String(row.bytes).padStart(10)}  ${perFrame(row.bytes).padStart(8)}/f  ${row.site}`,
        ),
    ].join("\n");
}

check(
    "first-person page frames allocate nothing on a real display adapter",
    {
        claim: "a warm frame of the production first-person web build, stepped by its own page loop in a headed browser on a real adapter, allocates JavaScript heap in the rendering, input or loop code the Node row cannot load",
        size: "integration",
        requires: ["display"],
        subject: ["examples/first-person"],
        budget: 20_000,
    },
    async () => {
        // The page loop runs at the display's rate (144 to 240 Hz here), so 480 warm frames and 120-frame windows
        // are what fit the build, the headed launch and five profiled spans inside the budget. With V8's tier
        // thresholds lowered, a per-frame function reaches TurboFan in about 50 frames; three agreeing windows
        // are the steadiness premise, not the warm's length.
        const sample = await samplePage(resolve(import.meta.dir, ".."), {
            warm: 480,
            frames: 120,
            deadline: performance.now() + 19_000,
        });
        const tables = [
            `${sample.runtime} on ${sample.adapter}`,
            ...sample.windows.map((window) => table(window.label, window.sites, sample.frames)),
            table("control over 60 frames", sample.control, 60),
        ].join("\n");
        console.log(tables);
        const metadata = { runtime: sample.runtime, hardware: sample.adapter };
        // The control literal, attributed under the run frame as the windows are, proves the sampler sees the
        // page's frame loop; without it an empty site set proves nothing.
        if (windowBytes({ label: "control", sites: sample.control }) <= 0)
            throw Object.assign(
                new Error(
                    "inconclusive: the sampler attributed no bytes to the frame-loop control",
                ),
                metadata,
            );
        if (!allocatesNothing(sample))
            throw Object.assign(
                new Error(`warm first-person page frames allocate:\n${tables}`),
                metadata,
            );
        return metadata;
    },
);
