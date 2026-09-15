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
        claim: "a warm requestAnimationFrame frame of the production first-person web build, stepped by its own page loop in a headed browser on a real adapter, allocates no JavaScript heap in the rendering, loop or other browser-only code under the engine's frame callback, so no periodic scavenge follows play",
        size: "integration",
        requires: ["display"],
        subject: ["examples/first-person"],
        budget: 20_000,
    },
    async () => {
        // 480 warm frames and 120-frame windows are what fit the build, the headed launch and five profiled
        // spans inside the budget on a page that presents at its display's rate. With V8's tier thresholds
        // lowered, a per-frame function reaches TurboFan in about 50 frames; three agreeing windows are the
        // steadiness premise, not the warm's length, so the sampler refuses a slow page by name rather than
        // trimming the warm. The deadline sits 4 s inside the budget, so teardown always runs before the
        // budget ends.
        const sample = await samplePage(resolve(import.meta.dir, ".."), {
            warm: 480,
            frames: 120,
            deadline: performance.now() + 16_000,
        });
        const tables = [
            `${sample.runtime} on ${sample.adapter}`,
            ...sample.windows.map((window) => table(window.label, window.sites, sample.frames)),
            table("control over 60 frames", sample.control, 60),
        ].join("\n");
        console.log(tables);
        const metadata = { runtime: sample.runtime, hardware: sample.adapter };
        // The control literal, attributed under the run frame as the windows are, proves the sampler sees the
        // page's frame loop: the loop's own site must read more with it than the A/A window read without it.
        // A bare nonzero total would pass on any red page's ordinary frame bytes. This gate compares against
        // the bytes under test, so it is a failure of the claim and never a refusal: enough live allocation
        // at the loop's own site to drown the control is itself the red this row exists to report.
        const at = (sites: readonly AllocationSite[]) =>
            sites.find((row) => row.site === sample.loopSite)?.bytes ?? 0;
        const controlAt = at(sample.control) / 60;
        const repeatAt = at(sample.windows[2].sites) / sample.frames;
        console.log(
            `control at ${sample.loopSite}: ${controlAt.toFixed(1)}/f against A/A ${repeatAt.toFixed(1)}/f`,
        );
        if (controlAt <= repeatAt)
            throw Object.assign(
                new Error(
                    `the control read ${controlAt.toFixed(1)}/f at ${sample.loopSite}, not above the A/A window's ${repeatAt.toFixed(1)}/f: either the sampler is not attributing the control literal to the frame loop, or the page allocates at least as much there on its own:\n${tables}`,
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
