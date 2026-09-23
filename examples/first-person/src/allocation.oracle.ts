import { resolve } from "node:path";
import {
    type AllocationSite,
    type AllocationWindow,
    allocationFailure,
    type PageSample,
    samplePage,
    windowBytes,
} from "@dylanebert/shallot/harness/allocation";
import { check } from "@dylanebert/shallot/harness/check";

function table(label: string, sites: readonly AllocationSite[], span?: AllocationWindow): string {
    const bytes = windowBytes({ sites });
    const bracket = span
        ? `; page frame-counter bracket ${span.frames}–${span.framesAtMost} frames`
        : "";
    return [
        `${label}: ${bytes} bytes${bracket} at ${sites.length} sites`,
        ...sites.map(
            (row) =>
                `  ${String(row.bytes).padStart(10)} B  ${String(row.count).padStart(6)} samples  ${row.site}`,
        ),
    ].join("\n");
}

function traceReport(sample: PageSample): string {
    const { trace } = sample;
    const own = trace.collections.filter((row) => !row.forced);
    const minor = own.filter((row) => row.kind === "minor");
    const major = own.filter((row) => row.kind === "major");
    const promoted = own.reduce((total, row) => total + row.promoted, 0);
    const forced = trace.collections.filter((row) => row.forced);
    const pause = (rows: typeof trace.collections) =>
        rows.length === 0
            ? "0"
            : `${(rows.reduce((total, row) => total + row.pause, 0) / rows.length).toFixed(2)} ms mean, ${Math.max(...rows.map((row) => row.pause)).toFixed(2)} ms max`;
    const withCollection = trace.longFrames.filter((row) => row.collection !== null);
    return [
        `gc over the steady windows: ${minor.length} minor (${pause(minor)}), ${major.length} major (${pause(major)}), ${promoted} bytes promoted; ${forced.length} further collections the harness forced before each window`,
        ...own.map(
            (row) =>
                `  ${row.kind} ${row.pause.toFixed(2)} ms, ${row.promoted} B promoted, reason "${row.reason}"`,
        ),
        `frame pacing: ${trace.frames} presented-frame intervals, median ${trace.period.toFixed(2)} ms; ${trace.longFrames.length} long frames above 1.5× that`,
        `  ${withCollection.length} coincide with a collection (${withCollection.filter((row) => row.collection?.kind === "major").length} major), ${trace.longFrames.length - withCollection.length} coincide with none`,
        ...trace.longFrames
            .slice(0, 20)
            .map(
                (row) =>
                    `  ${row.gap.toFixed(2)} ms  ${row.collection ? `${row.collection.kind} ${row.collection.pause.toFixed(2)} ms, ${row.collection.promoted} B promoted` : "no collection"}`,
            ),
    ].join("\n");
}

check(
    "first-person page frames allocate nothing on a real display adapter",
    {
        claim: "a warm requestAnimationFrame frame of the production first-person web build, stepped by its own page loop in a headed browser on a real adapter, allocates no JavaScript heap in any steady window, with nothing surviving a full collection and no major collection or promoted bytes, so no periodic scavenge follows play",
        size: "integration",
        requires: ["display"],
        subject: ["examples/first-person"],
        budget: 20_000,
    },
    async () => {
        // 480 warm frames and 120-frame windows are what fit the build, the headed launch and six profiled
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
        // The seat beside the adapter: which monitor the page was placed on and verified to have presented
        // on, what that monitor runs at, and what the page actually presented at. A window's wall-clock
        // length is its frames over that rate, so a reader can see which display a table below came from.
        const seat = `display seat ${sample.display.declared} at ${sample.display.refreshRate.toFixed(2)} Hz, page presented at ${sample.display.presented.toFixed(1)} Hz`;
        const tables = [
            `${sample.runtime} on ${sample.adapter}`,
            seat,
            ...sample.windows.map((window) => table(window.label, window.sites, window)),
            table("survivors after a full collection", sample.survivors),
            table("control", sample.control, sample.controlSpan),
            traceReport(sample),
        ].join("\n");
        console.log(tables);
        const metadata = { runtime: `${sample.runtime}; ${seat}`, hardware: sample.adapter };
        // Every red condition is read, then reported together: a page that fails more than one of them
        // shows all of them in one verdict, so a mutation aimed at one is visible beside the rest.
        const failures: string[] = [];
        // The control literal is sampled under the run frame. Compare conservative per-frame bounds from
        // the page's own brackets: the control's lower bound must exceed the A/A window's upper bound.
        const at = (sites: readonly AllocationSite[]) =>
            sites.find((row) => row.site === sample.loopSite)?.bytes ?? 0;
        const controlMinimum = at(sample.control) / sample.controlSpan.framesAtMost;
        const repeatMaximum = at(sample.windows[2].sites) / sample.windows[2].frames;
        console.log(
            `control at ${sample.loopSite}: at least ${controlMinimum.toFixed(1)}/f against A/A at most ${repeatMaximum.toFixed(1)}/f`,
        );
        if (controlMinimum <= repeatMaximum)
            failures.push(
                `the control's lower bound is ${controlMinimum.toFixed(1)}/f at ${sample.loopSite}, not above the A/A upper bound of ${repeatMaximum.toFixed(1)}/f: either the sampler did not detect the control literal or ordinary frame allocation obscured it`,
            );
        const allocation = allocationFailure(sample);
        if (allocation !== undefined) failures.push(allocation);
        if (sample.survivors.length > 0)
            failures.push(
                `the steady window allocated objects that survive a full collection:\n${sample.survivors.map((row) => `  ${row.bytes} B at ${row.site}`).join("\n")}`,
            );
        // the harness forces a full collection before each window and the sampling profiler runs its own,
        // so those are excluded from the gate and reported beside it; what steady play itself collects is
        // what this asserts
        const own = sample.trace.collections.filter((row) => !row.forced);
        const major = own.filter((row) => row.kind === "major");
        const promoted = own.reduce((total, row) => total + row.promoted, 0);
        if (major.length > 0 || promoted > 0)
            failures.push(
                `steady play ran ${major.length} major collections and promoted ${promoted} bytes`,
            );
        if (failures.length > 0)
            throw Object.assign(new Error(`${failures.join("\n")}\n${tables}`), metadata);
        return metadata;
    },
);
