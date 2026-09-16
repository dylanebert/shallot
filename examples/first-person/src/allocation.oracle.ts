import { resolve } from "node:path";
import {
    type AllocationSite,
    type PageSample,
    samplePage,
    windowBytes,
} from "@dylanebert/shallot/harness/allocation";
import { check } from "@dylanebert/shallot/harness/check";
import { readSanctions, type SanctionRow } from "@dylanebert/shallot/harness/surface";

function table(label: string, sites: readonly AllocationSite[], frames: number): string {
    const bytes = windowBytes({ label, sites });
    const perFrame = (value: number) => (value / frames).toFixed(1);
    return [
        `${label}: ${bytes} bytes (${perFrame(bytes)}/f) at ${sites.length} sites`,
        ...sites.map(
            (row) =>
                `  ${String(row.bytes).padStart(10)}  ${perFrame(row.bytes).padStart(8)}/f  ${(row.count / frames).toFixed(2).padStart(6)}×/f  ${row.site}`,
        ),
    ].join("\n");
}

// a measured site is `<function> <file>:<line>`; a sanction row names the `<file>:<line>` half, the one a
// minified production build keeps
function where(site: string): string {
    return site.slice(site.indexOf(" ") + 1);
}

// the declared sanctions beside their observed counts, printed with every verdict so the person sees what
// is tolerated and at what count each time memory is reviewed
function sanctionTable(rows: readonly SanctionRow[], sample: PageSample): string {
    const observed = (site: string) =>
        sample.windows.map(
            (window) =>
                (window.sites.find((row) => where(row.site) === site)?.count ?? 0) / sample.frames,
        );
    return [
        `sanctions: ${rows.length} rows, ${rows.filter((row) => row.approved === "").length} unapproved`,
        ...rows.map((row) => {
            const counts = observed(row.site)
                .map((value) => value.toFixed(2))
                .join(" / ");
            return `  ${row.count}×/f declared, ${counts} observed  ${row.approved || "(unapproved)"}  ${row.site} — ${row.reason}`;
        }),
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
        claim: "a warm requestAnimationFrame frame of the production first-person web build, stepped by its own page loop in a headed browser on a real adapter, allocates no JavaScript heap outside the declared per-frame sanctions, each at its derived count, with nothing surviving a full collection and no major collection or promoted bytes, so no periodic scavenge follows play",
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
        const declared = readSanctions(process.cwd());
        const tables = [
            `${sample.runtime} on ${sample.adapter}`,
            ...sample.windows.map((window) => table(window.label, window.sites, sample.frames)),
            table("survivors after a full collection", sample.survivors, sample.frames),
            table("control over 60 frames", sample.control, 60),
            sanctionTable(declared.rows, sample),
            traceReport(sample),
        ].join("\n");
        console.log(tables);
        const metadata = { runtime: sample.runtime, hardware: sample.adapter };
        // Every red condition is read, then reported together: a page that fails more than one of them
        // shows all of them in one verdict, so a mutation aimed at one is visible beside the rest.
        const failures: string[] = [];
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
            failures.push(
                `the control read ${controlAt.toFixed(1)}/f at ${sample.loopSite}, not above the A/A window's ${repeatAt.toFixed(1)}/f: either the sampler is not attributing the control literal to the frame loop, or the page allocates at least as much there on its own`,
            );
        failures.push(...declared.errors);

        // The Locked decision's red conditions: no byte outside the declared sanctions, every sanctioned
        // site at its derived count, no stale row, nothing alive after a full collection, and no major
        // collection or promoted bytes in steady play.
        const sanctioned = new Set(declared.rows.map((row) => row.site));
        const unsanctioned = sample.windows.flatMap((window) =>
            window.sites
                .filter((row) => !sanctioned.has(where(row.site)))
                .map((row) => `  ${window.label}: ${row.bytes} B at ${row.site}`),
        );
        if (unsanctioned.length > 0)
            failures.push(
                `warm first-person page frames allocate outside the sanctions:\n${unsanctioned.join("\n")}`,
            );
        const wrongCount: string[] = [];
        const stale: string[] = [];
        for (const row of declared.rows) {
            for (const window of sample.windows) {
                const seen = window.sites.find((site) => where(site.site) === row.site);
                if (!seen) {
                    stale.push(`  ${window.label}: ${row.site} allocates nothing`);
                    continue;
                }
                // the count is derived per frame, so the window's samples must round to it exactly
                const per = Math.round(seen.count / sample.frames);
                if (per !== row.count)
                    wrongCount.push(
                        `  ${window.label}: ${row.site} read ${(seen.count / sample.frames).toFixed(2)}×/f against the declared ${row.count}`,
                    );
            }
        }
        if (stale.length > 0) failures.push(`stale sanction rows:\n${stale.join("\n")}`);
        if (wrongCount.length > 0)
            failures.push(`sanctioned sites off their derived counts:\n${wrongCount.join("\n")}`);
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
