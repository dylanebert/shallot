import { skipReason, teardownBridge, verify } from "./verify";

// S1 of `shallot-demo-startup-stall`: discriminate the demo's startup ~1s stall by pipeline-label
// KIND — main-thread-eligible sync path vs genuinely-async GPU-thread await — in one instrumented
// `shallot verify --attribution` run (`bin/verify.ts`). Re-runnable — this is the oracle the table
// below is read from, not a one-off console poke.
//
// What this instrument establishes, and what it does NOT: `extras/profile/index.ts`'s own docblock on
// `Profile.compile` (and `.claude/rules/testing.md`'s pre-existing bullet on the compile/startup axis)
// already state that a synchronous `create*Pipeline`/`createComputePipeline` call's recorded duration
// is a near-zero STUB, not a measurement — Dawn defers the real shader compile past that call's
// return, so the cost never lands in the timed window at all, and summing stubs cannot recover it.
// So `isUnmeasuredSyncStub` below decides KIND only: a label recorded off an un-forced sync call ran on
// the main thread for whatever time it actually took (JS is single-threaded, so THAT much is true by
// construction), but the printed duration for that label is a floor on unknown real cost, never a
// measurement of it. The one exception is a label whose pipeline is later force-compiled through the
// `precompile()` drain — its `Compute.precompiled` completion overwrites the stub with a real,
// trustworthy span (`sear-typed-variants` is that case here; see below).
//
//  - Every OTHER registered label is genuinely async, for two distinct reasons that land the same way.
//    `sear-regather-a`/`-b` and `phys-pack-*` (`standard/sear/regather.ts`, `standard/avbd/step.ts`'s
//    `buildPass`) are raw `device.createComputePipelineAsync` calls, never routed through typegpu at
//    all. Every remaining label (`standard/avbd/step.ts`'s per-kernel forcers, `standard/bvh/bounds.ts`
//    /`build.ts`, `standard/part/part.ts`, `standard/slab/index.ts`, `extras/outline/index.ts`) IS a
//    typegpu pipeline, but its `precompile()` forcer hands the drain the RAW pipeline object without
//    unwrapping it first — confirmed by reading each site (`isUnmeasuredSyncStub`'s docblock below). The
//    drain's `initAsync()` is then that pipeline's first resolution attempt, which triggers typegpu's
//    real async constructor. Either way the recorded span is elapsed wall-clock that does NOT block the
//    main thread — UNLESS something else pinned it for that same window. The `longtask`
//    PerformanceObserver installed by `--attribution` (`ATTRIBUTION_INIT_SCRIPT`) is the independent,
//    concurrently-running instrument for exactly that question: it names every real main-thread block
//    ≥50ms across the whole boot, regardless of what caused it, so its total is read against the
//    stub sum below rather than trusted on its own.
//  - `sear-typed-variants` is `precompileVariants`'s `precompile()` forcer label, and it is the ONE
//    real, non-stub sync measurement this table has: its `Compute.precompiled` completion span is
//    timed from BEFORE `unwrapVariant`'s synchronous `Compute.root.unwrap` calls run to AFTER
//    (`initAsync()` then finds its memo already set and resolves instantly, per
//    `node_modules/typegpu/core/pipeline/renderPipeline.js`, so the forcer's span is effectively all
//    sync work, not padded by a real await). That is a genuine measured aggregate of "how long did
//    unwrapping the specializing variants actually take" — unlike the individual per-pipeline
//    `sear-typed-*` stubs it wraps (never overwritten, since `recordCompile` keys by the pipeline's OWN
//    descriptor label, a different string). `preparePipelines()`'s own loop has no equivalent: it is
//    never registered through `precompile()` at all, so none of ITS `sear-typed-*` labels ever get a
//    forcer completion to overwrite their stub — for those, no real number exists through this
//    instrument, full stop.
//
// What this instrument cannot see: `Profile.compile` exposes only a duration per label, not an
// absolute timestamp, so a sync label's block window cannot be overlaid against a `longtask` entry's
// `startTime` to prove they're the *same* interval rather than merely comparable totals — the totals
// comparison below is corroboration, not a per-millisecond alignment. Read the printed longtask list
// as "how much real main-thread block the whole boot produced", compared against the summed sync
// duration, not as a claim that a specific longtask entry IS a specific pipeline's compile.
//
// The lazy-escape question ("does any compile happen after the loading screen") is answered
// empirically, not by reading the source: `--attribution` reads `Profile.compile` once when the boot
// wait concludes (`compile`) and again `ATTRIBUTION_IDLE_MS` later with no page interaction
// (`compileAfterIdle`). A new label, or a growing `pipelineCount`/`totalMs` with no new label (an
// existing label recompiling), between the two reads is a compile that landed after the settle point a
// real loading screen would key off.

interface Args {
    dir: string;
    query: string[];
}

function parseArgs(argv: string[]): Args {
    const out: Args = { dir: "examples/showcase/sandbox", query: [] };
    for (let i = 0; i < argv.length; i++) {
        const arg = argv[i];
        if (arg === "--dir") out.dir = argv[++i] ?? out.dir;
        else if (arg === "--query") out.query.push(argv[++i] ?? "");
        else throw new Error(`unknown option: ${arg}`);
    }
    return out;
}

// the forcer-scope label whose own span wraps synchronous work already counted per-pipeline above —
// reported separately, excluded from the stub-sum/await-sum totals to avoid double-counting.
const VARIANTS_SCOPE_LABEL = "sear-typed-variants";

// classify a label by its OWN registration site, read from source (not a name-pattern guess) — KIND
// only, per the header above, never magnitude. Every `sear-typed-*` pipeline label is force-unwrapped
// (`Compute.root.unwrap`, sear's own `preparePipelines()`/`unwrapVariant`) BEFORE it ever reaches the
// `precompile()` queue, so its later `initAsync()` finds the memo already set and resolves instantly:
// the entire recorded span is the raw SYNCHRONOUS `create*Pipeline` call's near-zero stub (Dawn defers
// the real compile past its return — the header's caveat, not repeated here). `sear-typed-variants`
// itself is excluded by the caller before this runs — it is the one real sync measurement, handled
// separately in `main`. Every other registered label (`standard/avbd/step.ts`, `standard/bvh/bounds.ts`
// /`build.ts`, `standard/part/part.ts`, `standard/slab/index.ts`, `extras/outline/index.ts`, and the
// raw (non-typegpu) `sear-regather-a`/`-b` and `phys-pack-*` `device.createComputePipelineAsync` calls
// in `standard/sear/regather.ts` / `standard/avbd/step.ts`'s `buildPass`) hands its `precompile()`
// forcer the pipeline WITHOUT unwrapping it first — confirmed by reading each registration site, e.g.
// `standard/bvh/bounds.ts`: `precompile(`${scope}-${label}`, () => bound)` where `bound` is never
// passed through `unwrap`. The drain's `initAsync()` is then the FIRST resolution attempt, which
// triggers typegpu's real async path (`device.create*PipelineAsync`) — a genuine, measured GPU-thread
// elapsed span, exactly like the raw regather/pack pipelines.
function isUnmeasuredSyncStub(label: string): boolean {
    return label.startsWith("sear-typed-");
}

function r1(v: number): number {
    return Math.round(v * 10) / 10;
}

async function main(): Promise<void> {
    const args = parseArgs(process.argv.slice(2));
    const reason = skipReason();
    if (reason) {
        console.log(`skip: ${reason}`);
        return;
    }

    console.log(`booting ${args.dir} over the WSL bridge (--attribution)…`);
    const result = await verify(args.dir, [
        "--attribution",
        "--timeout",
        "30000",
        ...args.query.flatMap((q) => ["--query", q]),
    ]);

    if (!result) {
        console.log("no result — verify crashed before reporting (see its own stderr above)");
        process.exitCode = 1;
        return;
    }
    if (!result.pass) {
        console.log(`verify did not pass: ${result.error ?? JSON.stringify(result.errors)}`);
    }
    const attribution = result.attribution;
    if (!attribution) {
        console.log(
            "no attribution on the result — --attribution wasn't honored, or the page.evaluate " +
                "read failed (see verify's own errors above)",
        );
        process.exitCode = 1;
        return;
    }
    const compile = attribution.compile;
    if (!compile) {
        console.log(
            `${args.dir} has no ProfilePlugin (window.__benchmark absent) — Profile.compile is ` +
                "unreachable from this seat for this project. Add ProfilePlugin, or read this table " +
                "against a project that already carries it (examples/showcase/sandbox does).",
        );
        process.exitCode = 1;
        return;
    }

    const rows = Object.entries(compile.pipelines)
        .filter(([label]) => label !== VARIANTS_SCOPE_LABEL)
        .map(([label, ms]) => ({
            label,
            ms,
            kind: isUnmeasuredSyncStub(label)
                ? ("main-thread, UNMEASURED stub" as const)
                : ("gpu-thread await (measured)" as const),
        }))
        .sort((a, b) => b.ms - a.ms);

    const stubMs = rows
        .filter((r) => r.kind === "main-thread, UNMEASURED stub")
        .reduce((s, r) => s + r.ms, 0);
    const awaitMs = rows
        .filter((r) => r.kind === "gpu-thread await (measured)")
        .reduce((s, r) => s + r.ms, 0);
    const variantsScopeMs = compile.pipelines[VARIANTS_SCOPE_LABEL];

    console.log("\n## Attribution — pipeline-label kind, per label\n");
    console.log(
        "CAVEAT: 'main-thread, UNMEASURED stub' rows are near-zero because Dawn defers the real " +
            "shader compile past the sync constructor's return (extras/profile/index.ts's own " +
            "Profile.compile docblock, testing.md's compile/startup bullet) — the ms column there is a " +
            "floor on unknown real cost, NOT a measurement of it. Only 'gpu-thread await (measured)' " +
            "rows and the sear-typed-variants row below are real durations.\n",
    );
    console.log("| label | kind | ms |");
    console.log("|---|---|---|");
    for (const row of rows) {
        console.log(`| ${row.label} | ${row.kind} | ${r1(row.ms)} |`);
    }
    if (variantsScopeMs !== undefined) {
        console.log(
            `| ${VARIANTS_SCOPE_LABEL} (forcer completion — the ONE real, non-stub measurement of sear's sync unwrap cost; wraps the specializing-variant stub rows above without double-counting) | main-thread, MEASURED | ${r1(variantsScopeMs)} |`,
        );
    }
    console.log(
        `| **stub sum (NOT a measurement — see caveat above)** | ${r1(stubMs)}ms across ${rows.filter((r) => r.kind === "main-thread, UNMEASURED stub").length} unmeasured sync-stub labels | — |`,
    );
    console.log(
        `| **the one real sync measurement, excluded from the stub sum above** | ${VARIANTS_SCOPE_LABEL} | ${variantsScopeMs === undefined ? "absent from this run" : `${r1(variantsScopeMs)}ms`} |`,
    );
    console.log(
        `| **gpu-thread await total (measured, but a SUM not a wall-clock span — see note)** | ${r1(awaitMs)}ms | compile.totalMs (real wall span) = ${r1(compile.totalMs)} |`,
    );
    console.log(
        `\npipelineCount=${compile.pipelineCount} pipelineCalls=${compile.pipelineCalls} (labels: ${Object.keys(compile.pipelines).length})`,
    );
    console.log(
        `\nNote: the GPU-thread await total is a SUM of per-label durations, not a wall-clock span — ` +
            `several of these awaits run concurrently (prepareRegather's own Promise.all, AVBD's pack-` +
            `pipeline Promise.all, and separate plugins' warm() hooks all run under one outer ` +
            `Promise.all in build()), so the sum can and does exceed compile.totalMs (the actual first-` +
            `start-to-last-end wall-clock span of the whole compile phase, printed above).`,
    );

    const longTasks = attribution.longTasks;
    const longTaskMs = longTasks.reduce((s, t) => s + t.duration, 0);
    console.log(
        `\nlongtask corroboration: ${longTasks.length} entries, ${r1(longTaskMs)}ms total real main-` +
            `thread block across the whole boot. This is NOT comparable to the ${r1(stubMs)}ms stub ` +
            `sum above (a floor on unknown real cost, not a measurement — see the caveat), and a large ` +
            `gap between the two is not evidence the sync path is cheap: the same gap would print under ` +
            `a moderate real sync cost as under a negligible one, since the stub sum can't move to show ` +
            `the difference. It IS comparable to sear-typed-variants' real ${variantsScopeMs === undefined ? "(absent this run)" : `${r1(variantsScopeMs)}ms`} measurement as an order-of-magnitude check. This instrument ` +
            `cannot align an individual longtask entry to an individual label in either case (` +
            `Profile.compile carries no absolute per-entry timestamp) — only totals.`,
    );

    console.log("\n## Lazy-escape check (compile vs compileAfterIdle)\n");
    const after = attribution.compileAfterIdle;
    if (!after) {
        console.log("compileAfterIdle unavailable — same reason as compile above.");
    } else {
        const newLabels = Object.keys(after.pipelines).filter((l) => !(l in compile.pipelines));
        const grew = Object.entries(after.pipelines).filter(
            ([l, ms]) => l in compile.pipelines && ms !== compile.pipelines[l],
        );
        const escaped =
            newLabels.length > 0 ||
            grew.length > 0 ||
            after.pipelineCount !== compile.pipelineCount;
        // the spec's own named path (forward.ts's "registered after warm remains lazy" note) is
        // specifically sear's specializing-variant compile — read that sub-question apart from a
        // general escape, since a lazy AVBD/BVH kernel compiling well after boot is a different,
        // likely-intentional mechanism (per-feature physics kernels compiling on first use) that this
        // spec's Approach doesn't name.
        const searEscaped = newLabels.some((l) => l.startsWith("sear-typed-"));
        console.log(`escaped (any label): ${escaped}`);
        console.log(`escaped via sear's specializing-variant path specifically: ${searEscaped}`);
        console.log(`new labels after idle: ${newLabels.length ? newLabels.join(", ") : "none"}`);
        console.log(
            `changed-duration labels after idle: ${grew.length ? grew.map(([l]) => l).join(", ") : "none"}`,
        );
        console.log(
            `pipelineCount: ${compile.pipelineCount} → ${after.pipelineCount}; totalMs: ${r1(compile.totalMs)} → ${r1(after.totalMs)}`,
        );
    }

    // S1b: `Profile.compile` above can't see the sync path's magnitude — this is the instrument that
    // can. A CDP sample-based CPU profile of the same boot window, taken concurrently by `--attribution`
    // (bin/verify.ts's `Profiler.start`/`stop`), reads the real V8 call stack on a fixed cadence and
    // needs no engine-internal instrumentation, so it sees the sync `createRenderPipeline` loop's actual
    // cost directly rather than through a near-zero stub.
    console.log(
        "\n## S1b — CPU-profile self-time attribution (sees the sync path's real magnitude)\n",
    );
    const cpuProfile = result.cpuProfile;
    if (cpuProfile === undefined) {
        console.log(
            "cpuProfile absent from the result — this verify build predates S1b's CDP Profiler capture.",
        );
    } else if (cpuProfile === null) {
        console.log(
            "cpuProfile is null — CDP's Profiler domain wasn't reachable this run (no CDP session, or " +
                "Profiler.start/stop failed). Honest miss: no number below, not a fabricated zero.",
        );
    } else if (cpuProfile.totalMs === 0) {
        console.log(
            "cpuProfile captured but carries 0ms of sampled self time — the profile's samples/timeDeltas " +
                "stream was empty (a boot too short to take one sample at the 200us sampling interval, or " +
                "the tab never got a chance to run V8 between navigation and the boot wait's conclusion). " +
                "This is an instrument miss, not evidence the boot did no work.",
        );
    } else {
        console.log(
            `total sampled main-thread self time: ${r1(cpuProfile.totalMs)}ms (this is the CPU-profile's ` +
                `own total, independent of the longtask observer's total printed above; both describe the ` +
                `same boot window from two different instruments and should be the same order of magnitude ` +
                `— a big gap between them means one instrument is missing work the other sees).\n`,
        );

        console.log("### by named candidate mechanism (self time, descending)\n");
        console.log("| candidate mechanism | self ms | % of profiled total |");
        console.log("|---|---|---|");
        for (const b of cpuProfile.buckets) {
            const pct = cpuProfile.totalMs > 0 ? (100 * b.selfMs) / cpuProfile.totalMs : 0;
            console.log(`| ${b.name} | ${r1(b.selfMs)} | ${r1(pct)}% |`);
        }

        console.log("\n### top individual call-frame identities by self time (top 15)\n");
        console.log("| function | file:line | self ms |");
        console.log("|---|---|---|");
        for (const e of cpuProfile.entries.slice(0, 15)) {
            const file = e.url ? e.url.split("/").slice(-2).join("/") || e.url : "(no url)";
            console.log(`| ${e.functionName} | ${file}:${e.lineNumber + 1} | ${r1(e.selfMs)} |`);
        }
        console.log(
            "\nCAVEAT: self time is attributed per call-frame IDENTITY (function+url+line), merged across " +
                "every call path that reaches it — a function called from N sites reads as one row with " +
                "self time summed across all N, never split by caller. The candidate-mechanism buckets " +
                "(above) are a data-driven URL/path match (bin/verify.ts's CPU_FRAME_BUCKETS table), " +
                "audited by hand against the spec's named candidates — an unmatched frame falls through to " +
                "a per-file 'other' bucket rather than disappearing, so nothing is silently dropped, but a " +
                "candidate this table doesn't yet name for a given demo can still hide inside 'other'.",
        );
    }

    await teardownBridge();
}

await main();
