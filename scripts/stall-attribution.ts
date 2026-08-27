import { skipReason, teardownBridge, verify } from "./verify";

// S1 of `shallot-demo-startup-stall`: split the demo's startup ~1s stall into main-thread block vs
// GPU-thread compile await, per pipeline label, in one instrumented `shallot verify --attribution` run
// (`bin/verify.ts`). Re-runnable — this is the oracle the attribution table below is read from, not a
// one-off console poke.
//
// Why `--attribution`'s two numbers are classifiable without a heuristic, not just measurable:
//
//  - Every render-pipeline surface/background TypeGPU compiles (`preparePipelines()`'s loop over
//    `Surfaces` × `Backgrounds`, and `unwrapVariant`'s per-variant compiles for specializing surfaces,
//    both via `Compute.root.unwrap` → typegpu's `initSync()`) reaches `device.createRenderPipeline`,
//    the SYNCHRONOUS constructor. `ProfilePlugin` wraps that constructor in a bare
//    `performance.now()` delta around the call itself (`extras/profile/index.ts`). JS is
//    single-threaded, so the wall-clock span of a synchronous call is main-thread-blocked time by
//    construction — no probe, no inference from absence of contrary evidence needed. This holds
//    regardless of the recorded duration being small per call (gpu.md: Dawn defers the real shader
//    compile past the sync constructor's return) — many such calls back-to-back is exactly the
//    "co-scheduled synchronous work" the spec names as the prime suspect.
//  - Every OTHER registered label is genuinely async, for two distinct reasons that land the same way.
//    `sear-regather-a`/`-b` and `phys-pack-*` (`standard/sear/regather.ts`, `standard/avbd/step.ts`'s
//    `buildPass`) are raw `device.createComputePipelineAsync` calls, never routed through typegpu at
//    all. Every remaining label (`standard/avbd/step.ts`'s per-kernel forcers, `standard/bvh/bounds.ts`
//    /`build.ts`, `standard/part/part.ts`, `standard/slab/index.ts`, `extras/outline/index.ts`) IS a
//    typegpu pipeline, but its `precompile()` forcer hands the drain the RAW pipeline object without
//    unwrapping it first — confirmed by reading each site (`isMainThreadBlock`'s docblock below). The
//    drain's `initAsync()` is then that pipeline's first resolution attempt, which triggers typegpu's
//    real async constructor. Either way the recorded span is elapsed wall-clock that does NOT block the
//    main thread — UNLESS something else pinned it for that same window. The `longtask`
//    PerformanceObserver installed by `--attribution` (`ATTRIBUTION_INIT_SCRIPT`) is the independent,
//    concurrently-running instrument for exactly that question: it names every real main-thread block
//    ≥50ms across the whole boot, regardless of what caused it, so its total is read against the
//    sync-call sum below rather than trusted on its own.
//  - `sear-typed-variants` is `precompileVariants`'s `precompile()` forcer label — its
//    `Compute.precompiled` completion span wraps `unwrapVariant`'s own synchronous
//    `Compute.root.unwrap` calls (already counted individually above, under their own pipeline
//    labels) PLUS each resulting typegpu pipeline's `initAsync()`. Since `unwrap` already ran
//    `initSync()` first, `initAsync()` finds its memo already set and resolves immediately
//    (`node_modules/typegpu/core/pipeline/renderPipeline.js`) — so this span is not double-counted
//    against the per-pipeline sync entries; it is reported here to show that its own total tracks the
//    per-pipeline sum almost exactly, which is the empirical confirmation of that mechanism, not an
//    assumption.
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
// reported separately, excluded from the block/await totals to avoid double-counting.
const VARIANTS_SCOPE_LABEL = "sear-typed-variants";

// classify a label by its OWN registration site, read from source (not a name-pattern guess): every
// `sear-typed-*` pipeline label is force-unwrapped (`Compute.root.unwrap`, sear's own
// `preparePipelines()`/`unwrapVariant`) BEFORE it ever reaches the `precompile()` queue, so its later
// `initAsync()` finds the memo already set and resolves instantly — the entire recorded span is the
// SYNCHRONOUS `create*Pipeline` call, main-thread block by construction (single-threaded JS).
// Every other registered label (`standard/avbd/step.ts`, `standard/bvh/bounds.ts`/`build.ts`,
// `standard/part/part.ts`, `standard/slab/index.ts`, `extras/outline/index.ts`, and the raw
// (non-typegpu) `sear-regather-a`/`-b` and `phys-pack-*` `device.createComputePipelineAsync` calls in
// `standard/sear/regather.ts` / `standard/avbd/step.ts`'s `buildPass`) hands its `precompile()` forcer
// the pipeline WITHOUT unwrapping it first — confirmed by reading each registration site, e.g.
// `standard/bvh/bounds.ts`: `precompile(`${scope}-${label}`, () => bound)` where `bound` is never
// passed through `unwrap`. The drain's `initAsync()` is then the FIRST resolution attempt, which
// triggers typegpu's real async path (`device.create*PipelineAsync`) — genuine GPU-thread elapsed
// time, not main-thread block, exactly like the raw regather/pack pipelines.
function isMainThreadBlock(label: string): boolean {
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
            kind: isMainThreadBlock(label)
                ? ("main-thread block" as const)
                : ("gpu-thread await" as const),
        }))
        .sort((a, b) => b.ms - a.ms);

    const blockMs = rows
        .filter((r) => r.kind === "main-thread block")
        .reduce((s, r) => s + r.ms, 0);
    const awaitMs = rows.filter((r) => r.kind === "gpu-thread await").reduce((s, r) => s + r.ms, 0);
    const variantsScopeMs = compile.pipelines[VARIANTS_SCOPE_LABEL];

    console.log("\n## Attribution — main-thread block vs GPU-thread await, per pipeline label\n");
    console.log("| label | kind | ms |");
    console.log("|---|---|---|");
    for (const row of rows) {
        console.log(`| ${row.label} | ${row.kind} | ${r1(row.ms)} |`);
    }
    if (variantsScopeMs !== undefined) {
        console.log(
            `| ${VARIANTS_SCOPE_LABEL} (forcer scope — wraps the specializing-variant labels above, not summed separately) | main-thread block | ${r1(variantsScopeMs)} |`,
        );
    }
    console.log(
        `| **total** | main-thread block: **${r1(blockMs)}ms**, GPU-thread await: **${r1(awaitMs)}ms** | compile.totalMs = ${r1(compile.totalMs)} |`,
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
        `\nlongtask corroboration: ${longTasks.length} entries, ${r1(longTaskMs)}ms total main-thread` +
            ` block across the whole boot (compare against the ${r1(blockMs)}ms sync sum above — this` +
            ` instrument cannot align an individual longtask to an individual label, only totals).`,
    );
    if (longTaskMs > blockMs * 10) {
        console.log(
            `\n${r1(longTaskMs)}ms of real main-thread block was recorded while the summed sync ` +
                `pipeline-creation time was only ${r1(blockMs)}ms — the pipeline-creation calls ` +
                `themselves are NOT the main-thread stall's source in this run. Whatever produced the ` +
                `longtask entries is synchronous JS this instrument doesn't attribute (module ` +
                `evaluation, scene/ECS setup, asset decode, or something else in the boot path) —` +
                ` a negative this instrument can state (pipeline creation isn't it) but not resolve ` +
                `further; that needs a CPU profile or narrower marks around the boot's other phases.`,
        );
    }

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

    await teardownBridge();
}

await main();
