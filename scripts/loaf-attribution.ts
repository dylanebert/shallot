import { type LoAFEntry, loafByCompilePhase } from "../packages/shallot/bin/verify";
import { PIPELINE_COMPILE_MEASURE_PREFIX } from "../packages/shallot/src/engine/runtime/gpu";
import { queryFlags, skipReason, teardownBridge, verify } from "./verify";

// The probe the boot roadmap item names: the compile vital sees async compiles only, so two large
// pre-compile LoAF spikes had no owner. Entry-level LoAF fields say a frame was slow and never what
// ran in it; `PerformanceScriptTiming` (the entry's own `scripts` array, forwarded by
// `ATTRIBUTION_INIT_SCRIPT`) names the source function. This script is the permanent read over that
// pair, the same shape as `scripts/compile-concurrency.ts` and `scripts/stall-attribution.ts`: one
// `shallot verify --attribution` boot, fed through the pure reader `loafByCompilePhase`
// (`bin/verify.ts`) rather than re-deriving the split here.
//
//     bun run scripts/loaf-attribution.ts [--dir <project>] [--query k=v ...]
//
// Default `--dir` is `examples/showcase/sandbox` — the project the boot item's other readings ran
// against, so the spike population is comparable across them.
//
// What this instrument establishes: WHICH JS ran inside a pre-compile slow frame, on the same
// `performance.now()` axis the compile measures use, so the split against the compile chain is an
// alignment rather than a totals comparison (which is all `stall-attribution.ts` could offer —
// `Profile.compile` carries no absolute timestamp). What it does NOT: a frame's unattributed
// remainder. The spec reports per-script `duration` only above its own reporting threshold and
// attributes style/layout/paint separately, so `beforeUnattributedMs` is a real reading of
// non-script frame delay, never a gap to be summed into the top script. Both are printed.

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

function r1(v: number): number {
    return Math.round(v * 10) / 10;
}

/** the shortest string that still identifies a bundled module — the last two path segments, since a
 *  dev-server URL's origin and long prefix are noise in a table read by eye. */
function shortUrl(url: string): string {
    if (!url || url === "(no source url)") return "(no source url)";
    return url.split("?")[0].split("/").slice(-2).join("/");
}

function printEntries(label: string, entries: LoAFEntry[]): void {
    console.log(`\n### ${label} — ${entries.length} entries\n`);
    if (entries.length === 0) return;
    console.log(
        "| start (ms) | duration (ms) | blocking (ms) | top script | its ms | invokerType |",
    );
    console.log("|---|---|---|---|---|---|");
    for (const e of entries.slice().sort((a, b) => b.duration - a.duration)) {
        const top = e.scripts.reduce<(typeof e.scripts)[number] | null>(
            (best, s) => (best === null || s.duration > best.duration ? s : best),
            null,
        );
        const name = top
            ? `${top.sourceFunctionName || "(anonymous)"} (${shortUrl(top.sourceURL)})`
            : "— no script attributed";
        console.log(
            `| ${r1(e.start)} | ${r1(e.duration)} | ${r1(e.blockingDuration)} | ${name} | ` +
                `${top ? r1(top.duration) : "—"} | ${top ? top.invokerType || "n/a" : "—"} |`,
        );
    }
}

async function main(): Promise<void> {
    const args = parseArgs(process.argv.slice(2));
    const reason = skipReason();
    if (reason) {
        console.log(`skip: ${reason}`);
        return;
    }

    console.log(`booting ${args.dir} (--attribution, headed) to attribute the boot's LoAF spikes…`);
    // headed for the same reason `stall-attribution.ts` is: a display-less frame clock undershoots
    // real block durations, and every number below is a frame-timing reading.
    process.env.SHALLOT_HEADED = "1";
    const result = await verify(args.dir, [
        "--attribution",
        "--timeout",
        "30000",
        ...queryFlags(args.query),
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

    const userAgent = attribution.userAgent ?? "";
    if (userAgent.includes("HeadlessChrome")) {
        console.log(
            `FATAL: attribution run launched headless (UA contains "HeadlessChrome") — frame-timing ` +
                `readings are invalid on a display-less frame clock.`,
        );
        process.exitCode = 1;
        return;
    }

    const entries = attribution.longAnimationFrames ?? [];
    const report = loafByCompilePhase(
        entries,
        attribution.compileMeasures,
        PIPELINE_COMPILE_MEASURE_PREFIX,
    );

    console.log("\n## Boot LoAF attribution — pre-compile vs compile-chain\n");
    console.log(`adapter: ${result.hardware ?? "unknown"}`);
    console.log(`UA: ${userAgent}`);
    console.log(`LoAF entries over the boot window: ${entries.length}`);
    if (entries.length > 0 && entries.every((e) => e.scripts.length === 0)) {
        console.log(
            "NOTE: every entry carries an empty `scripts` array. Either this engine reports LoAF " +
                "without the script breakdown, or none of these frames was delayed by JS. The " +
                "unattributed totals below are the reading; do not treat them as a top script.",
        );
    }
    if (report.boundaryMs === null) {
        console.log(
            `\nno \`${PIPELINE_COMPILE_MEASURE_PREFIX}\` measure in this run — the compile chain never ` +
                `ran (or the project carries no ProfilePlugin), so no frame can be placed either side ` +
                `of it. ${report.unclassified.length} entries unclassified.`,
        );
        printEntries("unclassified (no compile boundary)", report.unclassified);
        await teardownBridge();
        return;
    }

    console.log(`compile-chain boundary (first compile measure start): ${r1(report.boundaryMs)}ms`);

    printEntries("pre-compile (frame ends at or before the boundary)", report.before);
    printEntries("straddling the boundary", report.straddling);
    printEntries("during/after the compile chain", report.after);

    console.log("\n### pre-compile script rollup (the roadmap item's open question)\n");
    if (report.beforeScripts.length === 0) {
        console.log("no scripts attributed to any pre-compile frame.");
    } else {
        console.log("| source function | source url | invokerType | total ms | occurrences |");
        console.log("|---|---|---|---|---|");
        for (const s of report.beforeScripts) {
            console.log(
                `| ${s.sourceFunctionName} | ${shortUrl(s.sourceURL)} | ${s.invokerType || "n/a"} | ` +
                    `${r1(s.totalMs)} | ${s.count} |`,
            );
        }
    }
    console.log(
        `\nattributed to scripts: ${r1(report.beforeAttributedMs)}ms — unattributed frame delay: ` +
            `${r1(report.beforeUnattributedMs)}ms. The second number is style/layout/paint or work ` +
            `below the spec's per-script reporting threshold, a real reading rather than a missing ` +
            `measurement (this file's header) — never fold it into the top script.`,
    );

    await teardownBridge();
}

await main();
