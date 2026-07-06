import { runExample } from "../core";
import { parseArgs } from "./args";
import { printChecks, printMeasurement } from "./format";
import type { Verdict } from "./verdict";

// Gym launcher: build on the core harness, drive one scenario in examples/gym, and route the
// returned verdict. metrics → printed frame-time; checks → a pass/fail gate that exits
// nonzero. Reads top-to-bottom — the arg parse, the run, the routing. No general
// page-driving lives here (that's the core); no scenario logic lives here (that's the gym
// example).

const PORT = 3002;

function buildUrl(scenario: string, seed: number, count: number | undefined, extra: string[]): string {
    const params = [`scenario=${scenario}`, `seed=${seed}`];
    if (count != null) params.push(`count=${count}`);
    params.push(...extra); // scenario-read `key=value` params (dist, viz, …)
    return `http://localhost:${PORT}/?${params.join("&")}`;
}

async function main(): Promise<void> {
    const args = parseArgs(process.argv.slice(2));

    // The stress CPU-memory axis drives its own collectGarbage-bracketed allocation probe
    // (window.__probeAlloc) from the assert; its parallel forced GCs would corrupt that no-GC window,
    // so swap the always-on retained-leak sampler for the probe binding on that scenario.
    const alloc = args.scenario === "stress";

    const result = await runExample({
        example: "gym",
        port: PORT,
        url: buildUrl(args.scenario, args.seed, args.count, args.params),
        warmup: args.warmup,
        frames: args.frames,
        // --timeout raises the overall budget AND the build/settle ready-window together (a heavy build
        // implies a long ready wait is acceptable); the page reserves a 30s tail for the measured run.
        timeoutMs: args.timeoutMs,
        readyTimeoutMs: args.timeoutMs,
        sampleMemory: !alloc,
        sampleAlloc: alloc,
        screenshot: args.screenshot,
        passthrough: args.passthrough,
    });
    if (args.screenshot) console.log(`\nscreenshot → ${args.screenshot}`);

    let failed = !result.ok;
    const verdict = result.verdict as Verdict | undefined;

    if (verdict?.metrics) {
        printMeasurement(args.scenario, verdict.metrics, result.memory ?? null);
    }
    if (verdict?.checks) {
        if (!printChecks(verdict.checks)) failed = true;
    }

    if (result.errors?.length) {
        console.error(`\n${result.errors.length} error(s) captured:`);
        for (const e of result.errors.slice(0, 5)) console.error(`  ${e.split("\n")[0]}`);
    }

    if (failed) {
        console.error("\ngym run FAILED");
        process.exit(1);
    }
    console.log("\ngym run passed");
}

main().catch((err) => {
    console.error(err instanceof Error ? err.message : err);
    process.exit(1);
});
