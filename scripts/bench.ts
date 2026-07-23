import { resolve } from "node:path";
import type { BenchmarkMeasurement } from "@dylanebert/shallot/extras";
import { SCENARIO_TIMEOUTS } from "../examples/gym/src/scenarios/timeouts";
import { type Check, type Memory, queryFlags, skipReason, teardownBridge, verify } from "./verify";

// `bun bench` — a thin wrapper over the shipped gate. It maps today's arg surface onto `shallot verify
// examples/gym --json` (verify boots the ejected gym vite app, picks its own port, drives the published
// `window.__harness`, returns a JSON verdict), then formats the profiler metrics + checks + memory the way
// the dissolved gym launcher did. No port logic, no server boot here — verify owns all of it.
//
// One gym scenario per run (default: render). `--scenario stress` drives the CPU-memory allocation probe
// (`--alloc`); every other scenario samples the retained-leak slope (`--memory`, informational). Exits
// nonzero when the run fails (a false verdict, a page error, or a setup failure).

const GYM = "examples/gym";

interface Args {
    scenario: string;
    seed: number;
    count?: number;
    warmup: number;
    frames: number;
    timeoutMs?: number;
    params: string[];
    screenshot?: string;
    leak?: number;
}

function help(): void {
    console.log(`Usage: bun bench [options]

Runs one gym scenario (examples/gym) through \`shallot verify\` on a real device and routes its
verdict: metrics → printed frame-time, checks → pass/fail gate.

Options:
  --scenario <name>    which scenario to run (default: render). See examples/gym.
  --seed <n>           determinism seed (default: 1)
  --count <n>          per-scenario size param (scenario default if omitted)
  --warmup <n>         warmup frames (default: 60)
  --frames <n>         measurement frames (default: 240)
  --timeout <ms>       overall run budget; also raises the build/settle ready-window for a heavy scenario
  --param <key=value>  extra URL param a scenario reads (repeatable; e.g. --param dist=clustered)
  --screenshot <path>  write a post-run canvas screenshot to <path> (PNG; visual smoke test)
  --leak <bytesPerSec> inject a retained allocation at this rate — red-proof for the leak detector`);
}

/**
 * the `--timeout` (ms) to drive a scenario under, or undefined to leave verify's 60s default. An explicit
 * `bun bench --timeout N` wins (operator override); otherwise a scenario that declared a budget in
 * {@link SCENARIO_TIMEOUTS} gets it, and everything else stays undefined so the tight default hang detector
 * holds. Pure — the resolution the run injects, unit-tested in bin/verify.test.ts.
 */
export function benchTimeout(scenario: string, cliTimeoutMs?: number): number | undefined {
    if (cliTimeoutMs != null) return cliTimeoutMs;
    return SCENARIO_TIMEOUTS[scenario];
}

function parseArgs(argv: string[]): Args {
    if (argv.includes("--help") || argv.includes("-h")) {
        help();
        process.exit(0);
    }
    const out: Args = { scenario: "render", seed: 1, warmup: 60, frames: 240, params: [] };
    const take = (name: string, i: number): string => {
        if (i + 1 >= argv.length) throw new Error(`--${name} requires a value`);
        return argv[i + 1];
    };
    for (let i = 0; i < argv.length; i++) {
        const arg = argv[i];
        if (!arg.startsWith("--")) continue; // passthrough was Playwright-runner-specific; verify has none
        const name = arg.slice(2);
        switch (name) {
            case "scenario":
                out.scenario = take(name, i++);
                break;
            case "seed":
                out.seed = parseInt(take(name, i++), 10);
                break;
            case "count":
                out.count = parseInt(take(name, i++), 10);
                break;
            case "warmup":
                out.warmup = parseInt(take(name, i++), 10);
                break;
            case "frames":
                out.frames = parseInt(take(name, i++), 10);
                break;
            case "timeout":
                out.timeoutMs = parseInt(take(name, i++), 10);
                break;
            case "param":
                out.params.push(take(name, i++));
                break;
            case "screenshot":
                out.screenshot = take(name, i++);
                break;
            case "leak":
                out.leak = parseInt(take(name, i++), 10);
                break;
            default:
                throw new Error(`unknown option: ${arg}`);
        }
    }
    return out;
}

// Mixed fixed/variable timing: the frame interval (variable, rAF/vsync paced) is reported with its
// decomposition (cpu + GPU fence-wait + idle gap). GPU is split by clock — sim passes per fixed step,
// render passes per frame. Salvaged from the dissolved gym launcher (harness/gym/format.ts).
function printMeasurement(label: string, r: BenchmarkMeasurement): void {
    const bar = "=".repeat(40);
    console.log(`\n${bar}`);
    console.log(`  ${label} Results`);
    console.log(bar);
    console.log(`  Frames measured: ${r.frames}`);
    if (r.frame) {
        const f = r.frame;
        const idlePct = f.avg > 0 ? Math.round((f.gapMs / f.avg) * 100) : 0;
        console.log(
            `  Frame:   avg ${f.avg.toFixed(2)}  median ${f.median.toFixed(2)}  p95 ${f.p95.toFixed(2)}  p99 ${f.p99.toFixed(2)}  max ${f.max.toFixed(2)} ms`,
        );
        console.log(
            `    = cpu ${f.cpuMs.toFixed(2)} + fence ${f.fenceMs.toFixed(2)} + idle ${f.gapMs.toFixed(2)} ms   (${idlePct}% idle — rAF/vsync paced)`,
        );
        console.log(
            `    fence p95 ${f.fenceP95.toFixed(2)} ms · ${f.stepsPerFrame.toFixed(2)} steps/frame · clamped ${f.clampedFrames} · pending ${f.maxPending}`,
        );
        console.log(
            `    stddev ${f.stddev.toFixed(2)} ms · spike(raw) p99 ${f.rawP99.toFixed(2)} / max ${f.rawMax.toFixed(2)} ms`,
        );
    }
    if (r.gpu) {
        const g = r.gpu;
        const steps = r.frame ? r.frame.stepsPerFrame.toFixed(2) : "?";
        console.log(
            `  GPU busy: ${g.busyPerFrameMs.toFixed(3)} ms/frame = render ${g.renderPerFrameMs.toFixed(3)}/frame + sim ${g.simPerStepMs.toFixed(3)}/step × ${steps}`,
        );
        const entries = Object.entries(g.passes);
        const sim = entries
            .filter(([, p]) => p.clock === "sim")
            .sort((a, b) => b[1].occMs - a[1].occMs);
        const render = entries
            .filter(([, p]) => p.clock === "render")
            .sort((a, b) => b[1].perFrameMs - a[1].perFrameMs);
        if (sim.length > 0) {
            console.log(`    sim (per step):`);
            for (const [name, p] of sim)
                console.log(
                    `      ${name.padEnd(20)} ${p.occMs.toFixed(3)} ms  (p99 ${p.occP99.toFixed(3)})`,
                );
        }
        if (render.length > 0) {
            console.log(`    render (per frame):`);
            for (const [name, p] of render)
                console.log(
                    `      ${name.padEnd(20)} ${p.perFrameMs.toFixed(3)} ms  (p99 ${p.occP99.toFixed(3)})`,
                );
        }
    } else {
        console.log(`  GPU timing unavailable (no profiler spans in the measure)`);
    }
    console.log(`${bar}\n`);
}

// Returns true if every check passed; prints a one-line verdict per check. `ok` is the published protocol's
// field (the gym launcher's `pass` translated at the harness boundary).
function printChecks(checks: Check[]): boolean {
    console.log(`  Checks:`);
    let allPass = true;
    for (const c of checks) {
        allPass = allPass && c.ok;
        console.log(`    ${c.ok ? "✓" : "✗"} ${c.name}${c.detail ? `  — ${c.detail}` : ""}`);
    }
    return allPass;
}

function printMemory(m: Memory): void {
    const mb = (b: number) => (b / 1024 / 1024).toFixed(1);
    console.log(`  Memory:  ${mb(m.start)} → ${mb(m.end)} MB`);
    console.log(
        `  Growth:  ${(m.growthPerSecond / 1024).toFixed(2)} KB/s${m.leak ? " ⚠ LEAK" : ""}`,
    );
}

async function main(): Promise<void> {
    const args = parseArgs(process.argv.slice(2));

    const skip = skipReason();
    if (skip) {
        console.log(`bun bench needs native hardware (${skip}). Skipping.`);
        process.exit(0);
    }

    const query = [
        `scenario=${args.scenario}`,
        `seed=${args.seed}`,
        `warmup=${args.warmup}`,
        `frames=${args.frames}`,
    ];
    if (args.count != null) query.push(`count=${args.count}`);
    query.push(...args.params);

    const extra = [
        ...queryFlags(query),
        // the stress CPU-memory axis drives its own no-forced-GC allocation probe (window.__probeAlloc); its
        // parallel forced GCs would corrupt that window, so swap the always-on retained sampler for the probe.
        args.scenario === "stress" ? "--alloc" : "--memory",
    ];
    if (args.screenshot) extra.push("--screenshot", resolve(args.screenshot));
    // a scenario that declared a budget (SCENARIO_TIMEOUTS) drives under it; an explicit --timeout wins.
    const timeoutMs = benchTimeout(args.scenario, args.timeoutMs);
    if (timeoutMs != null) extra.push("--timeout", String(timeoutMs));
    if (args.leak != null) extra.push("--leak", String(args.leak));

    const result = await verify(GYM, extra);
    if (!result) {
        console.error("\ngym run FAILED — no JSON result from shallot verify");
        process.exit(1);
    }
    if (args.screenshot) console.log(`\nscreenshot → ${resolve(args.screenshot)}`);
    if (result.rendered === "opt-out") {
        console.log(`\n  rendered: opt-out — ${args.scenario} renders nothing by design`);
    }

    let failed = !result.pass;
    const verdict = result.verdict;
    if (verdict?.metrics) printMeasurement(args.scenario, verdict.metrics as BenchmarkMeasurement);
    if (verdict?.checks) {
        if (!printChecks(verdict.checks)) failed = true;
    }
    if (result.memory) printMemory(result.memory);

    if (result.errors?.length) {
        console.error(`\n${result.errors.length} error(s) captured:`);
        for (const e of result.errors.slice(0, 5)) console.error(`  ${e.split("\n")[0]}`);
    }
    if (result.error) console.error(`\n${result.error}`);

    if (failed) {
        console.error("\ngym run FAILED");
        process.exit(1);
    }
    // release the shared WSL bridge so the process exits — its rendezvous + client servers and the host
    // browser subprocess otherwise keep the event loop alive past the verdict (a no-op off WSL). The
    // failure paths above `process.exit`, firing the bridge's sync exit hook; the pass path drains cleanly.
    await teardownBridge();
    console.log("\ngym run passed");
}

// guard so importing this module (bin/verify.test.ts exercises benchTimeout) doesn't launch a bench run.
if (import.meta.main) {
    main().catch((err) => {
        console.error(err instanceof Error ? err.message : err);
        process.exit(1);
    });
}
