import { isAbsolute, relative, resolve, sep } from "node:path";
import type { BenchmarkMeasurement } from "@dylanebert/shallot/extras";
import { globToRegExp } from "../examples/gym/src/scenarios/coverage";
import { SCENARIO_GATES, type ScenarioGate } from "../examples/gym/src/scenarios/timeouts";
import {
    type Check,
    type Memory,
    queryFlags,
    REPO_ROOT,
    skipReason,
    teardownBridge,
    type VerifyResult,
    verify,
    verifyBatch,
} from "./verify";

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
    list: boolean;
    for?: string[];
    sweep: boolean;
    memory: boolean;
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
  --leak <bytesPerSec> inject a retained allocation at this rate — red-proof for the leak detector
  --list               print every registered scenario name and exit (the real roster, not a guess)
  --for <paths...>     resolve changed source paths to the scenario(s) that gate them, via
                        SCENARIO_GATES' covers globs; prints the mapping and exits unless --sweep is
                        also given, in which case it selects the sweep's scenario set
  --sweep              run every scenario (or the --for subset) through shallot verify's batch mode —
                        one boot, N verdicts — spawning each declared-isolate scenario in its own process
  --memory             opt in to the retained-leak sample on the sweep path (single runs always sample it)`);
}

/**
 * the `--timeout` (ms) to drive a scenario under, or undefined to leave verify's 60s default. An explicit
 * `bun bench --timeout N` wins (operator override); otherwise a scenario that declared a budget in
 * {@link SCENARIO_GATES} gets it, and everything else stays undefined so the tight default hang detector
 * holds. Pure — the resolution the run injects, unit-tested in bin/verify.test.ts.
 */
export function benchTimeout(scenario: string, cliTimeoutMs?: number): number | undefined {
    if (cliTimeoutMs != null) return cliTimeoutMs;
    return SCENARIO_GATES[scenario]?.timeoutMs;
}

/** `--list`'s roster, sorted — pure so the sort/format is unit-tested without booting a page. */
export function formatRoster(names: readonly string[]): string {
    return [...names].sort().join("\n");
}

/** one `--for <path>` resolution: every scenario whose `covers` glob (from {@link SCENARIO_GATES}) matches
 *  the path. `scenarios` is empty when nothing covers it — a tumble path (excluded from the GPU-src
 *  population by design, `coverage.ts`) or any other path outside the table's tracked coverage — and
 *  the caller must say so explicitly (see {@link forUnmatchedReason}) rather than print an empty roster
 *  that reads like "nothing gates this file". */
export interface ForMatch {
    path: string;
    scenarios: string[];
}

/** a `--for` path as the `covers` globs spell it: repo-root-relative, forward-slashed. An operator pastes
 *  whatever their editor or tool hands them — an absolute path, or one relative to a cwd inside the repo —
 *  and an unnormalized path matches no glob and reports "no scenario declares coverage", which is the
 *  false-negative this table exists to kill (the naming trap that cost the last survey ~35 runs). A path
 *  outside `root` is returned unchanged, so it still reaches {@link forUnmatchedReason} honestly. Pure —
 *  cwd and root are passed in. */
export function normalizeForPath(path: string, cwd: string, root: string): string {
    const abs = isAbsolute(path) ? path : resolve(cwd, path);
    const rel = relative(root, abs).split(sep).join("/");
    return rel === "" || rel.startsWith("../") ? path : rel;
}

export function resolveFor(
    paths: readonly string[],
    table: Record<string, ScenarioGate>,
): ForMatch[] {
    return paths.map((path) => ({
        path,
        scenarios: Object.entries(table)
            .filter(([, gate]) => (gate.covers ?? []).some((glob) => globToRegExp(glob).test(path)))
            .map(([name]) => name)
            .sort(),
    }));
}

/** why a path matched no scenario — a tumble path is a declared exclusion (`tumble.md`'s own standing
 *  gates cover it, not this table); anything else is genuinely outside the table's tracked coverage. */
export function forUnmatchedReason(path: string): string {
    if (path.includes("standard/tumble/")) {
        return "no scenario declares coverage — tumble physics is gated by its own standing gates (tumble.md), not this table";
    }
    return "no scenario declares coverage in SCENARIO_GATES";
}

export function formatForResolution(matches: readonly ForMatch[]): string {
    return matches
        .map(({ path, scenarios }) =>
            scenarios.length > 0
                ? `${path} → ${scenarios.join(", ")}`
                : `${path} → ${forUnmatchedReason(path)}`,
        )
        .join("\n");
}

/** the sweep-vs-isolate partition (a perf-threshold gate measured under back-to-back sweep contention is
 *  not trustworthy, grounded on the `stress` finding). Every declared `isolate: true` scenario runs in
 *  its own process; everything else shares one boot through batch mode. Pure over the table so the
 *  partition is red-provable without a real sweep. */
export interface SweepPlan {
    batch: string[];
    isolate: string[];
}

/** `shallot verify --timeout` is one process-global budget, so a batch process can carry exactly one. A
 *  batched scenario's declared `timeoutMs` — or an operator's `bun bench --sweep --timeout N` — would
 *  otherwise be silently dropped and the run would drive under verify's tight 60s default. Group the batch
 *  by resolved budget rather than weakening every run to the longest: one `--run` batch per distinct
 *  timeout, in first-appearance order. Today every batched scenario resolves to the same value, so this is
 *  one group and costs no extra boot; it exists so a future `timeoutMs` entry can't be ignored. Pure. */
export interface TimeoutGroup {
    timeoutMs?: number;
    names: string[];
}

export function groupByTimeout(names: readonly string[], cliTimeoutMs?: number): TimeoutGroup[] {
    const groups: TimeoutGroup[] = [];
    for (const name of names) {
        const timeoutMs = benchTimeout(name, cliTimeoutMs);
        const group = groups.find((g) => g.timeoutMs === timeoutMs);
        if (group) group.names.push(name);
        else groups.push({ timeoutMs, names: [name] });
    }
    return groups;
}

export function partitionSweep(
    names: readonly string[],
    table: Record<string, ScenarioGate>,
): SweepPlan {
    const batch: string[] = [];
    const isolate: string[] = [];
    for (const name of names) {
        (table[name]?.isolate ? isolate : batch).push(name);
    }
    return { batch, isolate };
}

function parseArgs(argv: string[]): Args {
    if (argv.includes("--help") || argv.includes("-h")) {
        help();
        process.exit(0);
    }
    const out: Args = {
        scenario: "render",
        seed: 1,
        warmup: 60,
        frames: 240,
        params: [],
        list: false,
        sweep: false,
        memory: false,
    };
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
            case "list":
                out.list = true;
                break;
            case "sweep":
                out.sweep = true;
                break;
            case "memory":
                out.memory = true;
                break;
            case "for": {
                const paths: string[] = [];
                while (i + 1 < argv.length && !argv[i + 1].startsWith("--")) {
                    paths.push(argv[i + 1]);
                    i++;
                }
                if (paths.length === 0) throw new Error("--for requires at least one path");
                // normalized once here, at the arg edge, so every consumer of `args.for` matches the
                // `covers` globs' repo-root-relative spelling whatever the operator pasted.
                out.for = paths.map((p) => normalizeForPath(p, process.cwd(), REPO_ROOT));
                break;
            }
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
    if (r.memory) {
        const m = r.memory;
        const mb = (b: number) => (b / 1024 / 1024).toFixed(1);
        console.log(
            `  GPU memory: buffers ${mb(m.bufferBytes)} + textures ${mb(m.textureBytes)} = ${mb(m.bufferBytes + m.textureBytes)} MB`,
        );
        // the byte-budget gate reads the total minus lazy-pool bytes —
        // print both so a printed number always matches what the gate compares against.
        if (m.lazyBytes > 0) {
            console.log(
                `    excl. lazy pool: ${mb(m.lazyBytes)} MB  (gated total ${mb(m.bufferBytes + m.textureBytes - m.lazyBytes)} MB)`,
            );
        }
        const labels = Object.entries(m.byLabel).sort((a, b) => b[1] - a[1]);
        if (labels.length > 0) {
            console.log(`    by label:`);
            for (const [label, bytes] of labels)
                console.log(`      ${(label || "(unlabeled)").padEnd(20)} ${mb(bytes)} MB`);
        }
    }
    if (r.compile) {
        const c = r.compile;
        const entries = Object.keys(c.pipelines).length;
        // both gated quantities, printed beside the raw table they're filtered from — the same way the
        // memory block prints its gated total beside the raw one, so no number a gate compares against is
        // invisible here. `pipelineCount` is the distinct-label count (`budget:pipelines`), `pipelineCalls`
        // the raw constructor invocations (`budget:pipeline-calls`), and `entries` the unfiltered `compile`
        // table, which also holds `precompile` forcer-scope labels.
        console.log(
            `  Pipelines: ${c.pipelineCount} labels / ${c.pipelineCalls} calls gated  (${entries} compile entries)`,
        );
        console.log(`    compile span: ${c.totalMs.toFixed(1)} ms`);
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

// `--list`'s real-registration seam: registering a scenario module needs the WebGPU constants gym's own
// modules reference at import time (GPUTextureUsage etc.), which only exist once the polyfill installs —
// hence the dynamic imports, deferred until after `setupGlobals()`, rather than static ones that would
// hoist and run before it. `SCENARIO_GATES`' keys track this roster 1:1 (`coverage.ts`'s completeness
// check asserts both directions), but `--list` reads the real one anyway per the spec's naming-trap
// motivation, not the side table that exists to avoid booting a page.
async function registeredScenarios(): Promise<string[]> {
    const { setupGlobals } = await import("bun-webgpu");
    await setupGlobals();
    const { scenarioNames } = await import("../examples/gym/src/gym");
    await import("../examples/gym/src/scenarios/index");
    return scenarioNames();
}

// one sweep row's verdict, printed the same shape whether it ran batched or isolated — the caller can't
// tell which path produced it, which is the point (`shallot verify`'s one-verdict law holds either way).
// `bytes` (batch rows only — isolate rows have no separate byte-count channel) is the raw stdout size:
// null `result` covers a real crash and a truncated-or-unparseable stream alike, and there's no way from
// here to tell which, so the observable byte count is what gets reported (measured: a 64 KiB pipe-buffer
// cutoff).
function printSweepResult(name: string, result: VerifyResult | null, bytes?: number): boolean {
    if (!result) {
        const detail =
            bytes != null
                ? `no parseable verdict array on stdout (${bytes} bytes received)`
                : "no parseable verdict on stdout";
        console.log(`✗ ${name} — ${detail}`);
        return false;
    }
    const checks = result.verdict?.checks;
    let ok = result.pass;
    if (checks) for (const c of checks) if (!c.ok) ok = false;
    console.log(`${ok ? "✓" : "✗"} ${name}${result.error ? ` — ${result.error}` : ""}`);
    if (checks) {
        for (const c of checks) {
            if (!c.ok) console.log(`    ✗ ${c.name}${c.detail ? `  — ${c.detail}` : ""}`);
        }
    }
    return ok;
}

/** `--sweep`: run `names` through `shallot verify`'s batch mode, spawning every declared-isolate
 *  scenario in its own process instead (`partitionSweep`) — the sweep-vs-isolate law grounded on the
 *  `stress` sweep-contention finding. `--memory` is opt-in here (single runs always sample it): the
 *  retained-leak sampler adds ~1.2s/run and gates nothing, and a `⚠ LEAK` on a passing run trains agents
 *  to ignore warnings. Returns true iff every scenario passed. */
async function sweep(names: string[], args: Args): Promise<boolean> {
    const { batch, isolate } = partitionSweep(names, SCENARIO_GATES);
    const shared = [`seed=${args.seed}`, `warmup=${args.warmup}`, `frames=${args.frames}`];
    if (args.count != null) shared.push(`count=${args.count}`);
    shared.push(...args.params);

    let allPass = true;

    for (const group of groupByTimeout(batch, args.timeoutMs)) {
        const extra = [...queryFlags(shared), ...(args.memory ? ["--memory"] : [])];
        if (group.timeoutMs != null) extra.push("--timeout", String(group.timeoutMs));
        const { results, bytes } = await verifyBatch(
            GYM,
            group.names.map((name) => `scenario=${name}`),
            extra,
            true,
        );
        group.names.forEach((name, i) => {
            if (!printSweepResult(name, results?.[i] ?? null, results ? undefined : bytes)) {
                allPass = false;
            }
        });
    }

    for (const name of isolate) {
        const extra = [
            ...queryFlags([...shared, `scenario=${name}`]),
            // stress's CPU-memory probe is its own gate, not the informational leak sampler — always on.
            ...(name === "stress" ? ["--alloc"] : args.memory ? ["--memory"] : []),
        ];
        const timeoutMs = benchTimeout(name, args.timeoutMs);
        if (timeoutMs != null) extra.push("--timeout", String(timeoutMs));
        const result = await verify(GYM, extra, true);
        if (!printSweepResult(name, result)) allPass = false;
    }

    return allPass;
}

async function main(): Promise<void> {
    const args = parseArgs(process.argv.slice(2));

    if (args.list) {
        console.log(formatRoster(await registeredScenarios()));
        return;
    }

    if (args.for && !args.sweep) {
        console.log(formatForResolution(resolveFor(args.for, SCENARIO_GATES)));
        return;
    }

    const skip = skipReason();
    if (skip) {
        console.log(`bun bench needs native hardware (${skip}). Skipping.`);
        process.exit(0);
    }

    if (args.sweep) {
        let names: string[];
        if (args.for) {
            const matches = resolveFor(args.for, SCENARIO_GATES);
            for (const m of matches) {
                if (m.scenarios.length === 0)
                    console.log(`${m.path} → ${forUnmatchedReason(m.path)}`);
            }
            names = [...new Set(matches.flatMap((m) => m.scenarios))];
        } else {
            names = Object.keys(SCENARIO_GATES);
        }
        if (names.length === 0) {
            console.log("\nno scenario selected — nothing to sweep");
            await teardownBridge();
            return;
        }
        const started = Date.now();
        const passed = await sweep(names, args);
        console.log(
            `\nswept ${names.length} scenario(s) in ${((Date.now() - started) / 1000).toFixed(1)}s`,
        );
        await teardownBridge();
        if (!passed) {
            console.error("sweep FAILED");
            process.exit(1);
        }
        console.log("sweep passed");
        return;
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
    // a scenario that declared a budget (SCENARIO_GATES) drives under it; an explicit --timeout wins.
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
        console.log(`\n  rendered: opt-out — generic pixel gate inapplicable for ${args.scenario}`);
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
