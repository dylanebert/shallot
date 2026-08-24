import { existsSync } from "node:fs";
import { createRequire } from "node:module";
import { createServer as createTcpServer } from "node:net";
import { basename, join, resolve } from "node:path";
import { createServer, preview } from "vite";
import type { GpuDiagnostics, ShaderArtifact } from "../src/engine/runtime/gpu";
import type { GpuLog } from "../src/engine/runtime/log";
import type { Check, PixelProbe, Verdict } from "../src/harness";
import { REAL_GPU_LAUNCH } from "../src/harness/browser";
import { pixelProbePass, probePixels } from "../src/harness/pixels";
import { CROSS_ORIGIN_ISOLATION } from "../src/project/vite";
import { devConfig } from "./dev";
import { composeViteConfig, isProject, loadProjectConfig } from "./toolchain";

// `shallot verify [dir]` — the shipped, self-terminating verification gate. It boots the project in a
// real headless browser, waits for it to render (or for a `window.__harness` the project installs),
// reads a pass/fail Verdict, and exits 0 on pass / nonzero on fail. Playwright is lazy + optional (never
// a hard dep of @dylanebert/shallot); the browser path drives the Playwright LIBRARY API in-process — no
// @playwright/test runner, no config file. Boot = the dev server by default, `--dist` = the existing
// build served statically. One page load, one verdict, then everything tears down.

// exit codes: distinct so a caller (CI, an agent) can tell a real failure from a missing tool.
const EXIT_PASS = 0;
const EXIT_FAIL = 1; // booted, but the verdict was false (assertions / render / page errors)
const EXIT_SETUP = 2; // bad flags, no project, missing dist, boot failed — never reached a verdict
const EXIT_NO_PLAYWRIGHT = 3; // playwright module or its chromium browser isn't installed
export const EXIT_NO_DISPLAY = 4; // the browser only offered a software adapter — refused before any check ran

// Software rasterizers by the name they report in `GPUAdapterInfo`: Chromium's SwiftShader, Mesa's two,
// and D3D's WARP. Deliberately a name list rather than a capability probe — a software adapter clears
// shallot's whole base floor (every feature + limit check), so nothing about the floor distinguishes it;
// only *execution* dies (`GPU device lost`, oversized `mappedAtCreation` allocations that read as engine
// bugs). Mirrors the showcase drivers' proven shape (voxel.spec.ts/roads.spec.ts), biased narrow: an
// unlisted software adapter still fails loudly rather than silently pass having tested nothing.
const SOFTWARE_ADAPTER = /swiftshader|llvmpipe|lavapipe|warp|basic render/i;

/**
 * true when `hardware` (a joined `GPUAdapterInfo` identity string, or `readHardware`'s `"unknown"`
 * fallback when no adapter was offered at all) is not real GPU hardware. Pure — testable without a
 * browser. The CLI's own display gate: probed before any check runs, so a software adapter is refused
 * outright rather than left to crash mid-run.
 */
export function isSoftwareAdapter(hardware: string): boolean {
    return hardware === "unknown" || SOFTWARE_ADAPTER.test(hardware);
}

/** the display gate's decision for a probed adapter identity: {@link EXIT_NO_DISPLAY} to refuse, `null`
 *  to proceed. The seam `verifyCommand`'s refusal path reduces to, so the decision — and its exit code —
 *  is unit-testable without a browser (`testing.md`: never bind a device in `bun test`). */
export function displayGateExit(hardware: string): number | null {
    return isSoftwareAdapter(hardware) ? EXIT_NO_DISPLAY : null;
}

/** the refusal diagnostic for a non-hardware adapter: names what was offered and why the CLI won't run
 *  checks against it, never the caller or its environment (`branding.md` — point at the diagnostic). */
export function displayGateMessage(hardware: string): string {
    const reason =
        hardware === "unknown"
            ? "no GPU adapter was offered at all"
            : `the adapter is a software rasterizer ("${hardware}")`;
    return (
        `shallot verify needs a real GPU adapter, but ${reason}. A software adapter clears every ` +
        `feature/limit check and then crashes mid-run instead (GPU device lost, oversized buffer ` +
        `allocations) — a false engine-bug reading rather than a missing display. Run against a browser ` +
        `backed by real hardware, or attach to one via --connect.`
    );
}

// Match real failure signatures only — "adapter limits" / "requestAdapter" are normal startup chatter.
// Salvaged verbatim from harness/core/page.ts (the proven signature set).
const ERR_HINT =
    /\b(?:wgsl|shader compilation|pipeline.*invalid|destroyed|validation error|device.*lost|uncaptured|GPUValidationError|GPUInternalError|exceeds the max|crashed)\b/i;

const INSTALL_PLAYWRIGHT = "bun add -d playwright && bunx playwright install chromium";

// a setup failure the caller reports through the one failure path (JSON under --json) — distinct from
// a bug, which propagates as a plain Error.
/** @internal exported only so the serve* guard tests can assert the throw class directly. */
export class SetupError extends Error {}

export interface VerifyArgs {
    dir: string;
    dist: boolean;
    screenshot?: string;
    json: boolean;
    port?: number;
    query: string[];
    timeoutMs: number;
    /** sample retained JS heap for a leak slope over a post-run idle window (informational — never gates). */
    memory: boolean;
    /** expose `window.__probeAlloc(windowMs)` for a harness `run()` to measure allocation. */
    alloc: boolean;
    /** inject a deliberate retained allocation of N bytes/second once the harness is ready — the red-proof
     *  knob for the `--memory` leak detector (needs `--memory` to be observed). 0 = off. */
    leak: number;
    /** attach to a remote browser at this Playwright ws endpoint (`chromium.connect`); the endpoint's
     *  owner keeps the browser process, this run only drives it. Absent: launch a local browser. */
    connect?: string;
    /** print per-phase wall-clock spans (server boot, first page load, harness ready, run, memory idle,
     *  capture, teardown) after the verdict — makes a slow or hung run attributable to a phase. */
    timings: boolean;
    /** batch mode: one `&`-joined query-string spec per run, layered on top of the shared `--query`
     *  params. One boot, N page loads (fresh browser context + page each), one verdict each, a JSON
     *  array out, nonzero exit if any run fails. Empty (the default) keeps the single-run path. */
    run: string[];
    help: boolean;
}

/**
 * parse `shallot verify` flags. Pure — the CLI wiring and the tests share it. `--query k=v` repeats;
 * every other flag takes its last value. Unknown `--flags` and non-numeric `--port`/`--timeout`
 * values throw (a typo must not silently no-op or flow NaN into the wait math).
 */
export function parseVerifyArgs(raw: string[]): VerifyArgs {
    const num = (flag: string, v: string): number => {
        if (v.trim() === "") {
            throw new Error(`invalid ${flag} value "${v}" — must not be empty`);
        }
        const n = Number(v);
        if (!Number.isFinite(n) || n <= 0) {
            throw new Error(`invalid ${flag} value "${v}" — expected a positive number`);
        }
        return n;
    };
    const numNonNeg = (flag: string, v: string): number => {
        if (v.trim() === "") {
            throw new Error(`invalid ${flag} value "${v}" — must not be empty`);
        }
        const n = Number(v);
        if (!Number.isFinite(n) || n < 0) {
            throw new Error(`invalid ${flag} value "${v}" — expected a non-negative number`);
        }
        return n;
    };
    const args: VerifyArgs = {
        dir: ".",
        dist: false,
        json: false,
        query: [],
        timeoutMs: 60_000,
        memory: false,
        alloc: false,
        leak: 0,
        timings: false,
        run: [],
        help: false,
    };
    let sawDir = false;
    for (let i = 0; i < raw.length; i++) {
        const a = raw[i];
        if (a === "--dist") args.dist = true;
        else if (a === "--json") args.json = true;
        else if (a === "--memory") args.memory = true;
        else if (a === "--alloc") args.alloc = true;
        else if (a === "--timings") args.timings = true;
        else if (a === "--leak" && raw[i + 1]) args.leak = numNonNeg("--leak", raw[++i]);
        else if (a?.startsWith("--leak="))
            args.leak = numNonNeg("--leak", a.slice("--leak=".length));
        else if (a === "--help" || a === "-h") args.help = true;
        else if (a === "--screenshot" && raw[i + 1]) args.screenshot = raw[++i];
        else if (a === "--connect" && raw[i + 1]) args.connect = raw[++i];
        else if (a?.startsWith("--connect=")) args.connect = a.slice("--connect=".length);
        else if (a === "--port" && raw[i + 1]) args.port = num("--port", raw[++i]);
        else if (a?.startsWith("--port=")) args.port = num("--port", a.slice("--port=".length));
        else if (a === "--timeout" && raw[i + 1]) args.timeoutMs = num("--timeout", raw[++i]);
        else if (a?.startsWith("--timeout="))
            args.timeoutMs = num("--timeout", a.slice("--timeout=".length));
        else if (a === "--query" && raw[i + 1]) args.query.push(raw[++i]);
        else if (a?.startsWith("--query=")) args.query.push(a.slice("--query=".length));
        else if (a === "--run" && raw[i + 1]) args.run.push(raw[++i]);
        else if (a?.startsWith("--run=")) args.run.push(a.slice("--run=".length));
        else if (a?.startsWith("-")) throw new Error(`unknown option: ${a}`);
        else if (!sawDir) {
            args.dir = a;
            sawDir = true;
        } else throw new Error(`unexpected argument: ${a}`);
    }
    // the two heap samplers can't co-run: --memory forces a GC before each sample, which corrupts
    // --alloc's no-forced-GC window (page.ts's constraint). A run picks one.
    if (args.memory && args.alloc) {
        throw new Error("--memory and --alloc are mutually exclusive");
    }
    // --leak plants a retained allocation for the --memory slope to catch. Without --memory nothing samples
    // it (the injection is silent), and --leak requiring --memory keeps it clear of --alloc's no-forced-GC
    // window too (--memory/--alloc are already exclusive above).
    if (args.leak > 0 && !args.memory) {
        throw new Error("--leak requires --memory (nothing samples the injected allocation)");
    }
    return args;
}

/** append `--query k=v` params to a base URL (already carrying its own query, or none). */
export function buildUrl(base: string, query: string[]): string {
    if (query.length === 0) return base;
    const u = new URL(base);
    for (const q of query) {
        const eq = q.indexOf("=");
        if (eq === -1) u.searchParams.set(q, "");
        else u.searchParams.set(q.slice(0, eq), q.slice(eq + 1));
    }
    return u.toString();
}

/**
 * one query array per batch run: each `--run` spec is its own `&`-joined "k=v" query string, layered
 * on top of the `shared` (`--query`) params common to every run — so a batch of scenarios differs only
 * in what `--run` adds. Pure: string splitting only, no URL construction (`buildUrl` does that per run).
 */
export function resolveBatchQueries(shared: string[], runs: string[]): string[][] {
    return runs.map((spec) => [...shared, ...(spec.length === 0 ? [] : spec.split("&"))]);
}

/** the batch verdict: every run must pass. Empty input is never called live (`run.length === 0` takes
 *  the single-run path instead) but reads as a fail rather than a vacuous pass. */
export function batchPass(results: { pass: boolean }[]): boolean {
    return results.length > 0 && results.every((r) => r.pass);
}

/** one wall-clock marker taken at a phase boundary — `name` labels the phase that just *ended*. */
export interface Checkpoint {
    name: string;
    t: number;
}

/** one named phase span: how long `name` took, in milliseconds. */
export interface Span {
    name: string;
    ms: number;
}

/**
 * ordered checkpoints (marker 0 is a start fencepost, its own name unused) into the named spans
 * between them — the gap before checkpoint i is attributed to checkpoint i's name. Pure: durations
 * only, no clock reads, so the boundary math is unit-testable without a browser.
 */
export function spansFromCheckpoints(checkpoints: Checkpoint[]): Span[] {
    const spans: Span[] = [];
    for (let i = 1; i < checkpoints.length; i++) {
        spans.push({ name: checkpoints[i].name, ms: checkpoints[i].t - checkpoints[i - 1].t });
    }
    return spans;
}

/** render spans as aligned "name  123ms" lines, one per line, name column padded to the longest. */
export function formatTimings(spans: Span[]): string {
    const width = Math.max(0, ...spans.map((s) => s.name.length));
    return spans.map((s) => `  ${s.name.padEnd(width)}  ${s.ms}ms`).join("\n");
}

/**
 * install the `__harness` setter probe on `target`: writing `__harness` also stamps
 * `__harnessInstallAt` with the wall-clock time of that write. The `harness ready` checkpoint comes
 * from a 500ms poll (quantized, `stepWait`); this reads the exact in-page install moment instead.
 * Pure over a plain object and closure-free, so the same function both unit-tests without a browser
 * and ships in-page via {@link HARNESS_PROBE_SCRIPT} — one source of truth for a probe whose whole
 * value is reading the truth.
 */
export function installHarnessProbe(target: Record<string, unknown>): void {
    let value: unknown;
    Object.defineProperty(target, "__harness", {
        configurable: true,
        get: () => value,
        set: (v: unknown) => {
            value = v;
            target.__harnessInstallAt = Date.now();
        },
    });
}

/** {@link installHarnessProbe} as page source, applied to `window`. A Playwright init script is
 *  stringified and run in-page, so it can't close over an imported binding — serializing the tested
 *  function itself is what keeps the shipped probe and the proven one the same code. */
export const HARNESS_PROBE_SCRIPT = `(${installHarnessProbe.toString()})(window)`;

/** how many resource-timing entries the page keeps. The spec default is 250, and a workspace-symlinked
 *  dependency — resolving outside `node_modules`, so Vite's scanner never prebundles it — serves every
 *  module as its own unbundled request and blows past that default. A saturated buffer reports exactly
 *  250 however many modules were really fetched, silently flattening the one quantity this readout
 *  exists to measure. */
export const RESOURCE_BUFFER = 20_000;

/** everything `--timings` installs before navigation: the harness-install probe plus the raised
 *  resource-timing buffer. Both must be in place before the first request, so they share one init
 *  script rather than racing two. */
export const TIMINGS_INIT_SCRIPT = `performance.setResourceTimingBufferSize(${RESOURCE_BUFFER});${HARNESS_PROBE_SCRIPT}`;

/** ms from navigation start to the `__harness` setter firing, per {@link installHarnessProbe} —
 *  null when it never fired (no harness on this page, or the probe wasn't installed). */
export function harnessInstallMs(
    navStart: number,
    installedAt: number | null | undefined,
): number | null {
    return installedAt == null ? null : installedAt - navStart;
}

/** one `performance.getEntriesByType("resource")` entry, reduced to what the readout needs. */
export interface ResourceEntry {
    name: string;
    duration: number;
}

/** the resource-timing readout: request count, summed transfer duration, and the top-N slowest — the
 *  one signal that can see dev-server module transport directly (a workspace-symlinked dependency
 *  resolves outside `node_modules`, so Vite's scanner never prebundles it and serves every module as
 *  its own unbundled request). Pure: the page collects the raw entries, this reduces them. */
export interface ResourceTiming {
    count: number;
    totalMs: number;
    top: ResourceEntry[];
    /** the page's buffer filled, so `count` and `totalMs` are floors, not the real totals
     *  ({@link RESOURCE_BUFFER}). Reported rather than hidden — a silently capped count reads as a
     *  measurement of the transport this whole readout exists to attribute. */
    saturated: boolean;
}

export function summarizeResourceTiming(
    entries: ResourceEntry[],
    topN = 10,
    buffer = RESOURCE_BUFFER,
): ResourceTiming {
    const totalMs = entries.reduce((sum, e) => sum + e.duration, 0);
    const top = [...entries].sort((a, b) => b.duration - a.duration).slice(0, topN);
    return { count: entries.length, totalMs, top, saturated: entries.length >= buffer };
}

/** render the harness-install probe + resource-timing readout as extra `--timings` lines, appended
 *  after the phase spans. */
export function formatExtraTimings(
    harnessMs: number | null,
    resource: ResourceTiming | null,
): string {
    const lines: string[] = [
        `  harness install (probe)  ${harnessMs === null ? "n/a" : `${harnessMs}ms`}`,
    ];
    if (resource) {
        lines.push(
            `  resources  ${resource.count} requests, ${resource.totalMs.toFixed(0)}ms total${
                resource.saturated ? " (buffer full — both are floors)" : ""
            }`,
        );
        for (const e of resource.top) lines.push(`    ${e.duration.toFixed(0)}ms  ${e.name}`);
    }
    return lines.join("\n");
}

/** what the harness-install probe and resource-timing readout observed for one page — mutated by
 *  {@link drive} the same way `checkpoints` is, since both are opt-in extras under `--timings` rather
 *  than part of the published {@link Result} shape. */
export interface TimingProbes {
    harnessInstallMs: number | null;
    resourceTiming: ResourceTiming | null;
}

const spread = (a: number[], b: number[]): number =>
    Math.abs(a[0] - b[0]) + Math.abs(a[1] - b[1]) + Math.abs(a[2] - b[2]);

/** the centre-vs-corner spread a frame must exceed to count as structure. One constant because the
 *  render probe reports the spread *against* it — a second literal at the populate site would let a
 *  tuned threshold make the probe lie about the reading it just took. */
export const STRUCTURE_THRESHOLD = 12;

/** a frame carries visible structure (not a single flat clear color) — center vs corner region contrast. */
export function structured(center: number[], corner: number[]): boolean {
    return spread(center, corner) > STRUCTURE_THRESHOLD;
}

/** a captured frame shows visible structure — the pixel-honest signal the harness path gates `rendered` on.
 *  The settle path's center-vs-corner contrast: a scene the camera frames centrally lifts the centre off the
 *  cleared corner. A flat clear color, or a scene whose centre reads the clear color (a model that never
 *  rendered — the gym gltf symptom), or a missing sample (no capturable canvas / decode failure) is not
 *  rendered. Coarser whole-frame variants were rejected: on real hardware a "blank" canvas still carries a
 *  faint background gradient (measured spread ~49 on gltf), so a whole-frame max-min passes the very blanks
 *  this must catch. Known limit: a scene framed away from centre reads blank here (a finding to surface, not
 *  a reason to widen the check into one that stops catching gltf). */
export function hasStructure(sample: FrameSample | null): boolean {
    return sample !== null && structured(sample.center, sample.corner);
}

/** mean absolute RGB difference between two coarse frame grids, 0..255 — the magnitude of visible change. */
export function gridDiff(a: number[], b: number[]): number {
    const n = Math.min(a.length, b.length);
    let sum = 0;
    for (let i = 0; i < n; i++) sum += Math.abs(a[i] - b[i]);
    return sum / n;
}

/** one decoded frame reduced to a 64×64 RGB grid plus the two region averages the settle test reads. */
export interface FrameSample {
    grid: number[];
    center: number[];
    corner: number[];
}

/** the settle-tracking half of the unified wait loop, threaded between polls. */
export interface WaitState {
    /** a canvas appeared and produced at least one sample */
    booted: boolean;
    /** last structured grid seen, for the consecutive-shot diff */
    prev: number[] | null;
}

/**
 * one step of the unified wait: a defined `window.__harness` always wins — a slow-installing harness
 * must never downgrade to the settle-only smoke check (its own assertions would silently not run).
 * Otherwise fold the frame sample into the settle tracking: two consecutive structured shots below
 * the 0.5 diff epsilon = settled (the measured boot shape — one lighting/shadow transition after
 * first structure, then bit-static).
 */
export function stepWait(
    st: WaitState,
    harnessDefined: boolean,
    sample: FrameSample | null,
): "harness" | "settled" | "continue" {
    if (harnessDefined) return "harness";
    if (sample) {
        st.booted = true;
        if (structured(sample.center, sample.corner)) {
            if (st.prev && gridDiff(st.prev, sample.grid) < 0.5) return "settled";
            st.prev = sample.grid;
        }
    }
    return "continue";
}

/**
 * interpret what `__harness.run()` resolved: an object carrying a boolean `ok` passes through
 * (extra fields intact); anything else — undefined, a bare value, a missing/non-boolean `ok` — is a
 * clean FAIL, never a dereference of a non-verdict.
 */
export function coerceVerdict(value: unknown): Verdict {
    if (
        value !== null &&
        typeof value === "object" &&
        typeof (value as { ok?: unknown }).ok === "boolean"
    ) {
        return value as Verdict;
    }
    const got = value === null ? "null" : Array.isArray(value) ? "array" : typeof value;
    return {
        ok: false,
        checks: [
            {
                name: "run",
                ok: false,
                detail: `run() returned no verdict — expected { ok: boolean, ... }, got ${got}`,
            },
        ],
    };
}

/**
 * bound a promise that playwright doesn't bound itself — `page.evaluate` has no default timeout, so a
 * project `run()` that never resolves would hang the gate (and the shell that ran it) forever.
 */
export function withTimeout<T>(p: Promise<T>, ms: number, what: string): Promise<T> {
    return new Promise((res, rej) => {
        const t = setTimeout(() => rej(new Error(`${what} did not resolve within ${ms}ms`)), ms);
        p.then(
            (v) => {
                clearTimeout(t);
                res(v);
            },
            (e) => {
                clearTimeout(t);
                rej(e);
            },
        );
    });
}

/** harness path passes when the project's Verdict is ok, the canvas rendered visible structure, and no
 *  page errors were captured. A green verdict over a blank canvas (a scenario that draws nothing) fails —
 *  unless the harness declared `noRender`, which reports `rendered: "opt-out"` and passes the pixel gate. */
export function harnessPass(
    verdict: Verdict,
    rendered: boolean | "opt-out",
    errorCount: number,
): boolean {
    return verdict.ok === true && rendered !== false && errorCount === 0;
}

// A GPU-side `console.log` prints only when its kernel's ring buffer is read back, which is why the
// lines are worth surfacing at all: they are the one channel a shader has, and they otherwise scroll
// past in a headless browser nobody is watching. Errors are the free assert — a kernel that detects a
// bad state can say so and fail the gate from inside the shader.
const GPU_LOG_SHOWN = 8;

/** fold what GPU kernels printed into the verdict: one informational check carrying the lines, plus a
 *  failing check when any of them came through `console.error`. Empty when nothing logged. */
export function gpuLogChecks(log: GpuLog | null | undefined): Check[] {
    if (!log || log.lines.length === 0) return [];
    const shown = log.lines.slice(0, GPU_LOG_SHOWN);
    const more = log.lines.length - shown.length;
    const detail = shown.join(" | ") + (more > 0 ? ` | (+${more} more)` : "");
    const checks: Check[] = [{ name: "gpu.log", ok: true, detail }];
    if (log.errors.length > 0) {
        checks.push({ name: "gpu.error", ok: false, detail: log.errors.join(" | ") });
    }
    return checks;
}

/** retain generated WGSL only on a failed verify result. */
export function failureArtifacts(
    pass: boolean,
    diagnostics: GpuDiagnostics | null | undefined,
): ShaderArtifact[] | undefined {
    return !pass && diagnostics && diagnostics.artifacts.length > 0
        ? diagnostics.artifacts
        : undefined;
}

/** no-harness path passes when the scene booted, rendered a settled frame, and threw no page errors. */
export function settlePass(booted: boolean, rendered: boolean, errorCount: number): boolean {
    return booted && rendered && errorCount === 0;
}

// leak threshold, bytes/second of retained heap. Derivation (inherited, kept): the repo harness flagged
// 1024 B/frame; at 60fps that's 1024 × 60 = 61_440 B/s — the same physical growth re-expressed as a
// wall-clock rate, which is exactly what the post-run idle window fits (bytes/second, no frame count to
// divide by). Informational only: --memory never affects the verdict or exit code.
export const LEAK_BYTES_PER_SEC = 61_440;

// the post-run idle window the --memory slope is fitted over. run() has resolved, so the benchmark's own
// monotonically-growing stats arrays are unreachable and collect on the first forced GC — this window then
// measures the engine's steady-state retention alone. Sized for ≥5 samples at startMemory's 800ms cadence
// (fitMemory needs ≥3 after dropping the cold-start one), independent of how fast the scene rendered.
const LEAK_IDLE_MS = 5200;

/** one wall-clock heap reading: epoch ms + retained JS heap bytes (post forced-GC). */
export interface MemorySample {
    t: number;
    heap: number;
}

/** the --memory report: heap endpoints, the fitted leak slope, and the GC activity over the window. */
export interface MemoryStats {
    start: number;
    end: number;
    growthPerSecond: number;
    leak: boolean;
    gcCount: number;
    gcPauseMs: number;
}

/**
 * least-squares heap slope in bytes/second over wall-clock samples, plus the leak verdict. Null when
 * fewer than three samples — two would force fitting through the cold-start reading dropped below,
 * exactly the bias this exists to remove. Pure — the sampler collects, this fits.
 */
export function fitMemory(
    samples: MemorySample[],
    gcCount: number,
    gcPauseMs: number,
): MemoryStats | null {
    if (samples.length < 3) return null;
    // Drop the cold-start sample before fitting: the first reading precedes the workload's steady-state
    // working set (pipeline caches, profiler maps), so it biases the slope upward on a scene that retains
    // nothing. A leak is steady-state growth — same reason the timing measure discards warmup frames.
    const fit = samples.slice(1);
    const t0 = fit[0].t;
    const n = fit.length;
    let sumX = 0;
    let sumY = 0;
    let sumXY = 0;
    let sumX2 = 0;
    for (const p of fit) {
        const x = (p.t - t0) / 1000; // seconds from the first fitted sample
        sumX += x;
        sumY += p.heap;
        sumXY += x * p.heap;
        sumX2 += x * x;
    }
    const slope = (n * sumXY - sumX * sumY) / (n * sumX2 - sumX * sumX);
    return {
        start: fit[0].heap,
        end: fit[n - 1].heap,
        growthPerSecond: slope,
        leak: slope > LEAK_BYTES_PER_SEC,
        gcCount,
        gcPauseMs,
    };
}

/** @internal exported only so the driveHarness/withGpuLog page-stub tests can type their fixtures. */
export interface Result {
    project: string;
    timestamp: string;
    mode: "dev" | "dist";
    url: string;
    /** the GPU adapter's identity string (vendor / arch / device / description), "unknown" if unavailable. */
    hardware: string;
    harness: boolean;
    booted: boolean;
    /** pixel-honest render verdict: the canvas showed visible structure (`true`), was blank (`false`), or
     *  the harness declared `noRender` so the pixel gate was skipped (`"opt-out"` — visible, never a fake
     *  true). See {@link harnessPass} and the harness protocol's `noRender`. */
    rendered: boolean | "opt-out";
    verdict?: Verdict;
    /** the --memory leak sample, when requested: fitted across a post-run idle window (after run()
     *  resolves, when the benchmark's own growing stats arrays have gone collectible — so the slope reads
     *  the engine's steady-state retention, not the harness measuring itself). null when there's nothing
     *  to fit — no harness run() (settle path), too few samples, or any CDP failure. Informational, never
     *  gates. */
    memory?: MemoryStats | null;
    /** bounded generated WGSL records, present only on failure. */
    artifacts?: ShaderArtifact[];
    /** the render probe — samples taken, last centre/corner RGB, spread against `structured`'s
     *  threshold, elapsed wait, and how the wait concluded — so a blank-render red carries its
     *  measurement rather than printing `[]`. Absent on a setup failure. */
    renderProbe?: RenderProbe;
    errors: string[];
    pass: boolean;
}

/** the render probe the settle path records on its Result — the pixel evidence behind the `rendered`
 *  verdict. Carries the frame samples the wait loop took, the last centre/corner RGB and the spread
 *  against `structured`'s threshold, the elapsed wait, and how the wait concluded. */
export interface RenderProbe {
    /** frame samples the wait loop captured (non-null `sampleFrame` returns). */
    samples: number;
    /** last centre RGB [r,g,b] the probe measured, or null if no sample was ever taken. */
    center: number[] | null;
    /** last corner RGB [r,g,b] the probe measured, or null if no sample was ever taken. */
    corner: number[] | null;
    /** the centre-vs-corner spread (sum of abs channel diffs) of the last sample, or null. */
    spread: number | null;
    /** the threshold `structured` gates on (`STRUCTURE_THRESHOLD`). */
    threshold: number;
    /** elapsed milliseconds from the first poll to the wait outcome. */
    elapsed: number;
    /** how the wait concluded: `harness` (window.__harness appeared), `settled` (two consecutive
     *  structured shots below the diff epsilon), or `timeout` (the deadline expired). */
    outcome: "harness" | "settled" | "timeout";
}

// the Playwright Page — typed loosely because playwright is an optional dep with no types at build time.
type Page = any;

// the in-page `window.__harness` shape is the published protocol's — `../src/harness` declares the
// Window global, so the page functions below read it directly.

// decode a PNG (base64) with the browser's native decoder and reduce it to the 64×64 sample grid +
// region averages. Runs IN THE PAGE (playwright serializes this function's source). The bytes come
// from a node-side compositor element screenshot: drawImage(webgpuCanvas) reads blank on real
// hardware (probed 2026-07-13 — solid red read [0,0,0,0] while the element screenshot read the true
// pixels), so the screenshot is the capture and this only decodes it. Self-contained — closes over nothing.
async function decodeSample(b64: string): Promise<FrameSample | null> {
    const img = new Image();
    img.src = `data:image/png;base64,${b64}`;
    try {
        await img.decode();
    } catch {
        return null;
    }
    const W = 64;
    const H = 64;
    const off = document.createElement("canvas");
    off.width = W;
    off.height = H;
    const ctx = off.getContext("2d");
    if (!ctx) return null;
    ctx.drawImage(img, 0, 0, W, H);
    const d = ctx.getImageData(0, 0, W, H).data;
    const grid: number[] = [];
    for (let i = 0; i < d.length; i += 4) grid.push(d[i], d[i + 1], d[i + 2]);
    const region = (fx0: number, fy0: number, fx1: number, fy1: number): number[] => {
        const x0 = Math.floor(fx0 * W);
        const x1 = Math.max(x0 + 1, Math.floor(fx1 * W));
        const y0 = Math.floor(fy0 * H);
        const y1 = Math.max(y0 + 1, Math.floor(fy1 * H));
        let r = 0;
        let g = 0;
        let b = 0;
        let n = 0;
        for (let y = y0; y < y1; y++)
            for (let x = x0; x < x1; x++) {
                const i = (y * W + x) * 3;
                r += grid[i];
                g += grid[i + 1];
                b += grid[i + 2];
                n++;
            }
        return [r / n, g / n, b / n];
    };
    return { grid, center: region(0.4, 0.4, 0.6, 0.6), corner: region(0, 0, 0.15, 0.15) };
}

// decode a screenshot PNG (base64) to full-resolution RGBA + dimensions, native browser decode. Runs IN
// THE PAGE (self-contained, no closures — the decodeSample precedent); full resolution because a masked
// color footprint needs exact bytes, unlike decodeSample's 64×64 downsample for the coarser structure
// check. Classification itself stays node-side via the published `probePixels` — one source of truth for
// the mask math, only the decode is duplicated (forced by Playwright's page.evaluate serialization boundary).
async function decodeRgba(
    b64: string,
): Promise<{ width: number; height: number; rgba: string } | null> {
    const img = new Image();
    img.src = `data:image/png;base64,${b64}`;
    try {
        await img.decode();
    } catch {
        return null;
    }
    const canvas = document.createElement("canvas");
    canvas.width = img.naturalWidth;
    canvas.height = img.naturalHeight;
    const ctx = canvas.getContext("2d");
    if (!ctx) return null;
    ctx.drawImage(img, 0, 0);
    const bytes = ctx.getImageData(0, 0, canvas.width, canvas.height).data;
    let binary = "";
    const chunk = 0x8000;
    for (let i = 0; i < bytes.length; i += chunk) {
        binary += String.fromCharCode(...bytes.subarray(i, i + chunk));
    }
    return { width: canvas.width, height: canvas.height, rgba: btoa(binary) };
}

// capture the post-run compositor once, decode it full-resolution, and classify every declared
// `pixelProbe` against it via the published `probePixels`/`pixelProbePass`. A capture or decode failure
// fails every probe closed (never a silent skip) — the whole point is catching a scene that renders
// nothing, and a probe that can't observe the frame must read as "didn't observe the tag", not as absent.
async function samplePixelProbes(page: Page, probes: PixelProbe[]): Promise<Check[]> {
    let shot: { toString(encoding: string): string };
    try {
        shot = await page.locator("canvas").first().screenshot({ timeout: 5_000 });
    } catch {
        return probes.map((p) => ({ name: p.name, ok: false, detail: "canvas screenshot failed" }));
    }
    const decoded = (await page
        .evaluate(decodeRgba, shot.toString("base64"))
        .catch(() => null)) as { width: number; height: number; rgba: string } | null;
    if (!decoded)
        return probes.map((p) => ({ name: p.name, ok: false, detail: "pixel decode failed" }));
    const rgba = new Uint8Array(Buffer.from(decoded.rgba, "base64"));
    return probes.map((probe) => {
        const result = probePixels(rgba, decoded.width, decoded.height, probe);
        return {
            name: probe.name,
            ok: pixelProbePass(result, probe),
            detail: `${result.pixels} tagged px, ${result.width}×${result.height} footprint`,
        };
    });
}

// capture the canvas through the compositor (element screenshot — the proven mechanism the repo
// harness and eval gates read WebGPU pixels with), then hand the PNG into the page for native
// decoding. null when the canvas isn't there or can't be shot yet — the loop keeps polling.
async function sampleFrame(page: Page): Promise<FrameSample | null> {
    let shot: { toString(encoding: string): string };
    try {
        shot = await page.locator("canvas").first().screenshot({ timeout: 5_000 });
    } catch {
        return null;
    }
    try {
        return (await page.evaluate(decodeSample, shot.toString("base64"))) as FrameSample | null;
    } catch {
        return null;
    }
}

// the GPU adapter's identity string, read in the page. "unknown" when there's no adapter or the info is
// bare. Salvaged from harness/core/page.ts. The caller guards the evaluate with .catch — a broken page
// must not crash the gate.
async function readHardware(page: Page): Promise<string> {
    return (await page.evaluate(async () => {
        const adapter = await navigator.gpu?.requestAdapter();
        if (!adapter) return "unknown";
        const info = adapter.info;
        return (
            [info.vendor, info.architecture, info.device, info.description]
                .filter(Boolean)
                .join(" / ") || "unknown"
        );
    })) as string;
}

// one CDP Performance metric by name, 0 if absent.
type Metrics = { metrics: { name: string; value: number }[] };
const metricOf = (m: Metrics, name: string): number =>
    m.metrics.find((x) => x.name === name)?.value ?? 0;

/** a running --memory sampler; `stop()` ends sampling and resolves the fitted stats (null on any failure). */
interface MemorySampler {
    stop(): Promise<MemoryStats | null>;
}

// sample retained JS heap on a fixed wall-clock cadence over the post-run idle window (the caller starts it
// once run() has resolved). Forces a GC before each read so the slope measures *retained* memory, not the
// transient per-frame allocations (profiler sample arrays, overlay text churn) that pile up uncollected in a
// short window and read as a false leak on a scene that holds nothing. Best-effort: any CDP failure yields a
// null result, never a crash — the memory report is informational and never gates. Salvaged from
// harness/core/page.ts, re-cadenced to wall-clock.
async function startMemory(page: Page): Promise<MemorySampler> {
    const SampleMs = 800;
    try {
        const cdp = await page.context().newCDPSession(page);
        await cdp.send("Performance.enable");
        const initial = (await cdp.send("Performance.getMetrics")) as Metrics;
        const gcCountStart = metricOf(initial, "MinorGCCount") + metricOf(initial, "MajorGCCount");
        const gcDurStart =
            metricOf(initial, "MinorGCDuration") + metricOf(initial, "MajorGCDuration");

        const samples: MemorySample[] = [];
        let stopped = false;
        const loop = (async () => {
            while (!stopped) {
                // collectGarbage needs no domain enable; swallow if the backend lacks it (sampling proceeds).
                await cdp.send("HeapProfiler.collectGarbage").catch(() => {});
                const m = (await cdp
                    .send("Performance.getMetrics")
                    .catch(() => null)) as Metrics | null;
                if (m) {
                    const heap = metricOf(m, "JSHeapUsedSize");
                    if (heap) samples.push({ t: Date.now(), heap });
                }
                await new Promise((r) => setTimeout(r, SampleMs));
            }
        })();

        return {
            stop: async (): Promise<MemoryStats | null> => {
                stopped = true;
                await loop.catch(() => {});
                try {
                    const final_ = (await cdp.send("Performance.getMetrics")) as Metrics;
                    const gcCount =
                        metricOf(final_, "MinorGCCount") +
                        metricOf(final_, "MajorGCCount") -
                        gcCountStart;
                    const gcPauseMs =
                        (metricOf(final_, "MinorGCDuration") +
                            metricOf(final_, "MajorGCDuration") -
                            gcDurStart) *
                        1000;
                    await cdp.send("Performance.disable").catch(() => {});
                    await cdp.detach().catch(() => {});
                    return fitMemory(samples, gcCount, gcPauseMs);
                } catch {
                    return null;
                }
            },
        };
    } catch {
        return { stop: async () => null };
    }
}

// one allocating call site, attributed by the heap sampling profiler.
interface Allocator {
    name: string;
    location: string;
    bytes: number;
}

// CDP sampling-profile node: a call frame + its self-allocated bytes + children.
interface HeapNode {
    callFrame: { functionName: string; url: string; lineNumber: number };
    selfSize: number;
    children?: HeapNode[];
}

// reduce the sampling-profile tree to the top allocators by self-size — the attributed "where" a
// scenario's alloc assert reads. Salvaged from harness/core/page.ts.
function topAllocators(node: HeapNode | undefined, n: number): Allocator[] {
    const by = new Map<string, Allocator>();
    const walk = (h: HeapNode | undefined): void => {
        if (!h) return;
        if (h.selfSize > 0) {
            const cf = h.callFrame;
            const name = cf.functionName || "(anonymous)";
            const location = `${cf.url.split("/").pop() ?? cf.url}:${cf.lineNumber + 1}`;
            const key = `${name}@${location}`;
            const a = by.get(key);
            if (a) a.bytes += h.selfSize;
            else by.set(key, { name, location, bytes: h.selfSize });
        }
        if (h.children) for (const c of h.children) walk(c);
    };
    walk(node);
    return [...by.values()].sort((a, b) => b.bytes - a.bytes).slice(0, n);
}

// expose `window.__probeAlloc(windowMs)` — a scenario's allocation-hunt primitive. Each call forces one GC
// for a clean baseline, then samples allocations for `windowMs` WITHOUT forcing GC (so transient per-frame
// allocation is visible, not collected away), and returns heap growth + GC count + the top allocators by
// sampled self-size. A harness `run()` ramps a work knob between calls and reads the slope. Salvaged
// verbatim from harness/core/page.ts. Best-effort: a CDP/HeapProfiler failure leaves the binding uninstalled.
async function installAllocProbe(page: Page): Promise<void> {
    const cdp = await page.context().newCDPSession(page);
    await cdp.send("Performance.enable");
    await page.exposeFunction("__probeAlloc", async (windowMs: number) => {
        await cdp.send("HeapProfiler.collectGarbage").catch(() => {});
        const m0 = (await cdp.send("Performance.getMetrics")) as Metrics;
        const heap0 = metricOf(m0, "JSHeapUsedSize");
        const gc0 = metricOf(m0, "MinorGCCount") + metricOf(m0, "MajorGCCount");
        const dur0 = metricOf(m0, "MinorGCDuration") + metricOf(m0, "MajorGCDuration");

        // The HeapProfiler's allocation-tracking instrumentation taxes every system's timing while enabled,
        // so leaving it on for the whole run inflates per-system CPU p99. Confine it to the window. 4 KB
        // sampling interval (vs the 32 KB default) — fine enough to attribute a per-frame allocation to its
        // stack without taxing the window.
        await cdp.send("HeapProfiler.enable").catch(() => {});
        await cdp.send("HeapProfiler.startSampling", { samplingInterval: 4096 }).catch(() => {});
        await new Promise((r) => setTimeout(r, windowMs)); // no forced GC — the engine RAF loop runs
        const m1 = (await cdp.send("Performance.getMetrics")) as Metrics;
        const profile = await cdp
            .send("HeapProfiler.stopSampling")
            .then((p: unknown) => (p as { profile?: { head: HeapNode } }).profile?.head)
            .catch(() => undefined);
        await cdp.send("HeapProfiler.disable").catch(() => {});

        return {
            heapDelta: metricOf(m1, "JSHeapUsedSize") - heap0,
            gcCount: metricOf(m1, "MinorGCCount") + metricOf(m1, "MajorGCCount") - gc0,
            gcPauseMs:
                (metricOf(m1, "MinorGCDuration") + metricOf(m1, "MajorGCDuration") - dur0) * 1000,
            top: topAllocators(profile, 8),
        } satisfies AllocProbe;
    });
}

interface AllocProbe {
    heapDelta: number;
    gcCount: number;
    gcPauseMs: number;
    top: Allocator[];
}

// inject a deliberate retained allocation of `bytesPerSec` bytes/second — the red-proof for the --memory
// leak detector. A GC-rooted array on the page grows at a fixed 10 Hz wall-clock rate, so every forced-GC
// sample sees monotonic retained growth: a real leak the detector must flag, in the post-run idle window as
// much as during run(). Each chunk is a distinct-float array so V8 backs it as an on-heap double array
// counted in JSHeapUsedSize (an integer-fill array packs as SMIs and an external ArrayBuffer sits off-heap —
// both under-count). Best-effort; the page tears down at run end, so nothing unroots it.
async function injectLeak(page: Page, bytesPerSec: number): Promise<void> {
    await page
        .evaluate((rate: number) => {
            const held: number[][] = [];
            (globalThis as { __leakHeld?: number[][] }).__leakHeld = held; // root past forced GC
            const perTick = Math.max(1, Math.round(rate / 10 / 8)); // 10 Hz, ~8 B per double element
            setInterval(() => {
                const a = new Array<number>(perTick);
                for (let i = 0; i < perTick; i++) a[i] = Math.random();
                held.push(a);
            }, 100);
        }, bytesPerSec)
        .catch(() => {});
}

/** @internal exported only so serveDist/serveDev's return type resolves for the guard tests. */
export interface Booter {
    url: string;
    stop: () => Promise<void>;
    mode: "dev" | "dist";
}

// pick an ephemeral free port (or honor --port), via node:net so this runs under node too (the WSL bridge
// bundles this CLI and drives it with node — Bun is undefined there). Bind an OS-assigned port, read it,
// release it, hand it to the dev/dist server. A tiny race, acceptable for a one-shot gate.
function pickPort(explicit?: number): Promise<number> {
    if (explicit != null) return Promise.resolve(explicit);
    return new Promise((res, rej) => {
        const probe = createTcpServer();
        probe.on("error", rej);
        probe.listen(0, () => {
            const p = (probe.address() as { port: number } | null)?.port;
            probe.close(() => {
                if (p == null) rej(new Error("could not pick a free port"));
                else res(p);
            });
        });
    });
}

// serve an existing dist/ statically over vite's own preview server — the same mechanism serveDev/
// serveEjected already use. Together with pickPort's node:net probe, the whole boot path is now
// runtime-agnostic (the WSL bridge runs this CLI under node, where Bun is undefined).
// No implicit build — a missing dist is an actionable error, not a silent recovery.
/** @internal exported only so the SetupError guard can be driven by temp dir without booting vite. */
export async function serveDist(projectDir: string, port: number): Promise<Booter> {
    // hardcoded rather than read from the project's vite config (which preview() below derives outDir
    // from): build.ts pins `outDir: "dist"` for every manifest build, so this can't diverge in practice.
    const dist = resolve(projectDir, "dist");
    if (!existsSync(join(dist, "index.html"))) {
        throw new SetupError(
            `no build at ${dist} — "shallot verify --dist" serves an existing build; run "shallot build" first`,
        );
    }
    const server = await preview({
        root: projectDir,
        // cross-origin isolation so tumble physics multithreads (the same COOP/COEP the dev/preview
        // servers send — a --dist verify run must isolate the built page too)
        preview: { port, strictPort: true, open: false, headers: CROSS_ORIGIN_ISOLATION },
    });
    const url = server.resolvedUrls?.local?.[0] ?? `http://localhost:${port}/`;
    return {
        url,
        mode: "dist",
        stop: async () => {
            await server.close();
        },
    };
}

/**
 * which boot arm a dir selects: a shallot manifest/.scene project (the synthesized-entry dev server),
 * an ejected vite app that owns its own `index.html`, or neither. Pure over the two filesystem probes so
 * the arm choice is unit-testable without a disk.
 */
export function bootArm(
    isShallotProject: boolean,
    hasIndexHtml: boolean,
): "project" | "ejected" | "none" {
    if (isShallotProject) return "project";
    if (hasIndexHtml) return "ejected";
    return "none";
}

// boot the dev server. A shallot manifest/.scene project gets the exact vite config `shallot dev` uses
// (synthesized entry + projectPlugin); an ejected project that owns its own index.html gets a plain vite
// server rooted at it (vite auto-loads the project's own vite.config from root). Never opens a tab, always
// an auto-picked port. Fails loud through the setup path when the dir is neither shape.
/** @internal exported only so the SetupError guard can be driven by temp dir without booting vite. */
export async function serveDev(projectDir: string, port: number): Promise<Booter> {
    const arm = bootArm(isProject(projectDir), existsSync(join(projectDir, "index.html")));
    if (arm === "none") {
        throw new SetupError(
            `nothing to boot at ${projectDir} — expected a shallot.json manifest or a .scene file (scaffold one with "bun create shallot <name>"), or an index.html for an ejected vite app`,
        );
    }
    if (arm === "ejected") return serveEjected(projectDir, port);

    const name = basename(projectDir);
    const project = await loadProjectConfig(projectDir, "serve", "development");
    const server = await createServer(
        composeViteConfig(
            devConfig(projectDir, name, { port, strictPort: true, open: false }),
            project,
            new Set(["shallot-project", "shallot-synth-index", "unplugin-typegpu"]),
        ),
    );
    await server.listen();
    const url = server.resolvedUrls?.local?.[0] ?? `http://localhost:${port}/`;
    return {
        url,
        mode: "dev",
        stop: async () => {
            await server.close();
        },
    };
}

// an ejected project (gym, the flow fixtures) already owns its index.html and, optionally, its own
// vite.config; boot a plain vite server rooted at it. Passing no configFile lets vite auto-load that
// config from `root`; the server overrides pin our port and keep it headless. Holds for any vite app
// whose index.html sits at the dir root — `root: projectDir` wins over a config's own `root`.
async function serveEjected(projectDir: string, port: number): Promise<Booter> {
    const server = await createServer({
        root: projectDir,
        // cross-origin isolation so tumble physics multithreads (COOP/COEP → shared WebAssembly.Memory);
        // the ejected boot the gym physics bench uses, so the MT assert exercises the headers here
        server: { port, strictPort: true, open: false, headers: CROSS_ORIGIN_ISOLATION },
    });
    await server.listen();
    const url = server.resolvedUrls?.local?.[0] ?? `http://localhost:${port}/`;
    return {
        url,
        mode: "dev",
        stop: async () => {
            await server.close();
        },
    };
}

// resolve the optional playwright module from the CONSUMER's project (node walk-up from inside the
// installed package reaches its node_modules). Returns null when it isn't installed — the caller then
// prints the exact remedy and exits with the distinct EXIT_NO_PLAYWRIGHT.
async function importPlaywright(projectDir: string): Promise<{ chromium: unknown } | null> {
    try {
        return (await import("playwright")) as { chromium: unknown };
    } catch {
        // fall through to a project-rooted resolve
    }
    try {
        const req = createRequire(join(projectDir, "package.json"));
        return (await import(req.resolve("playwright"))) as { chromium: unknown };
    } catch {
        return null;
    }
}

const usage = `
  shallot verify [dir] — boot the project in a headless browser and check it renders

  By default runs the dev server and waits for a settled, non-blank frame (booted + rendered
  + zero page errors). A project that installs window.__harness (import { installHarness } from
  "@dylanebert/shallot/harness") gets its run(opts) Verdict driven instead.

  Options
    --dist                Serve an existing dist/ build instead of the dev server (run "shallot build" first)
    --screenshot <path>   Write a post-run canvas PNG
    --query k=v           URL param passed to the page (repeatable)
    --port <n>            Server port (default: an auto-picked free port)
    --timeout <ms>        Overall settle/ready budget (default: 60000)
    --memory              Sample retained JS heap over a post-run idle window for a leak slope (needs an
                          installed harness with run(); informational — never gates)
    --alloc               Expose window.__probeAlloc(windowMs) for a harness run() to measure allocation
    --leak <bytesPerSec>  Inject a retained allocation at this rate once the harness is ready — red-proof for
                          the --memory leak detector (pair with --memory; off by default)
    --connect <ws>        Drive a remote browser at this Playwright ws endpoint (chromium.connect) rather
                          than launching one — the endpoint owner supplies the browser's channel + flags
    --timings             Print per-phase wall-clock spans after the verdict (server boot, first page
                          load, harness ready, run, memory idle, capture, teardown)
    --run k=v[&k2=v2]     Batch mode (repeatable): one boot, one page load per --run (fresh browser
                          context + page each), one verdict each. Each spec is its own query string,
                          layered on the shared --query params. A JSON array on stdout; nonzero exit if
                          any run fails.
    --json                Machine-readable result on stdout
    -h, --help            Show this help

  --memory and --alloc are mutually exclusive (their heap samplers conflict).
  A dir with a shallot.json manifest or a .scene file boots the dev server; a dir with its own index.html
  (an ejected vite app) boots a plain vite server rooted at it.

  Requires playwright (optional): ${INSTALL_PLAYWRIGHT}
`;

// Node's stdout-to-a-pipe is async: `console.log`/`process.stdout.write` queue the write and return
// before the OS has it, so a `process.exit()` right after silently drops whatever's still queued past the
// pipe buffer (measured: cut at exactly 64 KiB under node; bun did not truncate, but node is what runs
// this CLI on the WSL bridge). Writes to the same handle are flushed in the order they were
// queued, so awaiting one more (empty) write's callback proves every prior write already landed — the one
// seam `runVerify` awaits before returning, covering every payload it can emit (batch array, `--timings`,
// shader artifacts, a lone large verdict) without each call site tracking its own flush. It runs in a
// `finally` around `verifyCommand`, so a throw that escapes it still delivers what was already printed.
/** @internal exported only so the pipe-truncation regression test can drive the exact production seam
 *  from a subprocess. */
export function flushStdout(): Promise<void> {
    return new Promise((resolve) => process.stdout.write("", () => resolve()));
}

// the one failure path for runs that never reach a verdict: a machine consumer always gets JSON on
// stdout under --json; a human gets the message on stderr.
function reportError(message: string, json: boolean): void {
    if (json) console.log(JSON.stringify({ pass: false, error: message }));
    else console.error(`\n  ✗ ${message}\n`);
}

// wire a page's console/error/crash events into a fresh errors array — the same capture the single-run
// path and every batch run install per page, factored so a batch run's fresh page gets its own array
// rather than one shared (and leaking) across runs.
function attachErrorCapture(page: Page): string[] {
    const errors: string[] = [];
    const record = (kind: string, text: string) => {
        if (kind === "err" || kind === "page-error" || ERR_HINT.test(text)) errors.push(text);
    };
    page.on("console", (msg: { type(): string; text(): string }) => {
        const type = msg.type();
        record(type === "error" ? "err" : type === "warning" ? "warn" : "log", msg.text());
    });
    page.on("pageerror", (err: Error) => record("page-error", `${err.name}: ${err.message}`));
    page.on("crash", () => record("page-error", "page crashed"));
    return errors;
}

/** run `shallot verify` from the CLI's remaining args. Returns the process exit code, only once every
 *  report it printed has actually reached stdout — see {@link flushStdout}. The flush runs in a
 *  `finally`, so a throw escaping `verifyCommand` still delivers whatever was already printed before
 *  propagating. */
export async function runVerify(raw: string[]): Promise<number> {
    try {
        return await verifyCommand(raw);
    } finally {
        await flushStdout();
    }
}

async function verifyCommand(raw: string[]): Promise<number> {
    let args: VerifyArgs;
    try {
        args = parseVerifyArgs(raw);
    } catch (err) {
        // args.json is unknown when the parse itself failed — honor a --json anywhere in raw.
        reportError(err instanceof Error ? err.message : String(err), raw.includes("--json"));
        return EXIT_SETUP;
    }
    if (args.help) {
        console.log(usage);
        return EXIT_PASS;
    }

    const projectDir = resolve(args.dir);
    const checkpoints: Checkpoint[] = [{ name: "start", t: Date.now() }];

    const pw = await importPlaywright(projectDir);
    if (!pw) {
        reportError(
            `playwright is not installed (shallot verify drives a real browser) — install it: ${INSTALL_PLAYWRIGHT}`,
            args.json,
        );
        return EXIT_NO_PLAYWRIGHT;
    }
    type Context = { newPage(): Promise<Page>; close(): Promise<void> };
    type Browser = {
        newContext(): Promise<Context>;
        close(): Promise<void>;
    };
    const chromium = pw.chromium as {
        launch(opts: unknown): Promise<Browser>;
        connect(endpoint: string, opts?: unknown): Promise<Browser>;
    };

    let booter: Booter;
    try {
        const port = await pickPort(args.port);
        booter = args.dist ? await serveDist(projectDir, port) : await serveDev(projectDir, port);
    } catch (err) {
        // every boot failure goes through the one failure path, not just SetupError — a project whose
        // own vite config throws, a busy --port under strictPort. Rethrowing would print a bare stack
        // and break the one-JSON-object-per-exit contract a machine consumer parses.
        const msg = err instanceof Error ? err.message : String(err);
        reportError(err instanceof SetupError ? msg : `boot failed: ${msg}`, args.json);
        return EXIT_SETUP;
    }
    checkpoints.push({ name: "server boot", t: Date.now() });
    const url = buildUrl(booter.url, args.query);

    let browser: Browser | undefined;
    let result: Result | undefined;
    const batchResults: Result[] = [];
    let batchTimings: Span[][] = [];
    const probes: TimingProbes = { harnessInstallMs: null, resourceTiming: null };
    let batchProbes: TimingProbes[] = [];
    try {
        try {
            // A remote endpoint (--connect) already carries the browser's channel + WebGPU flags: it was
            // launched with them server-side, so connect passes none — attaching to a differently-flagged
            // browser would silently drop the floor. A local launch sets them here.
            browser = args.connect
                ? await chromium.connect(args.connect, { timeout: 30_000 })
                : await chromium.launch({
                      headless: true,
                      // the published real-GPU recipe (`src/harness/browser.ts`) — its JSDoc carries the
                      // headless-shell/SwiftShader finding this channel exists to avoid.
                      ...REAL_GPU_LAUNCH,
                  });
        } catch (err) {
            const msg = err instanceof Error ? err.message : String(err);
            const how = args.connect
                ? `could not reach the browser at ${args.connect} (${msg.split("\n")[0]})`
                : `could not launch the chromium browser (${msg.split("\n")[0]}) — install it: ${INSTALL_PLAYWRIGHT}`;
            reportError(how, args.json);
            return EXIT_NO_PLAYWRIGHT;
        }

        // The display gate: probe the adapter identity on a throwaway page before running any check —
        // refuse a software adapter outright rather than let it clear the feature/limit floor and then
        // die mid-run. Refuse, not skip: this CLI is a user-invoked verifier, so a silent zero-exit here
        // would be the same class of lie in the other direction — a caller that wants skip-on-no-display
        // (this repo's own wrappers, `skipReason()`) owns that decision itself, upstream of this CLI.
        // One probe covers both the single-run and batch paths below (the adapter is a property of this
        // browser session, not of any one run). Navigates to the booted URL first: `requestAdapter`
        // returns no adapter at all on an opaque `about:blank` origin (measured), the same real page load
        // the actual run does — a nav failure here says nothing about the display, so it falls through to
        // the real run's own retry + setup-error path rather than misreporting as "no adapter".
        let gateContext: Context | undefined;
        try {
            gateContext = await browser.newContext();
            const gatePage = await gateContext.newPage();
            try {
                await gatePage.goto(booter.url, { timeout: 30_000 });
                const hardware = await readHardware(gatePage).catch(() => "unknown");
                const exit = displayGateExit(hardware);
                if (exit !== null) {
                    reportError(displayGateMessage(hardware), args.json);
                    return exit;
                }
            } catch {
                // navigation itself failed — the real run's own goto retry + setup-error path handles it.
            }
        } finally {
            await gateContext?.close().catch(() => {});
        }

        if (args.run.length === 0) {
            const context = await browser.newContext();
            const page = await context.newPage();
            const errors = attachErrorCapture(page);
            result = await drive(
                page,
                projectDir,
                url,
                booter.mode,
                args,
                errors,
                checkpoints,
                probes,
            );
        } else {
            const queries = resolveBatchQueries(args.query, args.run);
            const runSpans: Span[][] = [];
            const runProbesList: TimingProbes[] = [];
            for (const query of queries) {
                const runUrl = buildUrl(booter.url, query);
                const runCheckpoints: Checkpoint[] = [{ name: "start", t: Date.now() }];
                const runProbes: TimingProbes = { harnessInstallMs: null, resourceTiming: null };
                let runResult: Result;
                // a fresh context + page per run, never reused: the one-verdict law — module singletons
                // and GPU device state must not bleed between runs.
                // `newContext()` itself is inside the try: a page/browser crash (the `page.on("crash")`
                // case `attachErrorCapture` listens for) can take the whole browser process with it, and a
                // subsequent `newContext()` then throws — that must fail only this run, not lose every
                // already-collected verdict by escaping the loop (adversarial review, stage 2).
                let context: Context | undefined;
                try {
                    context = await browser.newContext();
                    const page = await context.newPage();
                    const errors = attachErrorCapture(page);
                    runResult = await drive(
                        page,
                        projectDir,
                        runUrl,
                        booter.mode,
                        args,
                        errors,
                        runCheckpoints,
                        runProbes,
                    );
                } catch (err) {
                    // a failure, timeout, or crash in this run must not poison the rest of the batch — it
                    // still owes a verdict, just a failing one, so the loop and the array both continue.
                    runResult = {
                        project: projectDir,
                        timestamp: new Date().toISOString(),
                        mode: booter.mode,
                        url: runUrl,
                        hardware: "unknown",
                        harness: false,
                        booted: false,
                        rendered: false,
                        errors: [err instanceof Error ? err.message : String(err)],
                        pass: false,
                    };
                } finally {
                    await context?.close().catch(() => {});
                }
                batchResults.push(runResult);
                runSpans.push(spansFromCheckpoints(runCheckpoints));
                runProbesList.push(runProbes);
            }
            checkpoints.push({ name: "batch", t: Date.now() });
            batchTimings = runSpans;
            batchProbes = runProbesList;
        }
    } finally {
        await browser?.close().catch(() => {});
        await booter.stop().catch(() => {});
        checkpoints.push({ name: "teardown", t: Date.now() });
    }

    if (args.run.length === 0) {
        // set on every path through the try block above that doesn't already return (parse/boot/launch
        // failures all return early) — never undefined here.
        const single = result as Result;
        report(single, args.json);
        if (args.timings) {
            console.log(formatTimings(spansFromCheckpoints(checkpoints)));
            console.log(formatExtraTimings(probes.harnessInstallMs, probes.resourceTiming));
        }
        // a run that never booted is a setup failure, not a verification failure — a distinct code.
        return single.pass ? EXIT_PASS : single.booted ? EXIT_FAIL : EXIT_SETUP;
    }

    reportBatch(batchResults, args.run, args.json);
    if (args.timings) {
        console.log(formatTimings(spansFromCheckpoints(checkpoints)));
        batchTimings.forEach((spans, i) => {
            console.log(`\n  run ${i + 1}/${batchResults.length} (${args.run[i]}):`);
            console.log(formatTimings(spans));
            const p = batchProbes[i];
            if (p) console.log(formatExtraTimings(p.harnessInstallMs, p.resourceTiming));
        });
    }
    return batchPass(batchResults) ? EXIT_PASS : EXIT_FAIL;
}

// the browser flow — ONE navigation, one unified wait. After goto, each poll checks for a
// `window.__harness` install and (absent one) folds a compositor frame sample into the settle
// tracking, on the same live page. A harness appearing at any point — however late — switches to the
// harness path; only a wait that concludes (settled or timed out) with no harness makes the settle
// result the verdict. A slow harness install can therefore never downgrade to the smoke check.
async function drive(
    page: Page,
    projectDir: string,
    url: string,
    mode: "dev" | "dist",
    args: VerifyArgs,
    errors: string[],
    checkpoints: Checkpoint[],
    probes: TimingProbes,
): Promise<Result> {
    const base: Omit<Result, "pass"> = {
        project: projectDir,
        timestamp: new Date().toISOString(),
        mode,
        url,
        hardware: "unknown",
        harness: false,
        booted: false,
        rendered: false,
        errors,
    };

    // Presence is the opt-in for both GPU log and shader artifact capture, before any app shader resolves.
    await page
        .addInitScript(() => {
            (window as unknown as { __gpuLog: unknown }).__gpuLog = { lines: [], errors: [] };
            (window as unknown as { __gpuDiagnostics: unknown }).__gpuDiagnostics = {
                artifacts: [],
            };
        })
        .catch(() => {});

    // Only installed when asked, so a normal run's __harness assignment isn't intercepted for no reason.
    if (args.timings) {
        await page.addInitScript({ content: TIMINGS_INIT_SCRIPT }).catch(() => {});
    }

    // retry the goto once only if the first attempt itself failed (a cold vite server can re-optimize
    // deps mid-load and strand it — the flows launcher's proven shape). Never re-navigate a page that
    // loaded: a second goto re-runs the app and diverges its first-load state. navStart is stamped
    // right before whichever attempt actually lands (adversarial review, stage 1): a stale mark from
    // the failed first attempt would inflate the harness-install probe by its own 30s timeout + the 1s
    // backoff, exactly the kind of retry a workspace-symlinked package's unbundled module population
    // (each a separate transform) makes more likely — the instrument lying about the very thing it
    // exists to attribute.
    let navStart = Date.now();
    try {
        await page.goto(url, { timeout: 30_000 });
    } catch {
        await page.waitForTimeout(1000);
        navStart = Date.now();
        try {
            await page.goto(url, { timeout: 30_000 });
        } catch {
            return { ...base, pass: false };
        }
    }
    checkpoints.push({ name: "first page load", t: Date.now() });

    // --alloc exposes a probe a harness run() can call, so it must exist before the wait loop.
    // Best-effort — a CDP failure never fails the run. (--memory starts later, inside the harness
    // path: sampling from page load would fit the boot phase's module/init growth and read a static
    // scene as a leak — the slope is meaningful only across run()'s steady state.)
    if (args.alloc) {
        await installAllocProbe(page).catch((err) => {
            if (!args.json) {
                console.error(
                    `  ! alloc probe unavailable: ${err instanceof Error ? err.message : String(err)}`,
                );
            }
        });
    }

    const deadline = Date.now() + args.timeoutMs;
    const waitStart = Date.now();
    const st: WaitState = { booted: false, prev: null };
    let outcome: "harness" | "settled" | "timeout" = "timeout";
    let sampleCount = 0;
    let lastSample: FrameSample | null = null;
    for (;;) {
        const harnessDefined = (await page
            .evaluate(() => typeof window.__harness !== "undefined")
            .catch(() => false)) as boolean;
        const sample = harnessDefined ? null : await sampleFrame(page);
        if (sample) {
            sampleCount++;
            lastSample = sample;
        }
        const step = stepWait(st, harnessDefined, sample);
        if (step !== "continue") {
            outcome = step;
            break;
        }
        if (Date.now() >= deadline) break;
        await page.waitForTimeout(500);
    }

    // the render probe — the pixel evidence behind the `rendered` verdict. Carries the wait loop's
    // samples, the last centre/corner RGB and spread against `structured`'s threshold, the elapsed
    // wait, and how the wait concluded. The harness path updates it with the ready/post samples; the
    // settle path carries it as-is. A blank-render red (no sample ever showed structure) is the
    // reading this probe exists to make legible.
    const renderProbe: RenderProbe = {
        samples: sampleCount,
        center: lastSample?.center ?? null,
        corner: lastSample?.corner ?? null,
        spread: lastSample ? spread(lastSample.center, lastSample.corner) : null,
        threshold: STRUCTURE_THRESHOLD,
        elapsed: Date.now() - waitStart,
        outcome,
    };

    // read the adapter identity once the wait has concluded; a broken page returns "unknown", never a crash.
    base.hardware = await readHardware(page).catch(() => "unknown");

    // (b)+(c): read the harness-install probe and the resource-timing readout once the wait has
    // concluded — the same point the adapter identity is read, after the page has had its whole boot
    // window to install `__harness` and fetch every module it needs.
    if (args.timings) {
        const installedAt = (await page
            .evaluate(
                () => (window as unknown as { __harnessInstallAt?: number }).__harnessInstallAt,
            )
            .catch(() => undefined)) as number | undefined;
        probes.harnessInstallMs = harnessInstallMs(navStart, installedAt ?? null);
        const entries = (await page
            .evaluate(() =>
                performance
                    .getEntriesByType("resource")
                    .map((e) => ({ name: e.name, duration: e.duration })),
            )
            .catch(() => [])) as ResourceEntry[];
        probes.resourceTiming = summarizeResourceTiming(entries);
    }

    if (outcome === "harness")
        return withGpuLog(
            page,
            await driveHarness(page, base, args, errors, checkpoints, renderProbe),
        );

    // no harness to wait for or run — both phases are instant, but the checkpoints stay uniform across
    // the harness and settle-only paths so a --timings reader compares the same phase list either way.
    checkpoints.push({ name: "harness ready", t: Date.now() });
    checkpoints.push({ name: "run", t: Date.now() });
    checkpoints.push({ name: "memory idle", t: Date.now() });
    await maybeScreenshot(page, args.screenshot);
    checkpoints.push({ name: "capture", t: Date.now() });
    // settled = the verdict; timed out structured-but-never-stable = an animated scene, rendered
    // (the caller's own gates judge motion); never structured = a blank canvas.
    const rendered = outcome === "settled" || st.prev != null;
    return withGpuLog(page, {
        ...base,
        booted: st.booted,
        rendered,
        // --memory needs a harness run() to measure across; a settle-only page has no steady state.
        ...(args.memory ? { memory: null } : {}),
        pass: settlePass(st.booted, rendered, errors.length),
        renderProbe,
    });
}

/** merge the page's captured GPU log into a finished result. A GPU `console.error` fails the run
 *  outright — a shader-side assert is a real verdict, not a note.
 *  @internal exported only so the merge arithmetic's branch surface can be driven by a page stub. */
export async function withGpuLog(page: Page, result: Result): Promise<Result> {
    const log = (await page
        .evaluate(() => (window as unknown as { __gpuLog?: GpuLog }).__gpuLog)
        .catch(() => null)) as GpuLog | null;
    const checks = gpuLogChecks(log);
    const failed = checks.some((c) => !c.ok);
    const completed =
        checks.length === 0
            ? result
            : {
                  ...result,
                  verdict: {
                      ...(result.verdict ?? { ok: result.pass }),
                      ok: (result.verdict?.ok ?? result.pass) && !failed,
                      checks: [...(result.verdict?.checks ?? []), ...checks],
                  },
                  pass: result.pass && !failed,
              };
    const diagnostics = (await page
        .evaluate(
            () => (window as unknown as { __gpuDiagnostics?: GpuDiagnostics }).__gpuDiagnostics,
        )
        .catch(() => null)) as GpuDiagnostics | null;
    const artifacts = failureArtifacts(completed.pass, diagnostics);
    return artifacts ? { ...completed, artifacts } : completed;
}

// the harness path: await ready with the full --timeout budget, probe + call run(), interpret the
// Verdict. Every page evaluate is guarded — a throwing run() (or a probe on a broken page) is a clean
// FAIL with the error as detail (the page.ts fatal-envelope precedent), never an unhandled rejection.
// A harness that never readies is a hard FAIL — never a downgrade to the settle check.
/** @internal exported only so the red arms (ready-timeout, noRender, probe-error, hasRun-fallback) can
 *  be driven directly by a duck-typed page stub. */
export async function driveHarness(
    page: Page,
    base: Omit<Result, "pass">,
    args: VerifyArgs,
    errors: string[],
    checkpoints: Checkpoint[],
    renderProbe?: RenderProbe,
): Promise<Result> {
    base.harness = true;
    if (renderProbe) base.renderProbe = renderProbe;
    try {
        await page.waitForFunction(() => window.__harness?.ready === true, null, {
            timeout: args.timeoutMs,
        });
    } catch {
        const verdict: Verdict = {
            ok: false,
            checks: [
                {
                    name: "ready",
                    ok: false,
                    detail: `window.__harness.ready never became true within ${args.timeoutMs}ms`,
                },
            ],
        };
        checkpoints.push({ name: "harness ready", t: Date.now() });
        checkpoints.push({ name: "run", t: Date.now() }, { name: "memory idle", t: Date.now() });
        await maybeScreenshot(page, args.screenshot);
        checkpoints.push({ name: "capture", t: Date.now() });
        return { ...base, booted: true, rendered: false, verdict, pass: false };
    }

    // a harness that renders no framed scene by design (a GPU-compute microbench, a solid-fill clip test)
    // declares `noRender`: the pixel gate is skipped and `rendered` reports "opt-out" — honest, visible, and
    // never a fake true. Every other harness is held to the pixel check below.
    const noRender = (await page
        .evaluate(() => window.__harness?.noRender === true)
        .catch(() => false)) as boolean;

    // pixel-honest `rendered`: capture the canvas now that the harness is ready, before run() drives it.
    // The post-run capture (below) is the primary signal; this early one covers a scenario that renders
    // during build then tears the scene down inside run(). A scenario that draws nothing fails on this,
    // not only on its own asserts. Reuses the settle path's compositor screenshot + structure check.
    const readySample = noRender ? null : await sampleFrame(page);
    if (readySample && renderProbe) {
        renderProbe.samples++;
        renderProbe.center = readySample.center;
        renderProbe.corner = readySample.corner;
        renderProbe.spread = spread(readySample.center, readySample.corner);
    }

    // --leak red-proof: start a known retained allocation now (harness is ready), so it runs through run()
    // and on into the post-run idle window the --memory slope is fitted over. Off (0) in every normal run.
    if (args.leak > 0) await injectLeak(page, args.leak);

    // "harness ready" closes here: the harness signaled ready, and every readiness-adjacent probe
    // above (noRender, the pre-run sample, --leak) has run — the next span is run() itself.
    checkpoints.push({ name: "harness ready", t: Date.now() });

    let verdict: Verdict;
    let hasRun = false;
    let probeError: string | null = null;
    try {
        hasRun = (await page.evaluate(
            () => typeof window.__harness?.run === "function",
        )) as boolean;
    } catch (err) {
        probeError = err instanceof Error ? err.message : String(err);
    }

    if (probeError) {
        verdict = {
            ok: false,
            checks: [
                { name: "run", ok: false, detail: `could not reach __harness.run: ${probeError}` },
            ],
        };
    } else if (hasRun) {
        const opts: Record<string, string> = {};
        for (const q of args.query) {
            const eq = q.indexOf("=");
            if (eq !== -1) opts[q.slice(0, eq)] = q.slice(eq + 1);
        }
        try {
            const value = await withTimeout(
                page.evaluate(
                    (o: Record<string, unknown>) => window.__harness!.run!(o),
                    opts,
                ) as Promise<unknown>,
                args.timeoutMs,
                "run()",
            );
            verdict = coerceVerdict(value);
        } catch (err) {
            const msg = err instanceof Error ? err.message : String(err);
            verdict = {
                ok: false,
                checks: [{ name: "run", ok: false, detail: `run() threw: ${msg}` }],
            };
        }
    } else {
        verdict = { ok: errors.length === 0, checks: [{ name: "ready", ok: true }] };
    }
    checkpoints.push({ name: "run", t: Date.now() });

    // the --memory leak slope reads a POST-run IDLE window, not run() itself. During run() the benchmark
    // retains monotonically-growing per-frame stats arrays (frameTimes + per-pass accumulators,
    // extras/profile/benchmark.ts), so a post-GC sample over run() measures the harness's own accumulation
    // and reads a false leak (~110 KB/s on a multi-second run, and hardware-speed-dependent — a fast GPU's
    // run is too short to even sample). Once run() resolves those arrays are unreachable and collect on the
    // first forced GC, so a fixed idle window here isolates the engine's steady-state retention — the honest
    // signal, deterministic across hardware. See fitMemory + testing.md.
    let memory: MemoryStats | null | undefined;
    if (args.memory && hasRun) {
        const sampler = await startMemory(page);
        await new Promise((r) => setTimeout(r, LEAK_IDLE_MS));
        memory = await sampler.stop();
    } else if (args.memory) {
        memory = null;
    }
    checkpoints.push({ name: "memory idle", t: Date.now() });

    // the post-run capture is the authoritative `rendered` signal — run() has driven the scene to its
    // final frame, and the RAF loop keeps drawing, so nextFrame lands a fresh composite before the shot.
    let rendered: boolean | "opt-out";
    let pixelChecks: Check[] = [];
    if (noRender) {
        rendered = "opt-out";
    } else {
        await nextFrame(page);
        const postSample = await sampleFrame(page);
        if (postSample && renderProbe) {
            renderProbe.samples++;
            renderProbe.center = postSample.center;
            renderProbe.corner = postSample.corner;
            renderProbe.spread = spread(postSample.center, postSample.corner);
        }
        rendered = hasStructure(readySample) || hasStructure(postSample);
        const probes = (await page
            .evaluate(() => window.__harness?.pixelProbe)
            .catch(() => undefined)) as PixelProbe[] | undefined;
        if (probes?.length) pixelChecks = await samplePixelProbes(page, probes);
    }
    if (pixelChecks.length) verdict.checks = [...(verdict.checks ?? []), ...pixelChecks];

    await maybeScreenshot(page, args.screenshot);
    checkpoints.push({ name: "capture", t: Date.now() });
    return {
        ...base,
        booted: true,
        rendered,
        verdict,
        ...(memory !== undefined ? { memory } : {}),
        pass: harnessPass(verdict, rendered, errors.length) && pixelChecks.every((c) => c.ok),
    };
}

// the scene's RAF loop keeps drawing after the verdict resolves, so awaiting two frames lands a fresh one
// before a capture (the harness/core/page.ts ordering). Best-effort — resolves without throwing on a
// broken page so the capture still proceeds.
async function nextFrame(page: Page): Promise<void> {
    await page
        .evaluate(
            () =>
                new Promise<void>((r) =>
                    requestAnimationFrame(() => requestAnimationFrame(() => r())),
                ),
        )
        .catch(() => {});
}

// write a post-run PNG when --screenshot asked for one. Best-effort — a screenshot failure never fails.
// Deliberately full-page rather than the canvas element: `scripts/flows.ts`'s ui-containment oracle samples
// host chrome in the 64px body padding *outside* the canvas, so a canvas-clipped capture reds that gate and
// makes the property unobservable from its own artifact (measured 2026-08-24, caught in review). A canvas
// capture is the better render diagnostic — this one is byte-identical across a green and a red boot — but
// it is an added mode behind its own flag, gated by `bun run flows`, never a change to this default.
async function maybeScreenshot(page: Page, path: string | undefined): Promise<void> {
    if (!path) return;
    try {
        await nextFrame(page);
        await page.screenshot({ path: resolve(path) });
    } catch {
        // a screenshot is a convenience, never a gate
    }
}

/** render N batch results: one JSON array under `--json`, one labeled {@link report} block per run
 *  otherwise — `labels` are the `--run` specs in the same order as `results`. @internal */
export function reportBatch(results: Result[], labels: string[], json: boolean): void {
    if (json) {
        console.log(JSON.stringify(results));
        return;
    }
    results.forEach((result, i) => {
        console.log(`\n[run ${i + 1}/${results.length}] ${labels[i] ?? ""}`.trimEnd());
        report(result, false);
    });
}

/** render one completed verify result for the human or JSON CLI boundary. @internal */
export function report(result: Result, json: boolean): void {
    if (json) {
        console.log(JSON.stringify(result));
        return;
    }
    const mark = (ok: boolean) => (ok ? "✓" : "✗");
    console.log(`\nverify: ${basename(result.project)}  (${result.mode}, ${result.url})`);
    console.log(`  ${mark(result.booted)} booted`);
    if (result.rendered === "opt-out") {
        console.log(`  ○ rendered — generic pixel gate opted out`);
    } else {
        console.log(`  ${mark(result.rendered)} rendered`);
    }
    if (result.harness && result.verdict) {
        for (const c of result.verdict.checks ?? []) {
            console.log(`  ${mark(c.ok)} ${c.name}${c.detail ? ` — ${c.detail}` : ""}`);
        }
    }
    console.log(`  hardware: ${result.hardware}`);
    if (result.memory) {
        const m = result.memory;
        console.log(
            `  memory: ${m.growthPerSecond.toFixed(0)} B/s${m.leak ? " ⚠ LEAK" : ""}` +
                ` (${(m.start / 1e6).toFixed(1)}→${(m.end / 1e6).toFixed(1)} MB, gc ${m.gcCount}× / ${m.gcPauseMs.toFixed(1)}ms)`,
        );
    }
    if (result.errors.length) {
        console.log(`  errors:`);
        for (const e of result.errors.slice(0, 5)) console.log(`    ${e.split("\n")[0]}`);
    }
    if (result.artifacts) {
        console.log(`  shader artifacts:`);
        for (const artifact of result.artifacts) {
            console.log(`    ${artifact.label} [${artifact.stage}] ${artifact.hash}`);
            if (artifact.compilationError) {
                console.log(
                    `      compilation ${artifact.compilationError.errorClass}: ${artifact.compilationError.message}`,
                );
            }
            for (const message of artifact.messages) {
                console.log(
                    `      ${message.type} ${message.lineNum}:${message.linePos} ${message.message}`,
                );
            }
            console.log(artifact.source);
        }
    }
    console.log(`  => ${result.pass ? "PASS" : "FAIL"}\n`);
}
