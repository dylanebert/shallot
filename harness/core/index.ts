import { copyFileSync, existsSync } from "node:fs";
import { join } from "node:path";
import { runPlaywright } from "./playwright";
import { exampleDir, startServer } from "./server";
import { detectDisplay } from "./wsl";

// page.ts writes the screenshot under this fixed name in its run dir; copied back to the caller's path.
const SHOT_NAME = "shot.png";

// Core layer: run a vite-served page under Playwright and return its ResultEnvelope. Knows
// nothing about gym scenarios — it boots the project's vite server, drives `page.ts` through the
// `window.__harness` contract, and decodes the result. The target is a shallot example (`example`)
// or any external project that installs the same contract (`cwd`, e.g. the orrstead game bench).
// The result is opaque here (`verdict: unknown`); a launcher (gym, or a consumer's bench script)
// interprets it.

/** how the core drives a page: the example installs this on `window.__harness`. */
export interface HarnessTarget {
    /** true once the scene is built and the first frame has drawn. */
    readonly ready: boolean;
    /** run the scenario for the given budget and return its verdict (opaque to the core). */
    run(opts: RunOpts): Promise<unknown>;
}

export interface RunOpts {
    warmup: number;
    frames: number;
}

export interface MemoryStats {
    start: number;
    end: number;
    growthPerFrame: number;
    leak: boolean;
    gcCount: number;
    gcPauseMs: number;
}

/** what `page.ts` emits and `runExample` decodes; a launcher narrows `verdict`. */
export interface ResultEnvelope {
    ok: boolean;
    hardware: string;
    verdict?: unknown;
    memory?: MemoryStats | null;
    errors?: string[];
    fatal?: string;
}

export interface RunSpec {
    // What to serve — exactly one of:
    //   example: a shallot example under examples/<example> (e.g. "gym" or "lab")
    //   cwd:     an external project dir with a `bun run dev` script (e.g. the orrstead game)
    example?: string;
    cwd?: string;
    // log/error label; defaults to `example`, else the cwd basename
    label?: string;
    port: number;
    // full page URL incl. scenario-selection params
    url: string;
    warmup: number;
    frames: number;
    timeoutMs?: number;
    // how long page.ts waits for window.__harness.ready (build + settle). Defaults to 30s; raise it for a
    // scenario whose build outlasts that (a large physics pile). The overall test budget is timeoutMs.
    readyTimeoutMs?: number;
    sampleMemory?: boolean;
    // expose `window.__probeAlloc(windowMs)` — a CDP collectGarbage-bracketed, no-forced-GC heap probe
    // a scenario's assert calls to measure allocation/frame + GC count + the top allocators over a
    // window (the stress CPU-memory axis). Mutually exclusive with sampleMemory: its parallel forced
    // GCs would corrupt the probe's no-GC window, so a run requests one or the other, not both.
    sampleAlloc?: boolean;
    // absolute path to copy a post-run canvas screenshot to (page.ts writes `shot.png` in its run
    // dir — staged.win under WSL — and this copies it back here). Undefined = no screenshot.
    screenshot?: string;
    // extra args forwarded to Playwright (e.g. --headed, --debug)
    passthrough?: string[];
}

declare global {
    interface Window {
        __harness?: HarnessTarget;
    }
}

const RESULT_RE = /__HARNESS_RESULT__([\s\S]+?)__HARNESS_RESULT__/;

const harnessDir = import.meta.dir;

function extractResult(stdout: string): ResultEnvelope | null {
    const match = stdout.match(RESULT_RE);
    return match ? (JSON.parse(match[1]) as ResultEnvelope) : null;
}

function defaultTimeout(spec: RunSpec): number {
    if (spec.timeoutMs != null) return spec.timeoutMs;
    return 30_000 + (spec.warmup + spec.frames) * 50 + 30_000;
}

// Boot the example dev server, run page.ts under Playwright, return the decoded
// envelope. Throws if Playwright exits non-zero or emits no result.
export async function runExample(spec: RunSpec): Promise<ResultEnvelope> {
    if (!detectDisplay()) throw new Error("no display available");
    if ((spec.example == null) === (spec.cwd == null)) {
        throw new Error("runExample needs exactly one of `example` or `cwd`");
    }

    const pageSpec = JSON.stringify({
        url: spec.url,
        warmup: spec.warmup,
        frames: spec.frames,
        timeoutMs: defaultTimeout(spec),
        readyTimeoutMs: spec.readyTimeoutMs ?? 30_000,
        sampleMemory: spec.sampleMemory ?? false,
        sampleAlloc: spec.sampleAlloc ?? false,
        screenshot: spec.screenshot ? SHOT_NAME : undefined,
    });

    const cwd = spec.cwd ?? exampleDir(spec.example!);
    const server = await startServer(cwd, spec.port, spec.label ?? spec.example);
    try {
        const run = runPlaywright({
            dir: harnessDir,
            config: "playwright.config.ts",
            args: ["page.ts", ...(spec.passthrough ?? [])],
            stage: { name: "shallot-harness", files: ["package.json", "playwright.config.ts", "page.ts"] },
            env: () => ({ HARNESS_SPEC: pageSpec }),
            // the page test bounds itself (page.ts: test.setTimeout); this ceiling sits above it +
            // browser launch so a wedged Playwright process is killed, not a slow-but-live bench.
            timeoutMs: defaultTimeout(spec) + 120_000,
        });
        const result = extractResult(run.stdout);
        // copy the screenshot back before the exit-code check — a failed render is the case worth
        // seeing. page.ts wrote it in the run dir (staged.win under WSL, else harnessDir).
        if (spec.screenshot) {
            const src = run.staged ? join(run.staged.wsl, SHOT_NAME) : join(harnessDir, SHOT_NAME);
            if (existsSync(src)) copyFileSync(src, spec.screenshot);
            else console.warn(`[harness] no screenshot produced at ${src}`);
        }
        if (run.exitCode !== 0) {
            throw new Error(
                `Playwright exited ${run.exitCode}${run.timedOut ? " (spawn ceiling)" : ""}${result?.fatal ? `: ${result.fatal}` : ""}`,
            );
        }
        if (!result) throw new Error("no __HARNESS_RESULT__ payload on stdout");
        return result;
    } finally {
        server.kill();
    }
}
