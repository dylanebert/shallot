import { copyFileSync, mkdtempSync, readdirSync, readFileSync, rmSync, symlinkSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join, resolve } from "node:path";
import { removeTestDirectory } from "../../scripts/test-directory";
import { DEFERRED_TURBOFAN_FLAGS, SAMPLE_RESULT } from "./allocation-sampler.mjs";

export interface AllocationFrame {
    functionName: string;
    site: string;
    native: boolean;
    boundary?: string;
}

export interface AllocationSite {
    /** top JavaScript frame's declaration position, admitted only for a checked API wrapper */
    site: string;
    /** sampled bytes attributed to this complete stack over the measured window, collected objects included */
    bytes: number;
    /** samples attributed to this stack over the window */
    count: number;
    /** ordered from the run frame's caller through the allocation and any trailing native frames */
    stack?: readonly AllocationFrame[];
    /** checked API wrapper declaration position when the top JavaScript frame is that boundary */
    boundary?: string;
}

export interface AllocationWindow {
    label: string;
    /** functions optimized during this window; the strict allocation qualification refuses it */
    optimizations?: readonly string[];
    /** attributed sites, most bytes first; builtin bytes credit their caller */
    sites: readonly AllocationSite[];
    /** the fewest frames this window can have stepped */
    frames: number;
    /** the most frames this window can have stepped; equal to {@link frames} for the Node sampler */
    framesAtMost: number;
}

export interface AllocationSample {
    runtime: string;
    warm: number;
    /** frames each window was asked to step; page samples bracket actual frames with `framesAtMost` */
    frames: number;
    /** windows read after `warm` frames, after twice that, and an A/A repeat, each after a collection */
    windows: readonly AllocationWindow[];
    /** sites of the bundle's `control` literal, attributed as the windows are; empty means the sampler saw nothing */
    control: readonly AllocationSite[];
}

export interface TransitionSample extends AllocationSample {
    /** every byte of the one frame that spawns, collected objects included */
    spawn: readonly AllocationSite[];
    /** every byte of the one frame that despawns, collected objects included */
    despawn: readonly AllocationSite[];
    /** the spawn frame of a second cycle at the same high-water mark */
    spawnAgain: readonly AllocationSite[];
    /** the despawn frame of that second cycle */
    despawnAgain: readonly AllocationSite[];
    /** every byte of the chunk of steady frames right after each of those four event frames, in order */
    afterEvents: readonly AllocationWindow[];
    /** objects allocated from spawn through despawn still live after a full collection */
    survivors: readonly AllocationSite[];
}

const SAMPLER = resolve(import.meta.dir, "allocation-sampler.mjs");
const TEXT_FONT = resolve(import.meta.dir, "../../src/extras/text/Inter-Regular.ttf");

/**
 * Lower thresholds only move compiles into warm-up; synchronous compilation keeps installation in its run.
 * Disable allocation-site pretenuring: a collection can otherwise change an allocation site's generation
 * and recompile its callers inside any window. Objects still allocate on the JavaScript heap (in young
 * space instead of directly in old space), and sampling includes objects collected by either GC.
 * This changes placement, not allocation elimination; no subject allocation becomes admitted or hidden.
 * V8's wait after a type-feedback change stays at its default, since shortening it changes what V8 inlines.
 * These flags do not guarantee a final tier, so Node refuses profiled runs that trace an optimization.
 * Keep production's lazy feedback: eager feedback changes steady allocation, not just compile timing.
 * No WebAssembly inlining: the feedback it collects allocates on the JavaScript heap outside the optimization
 * trace, and the sampler charges those bytes to the subject frame that called into the module.
 * Wrappers tier up on their first call: a JS-to-Wasm or Wasm-to-JS wrapper tiers up after a call budget,
 * allocating on the JavaScript heap outside the optimization trace, so a rarely called import, such as the
 * solver's spin pause, would otherwise tier up inside a window.
 * Chromium receives these same timing flags, without the Node trace refusal. The Node sampler also
 * starts with DEFERRED_TURBOFAN_FLAGS and enables Turbofan once the subject is created.
 */
export const TIER_FLAGS = [
    "--no-concurrent-recompilation",
    "--no-allocation-site-pretenuring",
    "--invocation-count-for-maglev=10",
    "--invocation-count-for-turbofan=10",
    "--no-wasm-inlining",
    "--wasm-wrapper-tiering-budget=1",
];

export const windowBytes = (window: { sites: readonly AllocationSite[] }) =>
    window.sites.reduce((total, row) => total + row.bytes, 0);

const expectedSteadyWindows = (warm: number) => [
    `after warm ${warm}`,
    `after warm ${2 * warm}`,
    "A/A repeat",
];

/** True only when every expected window is present, has no optimization, and reads zero bytes at zero sites. */
export const allocatesNothing = (sample: Pick<AllocationSample, "warm" | "windows">) => {
    const expected = expectedSteadyWindows(sample.warm);
    return (
        expected.every((label) => sample.windows.some((window) => window.label === label)) &&
        sample.windows.every((window) =>
            !window.optimizations?.length && window.sites.length === 0 && windowBytes(window) === 0
        )
    );
};

/** A binary steady-allocation failure, with sampled sites printed only to diagnose the red. */
export function allocationFailure(
    sample: Pick<AllocationSample, "warm" | "windows">,
): string | undefined {
    if (allocatesNothing(sample)) return undefined;
    const missing = expectedSteadyWindows(sample.warm).filter(
        (label) => !sample.windows.some((window) => window.label === label),
    );
    const failures: string[] = [];
    if (missing.length > 0)
        failures.push(
            `steady allocation sample is missing expected windows: ${missing.join(", ")}`,
        );
    for (const window of sample.windows) {
        if (window.optimizations?.length)
            failures.push(`steady allocation window refused: ${window.label}; optimized ${window.optimizations.join(", ")}; no byte reading`);
    }
    const allocating = sample.windows.filter(
        (window) => !window.optimizations?.length && (window.sites.length > 0 || windowBytes(window) !== 0),
    );
    if (allocating.length > 0) {
        const sites = allocating.flatMap((window) =>
            window.sites.length > 0
                ? window.sites.map((row) => {
                      const stack = row.stack?.map((frame) => `${frame.functionName} ${frame.site}`).join(" <- ");
                      return `  ${window.label}: ${row.bytes} B at ${row.site}${stack ? `\n    stack: ${stack}` : ""}`;
                  })
                : [`  ${window.label}: ${windowBytes(window)} B with no attributed sites`],
        );
        failures.push(
            `steady play allocated JavaScript heap; sampler sites are diagnosis only:\n${sites.join("\n")}`,
        );
    }
    return failures.join("\n");
}

/**
 * Bundle `entry` for Node, build its default export with `input`, step it `warm` frames, collect,
 * then sample `frames` more under V8's sampling heap profiler. Only the harness's chunk loops are
 * kept unoptimized; the subject tiers normally. Needs the `node` requirement resolved.
 * A subject that submits GPU work returns `allocationExpressions: gpuAllocationExpressions` and
 * `allocationFunctions: [world.gpu.sync]`: an undeclared wrapper is inlined, and WebGPU's pass,
 * encoder and fence objects are then charged to its callers instead of admitted.
 */
export function sampleAllocation(
    entry: string,
    options: {
        warm?: number;
        frames?: number;
        input?: string;
        /** Keep sampled sites for diagnosis even when a tier change makes strict qualification refuse a window. */
        retainOptimizedSites?: boolean;
    } = {},
): Promise<AllocationSample> {
    return runSampler(entry, options, [], options.retainOptimizedSites ?? false);
}

/**
 * As {@link sampleAllocation}, for an entry whose subject also has `spawn()` and `despawn()`: warms paired
 * cycles, samples each event frame and the chunk after it, the steady windows after them, and one cycle's live survivors.
 */
export function sampleTransition(
    entry: string,
    options: { warm?: number; frames?: number; input?: string } = {},
): Promise<TransitionSample> {
    return runSampler(entry, options, ["transition"], false) as Promise<TransitionSample>;
}

export interface AllocationSamplerOptions {
    warm?: number;
    frames?: number;
    input?: string;
    /** Keep sampled sites for diagnosis even when a tier change makes strict qualification refuse a window. */
    retainOptimizedSites?: boolean;
}

export interface PreparedAllocationSampler {
    sample(options?: AllocationSamplerOptions): Promise<AllocationSample>;
    dispose(): void;
}

/** Build an entry once, then sample it in fresh Node isolates without rebuilding the bundle. */
export async function prepareAllocationSampler(entry: string): Promise<PreparedAllocationSampler> {
    const bundle = await buildAllocationBundle(entry);
    let disposed = false;
    return {
        sample(options = {}) {
            if (disposed) throw new Error("allocation sampler was disposed");
            return runBuiltSampler(
                bundle,
                options,
                [],
                options.retainOptimizedSites ?? false,
            );
        },
        dispose() {
            if (disposed) return;
            disposed = true;
            removeTestDirectory(bundle.dir);
        },
    };
}

async function runSampler(
    entry: string,
    options: { warm?: number; frames?: number; input?: string },
    mode: string[],
    retainOptimizedSites: boolean,
): Promise<AllocationSample> {
    const bundle = await buildAllocationBundle(entry);
    try {
        return await runBuiltSampler(bundle, options, mode, retainOptimizedSites);
    } finally {
        removeTestDirectory(bundle.dir);
    }
}

async function buildAllocationBundle(entry: string): Promise<{ dir: string; path: string }> {
    const dir = mkdtempSync(join(tmpdir(), "shallot-allocation-"));
    try {
        const { shallot } = await import("@dylanebert/shallot/bun");
        const built = await Bun.build({
            entrypoints: [entry],
            outdir: dir,
            target: "node",
            format: "esm",
            sourcemap: "linked",
            naming: "subject.mjs",
            plugins: [shallot({ root: dirname(entry) })],
        });
        if (!built.success)
            throw new Error(`allocation bundle failed: ${built.logs.map(String).join("\n")}`);
        // Bun.build does not emit `new URL(asset, import.meta.url)` assets; Text's default font resolves beside the bundle.
        copyFileSync(TEXT_FONT, join(dir, "Inter-Regular.ttf"));
        return { dir, path: join(dir, "subject.mjs") };
    } catch (error) {
        rmSync(dir, { recursive: true, force: true });
        throw error;
    }
}

async function runBuiltSampler(
    bundle: { dir: string; path: string },
    { warm = 600, frames = 600, input = "" }: { warm?: number; frames?: number; input?: string },
    mode: string[],
    retainOptimizedSites: boolean,
): Promise<AllocationSample> {
    const dir = mkdtempSync(join(tmpdir(), "shallot-allocation-run-"));
    try {
        const subject = join(dir, "subject.mjs");
        symlinkSync(bundle.path, subject);
        const inputPath = join(dir, "input.txt");
        writeFileSync(inputPath, input);
        const proc = Bun.spawn(
            [
                "node",
                "--expose-gc",
                "--enable-source-maps",
                ...TIER_FLAGS,
                ...DEFERRED_TURBOFAN_FLAGS,
                "--trace-opt",
                // One stdout carries every isolate's traces; a worker's optimization would refuse
                // a main-thread window. Per-isolate files keep only the measured isolate's.
                "--redirect-code-traces",
                "--allow-natives-syntax",
                SAMPLER,
                subject,
                String(warm),
                String(frames),
                inputPath,
                ...mode,
            ],
            {
                cwd: dir,
                stdout: "pipe",
                stderr: "pipe",
                env: { ...process.env, SHALLOT_PROJECT_ROOT: process.cwd() },
            },
        );
        const [, stderr, code] = await Promise.all([
            new Response(proc.stdout).text(),
            new Response(proc.stderr).text(),
            proc.exited,
        ]);
        if (code !== 0) throw new Error(`allocation sampler exited ${code}: ${stderr.trim()}`);
        return tracedSample(`${measuredIsolateTrace(dir)}\n${readFileSync(join(dir, SAMPLE_RESULT), "utf8")}`, {
            retainOptimizedSites,
        });
    } finally {
        removeTestDirectory(dir);
    }
}

/** The one isolate trace in `dir` that carries the sampler's window markers; other isolates' are ignored. */
export function measuredIsolateTrace(dir: string): string {
    const marked = readdirSync(dir)
        .filter((file) => /^code-\d+-\d+\.asm$/.test(file))
        .map((file) => readFileSync(join(dir, file), "utf8"))
        .filter((text) => text.includes("<JSFunction SHALLOT_SAMPLE_BEGIN_"));
    if (marked.length !== 1)
        throw new Error(`expected one marked isolate trace, found ${marked.length}`);
    return marked[0];
}

/** Read synchronous V8 optimization traces bracketed by the measured isolate's run markers. */
export function tracedSample(
    stdout: string,
    { retainOptimizedSites = false }: { retainOptimizedSites?: boolean } = {},
): AllocationSample {
    const optimizations = new Map<string, Set<string>>();
    const marked = new Set<string>();
    let label: string | undefined;
    let json: string | undefined;
    for (const line of stdout.split("\n")) {
        // The sampler's markers are functions V8 compiles, named SHALLOT_SAMPLE_<BEGIN|END>_<label hex>;
        // each logs a compiling and a completed line, and the completed one moves the window.
        const marker = /<JSFunction SHALLOT_SAMPLE_(BEGIN|END|\w+?)_([0-9a-f]*) \(sfi =/.exec(line);
        if (marker) {
            if (!line.startsWith("[completed compiling ")) continue;
            const named = Buffer.from(marker[2], "hex").toString();
            if (marker[1] === "BEGIN") {
                if (label !== undefined) throw new Error("nested allocation trace marker");
                label = named;
                marked.add(label);
            } else if (marker[1] === "END") {
                if (label !== named) throw new Error(`unpaired allocation trace marker: ${named}`);
                label = undefined;
            }
        } else if (label !== undefined && /\[(?:compiling method|completed compiling|completed optimizing|optimizing) /.test(line)) {
            const match = /<JSFunction(.*?) \(sfi =/.exec(line);
            if (!match) throw new Error(`unrecognized V8 optimization trace: ${line}`);
            const names = optimizations.get(label) ?? new Set<string>();
            names.add(match[1].trim() || "(anonymous)");
            optimizations.set(label, names);
        } else if (line.startsWith('{"runtime":')) {
            json = line;
        }
    }
    if (label !== undefined) throw new Error(`incomplete allocation trace: window ${JSON.stringify(label)} has no end marker`);
    if (json === undefined) throw new Error("incomplete allocation trace: no sampler result line");
    const sample = JSON.parse(json) as TransitionSample;
    for (const window of sample.windows) {
        if (!marked.has(window.label)) throw new Error(`missing allocation trace markers: ${window.label}`);
        const names = optimizations.get(window.label);
        if (names?.size) {
            window.optimizations = [...names];
            if (!retainOptimizedSites) window.sites = [];
        }
    }
    if (optimizations.has("transition"))
        throw new Error(`transition allocation run refused; optimized ${[...optimizations.get("transition")!].join(", ")}; no byte reading`);
    return sample;
}

export function siteTable(sample: AllocationSample, limit = 30): string {
    const perFrame = (bytes: number, frames: number) => (bytes / frames).toFixed(1);
    const totals = sample.windows.map(
        (window) =>
            window.optimizations?.length
                ? `${window.label}: refused; optimized ${window.optimizations.join(", ")}; no byte reading`
                : `${window.label}: ${windowBytes(window)} bytes (${perFrame(windowBytes(window), window.frames)}/f) over ${window.frames} frames at ${window.sites.length} sites`,
    );
    const heaviest = sample.windows.reduce((a, b) => (windowBytes(b) > windowBytes(a) ? b : a));
    const rows = heaviest.sites
        .slice(0, limit)
        .map(
            (row) =>
                `  ${String(row.bytes).padStart(10)}  ${perFrame(row.bytes, heaviest.frames).padStart(8)}/f  ${row.site}`,
        );
    const more =
        heaviest.sites.length > limit ? [`  … ${heaviest.sites.length - limit} more sites`] : [];
    return [
        ...totals,
        `heaviest, ${heaviest.label}, over ${heaviest.frames} frames, ${sample.runtime}`,
        ...rows,
        ...more,
    ].join("\n");
}
