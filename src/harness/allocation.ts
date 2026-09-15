import { mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";

export interface AllocationSite {
    /** function name and original source line */
    site: string;
    /** sampled bytes over the measured window, collected objects included */
    bytes: number;
}

export interface AllocationWindow {
    label: string;
    /** allocating sites in the bundled subject, most bytes first; builtin bytes credit their caller */
    sites: readonly AllocationSite[];
}

export interface AllocationSample {
    runtime: string;
    warm: number;
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
    /** objects allocated from spawn through despawn still live after a full collection */
    survivors: readonly AllocationSite[];
}

const SAMPLER = resolve(import.meta.dir, "allocation-sampler.mjs");

/**
 * Lowered V8 tier thresholds (defaults 400 and 3,000 in Node 26), so every function in a stepped loop
 * reaches TurboFan inside the warm: warm-up boxing and Maglev-only literals are JIT transitions, not
 * steady-state cost. An unoptimized path allocates more, never less, so tiering can only redden a reading.
 */
export const TIER_FLAGS = [
    "--invocation-count-for-maglev=10",
    "--invocation-count-for-turbofan=50",
];

export const windowBytes = (window: AllocationWindow) =>
    window.sites.reduce((total, row) => total + row.bytes, 0);

/** True only when every window reads zero bytes at zero sites. */
export const allocatesNothing = (sample: AllocationSample) =>
    sample.windows.every((window) => window.sites.length === 0 && windowBytes(window) === 0);

/**
 * Bundle `entry` for Node, build its default export with `input`, step it `warm` frames, collect,
 * then sample `frames` more under V8's sampling heap profiler. Needs the `node` requirement resolved.
 */
export function sampleAllocation(
    entry: string,
    options: { warm?: number; frames?: number; input?: string } = {},
): Promise<AllocationSample> {
    return runSampler(entry, options, []);
}

/**
 * As {@link sampleAllocation}, for an entry whose subject also has `spawn()` and `despawn()`: warms paired
 * cycles, samples each event frame, the steady windows after it, and one cycle's live survivors.
 */
export function sampleTransition(
    entry: string,
    options: { warm?: number; frames?: number; input?: string } = {},
): Promise<TransitionSample> {
    return runSampler(entry, options, ["transition"]) as Promise<TransitionSample>;
}

async function runSampler(
    entry: string,
    { warm = 600, frames = 600, input = "" }: { warm?: number; frames?: number; input?: string },
    mode: string[],
): Promise<AllocationSample> {
    const dir = mkdtempSync(join(tmpdir(), "shallot-allocation-"));
    try {
        writeFileSync(join(dir, "input.txt"), input);
        const built = await Bun.build({
            entrypoints: [entry],
            outdir: dir,
            target: "node",
            format: "esm",
            sourcemap: "linked",
            naming: "subject.mjs",
        });
        if (!built.success)
            throw new Error(`allocation bundle failed: ${built.logs.map(String).join("\n")}`);
        const proc = Bun.spawn(
            [
                "node",
                "--expose-gc",
                "--enable-source-maps",
                ...TIER_FLAGS,
                SAMPLER,
                join(dir, "subject.mjs"),
                String(warm),
                String(frames),
                join(dir, "input.txt"),
                ...mode,
            ],
            { stdout: "pipe", stderr: "pipe" },
        );
        const [stdout, stderr, code] = await Promise.all([
            new Response(proc.stdout).text(),
            new Response(proc.stderr).text(),
            proc.exited,
        ]);
        if (code !== 0) throw new Error(`allocation sampler exited ${code}: ${stderr.trim()}`);
        return JSON.parse(stdout) as AllocationSample;
    } finally {
        rmSync(dir, { recursive: true, force: true });
    }
}

/** Empty when `sites` names exactly `named`; otherwise the unnamed sites by bytes and the named ones absent. */
export function siteSetMismatch(
    label: string,
    sites: readonly AllocationSite[],
    named: readonly string[],
): string {
    const extra = sites.filter((row) => !named.includes(row.site));
    const missing = named.filter((site) => !sites.some((row) => row.site === site));
    if (extra.length === 0 && missing.length === 0) return "";
    return [
        `${label}: ${extra.length} unnamed sites, ${missing.length} named sites absent`,
        ...extra.map((row) => `  + ${String(row.bytes).padStart(8)}  ${row.site}`),
        ...missing.map((site) => `  - ${site}`),
    ].join("\n");
}

/** Per-window totals, then the heaviest window's sites as a table, for a failure message. */
export function siteTable(sample: AllocationSample, limit = 30): string {
    const perFrame = (bytes: number) => (bytes / sample.frames).toFixed(1);
    const totals = sample.windows.map(
        (window) =>
            `${window.label}: ${windowBytes(window)} bytes (${perFrame(windowBytes(window))}/f) at ${window.sites.length} sites`,
    );
    const heaviest = sample.windows.reduce((a, b) => (windowBytes(b) > windowBytes(a) ? b : a));
    const rows = heaviest.sites
        .slice(0, limit)
        .map(
            (row) =>
                `  ${String(row.bytes).padStart(10)}  ${perFrame(row.bytes).padStart(8)}/f  ${row.site}`,
        );
    const more =
        heaviest.sites.length > limit ? [`  … ${heaviest.sites.length - limit} more sites`] : [];
    return [
        ...totals,
        `heaviest, ${heaviest.label}, over ${sample.frames} frames, ${sample.runtime}`,
        ...rows,
        ...more,
    ].join("\n");
}
