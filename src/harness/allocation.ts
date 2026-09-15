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
export async function sampleAllocation(
    entry: string,
    {
        warm = 600,
        frames = 600,
        input = "",
    }: { warm?: number; frames?: number; input?: string } = {},
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
