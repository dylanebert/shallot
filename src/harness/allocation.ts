import { mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";

export interface AllocationSite {
    /** function name and original source line */
    site: string;
    /** sampled bytes over the measured window, collected objects included */
    bytes: number;
}

export interface AllocationSample {
    runtime: string;
    warm: number;
    frames: number;
    totalBytes: number;
    /** allocating sites in the bundled subject, most bytes first; builtin bytes credit their caller */
    sites: readonly AllocationSite[];
    /** the same window read after `warm` frames, the first steadiness premise */
    atWarm: readonly AllocationSite[];
    /** the same window read after twice `warm` frames */
    atDoubleWarm: readonly AllocationSite[];
    /** an A/A repeat of the window after twice `warm` frames */
    repeat: readonly AllocationSite[];
    /** cost of one heap-statistics read */
    warmReadBytes: number;
    /** heap movement across an empty window of the same length */
    nullHeapDelta: number;
    /** sampled bytes of one known literal per frame; zero means the sampler saw nothing */
    controlBytes: number;
}

const SAMPLER = resolve(import.meta.dir, "allocation-sampler.mjs");

/**
 * Lowered V8 tier thresholds (defaults 400 and 3,000 in Node 26), so every function in a stepped loop
 * reaches TurboFan inside the warm: warm-up boxing and Maglev-only literals are JIT transitions, not
 * steady-state cost. A deopt loop under these would show as warm-N and warm-2N site sets that disagree.
 */
export const TIER_FLAGS = [
    "--invocation-count-for-maglev=10",
    "--invocation-count-for-turbofan=50",
];

const siteNames = (rows: readonly AllocationSite[]) => rows.map((row) => row.site).sort();

/**
 * The steadiness premise, or undefined when it holds: the warm-N and warm-2N windows allocate at the
 * same sites, and an A/A repeat reads the same bytes at the same sites.
 */
export function unsteady(sample: AllocationSample): string | undefined {
    const n = siteNames(sample.atWarm);
    const n2 = siteNames(sample.atDoubleWarm);
    if (n.join("\n") !== n2.join("\n")) {
        const only = (a: string[], b: string[]) =>
            a.filter((s) => !b.includes(s)).join(", ") || "none";
        return `warm ${sample.warm} sites differ from warm ${2 * sample.warm}: only at ${sample.warm}: ${only(n, n2)}; only at ${2 * sample.warm}: ${only(n2, n)}`;
    }
    const bytes = (rows: readonly AllocationSite[]) =>
        rows
            .map((r) => `${r.site}=${r.bytes}`)
            .sort()
            .join("\n");
    if (bytes(sample.atDoubleWarm) !== bytes(sample.repeat))
        return `A/A repeat disagrees: ${sum(sample.atDoubleWarm)} then ${sum(sample.repeat)} bytes`;
    return undefined;
}

const sum = (rows: readonly AllocationSite[]) => rows.reduce((total, row) => total + row.bytes, 0);

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

/** Render the heaviest sites as a table for a failure message. */
export function siteTable(sample: AllocationSample, limit = 30): string {
    const perFrame = (bytes: number) => (bytes / sample.frames).toFixed(1);
    const rows = sample.sites
        .slice(0, limit)
        .map(
            (row) =>
                `  ${String(row.bytes).padStart(10)}  ${perFrame(row.bytes).padStart(8)}/f  ${row.site}`,
        );
    const more =
        sample.sites.length > limit ? [`  … ${sample.sites.length - limit} more sites`] : [];
    return [
        `${sample.totalBytes} bytes over ${sample.frames} frames (${perFrame(sample.totalBytes)}/f) at ${sample.sites.length} sites, ${sample.runtime}`,
        ...rows,
        ...more,
    ].join("\n");
}
