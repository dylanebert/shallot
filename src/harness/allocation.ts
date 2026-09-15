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
    /** allocating sites in the bundled subject, most bytes first */
    sites: readonly AllocationSite[];
    /** cost of one heap-statistics read */
    warmReadBytes: number;
    /** heap movement across an empty window of the same length */
    nullHeapDelta: number;
    /** sampled bytes of one known literal per frame; zero means the sampler saw nothing */
    controlBytes: number;
}

const SAMPLER = resolve(import.meta.dir, "allocation-sampler.mjs");

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
export function siteTable(sample: AllocationSample, limit = 20): string {
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
