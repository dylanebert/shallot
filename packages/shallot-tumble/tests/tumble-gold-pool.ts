// Shared child-process pool for the tumble gold-replay tests. Both the default-suite sentinel
// (`tumble-golds.test.ts`) and the by-path tier (`tumble-golds.tier.ts`) drive the same runner
// (`tumble-gold-runner.ts`) through this pool — the only difference is which registry entries each
// iterates. One world per child is correctness (the kernel trap the runner's header names); the pool
// caps concurrency at 4 regardless of how bun schedules the tests.

import { resolve } from "node:path";
import type { GoldEntry } from "./tumble-registry";

const runner = resolve(import.meta.dir, "tumble-gold-runner.ts");
const shallotRoot = resolve(import.meta.dir, "..", "..", "..");
// The engine barrel's module graph references WebGPU globals (`GPUShaderStage`) at eval time; a plain
// `bun run` child has none. `bun test` gets them from bunfig's preload — the child gets the same setup.
const setup = resolve(shallotRoot, "packages/shallot/tests/setup.ts");
const concurrency = 4;

/** One child invocation: an entry's gold replay (`--slug`) or one knob probe (`--slug --knob N`). */
export interface Job {
    slug: string;
    args: string[];
}
export type JobResult = { ok: boolean; diagnostic: string };

/** Build the job list for a set of registry entries — one gold replay per entry, plus each knob probe. */
export function jobsFor(entries: GoldEntry[]): Job[] {
    return entries.flatMap((entry) => {
        const list: Job[] = [{ slug: entry.slug, args: ["--slug", entry.slug] }];
        entry.knobPoints?.forEach((_, i) => {
            list.push({ slug: entry.slug, args: ["--slug", entry.slug, "--knob", String(i)] });
        });
        return list;
    });
}

/** Spawn every job through a fixed-width pool, once. Returns a map of job to result. */
export async function runJobs(jobs: Job[]): Promise<Map<Job, JobResult>> {
    const results: Map<Job, JobResult> = new Map();
    let next = 0;
    async function worker(): Promise<void> {
        while (next < jobs.length) {
            const job = jobs[next++];
            const proc = Bun.spawn(["bun", "--preload", setup, runner, ...job.args], {
                cwd: shallotRoot,
                stdout: "pipe",
                stderr: "pipe",
            });
            const [out, err, code] = await Promise.all([
                new Response(proc.stdout).text(),
                new Response(proc.stderr).text(),
                proc.exited,
            ]);
            results.set(job, { ok: code === 0, diagnostic: `${out}${err}`.trim() });
        }
    }
    await Promise.all(Array.from({ length: concurrency }, worker));
    return results;
}
