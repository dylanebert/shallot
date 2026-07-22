// The shared standing gold test every stage-4 sample twin plugs into (spec tumble-inline stage 4). One
// registry entry (`tumble-registry.ts`) per ported sample; this file spawns an isolated child process per
// job — the entry's gold replay, plus one per knob point — through `tumble-gold-runner.ts`, ~4 at a time,
// and asserts each exits clean. On failure the child prints the entry, the check, and (for a gold
// divergence) the first divergent step + got/expected, which the entry's test surfaces verbatim. A later
// queue item adds ONE registry entry — this file needs no change.
//
// One world per child is correctness, not tidiness. The tumble kernel is a process-wide wasm singleton
// whose grow-only regions trap `queryPairs` after several rich worlds in one process, and below that
// threshold can silently lose determinism (a gold that matches in isolation diverges once other worlds run
// ahead of it). A shared in-process run made a verdict depend on registry order and, for shape-soup, even
// the entry's own oracle + knob probes exceeded the trap. A fresh kernel per world — the gold mint's own
// recipe (`scripts/gen-tumble-sample-golds.ts`) — removes both: registry order cannot affect any verdict
// (spec Residue "sequential-world kernel trap").
//
// Outside bunfig's `bun test` scope (`bunfig.toml` roots it at `packages/shallot`) — run explicitly:
//   bun test ./examples/gym/src/tumble-golds.test.ts

import { expect, test } from "bun:test";
import { resolve } from "node:path";
import { goldRegistry } from "./tumble-registry";

const runner = resolve(import.meta.dir, "tumble-gold-runner.ts");
const shallotRoot = resolve(import.meta.dir, "..", "..", "..");
// The engine barrel's module graph references WebGPU globals (`GPUShaderStage`) at eval time; a plain
// `bun run` child has none. `bun test` gets them from bunfig's preload — the child gets the same setup.
const setup = resolve(shallotRoot, "packages/shallot/tests/setup.ts");
const concurrency = 4;
// ~0.6 s per child at 4-wide; ample headroom for ~38+ entries × (1 gold + 2-3 knob) jobs on a slow box.
const timeout = 180_000;

// One child invocation: an entry's gold replay (`--slug`) or one knob probe (`--slug --knob N`).
interface Job {
    slug: string;
    args: string[];
}
type JobResult = { ok: boolean; diagnostic: string };

const jobs: Job[] = goldRegistry.flatMap((entry) => {
    const list: Job[] = [{ slug: entry.slug, args: ["--slug", entry.slug] }];
    entry.knobPoints?.forEach((_, i) => {
        list.push({ slug: entry.slug, args: ["--slug", entry.slug, "--knob", String(i)] });
    });
    return list;
});

// Spawn every job through a fixed-width pool, once. The first per-entry test awaits this and pays the whole
// cost; the rest resolve from the memoized map. One pool (not one Bun.spawn per test) caps concurrency at
// `concurrency` regardless of how bun schedules the tests.
async function runAll(): Promise<Map<Job, JobResult>> {
    const results = new Map<Job, JobResult>();
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

let pending: Promise<Map<Job, JobResult>> | null = null;
function allResults(): Promise<Map<Job, JobResult>> {
    if (!pending) pending = runAll();
    return pending;
}

for (const entry of goldRegistry) {
    test(
        entry.slug,
        async () => {
            const results = await allResults();
            const failures: string[] = [];
            for (const [job, result] of results) {
                if (job.slug === entry.slug && !result.ok) {
                    failures.push(
                        result.diagnostic || `[${entry.slug}] ${job.args.join(" ")} failed`,
                    );
                }
            }
            if (failures.length > 0) throw new Error(failures.join("\n"));
            expect(failures.length).toBe(0);
        },
        timeout,
    );
}
