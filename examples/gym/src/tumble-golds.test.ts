// Sentinel for the tumble gold-replay tier (`tumble-golds.tier.ts`): the first
// SENTINEL_ENTRIES registry entries, run against the same frozen gold fixtures, so the
// default suite keeps coverage when the full population moves tiers. Coverage moved tiers
// and did not shrink — the tier runs all 38 entries (90 child-process jobs: one gold replay
// per entry plus each entry's knob probes); this sentinel runs the first 5 (16 jobs) and
// preserves the same mechanism: spawn each job as an isolated child through the shared pool
// (`tumble-gold-pool.ts`), assert each exits clean.
//
// One world per child is correctness, not tidiness. The tumble kernel is a process-wide wasm singleton
// whose grow-only regions trap `queryPairs` after several rich worlds in one process, and below that
// threshold can silently lose determinism (a gold that matches in isolation diverges once other worlds run
// ahead of it). A shared in-process run made a verdict depend on registry order and, for shape-soup, even
// the entry's own oracle + knob probes exceeded the trap. A fresh kernel per world — the gold mint's own
// recipe (`scripts/gen-tumble-sample-golds.ts`) — removes both: registry order cannot affect any verdict
// (the "sequential-world kernel trap").
//
// Outside bunfig's `bun test` scope (`bunfig.toml` roots it at `packages/shallot`) — run explicitly:
//   bun test ./examples/gym/src/tumble-golds.test.ts

import { expect, test } from "bun:test";
import { type Job, type JobResult, jobsFor, runJobs } from "./tumble-gold-pool";
import { goldRegistry } from "./tumble-registry";

const timeout = 180_000;

// The first N entries — a cheap sentinel over the same frozen gold fixtures the tier runs in full.
const SENTINEL_ENTRIES = 5;
const sentinelEntries = goldRegistry.slice(0, SENTINEL_ENTRIES);
const jobs = jobsFor(sentinelEntries);

let pending: Promise<Map<Job, JobResult>> | null = null;
function allResults(): Promise<Map<Job, JobResult>> {
    if (!pending) pending = runJobs(jobs);
    return pending;
}

for (const entry of sentinelEntries) {
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
