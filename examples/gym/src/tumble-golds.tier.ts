// By-path tier (`testing.md` § `.test.ts` vs `.oracle.ts` vs `.lab.ts` vs `.tier.ts`):
// the full tumble gold-replay population, moved out of the default suite because it costs
// ~5.2 s of test time there against the 5 s per-file cap. The cost is the population —
// 38 ported tumble.js samples, each with a committed gold trajectory replayed bit-exact
// in a fresh child process, plus knob probes on 18 of 38 entries (52 probes total) that assert
// boundedness at non-default parameters. That is 90 child-process jobs (38 gold replays + 52 knob
// probes) at ~0.6 s per child on a slow box through a 4-wide pool, and N is the property: each entry is a real
// ported sample, not a scan step, so the count cannot be derived down. Coverage moved tiers
// and did not shrink — the default suite keeps `tumble-golds.test.ts` as a sentinel over
// the first 5 entries (16 jobs) against the same frozen gold fixtures.
//
// Run by path, one file per invocation:
//   bun test ./examples/gym/src/tumble-golds.tier.ts     (from the shallot root)
//
// Which edits are the cue to run it is this file's own transitive import cone — every `.ts`
// file reachable from this file's `./`-relative imports, walked mechanically. The tier
// imports `tumble-gold-pool.ts` and `tumble-registry.ts`; `tumble-registry.ts` imports all
// 38 `tumble-*.ts` build/update modules and 38 committed gold JSON files under
// `packages/shallot/tests/tumble/samples/`; `tumble-gold-pool.ts` imports only
// `tumble-registry.ts` (for the `GoldEntry` type) and resolves `tumble-gold-runner.ts` at
// runtime; `tumble-gold-runner.ts` imports `@dylanebert/shallot/tumble/core`,
// `tumble-oracle.ts`, and `./tumble-registry`. The cone is wider than the set of modules a gold verdict is a function
// of (a build module's import of an unrelated engine module arrives transitively but
// cannot move a gold trajectory): that over-inclusion is what a derived list costs and it is
// accepted rather than re-narrowed by hand (`checks.md`, by-path tier trigger lists). The
// tier reads no non-imported file at runtime beyond the gold JSON files the registry itself
// imports, so there is no runtime-read input to list beside the cone.
//
// Advisory, not a trigger: nothing runs this tier automatically — a person reads this header
// and runs the command above when an edit touches one of these paths.

import { expect, test } from "bun:test";
import { type Job, type JobResult, jobsFor, runJobs } from "./tumble-gold-pool";
import { goldRegistry } from "./tumble-registry";

const timeout = 180_000;

const jobs = jobsFor(goldRegistry);

let pending: Promise<Map<Job, JobResult>> | null = null;
function allResults(): Promise<Map<Job, JobResult>> {
    if (!pending) pending = runJobs(jobs);
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
