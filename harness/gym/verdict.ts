import type { BenchmarkMeasurement } from "@dylanebert/shallot/extras";

// The gym's result shape — what a gym scenario returns and the gym launcher routes. The core
// harness never reads this (it treats the run result as opaque); only the gym interprets it.
// A scenario fills the facets it produces:
// - `metrics` → printed (the profiler's GPU + frame timing, the source of truth)
// - `checks`  → pass/fail gate; any failure exits the launcher nonzero
// Both compose — a scenario can measure and assert in one run.

/** one behavioral assertion; `detail` is the human-readable value, `data` the machine-readable one.
 *  `data` is an optional flat number map for a bench script to consume directly (the physics scenario's
 *  `measured` reporter publishes its per-step spans + health counters here; the CLI ignores it). */
export interface Check {
    name: string;
    pass: boolean;
    detail?: string;
    data?: Record<string, number>;
}

export interface Verdict {
    metrics?: BenchmarkMeasurement;
    checks?: Check[];
}
