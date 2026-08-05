// Per-scenario compile/memory budget metadata — plain data with no imports, same shape and for the same
// reason as `timeouts.ts`: a driver (here, `gym.ts`'s runtime check, folded into every scenario's
// `installHarness` verdict) reads it without booting anything extra, and `budget-coverage.ts` proves the
// registry complete against fixtures with no filesystem or GPU.
//
// A budget is the golden `testing.md`'s exact-equality structural rung checks a scenario against, declared
// at DEFAULT params (`bun bench` with no `--count`/`--param` override — the scenario's own declared
// defaults, `gym.ts`'s `resolveParams`). Both fields are measured together, real numbers off a real run,
// never estimated: a scenario with GPU pipelines but no declared memory (or the reverse) is an incomplete
// measurement, not a partial budget.
//
// - `pipelines` — the exact real-pipeline count: `[...Profile.compile.keys()].filter(k =>
//   Profile.compiledPipelines.has(k)).length`, never `Profile.compile.size` directly — a `precompile`
//   scope-only span (`sear-typed-variants`) is not a pipeline (`shallot-perf-gates` stage 3a).
// - `gpuBytes` — the exact `Profile.bufferBytes + Profile.textureBytes` total.
//
// Exact equality, no tolerance (`shallot-perf-gates` Locked decision — counts and allocations are
// deterministic and device-independent for a fixed scenario at fixed params, so a tolerance would be
// invented, not derived). A scenario that legitimately can't be budgeted this way carries a
// `BUDGET_EXEMPTIONS` reason instead, never both — asserted in `budget-coverage.test.ts` the way
// `coverage.test.ts` already asserts `SCENARIO_GATES` / `GATE_EXEMPTIONS` both directions.
//
// Stage 3b lands the mechanism plus this one entry (`render`) end to end; stage 4 is the mechanical queue
// that measures and transcribes the rest of the roster — an unpopulated scenario here is deliberate, not
// a gap this file owes (`shallot-perf-gates` stage 3b).
export interface ScenarioBudget {
    pipelines: number;
    gpuBytes: number;
}

export const SCENARIO_BUDGETS: Record<string, ScenarioBudget> = {
    // measured 2026-08-05 on nvidia/lovelace via the WSL bridge, `bun bench --scenario render` (default
    // params: mode=cull, count=4096, lights=12, seed=1) — the filtered pipeline count
    // (`[...Profile.compile.keys()].filter(k => Profile.compiledPipelines.has(k)).length`), read directly
    // for the first time this stage (`shallot-perf-gates` stage 3a left it unprinted). It matches v0.8.1's
    // pre-port 29 (stage 1's Live log), the confirmation the instrument is sound. `gpuBytes` is the exact
    // `Profile.bufferBytes + Profile.textureBytes` off the same run (153.2 MB — stage 2's 155.6 MB was a
    // rounded display read, not this exact byte count).
    render: { pipelines: 29, gpuBytes: 160_584_704 },
};

/** every scenario explicitly exempted from a compile/memory budget, and why. Stage 4 populates the rest
 *  of the roster (`SCENARIO_BUDGETS` entries or exemptions here) — an honest partial table is deliberate
 *  (the same call `GATE_EXEMPTIONS` made in `timeouts.ts`), not a gap. */
export const BUDGET_EXEMPTIONS: Record<string, string> = {};
