// Per-scenario compile/memory budget metadata — plain data with no imports, same shape and for the same
// reason as `timeouts.ts`: a driver (here, `gym.ts`'s runtime check, folded into every scenario's
// `installHarness` verdict) reads it without booting anything extra, and `budget-coverage.ts` proves the
// registry complete against fixtures with no filesystem or GPU.
//
// A budget is the golden `testing.md`'s exact-equality structural rung checks a scenario against, declared
// at DEFAULT params (`bun bench` with no `--count`/`--param` override — the scenario's own declared
// defaults, `gym.ts`'s `resolveParams`). The exemption is **per axis**, not per scenario
// (`shallot-perf-gates` stage 4b): pipeline count is exact on every registered scenario by measurement, so
// every scenario gates on it, while GPU bytes is exact only where nothing allocates lazily off real-device
// timing. A row can carry `pipelines` as a golden while `gpuBytes` carries an exemption reason — and
// nothing in the type favors either direction, so the reverse (a byte golden with a pipeline-count
// exemption) is equally expressible, even though no row currently needs it. `SCENARIO_BUDGETS` and
// `BUDGET_EXEMPTIONS` share one scenario key; for each axis a scenario carries exactly one of a golden
// (in `SCENARIO_BUDGETS`) or a reason (in `BUDGET_EXEMPTIONS`), asserted both directions, per axis, in
// `budget-coverage.test.ts` the way `coverage.test.ts` already asserts `SCENARIO_GATES` / `GATE_EXEMPTIONS`.
//
// - `pipelines` — the exact real-pipeline count: `[...Profile.compile.keys()].filter(k =>
//   Profile.compiledPipelines.has(k)).length`, never `Profile.compile.size` directly — a `precompile`
//   scope-only span (`sear-typed-variants`) is not a pipeline (`shallot-perf-gates` stage 3a).
// - `gpuBytes` — the exact `Profile.bufferBytes + Profile.textureBytes` total.
//
// Exact equality, no tolerance (`shallot-perf-gates` Locked decision — counts and allocations are
// deterministic and device-independent for a fixed scenario at fixed params, so a tolerance would be
// invented, not derived). A scenario axis that legitimately can't be budgeted this way carries a
// `BUDGET_EXEMPTIONS` reason for that axis instead, never both.
//
// Stage 3b landed the mechanism plus one entry (`render`); stage 4 measured the rest of the roster off two
// `bun bench --sweep` runs (nvidia/lovelace via the WSL bridge, 2026-08-05) and populated every remaining
// registered scenario as a budget (both runs agreed exactly) or an exemption (the two runs disagreed).
// Stage 4b split the per-scenario exemption into per-axis form: all 6 rows below agreed exactly on
// `pipelines` across every sample (3-6 each) and disagreed only on `gpuBytes`, so a per-scenario exemption
// was silently dropping an exact, device-independent pipeline-count golden the Locked decision grants to
// every row — `render`'s 29 the sharpest instance, the number stage 3a exists to make sound.
export interface AxisBudget {
    pipelines?: number;
    gpuBytes?: number;
}

export interface AxisExemption {
    pipelines?: string;
    gpuBytes?: string;
}

export const SCENARIO_BUDGETS: Record<string, AxisBudget> = {
    // measured 2026-08-05, nvidia/lovelace via the WSL bridge, two full `bun bench --sweep` runs at default
    // params — every entry below agreed exactly across both runs (`shallot-perf-gates` stage 4's
    // determinism proof).
    accel: { pipelines: 44, gpuBytes: 45_101_556 },
    "bodies-body-type": { pipelines: 30, gpuBytes: 13_617_360 },
    "bodies-motion-locks": { pipelines: 30, gpuBytes: 13_615_024 },
    "bodies-spinning-book": { pipelines: 30, gpuBytes: 13_616_192 },
    chain: { pipelines: 26, gpuBytes: 33_869_708 },
    "character-mover": { pipelines: 30, gpuBytes: 13_620_864 },
    "collision-overlap-box": { pipelines: 30, gpuBytes: 13_616_192 },
    "collision-ray-curtain": { pipelines: 30, gpuBytes: 13_649_560 },
    "collision-shape-cast": { pipelines: 30, gpuBytes: 13_715_624 },
    "compound-simple": { pipelines: 30, gpuBytes: 13_714_072 },
    "compound-spheres": { pipelines: 30, gpuBytes: 13_968_560 },
    "compound-tile-floor": { pipelines: 30, gpuBytes: 13_616_192 },
    constraints: { pipelines: 66, gpuBytes: 21_807_244 },
    "continuous-bullet-vs-stack": { pipelines: 30, gpuBytes: 13_636_936 },
    "continuous-thin-wall": { pipelines: 30, gpuBytes: 13_650_392 },
    "determinism-falling-ragdolls": { pipelines: 30, gpuBytes: 14_947_168 },
    "events-hit": { pipelines: 30, gpuBytes: 13_772_800 },
    "events-joint-break": { pipelines: 30, gpuBytes: 13_616_192 },
    "events-sensor-sweep": { pipelines: 30, gpuBytes: 13_617_360 },
    "geometry-convex-hull": { pipelines: 30, gpuBytes: 13_641_304 },
    "geometry-convex-primitives": { pipelines: 30, gpuBytes: 13_624_240 },
    "geometry-hull-reduction": { pipelines: 30, gpuBytes: 13_622_032 },
    gltf: { pipelines: 74, gpuBytes: 113_679_528 },
    "gpu-diagnostic": { pipelines: 4, gpuBytes: 32_788 },
    "joints-bridge": { pipelines: 30, gpuBytes: 13_618_528 },
    "joints-cantilever": { pipelines: 30, gpuBytes: 13_617_360 },
    "joints-driving": { pipelines: 30, gpuBytes: 13_694_496 },
    "joints-elevator": { pipelines: 30, gpuBytes: 13_617_360 },
    "joints-filter": { pipelines: 30, gpuBytes: 13_617_360 },
    "joints-paddle": { pipelines: 30, gpuBytes: 13_617_360 },
    "joints-parallel": { pipelines: 30, gpuBytes: 13_616_192 },
    "joints-pendulum": { pipelines: 30, gpuBytes: 13_616_192 },
    "joints-rope": { pipelines: 30, gpuBytes: 13_749_584 },
    "joints-suspension": { pipelines: 30, gpuBytes: 13_618_528 },
    "mesh-fixture": { pipelines: 32, gpuBytes: 38_119_976 },
    "mesh-terrain": { pipelines: 30, gpuBytes: 13_714_640 },
    "mesh-torus": { pipelines: 30, gpuBytes: 13_994_256 },
    outline: { pipelines: 33, gpuBytes: 38_523_836 },
    pile: { pipelines: 66, gpuBytes: 26_972_672 },
    queries: { pipelines: 29, gpuBytes: 12_731_356 },
    "ragdoll-ragdoll": { pipelines: 30, gpuBytes: 13_763_040 },
    raining: { pipelines: 29, gpuBytes: 12_738_748 },
    rotation: { pipelines: 29, gpuBytes: 12_711_644 },
    "shapes-inclined-plane": { pipelines: 30, gpuBytes: 13_617_360 },
    "shapes-restitution": { pipelines: 30, gpuBytes: 14_006_544 },
    "shapes-shape-soup": { pipelines: 30, gpuBytes: 13_847_416 },
    sprite: { pipelines: 49, gpuBytes: 80_170_152 },
    "stacking-arch": { pipelines: 30, gpuBytes: 13_636_048 },
    "stacking-box-pyramid": { pipelines: 30, gpuBytes: 13_616_192 },
    "stacking-dominoes": { pipelines: 30, gpuBytes: 13_806_312 },
    text: { pipelines: 29, gpuBytes: 17_069_256 },

    // stage 4b: `pipelines` matched exactly across every sample stage 4 took of these 6 rows (3-6 each) —
    // reused here unchanged, no re-measurement. Only `gpuBytes` is exempt for these; see
    // `BUDGET_EXEMPTIONS` below for the reason each is timing-dependent on that axis.
    backend: { pipelines: 29 },
    render: { pipelines: 29 },
    character: { pipelines: 66 },
    motor: { pipelines: 66 },
    sat: { pipelines: 69 },
    stress: { pipelines: 31 },
};

/** every axis explicitly exempted from a compile/memory budget, and why — keyed by scenario, one optional
 *  reason per axis. A scenario/axis pair appears in exactly one of this table or `SCENARIO_BUDGETS`,
 *  never both, checked in `budget-coverage.test.ts`. */
export const BUDGET_EXEMPTIONS: Record<string, AxisExemption> = {
    // measured 2026-08-05, nvidia/lovelace via the WSL bridge, two full `bun bench --sweep` runs at default
    // params. Each row below disagreed on `gpuBytes` at least once across 3-6 same-day default-param
    // samples — exact equality has no tolerance to absorb that, so the byte axis is exempted rather than
    // budgeted (`shallot-perf-gates` stage 4: a scenario whose harvests disagree is not budgetable by exact
    // equality). `pipelines` is unaffected — it agreed exactly every time for all 6 and is budgeted above.
    // Each reason cites the mechanism actually read in source, not a guessed structural shape.
    backend: {
        gpuBytes:
            "gpuBytes disagreed once in three same-day default-param runs (12,743,332 twice, " +
            "12,740,404 once — Δ2,928 B). `backend.ts` holds a live `mirror()` readback of the transform " +
            "buffer (`xformMirror`), the same lazily-growing staging ring `render`'s exemption cites " +
            "(`standard/mirror/index.ts`: a new GPU staging buffer is allocated on real backpressure, up " +
            "to `ringSize`, not eagerly at construction) — so its GPU-resident byte total at sample time " +
            "depends on real-device readback timing.",
    },
    render: {
        gpuBytes:
            "gpuBytes took three distinct values across six same-day default-param runs (two full " +
            "`bun bench --sweep`s + two isolated `bun bench --scenario render` agreed on 163,206,152; a " +
            "third sweep measured 165,039,844; the stage 3b original single-sample read was " +
            "160,584,704). `render.ts` opens 8 concurrent `mirror()` readbacks at default `mode=cull` " +
            "params (drawArgs, color, cluster, lights, grid, indices, point-shadow, tile-rects — the " +
            "mode-gated probe mirrors add more on other modes); `Mirror`'s staging ring " +
            "(`standard/mirror/index.ts`) allocates a NEW GPU staging buffer lazily, on real backpressure, " +
            "up to `ringSize` (default 2) rather than eagerly at construction (`if (m._slots.length >= " +
            "m._ringSize) continue`) — so how many of those rings have grown to 1 vs 2 slots by the moment " +
            "`Profile` is sampled depends on real-device readback timing, not scenario code or params.",
    },
    character: {
        gpuBytes:
            "gpuBytes disagreed once in four same-day default-param runs (21,812,580 three times, " +
            "21,824,868 once — Δ12,288 B). `character.ts` combines the same two real-timing-dependent " +
            "mechanisms `motor`/`sat` (live AvbdPlugin physics, avbd.md's documented atomic-appended-" +
            "contact-pool non-determinism) and `render` (a `mirror()` readback of `Avbd.step.bodies` " +
            "through the same lazily-growing staging ring) cite individually.",
    },
    motor: {
        gpuBytes:
            "gpuBytes disagreed across two identical-default-param runs (88,897,840 vs 88,897,856 B, " +
            "Δ16 B). motor.ts runs a live AvbdPlugin scene (a real-time GPU physics solve, not the " +
            'bit-exact tumble backend) — avbd.md\'s "Cross-scene contact-order non-determinism" ' +
            "documents that the atomic-appended contact pool's f32 sum order, and with it the solver's " +
            "real-device-timing-dependent bookkeeping (contact/joint records, warmstart), varies run to " +
            "run even for an isolated topology, so its GPU-resident byte total isn't guaranteed exact-equal.",
    },
    sat: {
        gpuBytes:
            "gpuBytes disagreed across two identical-default-param runs (362,266,716 vs 364,888,164 B, " +
            "Δ2,621,448 B). sat.ts builds a live AvbdPlugin scene with ShapeKind.Hull bodies whose " +
            "`hullData` GPU buffer 'grows on demand' (avbd/step.ts `setHulls` — destroy + recreate at a " +
            "larger size only when the packed hull data exceeds the current buffer) on top of the same " +
            "real-time, non-bit-exact AVBD solve `motor`'s exemption cites — so the buffer's grown size at " +
            "sample time is not guaranteed identical run to run.",
    },
    stress: {
        gpuBytes:
            "gpuBytes disagreed across two identical-default-param runs (1,091,824,136 vs 1,091,660,292 " +
            "B, Δ163,844 B). stress.ts's assert phase (`rampAxis`) loops against a measured real-device " +
            "GPU-busy/CPU-total wall clock — not a fixed frame or byte count — doubling its induced load " +
            "until it crosses a felt-lag threshold that itself depends on this run's measured timing, and " +
            "separately allocates + destroys real transient GPU resources mid-run (the 32 MB " +
            "`readback-stress` buffer, the over-limit BVH/image-array/VAT probes in `gpuMemChecks`); " +
            "`Profile`'s byte counters are sampled once the assert returns, so the total reflects whatever " +
            "ramp level and mid-run allocation state that run's real-device timing produced.",
    },
};
