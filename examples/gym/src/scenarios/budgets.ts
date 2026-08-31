// Per-scenario compile/memory budget metadata — plain data with no imports, same shape and for the same
// reason as `timeouts.ts`: a driver (here, `gym.ts`'s runtime check, folded into every scenario's
// `installHarness` verdict) reads it without booting anything extra, and `budget-coverage.ts` proves the
// registry complete against fixtures with no filesystem or GPU.
//
// A budget is the golden `testing.md`'s exact-equality structural rung checks a scenario against, declared
// at DEFAULT params (`bun bench` with no `--count`/`--param` override — the scenario's own declared
// defaults, `gym.ts`'s `resolveParams`). The exemption is **per axis**, not per scenario:
// pipeline count is exact on every registered scenario by measurement, so
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
//   scope-only span (`sear-typed-variants`) is not a pipeline.
// - `pipelineCalls` — the exact raw `create*Pipeline(Async)` invocation count (`Profile.pipelineCalls`),
//   counted per CALL before labels collapse. It exists because `pipelines` counts DISTINCT labels and a
//   named label overwrites in `recordCompile`, so a pipeline built under an existing label moves that
//   count by zero — the blindness `gpu.md`'s per-pipeline-label law is meant to prevent, pinned at
//   exactly one surface (`sear/pipelines.test.ts`) and nowhere else. This axis closes it at the mechanism
//   instead of at the naming convention: a multiplied pipeline moves the call count whatever it's called.
//   Labels are not 1:1 with pipelines — TypeGPU derives several raw pipelines from one named typed
//   pipeline (`pile` builds 133 raw pipelines under 66 labels, `accel` 60 under 44) — so the count is
//   per call, never a label-equality form. The exact count is the sound form of the same intent.
// - `gpuBytes` — the exact `Profile.bufferBytes + Profile.textureBytes - Profile.lazyBytes` total: every
//   live allocation EXCEPT one an allocator marked `LazyAlloc.lazy` at creation (`Mirror`'s readback ring,
//   `Slab`'s staging pool — a pool that grows on real GPU backpressure, so its live-stager count is
//   timing-dependent even at fixed params). Excluding those bytes at the mechanism, rather than exempting
//   the whole scenario, is what makes this axis exact on nearly every row;
//   the excluded bytes print separately in `bun bench`'s measurement block, never silently dropped.
//
// Exact equality, no tolerance — counts and allocations are
// deterministic and device-independent for a fixed scenario at fixed params, so a tolerance would be
// invented, not derived. A scenario axis that legitimately can't be budgeted this way carries a
// `BUDGET_EXEMPTIONS` reason for that axis instead, never both.
//
// `BUDGET_EXEMPTIONS` is empty — every row is exact — because the two lazily-grown pools (`Mirror`'s
// readback ring, `Slab`'s staging pool) are excluded at the allocation site (`LazyAlloc`,
// `Profile.lazyBytes`), not by exempting the scenarios that hold one. `AvbdPlugin`'s other
// grow-on-demand buffer, `setHulls`'s `hullData` (`standard/avbd/step.ts`), is not a third lazy site:
// it re-uploads only when `Hulls.size` changes, a count fixed by which hulls a scenario registers at
// build, so it carries no `LazyAlloc` mark. The AVBD-driven rows (`motor`/`sat`/`character`/`stress`)
// are exact for the same reason — their measured disagreement traced to the shared `slab-staging`
// label, never to AVBD's contact-order non-determinism.
//
// `orbit-touch` is exact at DEFAULT params: 29 pipelines, 29 pipeline calls, 31_828_368 GPU bytes.
export interface AxisBudget {
    pipelines?: number;
    pipelineCalls?: number;
    gpuBytes?: number;
}

/** the gated quantities, derived from {@link AxisBudget} rather than listed beside it. `Record<Axis, true>`
 *  is the pin: a third quantity added to the interface and not here is a `bun check` error, where a
 *  hand-written list would have left the new axis silently uncovered by `budget-coverage.ts`'s entries
 *  check, its completeness check, AND `assertBudget` at once — all three iterate this list.
 *  Every other per-axis shape derives from {@link Axis} too:
 *  {@link AxisExemption} and `budget-coverage.ts`'s `MeasuredBudget` are mapped types, so a new axis
 *  reaches the exemption table and the measurement record without a second edit. */
export type Axis = keyof AxisBudget;
const AXIS_SET: Record<Axis, true> = { pipelines: true, pipelineCalls: true, gpuBytes: true };
export const AXES = Object.keys(AXIS_SET) as readonly Axis[];

export type AxisExemption = { [K in Axis]?: string };

export const SCENARIO_BUDGETS: Record<string, AxisBudget> = {
    // measured under the lazy-pool exclusion (`gpuBytes` = `Profile.bufferBytes + Profile.textureBytes -
    // `Profile.lazyBytes`) — a data-only read of the total, not a re-measurement of the scene. Every row
    // here rests on TWO independent agreeing samples: the harvest `bun bench --sweep` that produced these
    // numbers, plus a second sweep run against these already-final goldens through the real exact-equality
    // checks — a verifier confirming the harvest's numbers, not a second producer of them. `backend`,
    // `render`, `character`, `motor`, `sat`, `stress`, `chain` additionally carry two independent
    // same-day `--scenario` confirmation runs, so those 7 rows rest on FOUR independent agreeing samples.
    accel: { pipelines: 44, pipelineCalls: 60, gpuBytes: 34_574_780 },
    backend: { pipelines: 29, pipelineCalls: 29, gpuBytes: 12_723_708 },
    "bodies-body-type": { pipelines: 30, pipelineCalls: 30, gpuBytes: 13_519_028 },
    "bodies-motion-locks": { pipelines: 30, pipelineCalls: 30, gpuBytes: 13_516_692 },
    "bodies-spinning-book": { pipelines: 30, pipelineCalls: 30, gpuBytes: 13_517_860 },
    chain: { pipelines: 26, pipelineCalls: 26, gpuBytes: 29_151_084 },
    character: { pipelines: 66, pipelineCalls: 66, gpuBytes: 21_761_544 },
    "character-mover": { pipelines: 30, pipelineCalls: 30, gpuBytes: 13_522_532 },
    "collision-overlap-box": { pipelines: 30, pipelineCalls: 30, gpuBytes: 13_517_860 },
    "collision-ray-curtain": { pipelines: 30, pipelineCalls: 30, gpuBytes: 13_551_228 },
    "collision-shape-cast": { pipelines: 30, pipelineCalls: 30, gpuBytes: 13_617_292 },
    "compound-simple": { pipelines: 30, pipelineCalls: 30, gpuBytes: 13_615_740 },
    "compound-spheres": { pipelines: 30, pipelineCalls: 30, gpuBytes: 13_870_228 },
    "compound-tile-floor": { pipelines: 30, pipelineCalls: 30, gpuBytes: 13_517_860 },
    constraints: { pipelines: 66, pipelineCalls: 165, gpuBytes: 21_765_288 },
    "continuous-bullet-vs-stack": { pipelines: 30, pipelineCalls: 30, gpuBytes: 13_538_604 },
    "continuous-thin-wall": { pipelines: 30, pipelineCalls: 30, gpuBytes: 13_552_060 },
    "determinism-falling-ragdolls": { pipelines: 30, pipelineCalls: 30, gpuBytes: 14_848_836 },
    "events-hit": { pipelines: 30, pipelineCalls: 30, gpuBytes: 13_674_468 },
    "events-joint-break": { pipelines: 30, pipelineCalls: 30, gpuBytes: 13_517_860 },
    "events-sensor-sweep": { pipelines: 30, pipelineCalls: 30, gpuBytes: 13_519_028 },
    "geometry-convex-hull": { pipelines: 30, pipelineCalls: 30, gpuBytes: 13_542_972 },
    "geometry-convex-primitives": { pipelines: 30, pipelineCalls: 30, gpuBytes: 13_525_908 },
    "geometry-hull-reduction": { pipelines: 30, pipelineCalls: 30, gpuBytes: 13_523_700 },
    gltf: { pipelines: 74, pipelineCalls: 74, gpuBytes: 104_242_304 },
    "gpu-diagnostic": { pipelines: 4, pipelineCalls: 4, gpuBytes: 32_788 },
    "joints-bridge": { pipelines: 30, pipelineCalls: 30, gpuBytes: 13_520_196 },
    "joints-cantilever": { pipelines: 30, pipelineCalls: 30, gpuBytes: 13_519_028 },
    "joints-driving": { pipelines: 30, pipelineCalls: 30, gpuBytes: 13_596_164 },
    "joints-elevator": { pipelines: 30, pipelineCalls: 30, gpuBytes: 13_519_028 },
    "joints-filter": { pipelines: 30, pipelineCalls: 30, gpuBytes: 13_519_028 },
    "joints-paddle": { pipelines: 30, pipelineCalls: 30, gpuBytes: 13_519_028 },
    "joints-parallel": { pipelines: 30, pipelineCalls: 30, gpuBytes: 13_517_860 },
    "joints-pendulum": { pipelines: 30, pipelineCalls: 30, gpuBytes: 13_517_860 },
    "joints-rope": { pipelines: 30, pipelineCalls: 30, gpuBytes: 13_651_252 },
    "joints-suspension": { pipelines: 30, pipelineCalls: 30, gpuBytes: 13_520_196 },
    "mesh-fixture": { pipelines: 32, pipelineCalls: 32, gpuBytes: 31_828_492 },
    "mesh-terrain": { pipelines: 30, pipelineCalls: 30, gpuBytes: 13_616_308 },
    "mesh-torus": { pipelines: 30, pipelineCalls: 30, gpuBytes: 13_895_924 },
    motor: { pipelines: 66, pipelineCalls: 66, gpuBytes: 88_873_964 },
    "ocean-slope": { pipelines: 47, pipelineCalls: 47, gpuBytes: 56_386_840 },
    "orbit-touch": { pipelines: 29, pipelineCalls: 29, gpuBytes: 31_828_368 },
    outline: { pipelines: 33, pipelineCalls: 33, gpuBytes: 38_517_664 },
    pile: { pipelines: 66, pipelineCalls: 133, gpuBytes: 26_173_716 },
    queries: { pipelines: 29, pipelineCalls: 29, gpuBytes: 12_719_784 },
    "ragdoll-ragdoll": { pipelines: 30, pipelineCalls: 30, gpuBytes: 13_664_708 },
    raining: { pipelines: 29, pipelineCalls: 29, gpuBytes: 12_725_016 },
    render: { pipelines: 29, pipelineCalls: 29, gpuBytes: 150_362_360 },
    rotation: { pipelines: 29, pipelineCalls: 29, gpuBytes: 12_705_832 },
    sat: { pipelines: 69, pipelineCalls: 69, gpuBytes: 322_682_880 },
    "shapes-inclined-plane": { pipelines: 30, pipelineCalls: 30, gpuBytes: 13_519_028 },
    "shapes-restitution": { pipelines: 30, pipelineCalls: 30, gpuBytes: 13_908_212 },
    "shapes-shape-soup": { pipelines: 30, pipelineCalls: 30, gpuBytes: 13_749_084 },
    sprite: { pipelines: 49, pipelineCalls: 49, gpuBytes: 80_163_828 },
    "stacking-arch": { pipelines: 30, pipelineCalls: 30, gpuBytes: 13_537_716 },
    "stacking-box-pyramid": { pipelines: 30, pipelineCalls: 30, gpuBytes: 13_517_860 },
    "stacking-dominoes": { pipelines: 30, pipelineCalls: 30, gpuBytes: 13_660_940 },
    stress: { pipelines: 31, pipelineCalls: 31, gpuBytes: 1_089_137_076 },
    text: { pipelines: 29, pipelineCalls: 29, gpuBytes: 17_068_072 },
};

/** every axis explicitly exempted from a compile/memory budget, and why — keyed by scenario, one optional
 *  reason per axis. A scenario/axis pair appears in exactly one of this table or `SCENARIO_BUDGETS`, never
 *  both, checked in `budget-coverage.test.ts`. Empty: the two lazily-grown pools (`Mirror`'s ring or
 *  `Slab`'s staging pool) are excluded at the allocation site, so the remaining total is exact on every
 *  row. Kept non-empty-shaped (the
 *  type, not a value) because the mechanism this axis excludes is scoped to two known pools, not to "no
 *  scenario can ever disagree" — a future lazily-grown pool that isn't marked `LazyAlloc.lazy` yet would
 *  reintroduce a disagreeing row here, not a silent gate weakening. */
export const BUDGET_EXEMPTIONS: Record<string, AxisExemption> = {};
