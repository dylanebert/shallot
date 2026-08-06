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
// - `gpuBytes` — the exact `Profile.bufferBytes + Profile.textureBytes - Profile.lazyBytes` total: every
//   live allocation EXCEPT one an allocator marked `LazyAlloc.lazy` at creation (`Mirror`'s readback ring,
//   `Slab`'s staging pool — a pool that grows on real GPU backpressure, so its live-stager count is
//   timing-dependent even at fixed params). Excluding those bytes at the mechanism, rather than exempting
//   the whole scenario, is what makes this axis exact on nearly every row (`shallot-perf-gates` stage 4e);
//   the excluded bytes print separately in `bun bench`'s measurement block, never silently dropped.
//
// Exact equality, no tolerance (`shallot-perf-gates` Locked decision — counts and allocations are
// deterministic and device-independent for a fixed scenario at fixed params, so a tolerance would be
// invented, not derived). A scenario axis that legitimately can't be budgeted this way carries a
// `BUDGET_EXEMPTIONS` reason for that axis instead, never both.
//
// History: stage 3b landed the mechanism plus one entry (`render`); stage 4 measured the rest of the
// roster (2-sample reads, later shown too thin); stage 4b split the exemption per axis; stage 4c/4d traced
// every disagreement in the roster to ONE mechanism — `Mirror`'s readback ring and `Slab`'s staging pool,
// both lazily-grown pools under real GPU backpressure (`standard/mirror/index.ts`, `standard/slab/
// index.ts`). Stage 4e (2026-08-05) closed the axis by EXCLUDING those pools' bytes at the allocation site
// (`LazyAlloc`, `Profile.lazyBytes`) instead of exempting the scenarios that hold one, then re-harvested
// the entire roster off one `bun bench --sweep` (nvidia/lovelace via the WSL bridge) under the new
// definition — every number below changed because the total's definition changed, not because the scene
// did. The 7 rows previously byte-exempt (`backend`, `render`, `character`, `motor`, `sat`, `stress`,
// `chain`) were re-verified at the stage 3b/4c floor (3 independent same-day default-param runs each,
// beyond the one sweep sample) and all 7 came back exact — `BUDGET_EXEMPTIONS` is empty: excluding the
// lazy pools' bytes at the mechanism resolved every known disagreement, including the AVBD-driven rows
// (`motor`/`sat`/`character`/`stress`) whose prior exemption reasons cited AVBD's documented contact-order
// non-determinism (`avbd.md`) — their measured disagreement tracks back to the SAME lazy-pool bytes
// (`slab-staging`, shared by every scene with a dirty transform slab), not to a byte-affecting instance of
// that non-determinism, on the topologies these scenarios exercise.
//
// `AvbdPlugin`'s other grow-on-demand buffer, `setHulls`'s `hullData` (`standard/avbd/step.ts`), was
// checked and ruled out as a third lazy site: it re-uploads only when `Hulls.size` changes
// (`standard/avbd/index.ts`), a count fixed by which hulls a scenario registers at build — static
// registry content, not real-device readback timing — so it's deterministic for a fixed scenario at
// fixed params and carries no `LazyAlloc` mark. Stage 4d's attribution corroborates this independently:
// every one of the 34 disagreeing rows, including the AVBD-heavy ones, traced to the single label
// `slab-staging`, never `phys-hulls`.
export interface AxisBudget {
    pipelines?: number;
    gpuBytes?: number;
}

export interface AxisExemption {
    pipelines?: string;
    gpuBytes?: string;
}

export const SCENARIO_BUDGETS: Record<string, AxisBudget> = {
    // measured 2026-08-05, nvidia/lovelace via the WSL bridge, under the stage 4e exclusion (`gpuBytes` =
    // `Profile.bufferBytes + Profile.textureBytes - Profile.lazyBytes`) — a data-only re-harvest, not a
    // re-measurement of the scene (`shallot-perf-gates` stage 4e). Every row here rests on TWO independent
    // agreeing samples, not one: the harvest `bun bench --sweep` that produced these numbers finished
    // 22:17, the numbers were written here at 22:29, the temporary harvest check (`assertBudget`'s
    // `pass: false` instrument) was removed at 22:30, and a second sweep ran 22:31–22:37 — clean 57/57
    // against these already-final goldens through the real exact-equality checks, a verifier confirming
    // the harvest's numbers, not a second producer of them. `backend`, `render`, `character`, `motor`,
    // `sat`, `stress`, `chain` additionally carry two independent same-day `--scenario` confirmation runs
    // from the stage 4c floor (since each was previously byte-exempt), so those 7 rows rest on FOUR
    // independent agreeing samples.
    accel: { pipelines: 44, gpuBytes: 34_574_780 },
    backend: { pipelines: 29, gpuBytes: 12_723_708 },
    "bodies-body-type": { pipelines: 30, gpuBytes: 13_519_028 },
    "bodies-motion-locks": { pipelines: 30, gpuBytes: 13_516_692 },
    "bodies-spinning-book": { pipelines: 30, gpuBytes: 13_517_860 },
    chain: { pipelines: 26, gpuBytes: 29_151_084 },
    character: { pipelines: 66, gpuBytes: 21_761_544 },
    "character-mover": { pipelines: 30, gpuBytes: 13_522_532 },
    "collision-overlap-box": { pipelines: 30, gpuBytes: 13_517_860 },
    "collision-ray-curtain": { pipelines: 30, gpuBytes: 13_551_228 },
    "collision-shape-cast": { pipelines: 30, gpuBytes: 13_617_292 },
    "compound-simple": { pipelines: 30, gpuBytes: 13_615_740 },
    "compound-spheres": { pipelines: 30, gpuBytes: 13_870_228 },
    "compound-tile-floor": { pipelines: 30, gpuBytes: 13_517_860 },
    constraints: { pipelines: 66, gpuBytes: 21_765_288 },
    "continuous-bullet-vs-stack": { pipelines: 30, gpuBytes: 13_538_604 },
    "continuous-thin-wall": { pipelines: 30, gpuBytes: 13_552_060 },
    "determinism-falling-ragdolls": { pipelines: 30, gpuBytes: 14_848_836 },
    "events-hit": { pipelines: 30, gpuBytes: 13_674_468 },
    "events-joint-break": { pipelines: 30, gpuBytes: 13_517_860 },
    "events-sensor-sweep": { pipelines: 30, gpuBytes: 13_519_028 },
    "geometry-convex-hull": { pipelines: 30, gpuBytes: 13_542_972 },
    "geometry-convex-primitives": { pipelines: 30, gpuBytes: 13_525_908 },
    "geometry-hull-reduction": { pipelines: 30, gpuBytes: 13_523_700 },
    gltf: { pipelines: 74, gpuBytes: 104_242_304 },
    "gpu-diagnostic": { pipelines: 4, gpuBytes: 32_788 },
    "joints-bridge": { pipelines: 30, gpuBytes: 13_520_196 },
    "joints-cantilever": { pipelines: 30, gpuBytes: 13_519_028 },
    "joints-driving": { pipelines: 30, gpuBytes: 13_596_164 },
    "joints-elevator": { pipelines: 30, gpuBytes: 13_519_028 },
    "joints-filter": { pipelines: 30, gpuBytes: 13_519_028 },
    "joints-paddle": { pipelines: 30, gpuBytes: 13_519_028 },
    "joints-parallel": { pipelines: 30, gpuBytes: 13_517_860 },
    "joints-pendulum": { pipelines: 30, gpuBytes: 13_517_860 },
    "joints-rope": { pipelines: 30, gpuBytes: 13_651_252 },
    "joints-suspension": { pipelines: 30, gpuBytes: 13_520_196 },
    "mesh-fixture": { pipelines: 32, gpuBytes: 31_828_492 },
    "mesh-terrain": { pipelines: 30, gpuBytes: 13_616_308 },
    "mesh-torus": { pipelines: 30, gpuBytes: 13_895_924 },
    motor: { pipelines: 66, gpuBytes: 88_873_964 },
    outline: { pipelines: 33, gpuBytes: 38_517_664 },
    pile: { pipelines: 66, gpuBytes: 26_173_716 },
    queries: { pipelines: 29, gpuBytes: 12_719_784 },
    "ragdoll-ragdoll": { pipelines: 30, gpuBytes: 13_664_708 },
    raining: { pipelines: 29, gpuBytes: 12_725_016 },
    render: { pipelines: 29, gpuBytes: 150_362_360 },
    rotation: { pipelines: 29, gpuBytes: 12_705_832 },
    sat: { pipelines: 69, gpuBytes: 322_682_880 },
    "shapes-inclined-plane": { pipelines: 30, gpuBytes: 13_519_028 },
    "shapes-restitution": { pipelines: 30, gpuBytes: 13_908_212 },
    "shapes-shape-soup": { pipelines: 30, gpuBytes: 13_749_084 },
    sprite: { pipelines: 49, gpuBytes: 80_163_828 },
    "stacking-arch": { pipelines: 30, gpuBytes: 13_537_716 },
    "stacking-box-pyramid": { pipelines: 30, gpuBytes: 13_517_860 },
    "stacking-dominoes": { pipelines: 30, gpuBytes: 13_660_940 },
    stress: { pipelines: 31, gpuBytes: 1_089_137_076 },
    text: { pipelines: 29, gpuBytes: 17_068_072 },
};

/** every axis explicitly exempted from a compile/memory budget, and why — keyed by scenario, one optional
 *  reason per axis. A scenario/axis pair appears in exactly one of this table or `SCENARIO_BUDGETS`, never
 *  both, checked in `budget-coverage.test.ts`. Empty since stage 4e (`shallot-perf-gates`, 2026-08-05):
 *  every scenario previously exempt on `gpuBytes` was a lazily-grown pool (`Mirror`'s ring or `Slab`'s
 *  staging pool) under real GPU backpressure, and excluding those bytes at the allocation site made the
 *  remaining total exact on all 7 — see the history note on `SCENARIO_BUDGETS`. Kept non-empty-shaped (the
 *  type, not a value) because the mechanism this axis excludes is scoped to two known pools, not to "no
 *  scenario can ever disagree" — a future lazily-grown pool that isn't marked `LazyAlloc.lazy` yet would
 *  reintroduce a disagreeing row here, not a silent gate weakening. */
export const BUDGET_EXEMPTIONS: Record<string, AxisExemption> = {};
