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
    // measured 2026-09-01, `--scenario cells` (WSL/NVIDIA bridge, lovelace), re-measured the same day
    // after S3r item 2's ramp-monotonicity arm landed (`specs/shallot-tui.md`'s s3r item 2 — the owed
    // regression guard driving `draw.ts`'s real pipeline against a real font atlas, not the synthetic
    // solid atlas the draw differential below uses), again after the s3r item-8 background-detection
    // repair, which grew `SelectParams` (`select.ts`) by one `bg: vec4f` reference (16 B), and again after
    // criterion 9's arm (`assertFrameIsGrid`, `FramePlugin` below `MonoPlugin` in this file) — the two-
    // sided proof that `drawCells` replaces a target's prior contents rather than compositing over them
    // (`specs/shallot-tui.md`'s s3r item 9). Nine compute/render pipelines: `cells-fill` (S1's synthetic
    // producer), `cells-avg` + `cells-select` (S3's two-pass real content producer), `cells-draw` (the
    // draw differential's real render pipeline, shared by the mono-ramp arm and criterion 9's own
    // `FramePlugin` draw — reused under the same label three times over, which is why `pipelineCalls` (11)
    // exceeds `pipelines` (9), `budgets.ts`'s own module doc names this), `cells-gym-draw-probe` (the draw
    // differential's texture readback), `text-sdf-distance` + `text-sdf-finalize` (the real `GlyphAtlas`'s
    // SDF generator, first exercised here against a real font — `extras/text/sdf.ts`, warmed by
    // `buildGlyphUvTable`/`buildGlyphSizeTable`), `cells-gym-mono-probe` (the ramp arm's own per-cell
    // ink-average readback), and `cells-gym-frame-probe` (criterion 9's own sentinel-pixel counter, one
    // pipeline dispatched twice — against the drawn target and the undrawn control — which is the other
    // `pipelineCalls` unit past the reused `cells-draw`: 9 (prior pipelineCalls) + 1 (an extra `cells-draw`
    // create, criterion 9's own `drawCells` call) + 1 (`cells-gym-frame-probe`'s single create) = 11).
    // gpuBytes excludes the lazy `Mirror` staging pool (1_492 B, `mirror-staging`) — every other
    // `bun bench --scenario cells`-reported label, summed exactly: `cells-grid` 2_316 — four scenarios'
    // grids share this one label: `CellsPlugin`'s `COLS × ROWS` (10×6, 60 cells × 12 B/cell = 720),
    // `SelectPlugin`'s `SELECT_COLS × SELECT_ROWS` (8×4, 32 × 12 = 384), `DrawPlugin`'s `DRAW_COLS ×
    // DRAW_ROWS` (2×1, 2 × 12 = 24), `MonoPlugin`'s `MONO_GLYPH_COUNT × 1` (`= CELL_FILL_GLYPHS.length`,
    // 87 × 12 = 1_044, down from 91 × 12 = 1_092 pre-rule-2), and `FramePlugin`'s `FRAME_COLS × FRAME_ROWS`
    // (4×3, 12 × 12 = 144) — 720 + 384 + 24 + 1_044 + 144 = 2_316 — + `cells-select-src` 8_192 +
    // `cells-select-params` 32 (16 B dims uniform + the item-8 `bg: vec4f` reference above) +
    // `cells-select-avg` 512 + `cells-gym-draw-glyphuv`
    // 32 + `cells-gym-draw-glyphsize` 16 + `cells-gym-draw-atlas` 16 + `cells-gym-draw-target` 2_048 +
    // `cells-gym-draw-probe-out` 32 + `glyphAtlas` 4_194_304 (a real `createGlyphAtlas` 2048×2048
    // `r8unorm` texture, the same size every real-font cells/text scene allocates) + the SDF generator's
    // unlabeled 96×96 `rgba8unorm` intermediate texture 36_864 (`""` in `bun bench`'s by-label memory
    // breakdown) + `cells-glyph-uv` 1_456 + `cells-glyph-size` 728 + `cells-gym-mono-target` 200_448
    // (`2088×24 rgba8unorm`) + `cells-draw-params` 16 + `cells-gym-mono-out` 348 + `cells-gym-frame-glyphuv`
    // 32 + `cells-gym-frame-glyphsize` 16 + `cells-gym-frame-atlas` 16 + `cells-gym-frame-drawn` 12_288
    // (64×48 `rgba8unorm`) + `cells-gym-frame-control` 12_288 (same size, the undrawn sentinel control) +
    // `cells-gym-frame-out` 8 (two 4 B atomic-u32 counters sharing that label) = 4_472_008, matching
    // `gpuBytes` below exactly.
    //
    // Re-measured 2026-09-01, same day, after the s3r fill-treatment amendment (`specs/shallot-tui.md`):
    // rule 2's curated curved-glyph exclusion (`ramp.ts`'s `CELL_FILL_EXCLUDED_GLYPHS`) shrinks
    // `CELL_FILL_GLYPHS` from 91 to 87 and `MONO_GLYPH_COUNT` (this file, `= CELL_FILL_GLYPHS.length`)
    // with it, moving five labels that scale off the fill-ramp length: `cells-grid` (-48 B, `MonoPlugin`'s
    // own grid), `cells-glyph-uv` (-64 B, 16 B/glyph), `cells-glyph-size` (-32 B, 8 B/glyph),
    // `cells-gym-mono-target` (-9_216 B, `MONO_CELL_PX²` × 4 B/glyph) and `cells-gym-mono-out` (-16 B, 4
    // B/glyph) — a net -9_376 B, `4_481_384 → 4_472_008`. `pipelines`/`pipelineCalls` are unaffected (no
    // pipeline count changed, only buffer/texture sizes); rule 1's threshold re-derivation and rule 3's
    // now-retired `INK_DILATE_FRACTION` changed gpu *behavior*, not any allocation size, so neither moved
    // this row at that measurement. Two independent same-day confirming runs agreed exactly on `gpuBytes`.
    //
    // Re-measured 2026-09-02, after the fill-treatment amendment's s3r review repair
    // (`specs/shallot-tui.md`): `draw.ts`'s `INK_DILATE_FRACTION` is deleted outright (a behavior-only
    // change, so it still moves nothing on its own) and the facade-ink measurement (rule 3) moves off the
    // old per-glyph-average sweep onto a real per-pixel device readback — `examples/gym/src/scenarios/
    // cells.ts`'s new `FacadePlugin`, added to this scenario's plugin list between `MonoPlugin` and
    // `FramePlugin`. It builds its own real `GlyphAtlas` (independent of `MonoPlugin`'s own, so every
    // label a font-atlas load produces is now counted twice) and renders three FACADE_COLS×FACADE_ROWS
    // (12×8 = 96 cells) grids — the facade band, a densest-glyph control, a blank-glyph control — each its
    // own draw target and atomic-counter probe output.
    //
    // `pipelines` 9 → 10: one new label, `cells-gym-facade-probe` (`FacadePlugin`'s own threshold-count
    // kernel, `FramePlugin`'s `cells-gym-frame-probe` shape applied to a luma threshold instead of a
    // sentinel distance).
    //
    // `pipelineCalls` 11 → 13: `FacadePlugin` calls `resetDrawPipeline()` once (like every other draw-
    // pipeline consumer in this file) before its three `drawCells` calls, so `cells-draw` is re-created
    // once for all three (the pipeline memoizes across calls until the next reset) — +1; plus one create
    // of the new `cells-gym-facade-probe` pipeline — +1. 11 + 1 + 1 = 13.
    //
    // `gpuBytes` 4_472_008 → 9_372_380 (excl. `mirror-staging`, unchanged at 3_008 B). Every delta traces
    // to `FacadePlugin` building its own atlas/tables/grids rather than reusing `MonoPlugin`'s:
    // `glyphAtlas` doubles (4_194_304 → 8_388_608, a second real 2048×2048 `r8unorm` atlas texture), the
    // SDF generator's unlabeled 96×96 `rgba8unorm` intermediate texture doubles (36_864 → 73_728, a second
    // atlas's own SDF-generation scratch texture), `cells-glyph-uv` doubles (1_456 → 2_912) and
    // `cells-glyph-size` doubles (728 → 1_456) — both hardcoded labels in `glyphs.ts`, so a second
    // `buildGlyphUvTable`/`buildGlyphSizeTable` caller sums under the same label rather than a new one —
    // and `cells-grid` grows by 3 × (FACADE_COLS × FACADE_ROWS × 12 B/cell) = 3 × 1_152 = 3_456 B (2_316 →
    // 5_772, the three facade grids sharing that same label alongside the other four scenarios' grids).
    // Three new distinct-labeled entries: `cells-gym-facade-band`/`-ceiling`/`-floor`, each
    // FACADE_VIEW_W × FACADE_VIEW_H × 4 B = 288 × 192 × 4 = 221_184 B (rgba8unorm), and
    // `cells-gym-facade-out`, 3 × 4 B atomic-u32 counters sharing that label = 12 B. Net delta:
    // (8_388_608 − 4_194_304) + (73_728 − 36_864) + (2_912 − 1_456) + (1_456 − 728) + (5_772 − 2_316) +
    // 3 × 221_184 + 12 = 4_194_304 + 36_864 + 1_456 + 728 + 3_456 + 663_552 + 12 = 4_900_372;
    // 4_472_008 + 4_900_372 = 9_372_380, matching `gpuBytes` below exactly. Two independent same-day
    // confirming runs (`bun bench --scenario cells`, this seat) agree exactly.
    cells: { pipelines: 10, pipelineCalls: 13, gpuBytes: 9_372_380 },
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
    // The two exhaustive trig probes add device-only pipeline calls: FFT twiddles and the slope
    // phase at declared nonzero time. TypeGPU gives the phase probe an anonymous label already
    // represented by the existing `pipeline` label, so it increments calls without label count.
    "ocean-slope": { pipelines: 24, pipelineCalls: 25, gpuBytes: 40_808_844 },
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
