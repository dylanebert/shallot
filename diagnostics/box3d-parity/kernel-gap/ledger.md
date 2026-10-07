# Kernel-world stage 11 — the remaining kernel gap

Extends `../stage12/ledger.md`. Investigation on product revision `de5b8ebe85f165f0a54ea6fb5a3877c25e3d7be1`, measured 2026-10-06/07 UTC. **No product files or shipping artifacts changed.** Proposed work below is not an adopted stage.

## Reproduce

Set `BOX3D` to a checkout at `47d7f7cc7e091142c08d11dc7d2e493c5d34f536`.

```sh
bun diagnostics/box3d-parity/kernel-gap/build.ts HEAD
bun diagnostics/box3d-parity/kernel-gap/build.ts HEAD --counts
bun diagnostics/box3d-parity/kernel-gap/build-native.ts
# Use the DIAGNOSTIC_ROOT paths printed by the first two commands,
# and the NATIVE_COUNTS binary printed by the third:
bun diagnostics/box3d-parity/kernel-gap/measure.ts PROFILE_ROOT COUNT_ROOT NATIVE_COUNTS junkyard
# Replace junkyard with rain or joint_grid.
python3 diagnostics/box3d-parity/kernel-gap/report.py
```

Builds are setup, not included in the per-scene budget: compiling both Rust toolchains and rebuilding native may exceed a minute. Each build works in a temporary archive; dependencies are linked, never copied into the product. Delete those archives after use. The final complete scene commands took **29.251 s junkyard, 15.749 s rain, 7.779 s joint_grid**, including 1 and 4 threads, native and kernel profiles, both counted runs, native named timers and all hash comparisons. No speed assertion gates them.

The profiling build keeps function names with `CARGO_PROFILE_RELEASE_STRIP=false`, `CARGO_PROFILE_RELEASE_DEBUG=line-tables-only` and Binaryen `--debuginfo`; otherwise it uses the existing build's release flags and optimizer passes. The counting build instruments only its archive. Normal step timings never come from counting builds. The `build.ts` revision argument also accepts historical revisions; `ab.ts BEFORE_ROOT AFTER_ROOT SCENE THREAD LABEL` runs unsampled B A B A B A. `times.ts` runs native named timers alone.

For the carried-cost comparison, build profile archives at `eab6e033`, `7fb221fa`, `319ff2be` and `a603047f`. Run `run.ts ROOT SCENE` with `EVIDENCE_DIR` set to the corresponding `pairs-before`, `pairs-after`, `rain-before` or `rain-after` directory here. `code.py PAIRS_BEFORE_ROOT PAIRS_AFTER_ROOT RAIN_BEFORE_ROOT RAIN_AFTER_ROOT` disassembles both artifacts with `wasm-dis` and regenerates the code summary. For tiering evidence, after bundling through `run.ts`, run Node with `--trace-wasm-compilation-times` on `PROFILE_ROOT/diagnostic-bundle/scenes.js` for junkyard/1/200 and joint_grid/1/40, concatenate their compilation lines, then pass the single-thread optimized WASM and that log to `tiering.py`.

Hardware: Apple M4 Max, 64 GiB, arm64, macOS 26.7 (25G229). Node v26.10.0; Bun 1.4.2; stable rustc 1.98.1 (48a229cea 2026-09-01); shared rustc 1.100.0-nightly (a36d05efa 2026-09-09), pinned nightly-2026-09-10; pinned wasm-opt 132; diagnostic wasm-dis 131; Apple clang 21.0.0 (clang-2100.3.34.2), native CMake Release/default SIMD and harness `-O2 -ffp-contract=off`. Load averages observed near the end were 3.69/3.65/3.78; other workloads were not stopped. These are same-machine observations under load, not unloaded limits or browser measurements.

## Evidence and boundaries

- `tables.md` records **every profile field**, every counted category and every sampled named kernel routine, separately for each scene/thread count. It includes native named function times/call counts and the narrowest attribution for each routine. `measurements.json` is the same evidence as plain data.
- `*-kernel.txt` has main-thread self (`KS`) and inclusive (`KI`) samples, plus the three workers' raw inspector profiles (`WP`) for 4 threads. The table sums main and worker kernel samples; this is **CPU occupancy per step, not parallel wall time**. Main samples are restricted to `PhysicsWorld.step`, excluding hash/getter observations. Workers sample only the measured window; their final extra, unreported step stops sampling before doing further kernel work. Requested sampling interval is 100 µs, not a promise of that resolution. Zero samples mean unresolved, not free.
- `*-native.txt` and `F` rows in normal kernel output are aligned `b3Profile` wall timings. Windows remain rain 280–319, junkyard 180–199, joint_grid 20–39. Every reported step, including warm-up, hashes equal native in normal, counted and named-timer runs, at both thread counts.
- `*-counts-*.txt` reports counters and all 24 graph color occupancies. All geometric counters, colors and post-step graph-resident manifold points per color (`P`) match **at every measured step**, not just their medians. Tree query calls, node/leaf visits and gathered rebuild leaves also match at every measured step. Copies/initializers intentionally differ.
- `*-times-native.txt` supplies 22 named native timers, inclusive and self elapsed call-body sums, and invocation counts. These are a **separate instrumented experiment**, not the native production function throughput. Thread-local accumulation avoids a shared atomic on every timed return. Nested timer overhead is tracked; a 31-batch empty-timer median estimates remaining clock bias, printed as `B`. Subtraction clamps at zero. This is not a cure for timer-induced register pressure, code-layout changes, preemption or short-function uncertainty. In particular, do not derive a SIMD/support rate from these tiny timed bodies or divide their 4-thread elapsed sum by a wall-time phase. Classification uses the uncounted phase/substage envelopes and exact work counts; the named timers narrow attribution, not a new speed gate.
- Inclusive samples/timers overlap. Medians of different fields or counters need not add. In particular, the measured junkyard median SAT misses is **7,477.5**, not the subtraction of median calls and median hits used as a rough denominator in the earlier ledger.

### Counter definitions

`hull_sat`, hit and miss count Box3D's hull–hull SAT invocations/cache outcomes, including early returns. They deliberately follow `contact.c`'s SAT counter, not every triangle or distance query. `support_scalar`, `support_wide`, `support_face` and `triangle_support` count invocations, not vertices visited. `face_axes` counts faces evaluated by the full hull SAT search; `edge_candidates` counts real edge-pair lanes tested by its Gauss/tolerance gate, excluding padded lanes. Cached-axis work is additionally bounded by SAT outcomes and support calls; these fields are not relabeled as projected valid-edge axes.

`clip_input` counts vertices processed across planes; `clip_output` counts vertices emitted across planes. `clip_final` counts the polygon surviving the complete clip loop **before** capacity/speculative filtering/reduction, or the accepted segment's endpoints. Both sides include hull polygon clipping, hull/capsule segment clipping, hull/triangle polygons and capsule/triangle segment clipping. The three scenes do not exercise triangle clipping: rain's sphere/triangle path does not clip. Its zeros are not evidence that triangle clipping is uninstrumented.

`contact_points` counts newly emitted manifold points, including mesh-cluster results; it does not recount recycled manifolds. `P` separately counts resident manifold points per post-step graph color, including recycled points, with medians 20,529 junkyard, 2,167.5 rain and 0 joint_grid. This is a post-step snapshot, not a claim that points on constraints put to sleep during finalization did no solve work. Colors count scalar + convex contacts + joints per graph color. Consequently authoring totals are not solver-work denominators: joint_grid has 19,800 joints but **19,701 prepared**, 78,804 warm-start invocations and 157,608 solve/relax invocations per measured step. Rain's median prepared joints is 3,724, not its 4,200 authoring total.

`clip_copy_points` counts explicit caller prefix copies, or the two array contents exchanged by Rust's triangle buffer swap; native swaps pointers instead. `clip_zero_points` counts source-level initialized buffer slots. It is **not** measured write traffic: optimizers may remove initializations. `tree_queries`, `tree_nodes`, `tree_leaves` follow `b3DynamicTree_Query`'s returned statistics, including early termination; `rebuild_leaves` counts gathered rebuild roots/leaves, not all proxies.

## Per-scene findings

Exact normal timings, substages and named-function measurements are in `tables.md`; do not substitute timer-perturbed native data for its first table.

### Junkyard: extra clipping work, then a bounded dispatch/code-generation residual

The two runs do the same geometric search: medians 46,827 SAT calls, 39,404.5 hits, 7,477.5 misses, 145,008.5 full-search face axes, 2,038,040 edge-pair candidates, 43,698.5 scalar and 145,008.5 wide support calls. Both process 103,938.5 clipping inputs, emit 109,556.5 intermediate vertices, retain 36,886 final clipped vertices, and emit 19,457.5 contact points. Tree work also matches: 35,108.5 queries, 740,376 nodes, 83,841 leaves and 10,354.5 rebuild leaves. These medians are the same at 1 and 4 threads; equality assertions apply per step.

**More work:** `manifold::build_face_a_contact` copies each clipped prefix from scratch back to input; Box3D `b3BuildFaceAContact` swaps its two buffer pointers. Kernel median **109,556.5 copied points versus native 0**, or 2,191,130 logical copied bytes at 20 bytes/ClipVertex. Rust also initializes 1,333,184 clip-buffer slots per step versus native's uninitialized scratch buffers; that is a source-level upper bound of 26,663,680 initialized bytes, not a measured bandwidth claim. The original uncounted optimized WAT for the face builder retains `memory.copy` operations. Triangle builders swap owned `[ClipVertex;128]` contents rather than references; that additional work is named but contributes zero in these scenes. This extra work belongs to the face builders and their `collide_hulls`/`contact_block` ancestors, **not to the body of `clip_polygon` itself**.

**Same counted work slower:** pair traversal/rebuild; there is no hidden extra query/visit count. Within collide, `collide_hulls`, its full-axis search (inlined), support and clipping have identical geometric counts. Kernel self samples put the largest residual in **`arena::contact_block`**, followed by `manifold::collide_hulls` and `build_face_a_contact`; the table records their CPU time and native counterparts. Contact-block self includes inlined dispatch, recycling, resident manifold handling, material/default branches and writes. Its whole residual has not been divided into invented per-contact costs. The narrowest remaining point, after the counted clipping copies, is this optimized dispatch body and the inlined full-axis search. Native short-helper timers do not demonstrate that support/clip arithmetic alone explains the gap. Resident solver points match as well (20,529 median), so newly emitted-point equality is not substituted for the solver's recycled-point population.

Prepare has a positive wall-time residual at 1 thread. Warm-start, solve and relax contact substages have **no demonstrated material slowdown** in these windows; some are faster than native. Restitution is tiny. Store is a small same-count residual. Finalize (`transforms`) remains slower at both thread counts, with continuous work and tree enlargement sampled and equal query/rebuild work; the rest is bounded by finalization in `stages::execute_block`/the body finish tail. No complete finalize arithmetic cause was established. Thus the remaining solve envelope is not a generic contact-impulse throughput failure.

### Rain: same geometric and constraint work, slower execution/finalization

SAT/full hull axes/support/clip counters are zero on both sides. Newly emitted points are 358 median per step on both sides, including mesh clusters. Tree medians match: 10,714.5 queries, 142,377 nodes, 15,621.5 leaves and 2,938 rebuild leaves. All color occupancies and resident graph points (2,167.5 median total) match; no extra clipping copies are executed.

Pairs are **same work slower**, narrowed to `tree::query` and the native-form serial rebuild/callback envelope. Collide is **same counted geometry slower**, narrowed to `arena::contact_block`'s mesh dispatch/recycle/resident-contact path. A profile does not make its unisolated mesh cache/dispatch details a demonstrated WASM arithmetic penalty.

Prepare, warm start, solve and relax are **same counted constraint work slower**. `joint::solve` dominates sampled impulse CPU time; `joint::prepare_world` and `joint::warm_start` identify the other two routines. The counts/color ordering and native joint invocation counts rule out extra joints or passes as the explanation. The narrowest solve point is the compiled joint routine, not old TS marshalling. Restitution/store are small, with no material independent fixing brief. Finalize is slower; `continuous::solve`, tree enlargement and the inlined stage finalization are the named bounds. Geometry counts do not separate all finalization instructions, so no finer cause is claimed.

### Joint_grid: same joint work slower; collide absent

All geometric/contact counters are zero; colors and native joint invocation counts match the work described above. Tree medians match: 7,950 queries, 83,102 nodes, 2,650 leaves and 3,107 rebuild leaves.

Prepare, warm start, solve and relax are **same work slower**, narrowed respectively to `joint::prepare_world`, `joint::warm_start` and `joint::solve` (the latter includes biased and unbiased solves). Finalize has a smaller same-work residual. Restitution/store have no contact work and no independently demonstrated gap. Pairs are same query work slower, particularly visible at 4 threads on this small workload. There is no collide gap to assign.

`tiering.json` records an additional unprofiled V8 compilation trace: the hot single-thread query, contact block, hull collision, support, clip and joint solve functions all reached **TurboFan**, not only Liftoff. Contact block is 49,987 WASM body bytes / 76,420 optimized machine-code bytes; joint solve is 47,419 / 56,508. These are sizeable emitted bodies, not proof that size caused their timing. The investigation therefore stops at optimized function/code-generation shape; it does not blame an interpreter or invent a universal WASM factor.

### Other sampled functions and substages

The function inventory in `tables.md` keeps all observed symbols, including stage waits, graph/layout, event, sensor, manifold-write, distance/TOI and tree helpers. Where a native timer does not isolate the same body, the enclosing native phase is recorded instead of assigning it fabricated exclusive time. Wait/orchestrator samples cannot be compared to native's arithmetic-only execute-block timer as though their boundaries matched. Equality of graph/point/query work identifies these as coordination or same-work residuals, not evidence of more constraints. Small/unresolved samples have **no demonstrated independent gap**. The two gap classes apply to demonstrated gaps; a helper with no demonstrated slowdown is not falsely labeled slower merely because its parent is slower.

Solver setup/integration/refit and all smaller `b3Profile` fields are retained in the tables. Split-island, sensor-hit, joint/hit-event, bullet, sleep and sensor values below measurement precision are not assigned an optimization brief. The scene data do not establish a material independent gap there.

## Carried costs

### Pairs across `7fb221fa`

Compared `eab6e033` (parent) with `7fb221fa`, both named single/shared builds; same junkyard window. The symbolized run reproduced pairs 6.9096 → 7.6871 ms at 1 thread and 2.9198 → 3.0162 at 4. The main-thread profile points at the specialized **`tree::query< pairwork::run_query closure >`**, not contact creation or a compound workload.

Unsampled B A B A B A, with every hash equal:

| threads | A pair medians ms | B pair medians ms | median B−A |
|---|---|---|---|
| 1 | 6.80045, 7.48930, 6.76260 | 7.20765, 7.39655, 7.24085 | +0.44040 (+6.48%) |
| 4 | 2.79145, 2.79575, 2.82310 | 2.89245, 2.83835, 2.84630 | +0.05055 (+1.81%) |

The one-thread cost persists in this comparison, but the earlier +0.77 ms is not a hardware constant. The earlier +0.25 ms at 4 threads is not reproduced at that magnitude. Counts/source of the actual junkyard pair work were unchanged by the upload migration; current native/kernel query counts now independently agree as well.

`generated-code.json` records the optimized WAT comparison. The specialized query's instruction sequence **changes**: single-thread WAT text 63,591 → 49,376 characters, load sites 57 → 45, call sites 21 → 14; shared 62,573 → 48,120, loads 55 → 43, calls 21 → 14. It contains the compound callback path even when no compounds execute. This is evidence of changed callback/inlining/code-generation shape, **not** evidence that fewer instructions must run faster or that a particular spill caused +0.44 ms. Narrowest point: this specialized query's generated body and its callback call graph. LLVM/Binaryen versus V8 register allocation/layout/cache effects were not separated. No speculative source optimization is presented as a closed cause.

### Rain across stage 7

Compared `319ff2be` with `a603047f`, matching stage 7's original A/B boundary. Symbolized 4-thread step reproduced 2.6134 → 2.7851 ms (+6.57%); pairs .5835 → .6516 and collide .4757 → .5363. At 1 thread there was no analogous whole-step increase.

Unsampled B A B A B A at 4 threads: A step medians 2.53055, 2.56740, 2.73350; B 2.56815, 2.64825, 2.62550. Median difference **+.05810 ms (+2.26%)**, with drift in A. Therefore this experiment does not establish a stable +5.7% penalty; it also does not erase the historical observation.

For five sampled hot bodies (specialized pair query, record_child, contact_block, joint::solve, solve::run_job), the before/after **opcode sequences are identical** in both artifacts. After normalizing large static-address immediates and memory offsets, the compared WAT bodies are identical. Inspected differences include globals moving by 320 bytes and relocated panic/static-data descriptors. This narrows the observation to relocation/layout/JIT scheduling/cache effects and run variability, not a demonstrated changed step instruction sequence in these bodies. Native V8 machine-code layout/cache-conflict experiments were not performed. No source-level fix for this observation is justified here.

## Proposed stage brief — remove clipping buffer traffic

**Not adopted.** Shallot owns this cause and can close it without changing geometry or solve order.

Move the hull face builders, and the triangle face builders' analogous buffer exchange, from owned-array copying/swapping to Box3D's two-buffer **reference ping-pong over initialized active prefixes**. Keep clip output, early-return order, speculative filtering, point reduction and feature ids unchanged. Do not construct references to uninitialized `ClipVertex` values. Inspect and remove only scratch initialization demonstrated to survive code generation; the source-level zero counter alone does not justify unsafe changes.

Evidence: junkyard's geometric clip counts and every world hash match, while Shallot copies 109,556.5 extra clip vertices per median step and Box3D copies none. The original optimized face-builder WAT retains copies. The two face builders are sampled within the collide residual. Native `convex_manifold.c:1005–1042` swaps buffer pointers; Rust `manifold.rs:1515–1542` copies the active prefix, and `triangle_manifold.rs` swaps owned arrays.

**Check:** native/kernel clip inputs, intermediate outputs, final survivors, SAT/support/axis outcomes and contact points remain identical at every measured step for all three scenes at 1 and 4 threads; active hull/triangle and capsule/triangle fixtures prove both ping-pong orientations, clipped-away polygons, cached reconstruction and capacity/speculative filtering. `clip_copy_points` becomes zero on those paths, and generated code has no caller prefix copy/owned-array swap. Any initializer removal proves that only initialized prefixes are read. World hashes, manifold/feature golds, checked-kernel tests and `bun run check` pass. Interleave uninstrumented B A B A B A on junkyard collide at 1 and 4 threads before claiming a timing win; report any loss elsewhere. The stage closes counted extra work even if the full collide gap persists; it must not claim to explain all contact-block time.

## Check / disposition

Final complete diagnostic commands passed for each scene at both thread counts, including all hashes and counted geometric/color equality; tree query/rebuild counters also equal per step. `bun run check` passed on the final candidate; the commit records the repeat. The default product and its generated kernels stayed untouched. No speed number gates the unit. Extra clipping traffic is briefed above; optimized dispatch/joint/finalization and the carried code-generation/layout costs remain named, bounded points rather than invented platform penalties.
