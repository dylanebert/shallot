---
paths:
  - "packages/{shallot/src/standard/avbd/**/*.ts,shallot/tests/avbd/**/*.ts}"
---

# AVBD Backend

GPU backend: `AvbdPlugin`; escape: `Avbd.step`. `physics.md` owns the substrate, `tumble.md` the default. The committed f64 `tests/avbd/` oracle ports C++ operation-for-operation; golds need no external checkout. Build the augmented-Lagrangian ladder without skipping layers.

## The oracle is the spec

GPU is wrong until ruled out: the oracle is not the suspect. Never lower it to f32. Physics changes run `bun test ./packages/shallot/tests/avbd/*.oracle.ts`; real-device correctness/compile/perf use gym `pile`, hull kernels use `sat`. `headless.tier.ts` is by-path build/lifecycle evidence, never solver parity; `testing.md` owns device-execution exceptions.

Gate closed forms, then identical-start single-step GPU/oracle equality at derived tolerance; statistical energy/penetration/finite bands only for chaotic long horizons. Test hard topologies isolated near origin; diagnose coordinate cancellation by translation. Iterations tune performance, never relax parity: all six corpus topologies and gym gates stay at 10 independently of shipping iterations; per-step math must agree at any count.

## Reference fidelity

Preserve the oracle's contact set and expression order. Sphere filtering plus padded box-AABBs must be a SAT-equivalent superset. Derive speculative distance as four collision margins, distinct from equilibrium offset. Sweep axis/clip bands use max(static skin, closing displacement), not their sum; derive linear displacement from velocity × actual dt, never 1/60. Pad both motion bounds, static skin prim-only; angular sweep is unsupported.

Pair A is the higher creation index. SAT feature keys remain bit-identical body-local clip ordinals, never storage ranks. Jolt's verbatim reduce-to-four is independently gold-gated, without sorting/re-ordinaling. Preserve margin-shifted equilibrium, full versus adaptive gravity, active quaternion matrices, right-multiplied angular difference versus left-multiplied integration, BDF1 recovery and one cross-coupled 6×6 LDLᵀ solve. Friction ramp uses pre-clamp tangential magnitude; retain sliding parity, not merely energy/static-hold tests.

Rounded/hull contacts are producers, never solver rewrites; use analytic closest points/SAT, not GJK/EPA. Box×box keeps its validated analytic path; other polytopes share scaled hulls. Capsules clip their core segment, not endpoint samples. Rounded pairs always refresh arms but retain penalty/dual warmstart; store core arms and apply radius along the fixed normal in force AND initial-gap reconstruction. Keep free-spin-relative conservation tests; don't alter the reference friction Hessian to cancel BDF1 damping.

Hull tests require box-as-hull manifold equivalence, independent Bullet non-box separating-axis golds, packing roundtrip, production WGSL/oracle parity and full-pipeline rests. Rounded gates: rolling/spin, overhang, core arms, confined rests. Hull inertia remains an AABB approximation until non-box dynamics needs registration-time principal-axis polyhedral integration. Narrowphase pipelines partition shape classes exhaustively/disjointly; box owns invalid-slot lifecycle. Preserve `collide.ts`'s Metal-safe split and measure DXC compile on Windows, min-over-three; no constant-bound dynamic-break unrolling or duplicated SAT bodies.

## Storage and solve

Body/contact storage stays SoA, eid-indexed and persistent; pack deterministically sorts live eids, seeds only new/recycled slots and never CPU-iterates entities. Fixed per-owner pair blocks keep in-place warmstart by pair identity + feature; clear separated/invalid records. No global compaction/hash/cache pass. Overflow keeps nearest neighbors with static support pinned and fails loudly through counters; diagnose fast-approacher misranking. `checkContactStore` must report needed/available limits and remedy.

Keep source-agnostic tagged contact generation separate from solve. New constraints reuse merged adjacency/record storage: primal, coloring and LDS solve cannot exceed ten storage bindings. GPU stays warmstart-only; oracle retains penalty/dual/warmstart layers. Colored GS commits double-buffered within each color, never in-place races, global Jacobi or Chebyshev. Static bodies skip primal/velocity; all-static manifolds never ramp. Never repurpose solver initial poses for kinematic render interpolation.

Regimes share math/warmstart; gate both threshold directions from regime state, not stale spans. Keep cold-seeded colors/safe initial bounds; stale margins don't excuse parity. No per-color advance dispatch/pass: one pass per iteration, primal/commit pairs. Price frame/encode cost + dispatch count, not solve timestamps. Quantization follows `gpu.md`; subgroup solve needs a high-valence win. No megakernel without removing dependent storage round trips; no third tail threshold. Reprice tail only on a Metal budget breach.

## Joints and grab

Unchanged joint slots preserve live warmstart, activity and moved anchors. Seed/version-gate initialization; reject unsatisfiable endpoints every frame as a persistent gauge, reach only on fresh construction. World anchors have no body/contact/adjacency; grab uses a soft spherical dangle. Anchor motion updates only its record lane, after one fixed tick at the hit point. Pick retains static occluders and excludes the caster. Keep re-author, radius-aware reach, damping, coupling and occlusion gates.
