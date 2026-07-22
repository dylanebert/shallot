---
paths:
    - "packages/shallot/src/**/*.test.ts"
    - "packages/shallot/tests/**/*.ts"
---

# Testing

Shallot-specific testing patterns.

## GPU testing

Two surfaces, one source of truth each, no overlap. The boundary is one question: does the assertion's truth depend on the real GPU?

- **Logic — `bun test`.** Hardware-invariant correctness: WGSL structure, buffer layout/stride/offset, bind-group-layout construction, packing/encoding, ECS, audio DSP, physics math, plus CPU cross-checks of GPU algorithms (a debugging aid for sorting/BVH/spatial-hash, not parity). No device needed. `tests/setup.ts` preloads `bun-webgpu` globals so `build()` imports cleanly — an import shim, not a device under test. The preload is wired in the **root** `bunfig.toml`, so run `bun test` from the repo root; from `packages/shallot` the path doesn't resolve and modules die at import with `GPUShaderStage is not defined`. In WSL the only adapter is software (Mesa llvmpipe; dzn, the real-GPU path, is non-conformant and Dawn rejects it), so `bun test` is not a real-GPU signal. **Never bind a device in `bun test`** — the lavapipe adapter is slow, flaky, and feature-incomplete (no usable subgroups/timestamp, `build()` can time out), so a device-binding unit test is unreliable, not a signal ("GPU behavior lives in Playwright" below). Validate hardware-invariant GPU logic against a deterministic CPU oracle, or decouple it to `new State()` (no `build()`); don't assert hardware-variable behavior (subgroup width, f16, raster) here at all.
- **GPU behavior — `bun bench`.** The single source of truth for anything whose truth depends on the real device: pipeline/shader compile, bind-group validity, compute-kernel output (Mirror readback), raster, graphics-stage storage, hardware-variable behavior (subgroup width, f16), perf, visual. Playwright drives the real GPU via the gym example. Assert each GPU behavior exactly once, here — even where `bun-webgpu` binds a real adapter (native Mac / Windows / hardware Linux), so the suite reads identically everywhere instead of depending on the contributor's GPU. It's the **targeted** tier: run one atom at a time (`bun bench --scenario X`) after its domain changes — never the whole set in one go (a separate Playwright session per scenario is slow + flake-prone under WSL/vite). The one legitimate batch is a *within-scenario* param sweep (`scripts/physics-bench.ts`'s multi-row run over a single physics atom), not a run-all.

Nothing is skipped and nothing false-fails, because no test lives in a tier that can't run it. Raster needs graphics-stage storage the software adapter lacks, so it's bench-only by construction — never a skipped unit test. A "false failure" is a tier violation: a real-GPU assertion that leaked into `bun test`. Move it to bench; don't skip it.

### GPU behavior lives in Playwright, never in `bun test`

The real-GPU truth lives once, in the gym (Playwright → real device). The `render` scenario carries the forward pipeline — its `cull` mode the draw path (frustum-cull survivor counts + the slab transport round-trip explicitly, the membership gate + Mirror readback load-bearingly, its assert reading through Mirror snapshots), its shaded-look modes the framebuffer probe; the `pile` / `constraints` / `character` scenarios carry the solver. A `bun test` that binds a device and reads a buffer back is **not** a tier to rely on: the lavapipe software adapter is slow, flaky, and feature-incomplete, so such tests flake by construction. `bun test` is pure CPU logic — validate hardware-invariant GPU logic against a deterministic CPU/JS oracle (the BVH/physics oracle pattern), or decouple it to `new State()` (no `build()`, no device).

A unit readback's finer edges — the survivor identity / per-view slot offset / per-type scatter / Mirror staging-ring orchestration the gym's coarser assert doesn't reach — move into the gym scenario (the consolidated atom, or a narrow real-device check alongside it), not a device-binding unit test.

**Preload — import shim, no adapter.** `tests/setup.ts` installs bun-webgpu's GPU enum constants (`GPUShaderStage`, `GPUBufferUsage`, …) so modules referencing them at module scope import cleanly. Run `bun test` from the repo root (`shallot/`), where the `bunfig.toml` preload resolves; from elsewhere, modules die at import with `GPUShaderStage is not defined`. bun-webgpu exposes a constants-only `globals()` (no adapter) alongside the adapter-installing `setupGlobals()` — once every pure-logic test that still bootstraps via `build()` (plugin/ECS/tween) moves to a headless `build({gpu:false})` mode or `new State()`, swap the preload to `globals()` to drop the lavapipe adapter entirely, so a stray device-binding `bun test` fails fast and loud instead of flaking.

### `requestGPU` policy

`requestGPU(device?, features?, preferred?)` enforces shallot's documented floor — `BASE_FEATURES` (shader-f16, timestamp-query, indirect-first-instance, bgra8unorm-storage) plus the required `features` the active plugins declare and `maxStorageBuffersPerShaderStage: 10`. Adapters that don't meet `base ∪ required` get an `UnsupportedError` naming the missing features; this is production behavior, not a test concession. `preferred` features (e.g. `subgroups`, declared by a BVH-building plugin like physics) are requested only where the adapter has them — never gating the device, so a no-subgroup adapter still acquires. The required/preferred split is the pure `resolveFeatures` (`engine/runtime/gpu.ts`), unit-tested with no device.

Split-stage limits (`maxStorageBuffersInVertexStage` / `InFragmentStage` and texture variants) are a recent WebGPU spec addition, so `deviceLimits` forwards each only when the adapter reports it — never hard-required. Older mobile WebGPU omits them entirely, and forwarding the `undefined` would reject `requestDevice` with a `NaN`-range error (see the `deviceLimits` comment). Mesa lavapipe reports 0 in WSL2 — present, so forwarded — which lets dev environments start cleanly while a raster consumer that actually needs graphics-stage storage fails later at pipeline creation with a clear Dawn message.

## Component CRUD tests

Test the **mechanism**, not every component. One test of the query pattern covers 15 components using the same pattern. Write a CRUD test only when:

- A bug was reported
- The component uses a different lifecycle pattern (lazy init, async pipeline)
- Setup runs once and might miss dynamic adds

## Reload tier

A live `State` is reused across hot reloads — a live host swaps a reloaded plugin onto it in place (`ecs.md` "Reload-safety", engine `swap()`). Reload-safety is a standing tier, mechanism-level like CRUD: one test per pattern, not per component or plugin.

- **Component identity — `bun test`.** Re-register a component with a fresh object of the same schema (what a module reload hands in) and assert membership, queries, and stored data survive by id. Mechanism test: `ecs/ecs.test.ts` "stable component ids".
- **Plugin swap — `bun test`.** A simulated `swap` preserves a runtime-set value and runs the reloaded system's new behavior on the same scheduler slot; a schema / system-set / ordering / feature change falls back to a rebuild. Mechanism test: `app/plugin.test.ts` "swap (hot reload)".
- **Rebuild-from-document — `bun test`.** The fallback a rejected `swap` hands off to: `serialize` the live State, `build` from it with the device reused (no page reload). A runtime value on an authored entity carries over; a `warm`-derived entity re-derives once, never doubled. Mechanism test: `app/plugin.test.ts` "rebuild from the serialized document …".
- **Failure recovery — `bun test`.** The throwing paths never wedge: a system whose `update`/`setup` throws is paused after one throw and resumes on its next swap (`ecs/scheduler.test.ts` "Failure Recovery"); a mid-swap `initialize` throw returns `ok: false` (the State is half-updated, so the rebuild fallback is the recovery); a swap against a build-skipped plugin is rejected via `App.skipped`; a failed `build()` disposes what it initialized, in reverse, then the State (`app/plugin.test.ts` "build failure").
- **Survive-reload — flows.** The browser end-to-end consumer: `examples/flows/survive-reload/`, an ejected app that self-drives a real `location.reload()` and, on the restored boot, installs the published `window.__harness` asserting a runtime value restored and a `warm`-derived entity wasn't doubled. `shallot verify` drives it (its unified wait polls across the self-navigation). The app composes survival from the core atoms (`serialize`→`sessionStorage` on unload, restore via `run({ scene })` on boot) — no engine flag; the engine stays unopinionated about storage/trigger. Display-gated, `bun run flows` (`scripts/flows.ts`). The in-place hot-swap e2e died with the editor — `swap()` coverage is the unit tests above.
- **Per-plugin conformance — `bun test`.** `tests/conformance.test.ts` runs each roster plugin through two identical build→step→dispose passes against the same module singletons and diffs the observable signature — a double-registering registry or a doubling `warm` spawn goes red. A plugin-toggle sequence (changed-set rebuild) additionally catches a producer toggled off leaving stale registry entries; a seeded non-idempotent fixture pins the harness itself (detail in the file header). Physics/Character/Player can't build on the bun-webgpu adapter — they join at the real-GPU tier when a rebuild-loop gym scenario exists.

**A plugin earns a roster entry the way it earns a CRUD or GPU assertion** — when it introduces a new reload-relevant pattern, not by default. The trigger is carrying runtime state across a swap (a stateful `warm` spawn, runtime-set component values), a module-level registry, or a novel swap / re-registration shape; a plugin whose state is fully scene-derived is already covered by the mechanism tests above. The harness replaces ad-hoc per-plugin reload tests — grow the roster, don't write bespoke ones.

## Playwright test structure

One browser session per test file. Don't split related assertions into multiple `test()` blocks — use phases within a single test instead. Starting a new browser session is expensive and loses page state. Reconstruct state within the session if needed.

## Pairwise testing for combinatorial GPU features

Most bugs come from 2-factor interactions. For combinatorial feature spaces (surfaces × pipelines × opaque/transparent × shadow × instance fields), generate pairwise test matrices (~40-60 combos instead of thousands). `compileSurfaceBlock` is a pure function — feed it pairwise inputs, validate structurally + GPU-compile.

## Structural validation over visual regression

Prefer: (a) structural validation of generated WGSL (expected functions, bindings, dispatch cases), (b) GPU compilation validation (shader compiles, pipeline creates without errors), (c) GPU readback validation (run the actual compute, read results back). CPU cross-validation is optional for isolated debugging. Reserve screenshot comparison for integration-level Playwright tests only. Structural tests are fast, deterministic, pinpoint failures.

## `.test.ts` vs `.oracle.ts` vs `.lab.ts`

The suffix is the tier — bun only auto-discovers `.test.`/`.spec.`, so a different suffix opts a file out of the default `bun test` while staying runnable via a `./`-prefixed path.

- **`.test.ts`** — spec tests. First principles, conservation laws, tight tolerances. Permanent. Run by `bun test`. If one fails, the code is wrong. The fast inner loop — keep it lean.
- **`.oracle.ts`** — the heavy deterministic CPU reference tier (the f64 AVBD physics oracle, `tests/avbd/`). Permanent and load-bearing — the executable spec every GPU physics gate is validated against (physics.md "the oracle is not the suspect") — but slow (hundreds of f64 sim frames per test), so it's split out of the inner loop to keep `bun test` fast. Run via `bun run test:oracle` (`bun test ./packages/shallot/tests/avbd/*.oracle.ts`); `bun run test:full` runs `test` + `test:oracle` — the complete gate before a commit / PR. A new oracle-tier file just needs the `.oracle.ts` suffix in `tests/avbd/` — no script edit.
- **`.lab.ts`** — investigation files. Trace internals, compare against references, probe edge cases. Not discovered by `bun test` — run manually (`bun test ./packages/shallot/tests/foo.lab.ts`). Temporary — delete or promote when done.

## Tolerance tiers

- **Exact:** no floating-point reason to differ (mass invariance, quaternion normalization) → `1e-10` or tighter
- **Truncation error:** derive from integrator order + step size, not continuous-time physics
- **f32 precision:** `~1e-6` relative for single ops, accumulates with chain length
- **Solver convergence:** derive from iteration count + penalty schedule, not observation

## GPU timestamps, not FPS

`requestAnimationFrame` measures CPU, not GPU. WebGPU's `queue.submit()` returns immediately. GPU timestamp queries measure actual hardware execution time — this is the primary benchmark metric.

**CPU timing quantizes to 100 μs in non-isolated Chrome.** `performance.now()` returns multiples of 0.1 ms without COOP+COEP isolation, so per-frame samples bin to {0.0, 0.1, 0.2 …}. `min` over samples becomes biased toward zero; **avg over ~200 frames is the correct primary stat** for CPU-mixed measurements (the gym `extras/profile/benchmark.ts measure()` rig follows this — 60 warmup + 500 measured frames, avg as primary). Read avg for CPU-mixed rows, min for pure-GPU timestamp rows (sub-μs, one-sided noise).
