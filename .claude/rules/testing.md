---
paths:
    - "packages/shallot/tests/**/*.ts"
---

# Testing

Shallot-specific testing patterns.

## GPU testing

Two layers required for GPU features:

- **Unit tests** (`bun test`) — fast, bun-webgpu. Catches logic errors
- **GPU tests** (`bun bench`) — Playwright on real GPU via the gym example. Catches validation errors, pipeline mismatches, texture format issues, bind group errors, alignment problems. One invocation = one page load = one measurement. Supports scenarios (`--scenario pile`) and custom params (`--pipeline raytracing --frames 100`)

Unit tests alone miss real hardware failures. Always run `bun bench` after GPU code changes.

**Cross-validation:** Non-trivial GPU algorithms (sorting, BVH, spatial hash) can be validated against CPU reference implementations for isolated unit testing. Not a requirement — GPU is the primary target. CPU references are useful for early experimentation and pinpointing bugs, not for maintaining parity.

## Component CRUD tests

Test the **mechanism**, not every component. One test of the query pattern covers 15 components using the same pattern. Write a CRUD test only when:

- A bug was reported
- The component uses a different lifecycle pattern (lazy init, async pipeline)
- Setup runs once and might miss dynamic adds

## Playwright test structure

One browser session per test file. Don't split related assertions into multiple `test()` blocks — use phases within a single test instead. Starting a new browser session is expensive and loses page state. Reconstruct state within the session if needed.

## Pairwise testing for combinatorial GPU features

Most bugs come from 2-factor interactions. For combinatorial feature spaces (surfaces × pipelines × opaque/transparent × shadow × instance fields), generate pairwise test matrices (~40-60 combos instead of thousands). `compileSurfaceBlock` is a pure function — feed it pairwise inputs, validate structurally + GPU-compile.

## Structural validation over visual regression

Prefer: (a) structural validation of generated WGSL (expected functions, bindings, dispatch cases), (b) GPU compilation validation (shader compiles, pipeline creates without errors), (c) GPU readback validation (run the actual compute, read results back). CPU cross-validation is optional for isolated debugging. Reserve screenshot comparison for integration-level Playwright tests only. Structural tests are fast, deterministic, pinpoint failures.

## `.test.ts` vs `.lab.ts`

- **`.test.ts`** — spec tests. First principles, conservation laws, tight tolerances. Permanent. Run by `bun test`. If one fails, the code is wrong.
- **`.lab.ts`** — investigation files. Trace internals, compare against references, probe edge cases. Not discovered by `bun test` — run manually (`bun test ./packages/shallot/tests/foo.lab.ts`). Temporary — delete or promote when done.

## Tolerance tiers

- **Exact:** no floating-point reason to differ (mass invariance, quaternion normalization) → `1e-10` or tighter
- **Truncation error:** derive from integrator order + step size, not continuous-time physics
- **f32 precision:** `~1e-6` relative for single ops, accumulates with chain length
- **Solver convergence:** derive from iteration count + penalty schedule, not observation

## GPU timestamps, not FPS

`requestAnimationFrame` measures CPU, not GPU. WebGPU's `queue.submit()` returns immediately. GPU timestamp queries measure actual hardware execution time — this is the primary benchmark metric.
