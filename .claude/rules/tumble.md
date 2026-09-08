---
paths:
  - "packages/shallot-runtime/src/standard/tumble/**/*.ts"
  - "packages/shallot/rust/tumble/**"
  - "packages/shallot/tests/tumble/**"
  - "packages/shallot/scripts/build-tumble-kernel.ts"
  - "packages/shallot/scripts/run-tumble-fixtures.ts"
  - "packages/shallot/scripts/gen-tumble-fixtures.ts"
  - "packages/shallot/scripts/gen-tumble-gold.ts"
  - "packages/shallot/scripts/gen-tumble-sample-golds.ts"
  - "packages/shallot/scripts/tumble-exit-test.ts"
  - "examples/gym/src/tumble-*.ts"
  - "examples/gym/src/scenarios/**"
  - "scripts/bench-tumble.ts"
  - "scripts/tumble-interaction.ts"
  - "scripts/tumble-repro.ts"
  - "scripts/tumble-repro-driver.mjs"
  - "scripts/check-tumble-fp.ts"
---

# Tumble Backend

Default CPU Box3D backend; `physics.md` owns the substrate. `Tumble.world`/`Tumble.body` escape the core. Dispose the singleton; read it fresh. Marshal eid-sorted, stamp-check recycling and clear kinematics. Swap parity means settle/equilibrium; marshaling requires per-step hashes against independent raw scenes (`tumble.test.ts`).

`joints.ts`: reduced-mass critically damped springs, spherical/weld pins with spawn-relative rotation; unchanged defs retain warmstart. Reject invalid constraints loudly; defer pending marshal, pump on body-set changes locally. Dedupe def content AND endpoint-cause composition; re-arm on authored upload, never silence retries. Ragdolls stay examples; deferred joints need non-overlapping spawns and spawn-derived frames.

## Bit-exact contract

World hashes including sleep steps equal pinned C (colored solver, SIMD disabled, overflow off) at every thread count; no tolerance. Mirror C expression trees, scalar branches and portable trig, not JS/Rust min/max or libm. JS aliases `Math.fround`: round each op, non-exact literal/named operand and API float at creation/setters, not just stores. No FMA; Rust software helpers are non-fused. Mirror C file/function structure.

`check-tumble-fp.ts` misses named f64 operands. Only `createWaveMesh`/authoring `heightfield.ts` use sine: structural tests, never bit-exact libm claims. Emit required geometry from C; preserve op-for-op scene inputs/order, pre-step actions and auxiliary data.

## Kernel and memory

Hot loops: kernel; graph/world mutation, islands/events/API: TS. Phase FFI, never per-contact. Gather records, not fields. Growth preserves all columns; refresh after relocation without detachment and mid-step growth. Shared views key on byte length. Tree sync updates TS AND Rust; copy integer bits. New geometry/filter setters update columns.

No live wasm std/Vec allocation. Copy scratch refs; refresh views across migration. Set/index changes reclassify contacts/update indices; sleeping indices unread until wake. Fresh worlds zero reusable membership/caches; probe logical capacities. Invalidate destroyed-contact caches. Joint solves write back next-substep scratch, zero disabled fields; require multi-substep fixtures, not only isolated golds.

MT: host-resolved bounded pool, not all cores/scene API. Main instantiates first; stacks below regions. Reserve before wake; forbid grow/relocation until join and assert it. Colors write-disjoint, overflow/creation serial. Faults poison later access, no fallback. Unref workers; pool tests call `afterAll(shutdown)`. Browser MT requires COOP/COEP and isolation + threads>1 proof; nonisolated hosts log once/use ST. Deno needs exit-test proof.

## Fixtures and gates

Read `tests/tumble/fixtures/README.md`, `tests/tumble/samples/README.md` and `engine/upstream.json` before changes. Committed C fixtures, phase golds and sample trajectories are truth: never hand-edit or adjust for mismatch. Minting needs the external harness fork/C toolchain and must fail honestly without it. Freeze the pin by default; sync only to a tag/Fixes batch with observable math/staging fixes. Rebase harness, regenerate/reverify fixtures, port op-for-op, regenerate affected golds, update BOTH pin sites. Arithmetic/order changes need regen; data movement alone does not.

Kernel: `cargo test`, then from `packages/shallot` `bun run scripts/build-tumble-kernel.ts`; commit both wasm artifacts, even panic-line changes. MT changes also run `bun run test:fixture`, `bun run test:fixture:mt` at `TUMBLE_THREADS=2` and 8, `bun run test:fixture:auto`, `bun run test:exit`. Keep joint-event assertions at ST/2/8: hashes cannot see events. Decompose serial/kernel costs; memory-traffic wins need same-window interleaved pre-change A/B before landing.

Engine/host/twin changes run root `bun test ./examples/gym/src` (gold worlds in separate processes). Corpus/render changes run `bun run scripts/bench-tumble.ts`; grab/solids/overlay/input changes run `bun run scripts/tumble-interaction.ts`, input-to-pixels also `bun run scripts/tumble-repro.ts --gate`. Require synthetic AND trusted input, independent static-breach/finite guards. Derive visuals from world shapes. Serialize bridges; tiers: `testing.md`.
