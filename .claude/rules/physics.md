---
paths:
  - "src/standard/physics/**"
  - "rust/physics/**"
  - "scripts/physics/**"
  - "src/standard/character/**/*.ts"
  - "src/standard/player/**/*.ts"
---

# Physics

One module, no backend substrate: `PhysicsPlugin` runs the CPU Box3D port (TS + Rust/WASM kernel). `Physics.world`/`Physics.body` escape the atomic core; `Physics.readBody`/`setKinematic`/`setVelocity` drive it by eid. An outside solver plugs in through `physics/core` only (traits, defs, signatures, anchors, `Hulls`); physics never imports or names it.

## Contract

Register `Body`/`Spring`/`Joint` once. `Body` excludes `Transform`: spawn properties are authored, live poses belong to physics; mass ≤ 0 is kinematic. Warm creates the world and re-arms constraint uploads; dispose tears it down. Read the singleton fresh; guard pre-warm `null`.

Ordering: producers constrain against `StepSystem`. `ConstraintSystem` uploads before step. Compose after `BeginFrameSystem`, before `PrepassSystem`. Firehose ownership is a slot partition, never write order: Transform compose touches only its members, physics only Body slots.

Marshal eid-sorted, stamp-check recycling and clear kinematics. Marshaling requires per-step hashes against independent raw scenes (`plugin.test.ts`).

## Authoring

Each constraint is its own entity with body refs and local anchors. Joint angular stiffness defaults to spherical zero; fixed means infinite. Use components OR imperative upload per constraint type; an empty authored set never overwrites imperative state. Re-upload only when the authored signature changes, including endpoint stamps so same-update recycling rebinds. Retry state stays out of signatures; ecs.md "An eid is a borrow". Set fields before first observation; character registration reads them once.

`joints.ts`: reduced-mass critically damped springs, spherical/weld pins with spawn-relative rotation; unchanged defs retain warmstart. Reject invalid constraints loudly; defer pending marshal, pump on body-set changes locally. Dedupe def content AND endpoint-cause composition; re-arm on authored upload, never silence retries. Ragdolls stay examples; deferred joints need non-overlapping spawns and spawn-derived frames.

`Character` owns controller tuning on a capsule Body; `Player` composes it with look/camera. The eid-keyed move/jump/pose/grounded API is `@dylanebert/shallot/character/core`, same-frame CPU state.

## Kinematic character

One CPU controller. `CharacterSweepSystem` runs fixed before step; `PlayerSnapshotSystem` snapshots afterward and interpolates prev/current pose. Keep one-tick interpolation. Static geometry comes from authored Body data, moving/dynamic poses from `Physics.readBody`. Escape-hatch-only colliders are invisible; no swept TOI against them. Write the swept pose via `setKinematic` with realized velocity excluding cosmetic snap; push via `setVelocity`, waking dynamics.

Move then depenetrate using closest-point capsule-core MTV, not SAT face normals. Re-probe the penetrated pose for free step-up; no cached planes without a separate stair algorithm. Gravity applies only when previously airborne, with per-character override (zero reads world gravity). Skip snap while rising; grounded requires a walkable contact within snap range. Realized motion excludes snap.

Cull with an output-neutral sphere superset of every phase's reach, preserving scan order. Culled versus brute output must agree exactly; cap overflow and travel beyond budget report loudly. Jump requires AND consumes both coyote credit and buffer. Held input cannot re-fire airborne.

Depenetrate every contact, including dynamics; walkable dynamic tops support, sides block. Push on touch, not penetration: desired into-speed including ground velocity, never blocked realized speed; cancel downward impulse. Push is mass-independent, one character per dynamic. Carry and snap remain static-only. Translating/descending kinematic carry reads the deepest walkable starting contact; teleport zeros derived velocity. Angular carry is declined: powered rotating surfaces must be dynamic motor-driven bodies, not kinematics.

## Bit-exact contract

World hashes including sleep steps equal pinned C (colored solver, SIMD disabled, overflow off) at every thread count; no tolerance. Mirror C expression trees, scalar branches and portable trig, not JS/Rust min/max or libm. JS aliases `Math.fround`: round each op, non-exact literal/named operand and API float at creation/setters, not just stores. No FMA; Rust software helpers are non-fused. Mirror C file/function structure. Only `createWaveMesh`/authoring `heightfield.ts` use sine: structural tests, never bit-exact libm claims.

## Kernel and memory

Hot loops: kernel; graph/world mutation, islands/events/API: TS. Phase FFI, never per-contact. Gather records, not fields. Growth preserves all columns; refresh after relocation without detachment and mid-step growth. Tree sync updates TS AND Rust; copy integer bits. No live wasm std/Vec allocation. Fresh worlds zero reusable membership/caches. Invalidate destroyed-contact caches.

MT: host-resolved bounded pool, not all cores/scene API. Main instantiates first; stacks below regions. Reserve before wake; forbid grow/relocation until join and assert it. Colors write-disjoint, overflow/creation serial. Faults poison later access, no fallback. Unref workers; pool tests call `afterAll(shutdown)`. Never export `shutdown` past the engine barrel. Browser MT requires COOP/COEP and isolation + threads>1 proof; nonisolated hosts log once/use ST.

## Fixtures

Read `src/standard/physics/{engine/fixtures,samples}/README.md` and `engine/upstream.json` before solver changes. Committed C fixtures, phase golds and sample trajectories are truth: never hand-edit or adjust for mismatch. Minting needs the external harness fork/C toolchain and must fail honestly without it. Freeze the pin by default; arithmetic/order changes need regen, data movement alone does not.

Kernel: `cargo test` in `rust/physics`, then `bun run scripts/physics/build-kernel.ts`; commit both wasm artifacts. MT changes also run `scripts/physics/run-fixtures.ts` at `SHALLOT_PHYSICS_THREADS` unset, 2, 8 and `auto`, plus `scripts/physics/exit-test.ts`. Keep joint-event assertions at ST/2/8: hashes cannot see events.
