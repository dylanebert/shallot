---
paths:
    - "packages/shallot/src/standard/physics/**/*.ts"
    - "packages/shallot/src/standard/character/**/*.ts"
    - "packages/shallot/src/standard/player/**/*.ts"
---

# Physics Substrate

Shared authoring/queries and atomic `PhysicsBackend`, never simulation. `tumble.md` owns CPU, `avbd.md` GPU. Consumers never branch by backend. Richer capabilities belong on `Tumble.world`/`Avbd.step`, not a union interface or intersection-limited escape hatches.

## Backend contract

Register shared `Body`/`Spring`/`Joint` once. `Body` excludes `Transform`: spawn properties are authored, live poses belong to physics; mass ≤ 0 is static. Install the plain handle during warm, uninstall on dispose; double install throws. Guard the null pre-warm handle. Pose reads may be one fixed tick stale.

The substrate owns ordering: constrain producers against `StepSystem`, not a backend. `ConstraintSystem` uploads before step; installation re-arms uploads. Compose after `BeginFrameSystem`, before `PrepassSystem`. Firehose ownership is a slot partition, never write order: Transform compose touches only its members, physics only Body slots. Preserve both queue-write and encoder-ordered backend coverage; one can mask the other's overwrite bug.

## Authoring

Each constraint is its own entity with body refs and local anchors. Joint angular stiffness defaults to spherical zero; fixed means infinite. Use components OR imperative upload per constraint type; an empty authored set never overwrites imperative state. Re-upload only when the authored signature changes, including endpoint stamps so same-update recycling rebinds. Backend retry state stays out; ecs.md "An eid is a borrow". Set fields before first observation; character registration reads them once.

`Character` owns controller tuning on a capsule Body; `Player` composes it with look/camera. The eid-keyed move/jump/pose/grounded API is `@dylanebert/shallot/character/core`, same-frame CPU state, not GPU readback.

## Kinematic character

One CPU controller, no per-backend fork or GPU character pass. `CharacterSweepSystem` runs fixed before step; `PlayerSnapshotSystem` snapshots afterward and interpolates prev/current pose. Keep one-tick interpolation. Static geometry comes from authored Body data, moving/dynamic poses from the backend. Escape-hatch-only colliders are invisible; no swept TOI against them.

Write current swept pose via `setKinematic`, realized velocity excluding cosmetic snap. Push via `setVelocity`, waking dynamics; no GPU-additive path without a measured pile-push artifact. The f64 `tests/avbd/character.ts` is the spec: preserve `character-sweep.oracle.ts` parity and `character.oracle.ts` feel/conservation; real gym `character` gates backend coupling and input-camera latency.

Move then depenetrate using closest-point capsule-core MTV, not SAT face normals. Re-probe the penetrated pose for free step-up; no cached planes without a separate stair algorithm. Gravity applies only when previously airborne, with per-character override (zero reads world gravity). Skip snap while rising; grounded requires a walkable contact within snap range. Realized motion excludes snap.

Cull with an output-neutral sphere superset of every phase's reach, preserving scan order. Culled versus brute output must agree exactly; cap overflow and travel beyond budget report loudly. Jump requires AND consumes both coyote credit and buffer. Held input cannot re-fire airborne.

Depenetrate every contact, including dynamics; walkable dynamic tops support, sides block. Push on touch, not penetration: desired into-speed including ground velocity, never blocked realized speed; cancel downward impulse. Push is mass-independent, one character per dynamic. Carry and snap remain static-only. Translating/descending kinematic carry reads the deepest walkable starting contact; teleport zeros derived velocity. Angular carry is declined: powered rotating surfaces must be dynamic motor-driven bodies, not kinematics.
