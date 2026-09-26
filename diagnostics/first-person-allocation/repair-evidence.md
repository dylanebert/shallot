# First-person allocation repair evidence

## Identity

- Repair source: `e5db4695388e3b6199bc9b31989a5c71e7dc97b6` (parent: `f4409f372d0ac8667a90478f278d44da9d7e5838`).
- Original check, scene, authored inputs, warm-up (`6000`) and measured windows (`600` × three) were not changed.
- The temporary probes were local-only edits, removed before the repair commit. No sampler, exemptions, runtime flags, gameplay input or scene was changed.

## Controlled owner comparison

Using the original `first-person warm frame allocates nothing` check under Bun 1.4.2 and Node 26.8.1:

| Subject operation | Result | Evidence |
| --- | --- | --- |
| Full composed baseline | Fail before repair | Prior diagnosis: 115,200 B at the sampler run boundary; exact source was not previously established. |
| Temporarily omit Character's `setKinematic` upload call | Pass | Same original check and its in-entry known-allocation control; proves this call path is necessary for the red. |
| Temporarily omit only `SolverBody.setTransform` inside `setKinematic` | Pass | Same unchanged check and positive control; isolates the transform setter from the sweep and velocity upload. |
| Temporarily omit only `SolverBody.setLinearVelocity` | Fail | The unchanged check reported samples in `bodySetTransform`, `finalizeBodies`, `get flags`, and other physics consequences. Thus velocity upload alone is not the source. |

The omission comparisons distinguish the allocating operation: the unconditional `tb.setTransform(kinPos, kinQuat)` causes repeated transform/recompute work even when the kinematic pose has not changed. The repair gates that existing setter on pose change. The cached pose includes position and quaternion, preserving orientation-only updates. Velocity continues to be uploaded every tick, and changed positions still go through the same transform setter.

## Focused behavior checks

- `bun test examples/first-person/src/demo.test.ts --test-name-pattern 'first-person warm frame allocates nothing|first-person lift carries the actual character upward'`: 2 pass after repair. The allocation check's built-in positive allocation control remains nonzero, otherwise the check is inconclusive.
- `bun test examples/first-person/src/demo.test.ts`: 6 pass, including the unchanged allocation claim and composed lift behavior.
- `bunx biome check src/transitional/physics/index.ts` and `git diff --check`: pass.

## Declared-toolchain two-host qualification

Temporary workflow run: https://github.com/dylanebert/shallot/actions/runs/36259184521

It checked out and verified exactly the repair source SHA above. Both runners used Bun 1.4.2, Node v26.8.1, rustc 1.98.1 and Cargo 1.98.1; Ubuntu runner was Linux x86_64 and macOS runner was Darwin arm64.

| Invocation | Ubuntu | macOS |
| --- | --- | --- |
| `bun run check` | Exit 1: only known import-boundary failure | Exit 1: only known import-boundary failure |
| `bun run test` | 288 passed, 0 failed, 0 refused, 50 skipped | 288 passed, 0 failed, 0 refused, 50 skipped |
| `bun run test -- --integration --all --requires '!gpu' --requires '!browser' --requires '!display' --no-unit-fallback` | 43 passed, 0 failed, 0 refused, 0 unrun | 43 passed, 0 failed, 0 refused, 0 unrun |

The known `check-imports` reds are the rendering-to-input sibling import and 12 transitional Destination imports listed in the existing portable CPU verification report. Every other static gate passed. The full CPU integration population, including the unchanged first-person allocation row, passed on both hosts. No unexpected red occurred.

The temporary qualification workflow lived only on `qualify-first-person-allocation-e5db`; it was not added to permanent CI. Full host records and invocation logs were downloaded from run `36259184521` during qualification.
