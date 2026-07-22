# tumble sample golds — source-faithful physics oracle

The 40 `*.json` files here are per-sample **gold trajectories**: the ground truth a ported gym
scenario must reproduce bit-exact. Each is one tumble.js sample run headless at knob defaults, recording
the initial body snapshot, the per-step world-state hash, and the sample's camera pose + knob schema.
The corpus registered 41 samples; 40 minted a gold, 1 is excepted (below).

A gym scenario (spec `tumble-inline` stage 3+) reproduces a sample's `build()` through the escape hatch
(`Tumble.world`) and replays it against the same engine these golds were minted with. Both sides are the
same engine, so only authoring can differ — a wrong axis, wrong joint, or wrong shape mismatches the hash
at the first divergent step. That is the oracle the earlier examples port lacked.

## Provenance

- **Sample corpus:** `tumble.js/samples` at commit `128a4efeb7d28024c338a4b5beaefc49d3dc4345` — the reference implementation.
- **Minting engine:** shallot's inlined `src/standard/tumble/engine` (box3d pin `29bf523`), **not**
  tumble.js's own engine copy. The samples' `import "tumble.js"` is aliased to the shipping engine barrel
  during the mint, so the hashes are exactly what a gym replay against `Tumble.world` produces.
- **Hash:** the engine's own `hashWorldState` (FNV-1a over every live body's transform + velocity), the
  same function the C fixtures compare against — so a ported scenario compares identically on both sides.

## Mint recipe

```
bun run scripts/gen-tumble-sample-golds.ts     # from packages/shallot
```

- **Defaults** (`defaultContext()`): 60 Hz (timeStep 1/60), 4 substeps, gravity (0, -10, 0), sleep on,
  continuous on, every sample at its declared knob defaults — no interaction (no mouse grab).
- **Threads:** single-thread (`init({ threads: 0 })`). The gold contract is thread-count-independent by
  construction; a gym replay at any thread count reproduces it.
- **Step count:** 600 steps (10 s at 60 Hz). The hash is bit-exact every step, so a static authoring
  error diverges at step 0; the horizon is what pins the behavior phases that fire late — a break
  threshold, a motor reversal, a settling stack, a chaotic divergence — while keeping the golds trivially
  committable.
- **Isolation:** one child process per sample — a fresh wasm kernel per gold. The kernel is a
  process-wide singleton whose grow-only regions carry a high-water across sequential worlds and trap
  after several; a pristine kernel is also what a single-scenario gym replay gets. Output is
  timestamp-free, so a double mint is byte-identical.

**The source project is retired** (tumble.js was deorbited when its engine folded into shallot, 2026-07),
so regeneration is no longer possible: these golds are **frozen truth**, the same status as the C fixtures —
never hand-edit or re-mint; a mismatch is an engine or authoring bug, never a gold to adjust. The source
commit above is historical provenance. `scripts/gen-tumble-sample-golds.ts` is kept as the mint's recipe
record; run without the retired checkout it errors honestly. The committed golds carry the whole contract —
no checkout ever needs the source corpus to run the ported scenarios' oracle.

## Exceptions

Samples that can't mint a gold (recorded, not worked around — the source is read-only):

- **Benchmark / Rain** — engine hangs inside a step once a second ragdoll column lands on the mesh field (~step 40), short of any horizon. Throughput benchmark, not an authoring scene.

## Layout

- `<slug>.json` — one per minted sample: `{ slug, category, name, description, timeStep, subStepCount,
  gravity, enableSleep, enableContinuous, stepCount, camera, knobs, bodyCount, initial, hashes }`.
- `index.json` — the manifest: source commit, engine, step count, `registeredCount` / `goldCount`, every
  minted sample's slug + category + name + description (registration order), and `exceptions`.
