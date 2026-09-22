# First-person allocation diagnosis — stage 1

## Verdict

The landed `c6a7fc16` baseline is still red under the pinned Node child runtime. The red is in the stepped CPU subject path, not the sampler's control path. Bounded controls localize the steady allocation to the `Character` fixed-step path (the path is necessary for the red), with the smallest plausible ownership boundary at the Character/physics kinematic-upload seam. The exact allocating operation is **not proved**: the harness reports the run caller, `stepChunk`, rather than a Character or physics source line.

No repair, sanction, red-circle, or production test change is included. The allocation claim remains red.

## Identity and exact reproduction

The diagnosed tree began at landed Shallot `c6a7fc16` (`docs: refine browser input API wording`). The allocation subject is the landed path in:

- `examples/first-person/src/demo.test.ts`
- `examples/first-person/src/allocation.entry.ts`
- `src/harness/allocation.ts`
- `src/harness/allocation-sampler.mjs`

`allocation.entry.ts` composes `PhysicsPlugin`, `CharacterPlugin`, `InputPlugin`, and `Demo`; it does **not** compose `PlayerPlugin`. The input is the exact authored `examples/first-person/public/scenes/first-person.scene` text. The sampler is asked for `warm: 6000` and `frames: 600`, with the pinned child flags shown in the output below.

Node 26.8.1 was installed outside the worktree for the run (the worktree's `.node-version` is `26.8.1`):

```sh
mkdir -p /tmp/shallot-node
curl -fsSL https://nodejs.org/dist/v26.8.1/node-v26.8.1-darwin-arm64.tar.gz \
  -o /tmp/shallot-node/node.tar.gz
tar -xzf /tmp/shallot-node/node.tar.gz -C /tmp/shallot-node
export PATH=/tmp/shallot-node/node-v26.8.1-darwin-arm64/bin:$PATH
node --version                 # v26.8.1
bun --version                  # 1.4.2
```

Exact selected red command:

```sh
bun test examples/first-person/src/demo.test.ts \
  --test-name-pattern 'first-person warm frame allocates nothing'
```

Observed output from the selected reproduction (exit 1):

```text
bun test v1.4.2 (744846f84)
shallot verdict {"claim":"a warm fixed step of the actual first-person CPU composition allocates no JavaScript heap, so no periodic scavenge follows play","size":"integration","runtime":"bun 1.4.2","hardware":"none","result":"fail"}
error: warm page frames allocate outside the sanctions and red circles:
  A/A repeat: 115200 B at stepChunk src/harness/allocation-sampler.mjs:140
(fail) first-person warm frame allocates nothing
 0 pass
 4 filtered out
 1 fail
Ran 1 test across 1 file.
EXIT=1
```

A later identical command on the same pinned runtime also remained red but read `230400 B` at the same `stepChunk` site. This repeat is recorded as an observation, not hidden: the red is repeatable, while the sampled byte/count is not fully repeatable even with Node 26.8.1. The first exact output above is the selected reproduction; `all-results.json` records the bounded run whose baseline A/A window read `115200 B`.

## Controlled comparisons

All sampler comparisons use the same `warm: 6000`, `frames: 600`, scene text, V8 tier flags, and three windows unless noted. The complete JSON is [`all-results.json`](./all-results.json). Run them with:

```sh
bun diagnostics/first-person-allocation/run.ts
```

The committed diagnostic entries are scratch subjects only; they do not alter the engine or its ledgers.

| Comparison | One controlled input change | A/A observation | What it supports |
| --- | --- | --- | --- |
| `baseline` | Landed allocation entry | `115200 B`, `1200` samples, only `stepChunk ...:140` | The landed claim is red under the pinned runtime. |
| `noopStep` | Same build and scene; returned `step()` does no work | `0 B`, no sites | The sampler/control and build alone do not produce the steady red; the stepped subject path is required. |
| `knownSubjectAllocation` | Same baseline step, plus one live `{ frame: 0 }` object assigned in the subject's `step()` | `134400 B`, `1800` samples, still only `stepChunk ...:140` | Positive control: a known subject allocation is detected. The extra `19200 B` is not attributed to its source line, so the `stepChunk` label is a caller bucket, not an owner proof. |
| `noCharacterSystem` | Same scene and physics/input composition, but the Character plugin's fixed system is removed | `0 B`, no sites | The steady red requires the Character fixed-step path. Warm-window tiering bytes are present but do not survive the A/A gate. |
| `noPhysicsStep` | Same Character path and warmed physics world, but only `StepSystem` is removed | `369600 B`, `3600` samples, `stepChunk ...:140` | Character-side work still allocates without the solver step. This rules out assigning the red solely to the physics solver step, but does not identify the individual Character/FFI operation. |
| `empty` scene | Same landed composition and sampler, no authored entities | `0 B`, no sites | The red needs the authored subject state, rather than being a fixed per-call sampler artifact. |
| `character` scene | Same landed composition, reduced input containing one Character+Body and no ground | `115200 B`, `1200` samples, `stepChunk ...:140` | A single Character+Body is sufficient; the full route and its extra static geometry are not needed for the baseline-sized red. |

The `noCharacterSystem` and `noPhysicsStep` controls preserve the relevant fixed-clock composition boundary while removing one named system at a time. They are not claims that the removed variants are valid gameplay configurations.

The control literal already in the landed entry was also nonzero in every comparison (`controlChunk ...:143`, roughly 22–25 KB over its 600 calls). Together with `noopStep`, this is a positive check that the sampler is live and does not need a subject red to emit data.

## Source versus caller attribution

`src/harness/allocation-sampler.mjs` calls `attribute(profile, runSite, siteOf)`. `runSite` recognizes `stepChunk`; it is deliberately the nearest run boundary, and the attribution walk credits samples there before a deeper subject frame can name them. The baseline result therefore says:

```text
stepChunk src/harness/allocation-sampler.mjs:140
```

It does **not** say that `allocation-sampler.mjs:140` allocates. The known-subject positive control is the distinguishing evidence: adding a literal in the subject raises the measured total, but the report remains at the same caller label. The source map and the profile consequently prove detection at the run boundary, not the allocating owner.

The relevant source-side boundary is:

```text
CharacterSweepSystem.update
  -> sweepEach / sweepEid
  -> sweepCharacter
  -> setKinematic
  -> physics body transform / linear-velocity bindings
```

`src/standard/character/sweep.ts` reuses its geometry scratch, and `src/standard/physics/index.ts` reuses the `kinPos`, `kinQuat`, and `kinVel` records in `setKinematic`; those facts make an avoidable allocation at this boundary surprising, but they do not prove that no binding or runtime wrapper allocates. The controls localize the path, not the exact line.

## Observations, inferences, unknowns

### Observed

- The exact landed first-person check fails under Node `v26.8.1` as the child sampler runtime and Bun `1.4.2` as the test host.
- The selected reproduction read `115200 B` at `stepChunk`; an identical later invocation read `230400 B` at the same site.
- Three steady windows agree within each captured baseline run, and the A/A site remains the sampler caller.
- A no-op subject is steady-zero; a known subject literal is detected and increases the total; both facts use the same sampler and control mechanism.
- Removing Character's fixed system makes the A/A window empty. Retaining Character while removing only the physics `StepSystem` remains red.
- A minimal one-Character scene is sufficient for the baseline-sized steady red.

### Inferred

- This is not justified as a sampler-only allocation: the subject step is required and a known subject allocation is observable.
- `stepChunk` is attribution machinery / caller context, not the responsible source.
- The least justified ownership boundary is the Character fixed-step kinematic path, with the Character-to-physics body setter seam the first boundary to instrument or correct. The `Demo`/`PlayerPlugin` extraction is not implicated by this evidence; `PlayerPlugin` is absent from the subject.

### Unknown

- Which operation accounts for the baseline bytes: Character sweep arithmetic/call boundary, `setKinematic`, a physics WASM/FFI binding, or another callee under that fixed path.
- Why the pinned-runtime sampler alternates between at least `115200 B` and `230400 B` on otherwise identical red runs. This is an evidence-quality issue, not permission to round, retry until green, or change the claim.
- Whether the bytes are avoidable engine work or platform/runtime-forced work. No sanction decision is justified.

## Least correction and distinguishing check

The least proposed correction is **not** a change to `stepChunk`, the sampler, `sanctions.json`, or `red-circles.json`. It is a targeted allocation-free correction at the Character/physics kinematic upload boundary, if and only if a follow-up measurement proves an avoidable allocation there. Do not land that correction in this stage.

The distinguishing check for the architect/person is a source-level boundary experiment that keeps the same scene, fixed input, warm/window lengths, and positive control, and separates:

1. `sweepCharacter` with its reused scratch and no body upload;
2. the body upload (`setKinematic` / its binding) with the sweep result supplied from the same subject; and
3. the full Character system.

The repair verification is the original exact red command: every A/A steady window must be zero at the corrected boundary, the known-subject positive control must still be nonzero, and attribution must no longer rely on the `stepChunk` caller to claim an owner. If the boundary experiment cannot distinguish JS from WASM/runtime-forced work, that missing platform decision is the blocker; it is not a green allocation claim.

## Artifacts and commits

- `diagnostics/first-person-allocation/run.ts` — bounded comparison runner.
- `diagnostics/first-person-allocation/entries/*.ts` — scratch subjects for no-op, known-allocation, no-Character, and no-physics-Step controls.
- `diagnostics/first-person-allocation/all-results.json` — exact JSON output for the committed runner invocation.
- Candidate/evidence commit: `5b801b1` (`diagnostic: bound first-person allocation failure`).
- This report is the final-report commit's tree artifact; no production or ledger file is modified.
