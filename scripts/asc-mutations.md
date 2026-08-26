# audit-stale-claim-sweep — round 7 mutation captures and verdicts

Committed in-repo capture path for the check-docs citation-resolution arm's
mutation proofs. Each mutation was applied in place (count-neutral: a
single-occurrence LIVE citation swapped for the dead one, never appended),
one `check-docs.ts` run, exit code and failing leg read separately, then the
file was restored from a pristine copy. All seven mutations red (exit 1);
the clean baseline is green (exit 0).

The citation-count pin runs before the resolution loop, so ANY appended line
moves the count and reds — including one citing a live symbol (measured 1665
vs 1664 at `f6b302e`). A seed mutation therefore swaps a single-occurrence
live citation for the dead one IN PLACE, and the record carries the leg name
(`stale citation`, `roster entry count mismatch`, …) beside the exit code,
never the exit code alone.

Witnessed at the round 7 working state on branch `audit-stale-claim-sweep/S1`.

## Mutations

| # | Mutation | Exit | Failing leg |
|---|----------|------|-------------|
| (i) | Seeded backticked dead symbol: `` `advanceColor` `` → `` `zombieUploadPass` `` in avbd.md:114 (in place) | 1 | stale citation |
| (ii) | Bare dead symbol: `` `advanceColor` `` → bare `zombieUploadPass` in avbd.md:114 (in place) | 1 | stale citation |
| (iii) | Roster launder: `zombieUploadPass` added to `FOREIGN_NAMESPACES.Tools` (no rule file edit) | 1 | roster entry count mismatch (44 vs 43) |
| (iv) | Substring: `` `advanceColor` `` → `` `spotInner` `` in avbd.md:114 (substring of live `spotInnerF`) | 1 | stale citation |
| (v) | Launder-via-marker: `` `advanceColor` `` → `` `zombieUploadPass` (retired) `` in avbd.md:114 | 1 | marker-exempted count mismatch (21 vs 20) |
| (vi) | Weak-shape bare: `` `advanceColor` `` → bare `zombie_upload_pass` (snake) in avbd.md:114 | 1 | stale citation |
| (vii) | Predicate narrowing: `matchesWeakShape` call removed from `matchesShape` | 1 | citation count mismatch (1508 vs 1735) |
| baseline | Clean tree | 0 | — |

## (viii) — SHAPE_FALSE_POSITIVES, retired honestly

The round-6 `SHAPE_FALSE_POSITIVES` set was an unpinned per-entry allowlist
that could green a dead symbol in one line (measured at `6ee6ed6`: adding a
string to the set took a seeded dead symbol from exit 1 to exit 0). Round 6b
deleted the set. The round-6b capture row (viii) claimed re-introducing the
set as an unread variable + seeding a dead symbol "still reds" — but that is
a tautology about dead code, not a witness about the gate: a variable no
function reads has no effect, so the dead symbol is caught by the population
predicate regardless of the set's presence. The real channel
(`SHAPE_FALSE_POSITIVES` greening a dead symbol) was closed by **deletion** of
the set, not by the gate catching a re-introduction. There is no mutation to
witness — the escape no longer exists in the code, and a re-introduction as
an unread variable is inert by construction. Row (viii) is retired.

## A — Roster disjointness (round 7, item a)

34 of 75 roster entries also resolved in the tree token index (measured at
`f6b302e`), making them redundant with disjunct 1 (tree resolution). A
swap-in — dropping one redundant entry and adding a dead symbol in its place
— read exit 0 with every pinned cardinality unchanged, because the pinned
total 75 bought nothing against a swap-in. Round 7 prunes the 34 redundant
entries (75 → 41 clean) and adds the disjointness assertion: every roster
entry is asserted ABSENT from the tree token index, so each surviving entry
is load-bearing and removing one reds. Two new roster entries are added for
round 7 item (b) — `gain_effect` and `direct_effect` (Steam Audio upstream
filenames, SteamAudio roster) — bringing the final roster count to 43.

## B — Weak shapes in-span (round 7, item b)

Admitting weak shapes (snake_case, lowercase-with-digits) in multi-token
backtick spans (`matchesShape(ref)` at the in-span branch) adds 67 new
candidates (1664 → 1731). Six additional candidates come from the
EndFrameSystem sentence fix in render.md (item c: `MirrorSystem`,
`InputResetSystem`, `OrbitOverlaySystem`, `ProfileRenderSystem`,
`sortSystems`, `scheduler.ts`), and 2 are excluded by widening `ARITH_RE`
with comparison operators (`≤`, `≥`) to exclude formula variables
(`working_set`, `L2_size` at gpu.md:219). Net: 1664 → 1735.

The 5 new unresolved sites from the in-span admission are adjudicated:
- `gain_effect` (audio.md:43) and `direct_effect` (audio.md:44–45) — Steam
  Audio upstream C++ filenames, added to the `SteamAudio` roster class.
- `working_set` and `L2_size` (gpu.md:219) — formula variables in
  `working_set ≤ L2_size`, excluded by widening `ARITH_RE` with `≤`/`≥`
  (not `<`/`>`, which are TypeScript angle brackets in spans).

## C — Bare weak shapes re-admitted (round 6b, carried forward)

74 resolving bare weak-shape occurrences become new candidates (e.g. `u32`
15×, `box3d` 14×, `vec4` 5×, `base64` 4×, `unorm16` 8×). The 11 distinct
bare weak-shape tokens that did not resolve at `6ee6ed6` are adjudicated per
site: `webgpu_inspector` is a genuine foreign tool name in the Tools roster;
`memory64` is a Wasm feature (not a tool) filed under `WasmFeatures`; 9 prose
terms (bench metrics, formula variables, hardware terms, data formats,
benchmark labels) are fixed in the rule prose.

## D — EndFrameSystem sentence fix (round 7, item c)

The re-spelling of `EndFrame` → `EndFrameSystem` (round 6b) was correct, but
its sentence — "the sole `last: true` system" / "owns that slot" — was false.
The scheduler (`scheduler.ts:259-265`) splits systems into `first` / `normal`
/ `last` buckets and kahn-sorts each bucket, so multiple `last: true` systems
coexist (in the draw group: `EndFrameSystem`, `MirrorSystem`,
`InputResetSystem`, `OrbitOverlaySystem`, `ProfileRenderSystem`; in the
simulation group: `ReadbackSystem`). The sentence is fixed in render.md:68,
render.md:161, and standard/render/index.ts:75 to state that EndFrameSystem
submits the encoder and a renderer declaring `last: true` without a
`before: [EndFrameSystem]` edge risks the kahn-sort placing it after the
submit.

## E — Instrument residue (round 7, item e)

- `stale-claim-predicates.ts:151` (extractCandidates docblock) said weak
  shapes "only match when backticked" — false at HEAD, written by round 6b.
  Fixed: "All identifier shapes (including weak shapes) are caught bare or
  backticked."
- `matchesShape`'s ignored `_backticked` parameter was passed by four call
  sites as if it mattered. Removed: the parameter is gone, all call sites
  updated.
- `asc-mutations.md`'s "77 … become new candidates" was a raw-occurrence
  count where the candidate delta is 74. Fixed: "74 resolving bare
  weak-shape occurrences become new candidates."
- `memory64` was misfiled under `FOREIGN_NAMESPACES.Tools` though it is a
  Wasm feature. Moved to `FOREIGN_NAMESPACES.WasmFeatures`.
