# audit-stale-claim-sweep — round 6b mutation captures and verdicts

Committed in-repo capture path for the check-docs citation-resolution arm's
mutation proofs. Each mutation was applied in place, one `check-docs.ts`
run, exit code captured separately, then the file was restored from a
pristine copy. All mutations red (exit 1); the clean baseline is green
(exit 0); the (viii) control (set re-introduced without seeding) is green
(exit 0).

Witnessed at commit `20ea760` on branch `audit-stale-claim-sweep/S1`.

## Mutations

| # | Mutation | Exit | Description |
|---|----------|------|-------------|
| (i) | Seeded backticked dead symbol | 1 | `` `zombieUploadPass` `` appended to `gpu.md` → stale citation |
| (ii) | Bare dead symbol | 1 | Bare `zombieUploadPass` appended to `gpu.md` → caught by formatting-invariant predicate |
| (iii) | Roster launder | 1 | `zombieUploadPass` added to `FOREIGN_NAMESPACES.Tools` → pinned roster entry count mismatch (76 vs 75) |
| (iv) | Substring | 1 | `` `spotInner` `` appended to `gpu.md` → does not resolve (token index is exact, not substring; `spotInnerF` is live but `spotInner` is not) |
| (v) | Launder-via-marker | 1 | `` `zombieUploadPass` (retired) `` appended to `gpu.md` → pinned marker count mismatch (21 vs 20) |
| (vi-a) | Bare SCREAMING_SNAKE | 1 | Bare `ENTITY_COLS_WGSL` appended to `gpu.md` → caught bare by SCREAMING_SNAKE predicate |
| (vi-b) | Bare snake_case | 1 | Bare `zombie_upload_pass` appended to `gpu.md` → caught bare by re-admitted snake_case predicate |
| (vi-c) | Bare lowercase-with-digits | 1 | Bare `zombiepass2` appended to `gpu.md` → caught bare by re-admitted lowercase-with-digits predicate |
| (vi-d) | `*`-prefixed | 1 | `*zombieUploadPass` appended to `gpu.md` → `*`-prefix drop inadmissible for letter-starting tokens |
| (vii) | Predicate narrowing | 1 | `matchesWeakShape` call removed from `matchesShape` in `stale-claim-predicates.ts` → pinned citation count mismatch (lower than 1664) |
| (viii) | SHAPE_FALSE_POSITIVES launder + seed | 1 | `SHAPE_FALSE_POSITIVES` set re-introduced (variable only, no checks in shape functions) + bare `zombieUploadPass` seeded into `gpu.md` → still reds (set is not checked by any function; dead symbol caught by population predicate) |
| (viii-ctrl) | SHAPE_FALSE_POSITIVES control | 0 | `SHAPE_FALSE_POSITIVES` set re-introduced (variable only, no checks) without seeding a dead symbol → green (set has no effect; not checked by any function) |
| baseline | Clean tree | 0 | No mutation applied → green (1664 citations, 75 roster entries, 20 marker-exempted) |

## A — SHAPE_FALSE_POSITIVES deletion (three former entries)

The round-6 `SHAPE_FALSE_POSITIVES` set was an unpinned per-entry allowlist:
adding one string took a seeded dead symbol from exit 1 to exit 0 (measured
at `6ee6ed6`). It is deleted this round. The three former entries are
handled per site:

### `AoSoA` → prose fix (gpu.md:110)
**Move (3): edit the rule prose.** `AoSoA` (Array of Structures of Arrays)
is a GPU data-layout pattern name, not a code symbol. It was bare and
PascalCase-shaped, so it would match `matchesStrongShape` without the
exclusion. The prose is fixed: "**AoSoA tile = 32 elements.**" → "**Array
of Structures of Arrays tile = 32 elements.**" — the identifier-shaped
token is removed.

### `iGPUs` → prose fix (gpu.md:242)
**Move (3): edit the rule prose.** `iGPUs` (integrated GPUs) is a prose
term, not a code symbol. It was bare and camelCase-shaped. The prose is
fixed: "Intel iGPUs" → "Intel integrated GPUs" — the identifier-shaped
token is removed.

### `EndFrame` → re-spell onto live member (render.md:68, :161)
**Move (3): edit the rule prose.** `EndFrame` is a shorthand for the live
symbol `EndFrameSystem` (`packages/shallot/src/standard/render/index.ts:243`).
It was bare and PascalCase-shaped. The prose is re-spelled onto the live
member: bare `EndFrame` → `EndFrameSystem` (which resolves in the tree token
index). `EndFrameSystem` is already backticked earlier on the same lines,
so the bare re-spelling deduplicates against the existing backticked
candidate (no count change).

## B — bare weak shapes re-admitted

**Move (1): re-admit bare snake_case and bare lowercase-with-digits into
the population.** Measured at `6ee6ed6` before the fix: 11 distinct bare
weak-shape tokens that did not resolve — 5 snake_case (`warp_size`,
`theoretical_min`, `bytes_moved`, `peak_BW`, `webgpu_inspector`) and 6
lowercase-with-digits (`l12`, `l4`, `snorm8`, `f16x2`, `wg32`, `memory64`).
77 resolving bare weak-shape occurrences also become new candidates (e.g.
`u32` 15×, `box3d` 14×, `vec4` 5×, `base64` 4×, `unorm16` 8×).

Each of the 11 non-resolving sites is adjudicated:

### `webgpu_inspector` → roster (gpu.md:321)
**Genuine foreign tool name.** Added to `FOREIGN_NAMESPACES.Tools` in
`scripts/rosters.ts` (alongside `PowerVR` and `RenderDoc`).

### `memory64` → roster (tumble.md:105)
**Genuine foreign tool name.** Added to `FOREIGN_NAMESPACES.Tools` in
`scripts/rosters.ts`.

### `warp_size` → prose fix (gpu.md:169)
**Prose term** (GPU hardware: warp/wavefront size). Fixed: "(warp_size + 1)"
→ "(warp size + 1)".

### `theoretical_min` → prose fix (gpu.md:215)
**Prose term** (formula variable). Fixed: "5× theoretical_min," → "5× the
theoretical minimum,".

### `bytes_moved` → prose fix (gpu.md:219)
**Prose term** (benchmark metric). Fixed: "bytes_moved/peak_BW floor" →
"bytes-moved / peak-BW floor". (Also inside a fenced code block at
gpu.md:211, which the arm skips.)

### `peak_BW` → prose fix (gpu.md:219)
**Prose term** (benchmark metric: peak bandwidth). Fixed in the same edit
as `bytes_moved`.

### `snorm8` → prose fix (gpu.md:134)
**Prose term** (data format: signed-normalized 8-bit). Fixed: "→ snorm8
via" → "→ snorm-8 via".

### `f16x2` → prose fix (gpu.md:153)
**Prose term** (data format: half-precision float × 2). Fixed: "plain f16x2."
→ "plain F16 × 2.".

### `wg32` → prose fix (gpu.md:338)
**Prose term** (GPU hardware: workgroup 32). Fixed: "or wg32 multi-lane)"
→ "or WG32 multi-lane)".

### `l12` → prose fix (avbd.md:76)
**Prose term** (benchmark label: 12-layer pile). Fixed: "16384 l12" →
"16384 L12".

### `l4` → prose fix (avbd.md:76)
**Prose term** (benchmark label: 4-layer pile). Fixed: "32768 l4 churn" →
"32768 L4 churn".

## Punch item 5 — detect-stale-claims.ts deletion residue

The file `scripts/detect-stale-claims.ts` was deleted in round 5, but its
name survived in `scripts/check-docs.ts:803` (the token-index exclusion
comment). Fixed: the dead name is removed from the comment. The actual
exclusion in `buildTokenIndex` (`stale-claim-predicates.ts`) never listed
`detect-stale-claims.ts` — only `check-docs.ts`, `rosters.ts`, and
`stale-claim-predicates.ts` are excluded. The `stale-claim-predicates.ts:2`
comment ("One copy of the shape functions — two copies is two detectors
that disagree") is updated to not imply a second copy exists. The arm (e)
section comment's stale "per-site residue" mention is updated.

## Punch item 7 — sweep candidate verdicts

### `AGENTS.md:22` — index-order claim
**Live.** The line maps file path patterns to `.claude/rules/tumble.md`.
All listed paths exist (`packages/shallot/src/standard/tumble/`,
`packages/shallot/rust/tumble/`, `packages/shallot/tests/tumble/`,
`scripts/bench-tumble.ts`, `scripts/tumble-interaction.ts`,
`scripts/check-tumble-fp.ts`, and the `packages/shallot/scripts/tumble-*`
scripts). No stale claim.

### `extras/outline/passes.ts:12` — JFA pass-count comment
**Live.** The comment says `MAX_WIDTH = 64` bounds the JFA pass count
(`ceil(log2(width))`). The code at line 14 confirms `export const MAX_WIDTH
= 64;` and line 19 confirms the `ceil(log2(width))` formula.
`ceil(log2(64)) = 6` passes. No stale claim.

### `scripts/check-site.ts:220` — clause-6 stale reason
**Live.** The comment describes a limitation of substring checks: they pin
individual fragments but not the composition. The example (a paren dropped
between `RUM_ENV_USAGE` and `JSON.stringify(RUM_CONFIG)`) is a valid
description of the code's own limitation. No stale claim.

### `harness/pixels.ts:93` — doc-vs-predicate
**Live.** The docblock says "whether a `probePixels` measurement clears
`probe`'s thresholds." The function checks `result.pixels >=
probe.minPixels && Math.max(result.width, result.height) >=
probe.minSpan` — the two thresholds in `PixelProbe` (`minPixels`,
`minSpan`). `probePixels` is a live function at `pixels.ts:49`.
`PixelProbe` is a live interface at `pixels.ts:26`. The docblock accurately
describes what the function checks. No stale claim.

### `harness/index.ts:117` — ready comment
**Live.** The comment says "elapsed advances only after the first frame
steps — so a truthy read means build finished (this ran) and the RAF loop
has driven at least one draw." The `ready` getter returns
`state.time.elapsed > 0`. `elapsed` is advanced by the frame loop
(confirmed by `index.test.ts:15`: "ready follows the elapsed clock (built
+ drawn once a frame has stepped)"). No stale claim.

### `AGENTS.md`/`CLAUDE.md` doc tier — population exclusion
**Stated exclusion.** The arm's citation resolution scans only
`.claude/rules/**/*.md` (the population is derived from `git ls-files
'*.md'` filtered to `.claude/rules/`). `AGENTS.md` and `CLAUDE.md` are
not under `.claude/rules/`, so they are excluded from the citation
population. This is a stated, intentional exclusion — the arm's scope is
the rules corpus, not all markdown files. A stale claim in `AGENTS.md` or
`CLAUDE.md` is not caught by this arm; it is out of scope for this spec's
sweep. The reason the doc tier is out: `AGENTS.md` and `CLAUDE.md` are
context-loader entry points, not rules — they are read by every session
and change for operational reasons (path mappings, workflow notes) that
are not the stale-citation defect class this spec exists to sweep. The
rules corpus is the set of files that teach durable contracts, which is
where a stale symbol citation is a defect at any age.
