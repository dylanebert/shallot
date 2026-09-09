---
paths:
  - "packages/{shallot-runtime/src/**/*.test.ts,shallot-runtime/scripts/*.ts,shallot-tumble/**/*.ts,shallot/tests/**/*.ts,shallot-cli/**/*.{test,probes}.ts,shallot-cli/scripts/build.ts}"
  - "scripts/{install-test.ts,install-test/**/*.ts,boot-cost.ts}"
  - "bench/**/*.ts"
  - "examples/showcase/ocean/test/**/*.{test,oracle}.ts"
  - "site/rum-*.ts"
---

# Testing

Gates never write. Root commands: AGENTS.md; budgets: style.md; paths: manifest.

## What a change class owes

Keep AGENTS.md's Verification triggers; price selected page boots. Narrow bench/recipe selectors, not coverage. A row's cone is its own directory plus the modules its check claims about; the per-file ceiling ratchets down only (`example-gates.test.ts`). Close: `bun run test:changed -- --base <ref> --diff <ref>`. CPU precedes display; deleted covers select. Zero/unavailable/success differ; `SHALLOT_DISPLAY_REQUIRED=1` refuses unavailable.

Ocean gates: `bun run test:ocean-realization`, `bun run test:ocean-slope`, `bun run test:ocean-mesh-inversion`, `bun run test:ocean-fold`. Headers own bounds/controls, `scripts/ocean-oracle-gates.ts` the cones. Reduced fold isn't production proof; mesh flips aren't fidelity.

## GPU testing

CPU TGSL proves logic; JS mirrors need bit-exact differentials. Default tests may bind adapters but cannot depend on execution, except slab repooling. Software adapters de-risk lifecycle, not correctness. Device truth lives once in the bench or a headed tier; move misplaced assertions, never skip. Constants-only preload waits for build tests/probes to own setup. Feature/limit law: `gpu.md`.

Readback: exported columns, bitcast IDs as integers. Separate CPU/compile/API/resource/product proof; justify omissions. CPU optional only without a meaningful oracle. Shader diagnostics perturb until resource/output agrees. Record failing boundary/label/hash.

### Selecting and sweeping bench scenarios

Select by a scenario's own `covers` — one table, read by `bun bench --for` and the roster row. Fresh context/page per verdict; isolate ratios, not counts. No GPU parallelism or page reuse. Correlate WGSL within functions; call syntax isn't result proof.

Asserts own bounded advancement in derived units; exhaustion fails. Free space budgets both swept reaches; name prerequisite asserts. Prove `--warmup 0 --frames 1`. Setup dominates; shorter budgets/dist/parallelism/reuse rejected. CI/device baselines are non-goals.

## Perf gates

Exact goldens: labels, raw calls (needn't equal), non-lazy GPU bytes. Pin producer marks at allocators; exclusion misses stager size/capacity. AVBD exactness is topology-limited. Timing gates: in-run scaling/sibling ratios only. No CPU pacing thresholds.

### Freezing a golden

Harvest through production: three agreeing samples unless mechanism-exact with corroboration. Samples cannot make nondeterminism exact. Tighten quantities; exemptions key subject/quantity. Attribute bytes per label. Control reds against recorded-green trees; retire proxies when mechanisms are gated. Larger windows cannot suppress unbounded tails.

Counts miss compile/transport time; sync spans are stubs unless forced. Compare built/dev before claiming player cost. `scripts/boot-cost.ts`: release-scale, not cold-start gate. Asset counts blind, bundle bytes ungated.

GPU timestamps, not RAF; CPU-mixed averages, pure-GPU minimum.

## Verdicts

Observe, don't guess causes; empty diagnostics mean incomplete. Flush before exit; reject empty/non-numeric flags. RUM reads emitters on clean profiles.

`verify` reads console/pixels; motion needs differing frames. Bench needs a scenario. `bin/verify.ts` owns center-vs-corner ready/post-run OR: flashes pass, capture fails closed, paint isn't identity. Reframe, don't soften; assert buffer identity. Only app `noRender` opts out visibly. Blank tier is the negative control. Leak: idle, derived rate, manual `--leak` control; GC ungated. Instrument failure is incomplete; determined failure outranks unavailable.

## Tiers

A tier is what a check may touch, not a folder name; a verdict is as real as its tier. Step is the default, headed only where a claim names the browser.

| Tier | May touch | Cannot establish |
|---|---|---|
| step | CPU, stepped clock | device input, anything drawn |
| headless GPU | native adapter, readback | presentation, browser input |
| headed | browser on the seat | other hardware, deployment |

Quarantine is a file, not a comment: path, expiry, filed unit per line, counted against the gate budget; expired rows red.

CRUD/reload: bugs/novel lifecycles/dynamic adds; conformance: novel state/registries. ECS/app tests own atoms, tiers end-to-end. Pairwise GPU matrices plus compile.

One browser session/file, phased assertions. Probe adapter names before waits; software passes feature floors. Only device-dependent gates skip; skips aren't proof.

Suffix owner: `tests/test-tiers.ts`. Fast `.test.ts`; heavy CPU `.oracle.ts`/corpus `.tier.ts` by explicit `./` path on header triggers, keeping split sentinels. Browser/subprocess `.probes.ts` by-path, pure siblings fast: `bun run test:tui-probes`. `.lab.ts` temporary; `.playwright.ts` uses project configs.

Outside-cone tests need invoking gates. Headed by-path tiers run alone under `SHALLOT_DISPLAY_REQUIRED=1` — `flow-*.tier.ts`, `verify-blank.tier.ts`, `tests/orbit-touch` — triggered by AGENTS.md prose, not a selector row the ratchet can see.

Isolation diagnoses, never exempts: derive smaller scans or promote to oracle/tier with reasons; no cap raises.

## Tolerance tiers

Exact: `1e-10` or tighter. Truncation: integrator order/step size. Single f32 ops: about `1e-6` relative, accumulated by chain. Solver: iterations/penalty schedule. Never observation-tune.

## Fixtures and pins

Floor needs member install, matching audio JS/declarations/WASM and AVBD inputs. `GLTF_CORPUS_REQUIRED=1`; reconcile passes, not files/fails. Bench mounts stay separate. Missing artifacts block shared imports; build checks need required-mode absence reds. Native-FFI needs a conformant adapter (macOS arm64 and nvidia lovelace), else incomplete, not red; no browser-golden claim.

Bump pins together, resolve lock, confirm one TypeGPU identity before gates. Dependency/manifest edits: `bun run test:changed --all`.

## Install gate

Evals tools: `bun run evals/setup.ts`, `bun run evals/grade.ts`.

`bun run test:install`: pack, not link; browser plus build. Plain-Node export/brand checks precede display guards, on intended realpaths. Boot doc recipes; inspect artifact imports, not erasable markers. Controls must yield false, not absence. Brand identity, not autonaming; pnpm needs distinct versions. Native packed build is manual on packaging edits.

## Release gate

Cycle order: bump all `check-versions.ts` sites/deferred changelog entries; `bun run scripts/check-versions.ts --release` before pack; RC dogfood; publish engine and scaffold; then `SHALLOT_DISPLAY_REQUIRED=1 bun run test:changed --all` (every row, no skips), then separately `bun run demos` (built distribution through published-package ejected consumers; failures/skips nonzero), then merge and tag. Publish/tag together.

Dogfood registry-installed tracked-pin floors, never links. Peers/dev deps first, tarball last; reread installed versions. Preserve ignored artifacts; restore manifests/locks, inspect gitlinks/shared-root installs, leave RC runnable. Human live render required: hand over the command; decoded stills only support it.

Read prose against artifacts. Migration changes only for major/minors; historical facts aren't version sites. Deploy via `site.yml`, not disabled `pages.yml`. Confirm deployed version, archives/checksums and covered bytes. `bun run scripts/e2e-prebuilt.ts`: no-cargo proof.
