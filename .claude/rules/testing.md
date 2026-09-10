---
paths:
  - "src/**/*.test.ts"
  - "scripts/*.ts"
  - "tests/**/*.ts"
  - "{src,bin,tests,scripts}/**/*.{test,probes}.ts"
  - "scripts/build.ts"
  - "scripts/{install-test.ts,install-test/**/*.ts}"
---

# Testing

Gates never write. Root commands: AGENTS.md; budgets: style.md; test paths: manifest.

## What a change class owes

Keep AGENTS.md's Verification triggers; price selected page boots. Narrow bench/flow/recipe selectors, not coverage; install has none. Close: `bun run test:changed -- --base <ref> --diff <ref>`. CPU precedes display; deleted covers select. Zero/unavailable/success differ; `SHALLOT_DISPLAY_REQUIRED=1` refuses unavailable.

## GPU testing

CPU TGSL proves logic; JS mirrors need bit-exact differentials. Default tests may bind adapters but cannot depend on execution except slab repooling. Software-adapter tiers de-risk lifecycle, not correctness. Device truth lives once in gym/Playwright; move misplaced assertions, never skip. Constants-only preload waits for build tests/probes to own setup. Feature/limit law: `gpu.md`.

Readback: exported columns, bitcast IDs as integers. Separate CPU/compile/API/resource/product proof; justify omissions. CPU optional only without meaningful oracle. Shader diagnostics perturb until resource/output agrees; Mirror isn't causal. Record failing boundary/label/hash.

### Selecting and sweeping bench scenarios

Use registration/coverage. Fresh context/page per verdict; isolate ratios, not counts. No GPU parallelism/page reuse. Run `bun test ./examples/gym/src` on module/scenario/cover changes. Exempt by property/resolved subject; correlate WGSL within functions. Registries need named gaps and one checkable row/subject; call/import syntax isn't module/result proof.

Asserts own bounded advancement in derived units; exhaustion fails. Free space budgets both swept reaches; name prerequisite asserts. Prove `--warmup 0 --frames 1`. Setup dominates; shorter budgets/dist/parallelism/reuse rejected. CI/device baselines: non-goals. Install cost: boots; identity excess unexplained.

## Perf gates

Exact goldens: labels, raw calls (needn't equal), non-lazy GPU bytes. Pin producer marks at allocators; exclusion misses stager size/capacity. AVBD exactness is topology-limited. Timing gates: in-run scaling/sibling ratios only, otherwise ungated. No CPU pacing thresholds.

### Freezing a golden

Harvest through production: three independent agreeing samples unless mechanism-exact with corroboration. Samples cannot make nondeterminism exact. Tighten quantities; exemptions key subject/quantity. Attribute bytes per label. Control reds against recorded-green trees; retire proxies when mechanisms are gated. Larger windows cannot suppress unbounded tails.

Counts miss compile/transport time; sync spans are stubs unless forced. Compare built/dev before claiming player cost. Asset counts blind, bundle bytes ungated, cold-pipeline cost unmeasured.

Attribution kind isn't magnitude; non-script remainder isn't a JS owner. Assert launched conditions; no cross-seat causal deltas. GPU timestamps, not RAF; CPU-mixed averages, pure-GPU minimum.

## Verdicts

Observe, don't guess causes; empty diagnostics mean incomplete. Flush before exit; reject empty/non-numeric flags. RUM: read emitters, clean profiles.

`verify` reads console/pixels; motion needs differing frames. Gym needs a scenario. `bin/verify.ts` owns center-vs-corner ready/post-run OR: flashes pass, capture fails closed, paint isn't identity. Reframe, don't soften; assert buffer identity. Only app `noRender` opts out visibly. Blank flow is negative control. Leak: idle, derived rate, manual `--leak` control on changes; GC ungated. Instrument failure is incomplete; determined failure outranks unavailable.

## Tiers

CRUD/reload: bugs/novel lifecycles/dynamic adds; conformance: novel state/registries. ECS/app tests own atoms. Pairwise GPU matrices plus compile. `tests/standards.ts`: corpus/limits.

One browser session/file, phased assertions. Probe/log adapter names before waits; software can pass feature floors. Only device-dependent gates skip; skips aren't proof.

Suffix owner: `tests/test-tiers.ts`. Fast `.test.ts`; heavy CPU `.oracle.ts`/corpus `.tier.ts` by explicit `./` path on header triggers, retaining split sentinels. Browser/subprocess `.probes.ts` by-path, pure siblings fast. `.lab.ts` temporary; `.playwright.ts` uses project configs. Outside-cone tests need invoking gates. Walk all tiers/unloaded files; no duplicate roster.

Five-second cap stands until consumed distributions exist. Isolation diagnoses, never exempts: derive smaller scans or promote to oracle/tier with reasons; no cap raises. Only standard header classes earn scripts. Read oracle results, not reach sentinels.

## Tolerance tiers

Exact: `1e-10` or tighter. Truncation: integrator order/step size. Single f32 ops: about `1e-6` relative, accumulated by chain. Solver: iterations/penalty schedule. Never observation-tune.

## Fixtures and pins

Floor needs member install, matching audio JS/declarations/WASM and AVBD inputs. `GLTF_CORPUS_REQUIRED=1`; reconcile passes, not just files/fails. Gym mounts stay separate. Missing artifacts block shared imports; build checks need required-mode absence reds. In-tree isn't pack proof. Native-FFI needs a conformant adapter (macOS arm64 and nvidia lovelace), else incomplete, not red; no browser-golden claim.

Bump all pins together, resolve lock, confirm one TypeGPU identity before gates. Dependency/manifest edits: `bun run test:changed --all`. `check-docs.ts` pins commands, not history.

## Install gate

Evals tools: `bun run evals/setup.ts`, `bun run evals/grade.ts`.

`bun run test:install`: pack not link; browser/build; plain-Node export/brand/realpath/docs/imports, fail-closed; no selector. Reachability repairs owe `bun test --timeout 120000 ./bin/build.probes.ts`: source + physical public-build proof, retain/prune. Other CLI/manifest/deps/prebundle/launch/runtime/scaffold/native-package changes retain full install.

## Release gate

Cycle order: bump all `check-versions.ts` sites/deferred changelog entries; `bun run scripts/check-versions.ts --release` before pack; RC dogfood; publish engine and scaffold; then `SHALLOT_DISPLAY_REQUIRED=1 bun run test:changed --all` (every row, no green skips), then merge and tag. Publish/tag together.

Dogfood against registry-installed tracked-pin floors, never links. Peers/dev deps first, tarball last; reread installed versions. Preserve ignored artifacts; restore manifests/locks, inspect gitlinks/shared-root installs, leave RC runnable. Human live render required: hand over command; decoded stills only support it; name absent image proof.

Read prose against artifacts. Migration changes only for major/minors; historical facts aren't version sites. Confirm deployed version, six archives/checksums and covered bytes.
