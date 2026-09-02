# Shallot

WebGPU game engine. The repo is the documentation: readable source with JSDoc contracts on every public export, a problem-indexed examples corpus, and this file plus `.claude/rules/` as the behavioral contract. There is no docs site and no editor — agents and humans both read the source, the examples index (`examples/AGENTS.md`), and `packages/shallot/AGENTS.md` (the consumer-facing contract — how to build games on Shallot: ECS, plugins, GPU, render, physics, testing — shipped with the npm package). This file covers engine-internal layout and commands; behavioral constraints live in `.claude/rules/`.

**Layout:** `packages/shallot/` (engine — `src/engine/`, `src/standard/`, `src/extras/`, `src/project/` (manifest/generate/vite toolchain behind the CLI)), `packages/create-shallot/` (the `bun create shallot` scaffold), `evals/` (the eval harness), `examples/` (standalone projects).

Code is source of truth — elegant first, document what's non-obvious.

## Rules

`.claude/rules/style.md` has no `paths:` frontmatter and always applies. Other globs match repo-root paths (`packages/shallot/…`), not cwd:

- `packages/shallot/src/standard/audio/**/*.ts`, `packages/shallot/rust/audio/**/*.rs` → `.claude/rules/audio.md`
- `packages/shallot/src/standard/avbd/**/*.ts`, `packages/shallot/tests/avbd/**/*.ts` → `.claude/rules/avbd.md`
- `packages/shallot/src/engine/**/*.ts`, `packages/shallot/src/standard/**/*.ts`, `packages/shallot/src/extras/**/*.ts`, `examples/**/*.scene`, `examples/**/*.ts` → `.claude/rules/ecs.md`
- `examples/**/*.ts`, `examples/**/*.scene`, `examples/AGENTS.md` → `.claude/rules/examples.md`
- `packages/shallot/src/**/*.ts`, `packages/shallot/package.json`, `examples/showcase/ocean/src/ocean/**` → `.claude/rules/exports.md`
- `packages/shallot/src/engine/runtime/**/*.ts`, `packages/shallot/src/engine/utils/encode.ts`, `packages/shallot/src/standard/render/**/*.ts`, `packages/shallot/src/standard/sear/**/*.ts`, `packages/shallot/src/standard/part/**/*.ts`, `packages/shallot/src/standard/slab/**/*.ts`, `packages/shallot/src/standard/bvh/**/*.ts`, `packages/shallot/src/extras/{cells,gltf,lines,outline,profile,skin,sky,sprite,text}/**/*.ts`, `examples/showcase/ocean/src/ocean/**` → `.claude/rules/gpu.md`
- `packages/shallot/src/standard/physics/**/*.ts`, `packages/shallot/src/standard/character/**/*.ts`, `packages/shallot/src/standard/player/**/*.ts` → `.claude/rules/physics.md`
- `packages/shallot/src/standard/render/**/*.ts`, `packages/shallot/src/standard/sear/**/*.ts`, `packages/shallot/src/standard/glaze/**/*.ts`, `packages/shallot/src/standard/part/**/*.ts` → `.claude/rules/render.md`
- `packages/shallot/src/**/*.test.ts`, `packages/shallot/tests/**/*.ts`, `packages/shallot/bin/*.test.ts`, `packages/shallot/bin/*.probes.ts`, `scripts/install-test.ts`, `packages/shallot/scripts/build-tooling.ts`, `examples/showcase/ocean/test/**/*.test.ts`, `examples/showcase/ocean/test/**/*.oracle.ts` → `.claude/rules/testing.md`
- `packages/shallot/src/standard/tumble/**/*.ts`, `packages/shallot/rust/tumble/**`, `packages/shallot/tests/tumble/**`, `packages/shallot/scripts/build-tumble-kernel.ts`, `packages/shallot/scripts/run-tumble-fixtures.ts`, `packages/shallot/scripts/gen-tumble-fixtures.ts`, `packages/shallot/scripts/gen-tumble-gold.ts`, `packages/shallot/scripts/gen-tumble-sample-golds.ts`, `packages/shallot/scripts/tumble-exit-test.ts`, `examples/gym/src/tumble-*.ts`, `examples/gym/src/scenarios/**`, `scripts/bench-tumble.ts`, `scripts/tumble-interaction.ts`, `scripts/tumble-repro*`, `scripts/check-tumble-fp.ts` → `.claude/rules/tumble.md`
- `examples/**/*.html` → `.claude/rules/visual-identity.md`

Read the ones whose globs match the files you're editing. Claude Code loads them for you on a matching read; other runtimes read them from this index. The `paths:` frontmatter in each rule is the source of truth — edit it first, then mirror it here, in order.

---

## Architecture

Bevy is the structural reference for ECS, plugin layout, and frame-graph shape — taken where it earns its place, skipped where it exists for Bevy's scale and constraints. The minimal scheduler/ECS isn't a TODO; it's a deliberate shape that compounds with TS hot reload and a small mental model to give shallot its iteration speed. Most of Bevy's mature ECS apparatus (parallel executor, change detection, deferred commands, typed system params, generational entities) solves multi-threaded CPU parallelism problems shallot doesn't have. Before adopting any Bevy-style machinery — or inventing an analogue where none is clean — read the take/skip lists, the two decision axes, and the frame shape (views, `BeginFrameSystem`, no compositor) in `.claude/rules/ecs.md` "Bevy as the structural reference".

## Platform support floor

Single modern WebGPU feature floor. No conditional fallback paths — a plugin has its required features or fails loud; the one sanctioned fallback site is a *preferred* feature (a fast arm where present, a fallback arm where absent — the BVH builder's subgroups→LDS). Assume the base floor + your required features are present; don't gate or write a fallback for those. Use features behind narrow interfaces (codec module, helper chunk) so a spec shift is a contained edit.

- **Base floor (every app):** WebGPU 1.0 + `indirect-first-instance` + `bgra8unorm-storage` + `rg11b10ufloat-renderable` — what the default renderer + slab substrate need, so they hold for any Shallot app. A floor entry earns its place only by being a `DEFAULT_PLUGINS` need; anything else rides the plugin that uses it, as `Plugin.features` (required — a missing one throws `UnsupportedError` before any plugin loads) or `Plugin.preferredFeatures` (best-effort). The deliberate absentees (`timestamp-query` → `ProfilePlugin`, the three texture-compression families → `GltfPlugin`, `shader-f16`) and the full feature-resolution mechanics are in `.claude/rules/gpu.md`.
- **Targets:** Chrome / Edge on desktop, recent Android Chrome, Safari 26+ on Apple Silicon (WebKit lacks only `subgroups`), Steam Deck. Native builds default to the platform's system webview (wry: WebView2 / WKWebView / WebKitGTK); `--portable` bundles CEF instead — required on Linux (WebKitGTK has no usable WebGPU). The backend/feature matrix, the WKWebView audit, and the `bin/features.ts` build-time warning are in `.claude/rules/gpu.md` "Native targets and webview backends".
- **Diagnostic tier:** boot, display unsupported-configuration message, exit. Not a degraded path — a clear boundary. Firefox and pre-Gen11 Intel iGPUs sit here until they ship the floor.

---

## Commands

```bash
bun run test                                       # Fast unit tests (bun-webgpu) — the default gate
bun test ./packages/shallot/tests/avbd/*.oracle.ts # The f64 AVBD physics oracle — slow, run when you touch AVBD/physics
bun test ./examples/gym/src                        # Gym host-layer + tumble gold oracle — run when you touch the engine, host layer, or a twin
bun bench [--scenario <name> --seed --count --warmup --frames --param k=v --screenshot <path>]  # Gym scenario via `shallot verify` on a real device. --screenshot writes a post-run canvas PNG (visual check, not a gate)
bun bench --list | --for <paths...> | --sweep [--for <paths...>]  # Scenario roster (registration slugs, not filenames); which scenarios gate changed paths; that selection through verify's batch mode (testing.md)
bun run scripts/physics-bench.ts                    # AVBD physics perf + scaling sweep (drives the gym pile scenario + constraints/character rows)
bun check                                          # Read-only: tsc + Biome + checks
bun run format                                     # Biome (.ts/.js/.json) + scene formatter
bun run build                                      # All Rust artifacts (WASM + native window)
bunx shallot dev [dir]                             # Run the project standalone (vite HMR over its shallot.json; native --target = debug build + run)
bunx shallot build [dir]                           # Web build (Vite → dist/)
bunx shallot build --target <os> [--portable] [--release]  # Native build; <os> = windows | mac | linux. Default = system webview (WebView2 / WKWebView / WebKitGTK); --portable = CEF. Linux needs --portable; warns (never blocks) on mismatch
bunx shallot run [dir] [--target <os>] [--portable]  # Build + run (web preview, or native; windows via WSL→Windows)
bun local [name]                                   # Scaffold local test project with packed engine (manual poking)
bun run test:install                               # Real-install gate: pack engine + a plugin lib, bun install, assert build/dev/create flows
bun run flows [--flow <name>]                      # Standalone-app engine flows — ejected apps under `examples/flows/` driven by `shallot verify` (blank, no-walls, survive-reload, ui-containment)
bun run recipes [--recipe <name>]                  # Physics recipes' dynamics smoke — each ported recipe's `window.__harness` asserts its observable (platform slides, rotor spins, joint breaks…), driven by `shallot verify`
```

The verification vehicle is the **shipped gate**: `shallot verify [dir]` (`packages/shallot/bin/verify.ts`) boots a project in a real headless browser, waits for it to render (or drives the `window.__harness` a project installs — the published protocol on `@dylanebert/shallot/harness`), reads a pass/fail Verdict, exits 0/nonzero. There is no repo-private harness tier: `bun bench` (`scripts/bench.ts`, over `examples/gym`) and `bun run flows` (`scripts/flows.ts`, over the ejected apps in `examples/flows/`) are thin wrappers over that CLI.

Gym is the single real-device surface: `bun bench` drives `examples/gym` (default `render`), prints the measure, and gates on its checks (failure exits nonzero). It's the **targeted** tier — one atom run after its domain changes, opposite `bun test`'s hardware-invariant run-all (split: `testing.md`) — holding permanent param-driven regression atoms and in-flight scenarios that run + render before earning a gate (`assert` optional until then); maintained like the test suite, not held pristine. The scenario contract is `.claude/rules/examples.md` "Gym scenario contract".

### Verification

Run `bun run format`, `bun check`, `bun run test` before completing work. The slow gates are separate files run by path, when you touch what they gate, never as a routine sweep: `bun test ./packages/shallot/tests/avbd/*.oracle.ts` for AVBD/physics, `bun test ./examples/gym/src` for engine / host-layer / twin changes, and the tumble fixture gates from `packages/shallot` (`tumble.md`). `cargo test` after Rust audio changes (from `packages/shallot/rust/audio`). `bun bench` after GPU code changes (`testing.md`). `bun run flows` after serialize/restore, `config.ui`/`mountOverlay`, or `shallot dev` server changes — standalone-app Playwright flows (display-gated, self-terminating, run alone). `bun run recipes` after a physics-recipe or substrate/tumble change — dynamics smoke (display-gated, same shape as flows). On WSL these three run for real against the Windows host's GPU via `scripts/wsl-bridge.ts` (browser server + reverse tunnel, verify CLI `--connect`), skipping only when the host lacks the node/bun it needs. `bun run test:install` after packaging / CLI / manifest-resolution / asset-shipping changes and after `packages/create-shallot` changes — the dev symlink hides real-install bugs (`testing.md` "Install gate").

---

## Examples

Four groups under `examples/`, indexed by `examples/AGENTS.md` — one line per entry: the problem, the path, what it shows. The index is the retrieval surface; grep it before writing a pattern from scratch. The corpus contract — what each tier is and what an entry must be — is `.claude/rules/examples.md`. Every tier runs through the same shipped gate (`shallot verify`); none reaches into repo tooling.

- `recipes/` — the teaching tier: one minimal manifest project per problem a game developer actually has, named by the problem, compile-gated
- `gym/` — the testing-harness tier: one project, `?scenario=`-selected param-driven atoms (above)
- `flows/` — standalone-app engine flows a `bun test` can't reach (`blank`, `no-walls`, `survive-reload`, `ui-containment`), verify-only via `bun run flows`
- `showcase/` — the capability tier: one self-contained project per subdir, each owning its own testing against the published surface — never repo `scripts/`. Live at dylanebert.com/shallot

The hello path is the scaffold: `bun create shallot` (`packages/create-shallot/index.ts`, the single source — no committed starter copy) emits a minimal project whose AGENTS.md points back at the engine's agent surface, with CLAUDE.md importing it; `bun run test:install` gates the create flow. Every example ships `public/icon.svg`; always `dispose()` State on HMR/unmount — without it, each reload stacks another State + RAF loop. App UI follows the sandboxed-container contract in `packages/shallot/AGENTS.md` "UI"; the one exemption — an *ejected* example that owns its full page and is never embedded (`gym`, `showcase/visualization`) may own the viewport with `position: fixed`. Project shape and per-tier run commands, icon details, and the Svelte UI pattern: `.claude/rules/examples.md` "Corpus-wide conventions".
