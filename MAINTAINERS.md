# Maintaining Shallot

Repo-level contract; `AGENTS.md` is the consumer contract and ships in the npm package, this file does not. Grep `examples/AGENTS.md` first.

Owners: the root package (`src`, `bin`, `rust`, `tests`), `packages/{shallot-tumble,create-shallot}`; evals, examples.

## Rules

Read `.claude/rules/style.md` always; matching rules below via authoritative `paths:` frontmatter. Repo-root globs govern delivery, not authority. Claude Code loads on matching reads; others read manually. Edit frontmatter, not duplicate globs.

In `.claude/rules/`: `audio.md`, `avbd.md`, `ecs.md`, `examples.md`, `exports.md`, `gpu.md`, `physics.md`, `render.md`, `testing.md`, `tumble.md`, `visual-identity.md`. Testing includes `scripts/stall-attribution.ts`, `scripts/compile-concurrency.ts`, `scripts/loaf-attribution.ts`, `site/rum-*.ts` and its frontmatter paths. Also read `tumble.md` for `scripts/tumble-repro*`.

## Architecture

Before Bevy machinery/analogues, read ecs.md "Bevy as the structural reference": take/skip, adoption axes, views/frame. Small immediate ECS/scheduler is deliberate.

## Platform support floor

One WebGPU 1.0 floor: indirect-first-instance, bgra8unorm-storage, rg11b10ufloat-renderable. Only default-plugin needs belong here; others declare required/preferred plugin features. Missing required fails before loading; only preferred features may fall back (BVH subgroups→LDS). Don't gate guaranteed features. Keep feature use behind narrow interfaces.

Targets: desktop Chrome/Edge, recent Android Chrome, Safari26+ Apple Silicon, Steam Deck. Native defaults: wry WebView2/WKWebView/WebKitGTK; portable CEF required on Linux. Admission by features/limits, not browser name; a missing floor boots, explains, exits. Details: `gpu.md`.

## Commands

```bash
bun run build                             # After bun install: audio WASM, dist/ tooling, native window
bun run test                              # Unit gate; needs build and a native adapter
bun run test:changed -- --base <ref> --diff <ref>
bun run demos                             # Separate release site build + verify --dist; skips fail
bun run site; bun run site:pages          # Ejected demos, not runtime proof; pages only
bun run rum-intake [--demo <slug>]         # RUM wire check; locally fulfilled intake
bun check                                 # Read-only tsc/Biome/checks; pack test is check-pack
bun run format                            # Biome + scenes
bun run prepack                           # dist/ tooling only; bun pm pack runs it
```

```bash
bun bench --scenario <name> [--seed --count --warmup --frames --param k=v --screenshot <path>]
bun bench --list | --for <paths...> | --sweep [--for <paths...>]
bun run scripts/physics-bench.ts
bun run audio:wasm-bench                  # Audio WASM throughput
bun run dump-cells-ascii                  # Cells debug dump
bun local [name]                          # Packed local scaffold
bun run test:install                      # Pack engine/plugin, install, build/dev/create
bun run flows [--flow <name>]
bun run recipes [--recipe <name>]
```

```bash
bun bin/cli.ts <dev|build|run|verify> [dir]
# build/run: [--target <os>] [--portable]; build: [--release]
```

OS: windows/mac/linux; web emits dist; native uses platform tools. Verify owns Verdict/exit, full-Chromium headless; `--headed` for display, `--connect` remote. Published `/harness`; bench/flows/recipes wrap it. Gym defaults render; slugs select atoms. Screenshots never gate. Laws: `examples.md`.

### Verification

Before completion: format, check, test above. Release order: `testing.md` (all-roster AND separate demos). After AVBD/physics: `bun test ./tests/avbd/*.oracle.ts`; engine/host/twin: `bun test ./examples/gym/src`; tumble fixtures from package per `tumble.md`; Rust audio: `cargo test` from `rust/audio`.

GPU/serialize/restore/config.ui/dev-server/physics changes owe bench/flow/recipe gates. Verify is headless; hardware refusal is nonzero; display callers use `--headed`. Reachability repairs owe source + physical public-build proof via build.probes.ts; other CLI/manifest/dependency/launch/runtime/scaffold/native-package changes owe test:install; links don’t prove it.

## Examples

Tiers/conventions: `examples/AGENTS.md`, `examples.md`. Sole hello source: `packages/create-shallot/index.ts`, no starter copy; emitted AGENTS points to engine, CLAUDE imports it. Examples ship icon.svg, dispose State on HMR/unmount, obey package AGENTS UI containment; only ejected never-embedded gym/visualization may own viewport.
