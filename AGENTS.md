# Shallot

WebGPU engine. JSDoc owns APIs. Read `packages/shallot/AGENTS.md` (consumer); grep `examples/AGENTS.md` first.

Owners: `packages/shallot{,-runtime,-cli,-tumble}`, `packages/create-shallot`; evals, examples.

## Rules

Read `.claude/rules/style.md` always; matching rules below via authoritative `paths:` frontmatter. Repo-root globs govern delivery, not authority. Claude Code loads on matching reads; others read manually. Edit frontmatter, not duplicate globs.

In `.claude/rules/`: `audio.md`, `avbd.md`, `ecs.md`, `examples.md`, `exports.md`, `gpu.md`, `physics.md`, `render.md`, `testing.md`, `tumble.md`, `visual-identity.md`. Testing includes `scripts/stall-attribution.ts`, `scripts/compile-concurrency.ts`, `scripts/loaf-attribution.ts`, `site/rum-*.ts` and its frontmatter paths.

## Architecture

Before Bevy machinery/analogues, read ecs.md "Bevy as the structural reference": take/skip, adoption axes, views/frame. Small immediate ECS/scheduler is deliberate.

## Platform support floor

One WebGPU 1.0 floor: indirect-first-instance, bgra8unorm-storage, rg11b10ufloat-renderable. Only default-plugin needs belong here; others declare required/preferred plugin features. Missing required fails before loading; only preferred features may fall back (BVH subgroups→LDS). Don't gate guaranteed features. Keep feature use behind narrow interfaces.

Targets: desktop Chrome/Edge, recent Android Chrome, Safari26+ Apple Silicon, Steam Deck. Native defaults: wry WebView2/WKWebView/WebKitGTK; portable CEF required on Linux. Admission by features/limits, not browser name; a missing floor boots, explains, exits. Details: `gpu.md`.

## Commands

```bash
bun run test                              # Unit gate; needs a native adapter
bun run test:changed -- --base <ref> --diff <ref>
bun run demos                             # Separate release site build + verify --dist; skips fail
bun run site; bun run site:pages          # Ejected demos, not runtime proof; pages only
bun run rum-intake [--demo <slug>]         # RUM wire check; locally fulfilled intake
bun check                                 # Read-only tsc/Biome/checks
bun run format                            # Biome + scenes
bun run build                             # Rust WASM + native window
```

```bash
bun bench --scenario <name> [--seed --count --warmup --frames --param k=v --screenshot <path>]
bun bench --list | --for <paths...> | --sweep [--for <paths...>]
bun run scripts/physics-bench.ts
bun local [name]                          # Packed local scaffold
bun run test:install                      # Pack engine/plugin, install, build/dev/create
bun run recipes [--recipe <name>]
```

```bash
bun packages/shallot-cli/bin/cli.ts <dev|build|run|verify> [dir]
# build/run: [--target <os>] [--portable]; build: [--release]
```

OS: windows/mac/linux; web build emits dist, run builds/previews; native dev runs debug, Windows cross-compiled with cargo-xwin. Shipped verify owns browser Verdict/exit, published `/harness`; bench/recipes wrap it, no private tier. Gym defaults render; slugs select atoms. Screenshots never gate. Laws: `examples.md`.

### Verification

Before completion: format, check, test above. Release order: `testing.md` (all-roster AND separate demos). After AVBD/physics: `bun test ./packages/shallot/tests/avbd/*.oracle.ts`; tumble golds: `bun test ./packages/shallot-tumble/tests/tumble-golds.tier.ts`; tumble fixtures from package per `tumble.md`; Rust audio: `cargo test` from `packages/shallot-runtime/rust/audio`.

GPU changes owe bench; serialize/restore, config.ui/mountOverlay or dev-server changes owe `flow-*.tier.ts`; physics-recipe/substrate/tumble changes owe recipes. Display gates self-terminate, run headed on the seat's own display and run alone; no display refuses, never skips green. Packaging/CLI/manifest/assets/scaffold changes owe test:install; symlinks hide install defects.

## Examples

Tiers/conventions: `examples/AGENTS.md`, `examples.md`. Sole hello source: `packages/create-shallot/index.ts`, no starter copy; emitted AGENTS points to engine, CLAUDE imports it. Examples ship icon.svg, dispose State on HMR/unmount, obey package AGENTS UI containment; only ejected never-embedded gym/visualization may own viewport.
