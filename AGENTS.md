# Shallot

WebGPU engine. JSDoc owns APIs. Read `packages/shallot/AGENTS.md` (consumer); grep `examples/AGENTS.md` first.

Owners: `packages/shallot{,-runtime,-cli,-tumble}`, `packages/create-shallot`; evals, examples.

## Rules

Always read `.claude/rules/style.md`; others by authoritative `paths:`. Root globs govern delivery, not authority. Non-Claude runtimes load manually; edit frontmatter, not duplicate globs.

In `.claude/rules/`: `audio.md`, `avbd.md`, `ecs.md`, `examples.md`, `exports.md`, `gpu.md`, `physics.md`, `render.md`, `testing.md`, `tumble.md`, `visual-identity.md`.

## Architecture

Before Bevy analogues, read ecs.md "Bevy as the structural reference". Small immediate ECS/scheduler is deliberate.

## Platform support floor

One WebGPU 1.0 floor: indirect-first-instance, bgra8unorm-storage, rg11b10ufloat-renderable. Only default-plugin needs belong here; others declare required/preferred plugin features. Missing required fails before loading; only preferred features may fall back (BVH subgroups→LDS). Don't gate guaranteed features. Keep feature use behind narrow interfaces.

Targets: Chrome/Edge, recent Android Chrome, Safari26+ Apple Silicon, Steam Deck. Native: wry WebView2/WKWebView/WebKitGTK; Linux needs portable CEF. Admit by features/limits, not browser name; missing floor boots, explains, exits (`gpu.md`).

## Commands

```bash
bun run test # Unit gate; native adapter required
bun run test:changed -- --base <ref> --diff <ref>
bun run demos # Separate release site build + verify --dist; skips fail
bun run site; bun run site:pages # Ejected demos, not runtime proof; pages only
bun run rum-intake [--demo <slug>] # RUM wire; local intake
bun check # Read-only tsc/Biome/checks
bun run format # Biome + scenes
bun run build # Rust WASM + native window
```

```bash
bun bench --scenario <name> [--seed --count --warmup --frames --param k=v --screenshot <path>]
bun bench --list | --for <paths...> | --sweep [--for <paths...>]
bun run scripts/physics-bench.ts
bun local [name] # Packed local scaffold
bun run test:install                      # Pack engine/plugin, install, build/dev/create
bun run recipes [--recipe <name>]
```

```bash
bun packages/shallot-cli/bin/cli.ts <dev|build|run|verify> [dir]
# build/run: [--target <os>] [--portable]; build: [--release]
```

OS: windows/mac/linux. Web build: dist; run: build/preview; native dev: debug; Windows cross-compile: cargo-xwin. Verify owns browser Verdict/exit and published `/harness`; bench/recipes wrap it. Bench defaults render; slugs select atoms. Screenshots never gate (`examples.md`).

### Verification

Before completion: format, check, test above. Release order: `testing.md` (all-roster AND separate demos). After AVBD/physics: `bun test ./packages/shallot/tests/avbd/*.oracle.ts`; tumble golds: `bun test ./packages/shallot-tumble/tests/tumble-golds.tier.ts`; tumble fixtures from package per `tumble.md`; Rust audio: `cargo test` from `packages/shallot-runtime/rust/audio`.

GPU changes owe bench; serialize/restore, config.ui/mountOverlay or dev-server changes owe `flow-*.tier.ts`; physics-recipe/substrate/tumble changes owe recipes. Orbit/pointer/touch edits: `bunx playwright test -c packages/shallot/tests/orbit-touch/playwright.config.ts`. Verify pixel/wait edits: `bun test ./packages/shallot-cli/bin/verify-blank.tier.ts`.

Attribution capture/transport, in-page sampling/polling, launch edits or real attribution evidence: `SHALLOT_DISPLAY_REQUIRED=1 bun run scripts/stall-attribution.ts --dir examples/showcase/sandbox` checks live contamination and observed headed UA. Fixtures aren't capture evidence.

Display gates self-terminate, headed on the seat, alone; absent display/skips aren't green. Packaging/CLI/manifest/assets/scaffold edits owe test:install; symlinks hide install defects.

## Examples

`examples/AGENTS.md` and `examples.md` own conventions. Sole hello: `packages/create-shallot/index.ts`; emitted AGENTS points to engine, CLAUDE imports it. Ship icon.svg, dispose State on HMR/unmount, obey package UI containment; only ejected never-embedded bench/visualization may own viewport.
