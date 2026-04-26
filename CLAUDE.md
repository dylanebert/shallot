# Shallot

WebGPU game engine. In-development.

For consumer-facing patterns (how to build games on Shallot — ECS, plugins, GPU, render, physics, testing), see `packages/shallot/AGENTS.md`. This file covers engine-internal layout and commands. Behavioral constraints live in `.claude/rules/`.

**Layout:** `packages/shallot/` (engine — `src/engine/`, `src/standard/`, `src/extras/`, `src/editor/`), `packages/shallot/editor/` (Svelte editor app), `examples/` (standalone projects).

**Docs:** `docs/guide/`, `docs/engine/`, `docs/standard/`, `docs/extras/`, `docs/editor/`. Code is source of truth — make it elegant first, document what's non-obvious. Three tabs (UI/Code/Reference) with distinct audiences — see `.claude/rules/docs.md`. `bun run build` generates `docs/dist/` with API tables from source exports.

---

## Commands

```bash
bun test                                           # Unit tests (bun-webgpu)
bun bench                                          # Benchmark — default (raster)
bun bench -- --scenario pile                       # Physics scenario (default 100 bodies)
bun bench -- --pipeline raytracing --frames 100     # Custom params
bun run scripts/shader-compile.ts                   # Shader compilation profiling (Vulkan/bun-webgpu)
bun run scripts/shader-compile-chrome.ts             # Shader compilation profiling (Chrome/DX12 via WSL)
bun check                                          # Format + type check (Biome + tsc)
bun run check:svelte [filter]                       # Svelte compiler diagnostics (optional: filter by dir name)
bun run format                                     # Biome + scene formatter
bun run build                                      # All Rust artifacts (WASM + native window) + docs
shallot build [dir]                                # Web build (Vite → dist/)
shallot build --target windows [dir]               # Windows webview build (→ build/windows/debug/)
shallot build --target windows --release [dir]     # Windows webview release (→ build/windows/release/)
shallot build --target mac [dir]                   # macOS .app build (→ build/mac/debug/)
shallot build --target mac --release [dir]         # macOS .app release (→ build/mac/release/)
shallot build --target linux [dir]                 # Linux build (→ build/linux/debug/)
shallot build --target linux --release [dir]       # Linux release (→ build/linux/release/)
shallot run [dir]                                  # Web build + preview server
shallot run [dir] --target windows                 # Windows debug build + run (WSL→Windows)
shallot run [dir] --target windows --release       # Windows release build + run (single exe)
shallot run [dir] --target mac                     # macOS build + run
shallot run [dir] --target linux                   # Linux build + run (CEF)
bun local [name]                                   # Scaffold local test project with packed engine
bun run capture --out <dir>                        # Editor screenshot capture (WebP + manifest)
bun run capture --out <dir> --flow editor-layout   # Single capture flow
```

GPU test args: `--scenario`, `--pipeline`, `--frames`, `--warmup`, `--count`, `--effects`, `--camera`, `--layout`, `--test`. One invocation = one page load = one measurement. Physics scenarios: `pile` (body stress, scales with `--count`, default 100), `physics --test <name>` (AVBD solver variant: box, sphere, capsule, cone, mixed, stack, pyramid, rope, heavy-rope, spring, spring-chain, bridge, gravity, force, impulse, velocity, filter).

### Verification

Run `bun run format`, `bun check`, `bun test` before completing work. `cargo test` after Rust audio changes (run from `packages/shallot/rust/audio`). `bun bench` required after GPU code changes. `bun run capture` after editor UI changes.

---

## Examples

`cd examples/<name> && bun dev`. Editor: `shallot examples/<name>/`.

Every example must include `public/icon.svg` (the shallot icon) and a `<link rel="icon" type="image/svg+xml" href="/icon.svg" />` in `index.html`. For native builds, `public/icon.png` becomes the window icon; if absent, the default shallot icon is used. Always `dispose()` State on HMR/unmount — without it, each hot-reload stacks another State + RAF loop.

**UI convention:** Control panels use `config.ui`: `(container: HTMLElement, state: State) => () => void`. UI receives State directly — read/write ECS components, query entities, access resources. No bridge pattern needed. Framework-agnostic — any framework that mounts into a DOM element works. **Must return a cleanup function** — called on dispose to cancel animation loops, unmount frameworks, etc. Return `() => {}` if no cleanup is needed. Examples with complex UI (e.g. `workstation/`) can own their full page layout with Svelte — add `svelte`, `@sveltejs/vite-plugin-svelte` as deps, copy `svelte.config.js` from the editor, and mount via `svelte`'s `mount()`/`unmount()`.
