# Shallot

WebGPU game engine: ECS, scenes, plugins. This is the repository's agent contract; consumers read the README and JSDoc, which owns APIs. Search `examples/AGENTS.md` and read a recipe first.

## Commands

```bash
bun run build          # audio WASM, dist/vite.js, physics kernel
bun run check          # tsc, biome, check-*.ts, scenes
bun run test
bun run format
bun run test:install   # pack, install, build/dev/add
bun run assets [name]  # --icons rewrites mark icons
bun run prepack
bun bin/shallot.ts <dev|build|run|add> [dir]
```

CLI, manifest, dependency, runtime and native-package changes owe `test:install`; a `bun link` of a local engine doesn't prove the packed shape.

## Toolchain and pins

- Bun per `packageManager`, TypeScript 7, Rust per `rust-toolchain.toml`. Biome is the only linter.
- `@types/node` and `@webgpu/types` are runtime dependencies because `types` points at source.
- `bun-webgpu` pins a `dylanebert/bun-webgpu` main commit until kommander/bun-webgpu#10 merges, then returns to upstream.
- Bump a pin with every doc and fixture site in one commit; `check-docs` reds on drift.
- Retired units are a Git tag plus an `ARCHIVE.md` row, never a directory.

## Philosophy

Components are data, systems behavior, not methods/managers. Scenes author composition; imperative setup is procedural/tests. Derive one truth; compose plugins, depend inward on pure core. Declare group/after/before, not manual sequencing; one terminal runs last (RenderPlugin owns draw submit).

## Imports

`@dylanebert/shallot`: author APIs/defaults; `/extras`: opt-in convenience, also bare. Audio/mirror are bare-only. `/runtime`: platform/device; `/<module>`: extensions. Never deep-import src; file issues for missing seams. Plugins register components/traits.

## ECS & Plugins

Initialize is pre-scene; warm handles scene data; system setup is lazy first-frame. Mesh/surface plugins depend on RenderPlugin after its registry wipe. Missing dependencies fail before side effects; optional peers use conditional inclusion, nullable hooks or absent-system ordering. Position producers run before PrepassSystem.

Marker + `not()` gates once; initialized singletons hold services, component eids relations (`@name`, flat scenes, no parent). Eids recycle: re-query, no module handles/ownership maps. Derive from `state.time.elapsed`; scope state, not last-State/exists guards.

## UI

Mount only in `config.ui(container, state)` or `mountOverlay(canvas, state)`, sandboxed to the canvas by layout/paint containment and clipping. Position relative, never fixed or on document.body. Container ignores pointer events; interactive children enable them. Return/register real framework unmount and cancel timers/rAF via State cleanup; use `state.signal` for listeners/fetch. Removing DOM alone leaks effects. Re-warm without dispose must clear its prior mount first; retain dispose cleanup too.

## GPU

Transform TGSL exactly once. Compute on Render.encoder; register on render. Only allocators destroy. CPU truth is typed arrays, not per-frame objects.

Hard ceiling: 10 storage bindings/stage across ALL groups, including read-only. Consolidate buffers/headers/uploads, not per-entity CPU iteration. Batch async raw compilation; label raw modules/pipelines, name TypeGPU factories. Use preferred canvas format. DXC needs constant loop bounds/dynamic break, not large dynamic-loop functions.

## Render, physics, assets

Decode scene sRGB hex to linear; surfaces stay linear, composite alone encodes sRGB. Physics is opt-in PhysicsPlugin; Body/Spring/Joint author it, Physics.world extends it. Hand-wired joint bodies must spawn non-overlapping to avoid persistent fighting contacts.

Procedural-first, no format-shaped substrate. GltfPlugin converts to mesh/material/VAT/rig data; engine-owned SkinPlugin accepts glTF/physics/procedural poses.

## Testing

Unit verdicts need a native adapter; real GPU gates cover compile/raster/readback. Derive tolerances (exact ~1e-10, f32 ~1e-6 relative, convergence from order/steps), never tune. Measure GPU timestamps, not FPS. Hardware refusal fails.

Publish `window.__harness`: pin `{ready:false}` immediately, then installHarness from `/harness`; run resolves entities and returns a Verdict.
