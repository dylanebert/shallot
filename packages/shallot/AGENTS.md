# Shallot

WebGPU game engine: ECS, scenes, plugins. JSDoc owns APIs. Search `examples/AGENTS.md` and read a recipe first; `bunx shallot recipe <name> [dir]` copies one (bare lists).

## Commands

```bash
bun create shallot <name>
bunx shallot dev [dir]
bunx shallot build [dir]
bunx shallot run [dir]
bunx shallot verify [dir]
```

Verify defaults full-Chromium headless; display uses `--headed`, remote mode `--connect`. `bunx tsc --noEmit`. `shallot.json` names scene/plugins; CLI supplies HTML/Vite. Native targets use Rust/system dependencies; `--portable` bundles Chromium.

## Philosophy

Components are data, systems behavior, not methods/managers. Scenes author composition; imperative setup is procedural/tests. Derive one truth; compose plugins, depend inward on pure core. Declare group/after/before, not manual sequencing; one terminal runs last (RenderPlugin owns draw submit).

## Imports

`@dylanebert/shallot`: author APIs/defaults; `/extras`: opt-in convenience, also bare. Audio/mirror are bare-only. `/runtime`: platform/device; `*/core` and `/glaze`: extensions. Never deep-import src; file issues for missing seams. Plugins register components/traits.

## ECS & Plugins

Initialize is pre-scene; warm handles scene data; system setup is lazy first-frame. Mesh/surface plugins depend on RenderPlugin after its registry wipe. Missing dependencies fail before side effects; optional peers use conditional inclusion, nullable hooks or absent-system ordering. Position producers run before PrepassSystem.

Marker + `not()` gates once; initialized singletons hold services, component eids relations (`@name`, flat scenes, no parent). Eids recycle: re-query, no module handles/ownership maps. Derive from `state.time.elapsed`; scope state, not last-State/exists guards.

## UI

Mount only in `config.ui(container, state)` or `mountOverlay(canvas, state)`, sandboxed to the canvas by layout/paint containment and clipping. Position relative, never fixed or on document.body. Container ignores pointer events; interactive children enable them. Return/register real framework unmount and cancel timers/rAF via State cleanup; use `state.signal` for listeners/fetch. Removing DOM alone leaks effects. Re-warm without dispose must clear its prior mount first; retain dispose cleanup too.

## GPU

[MIGRATION.md](./MIGRATION.md) governs new code too: exact-once transforms, ejected/framework inclusion, consumer tests, schemas, ownership, TGSL integers/lint, surfaces/varyings/fragment inputs, warm queues. Compute on Render.encoder; register on render/core. Only allocators destroy. CPU truth is typed arrays, not per-frame objects.

Hard ceiling: 10 storage bindings/stage across ALL groups, including read-only. Consolidate buffers/headers/uploads, not per-entity CPU iteration. Batch async raw compilation; label raw modules/pipelines, name TypeGPU factories. Use preferred canvas format. DXC needs constant loop bounds/dynamic break, not large dynamic-loop functions.

Debug CPU → labeled WGSL/API → safe fragment/compute log → resource probe → verify; no rung proves the next. Logging perturbs bindings/atomics, is bounded/delayed, excludes vertices; external-pass drains need replay. Probe if replay changes behavior; Mirror is delayed telemetry. No atomic debug buffer if logging suffices; capture last after naming pass/draw.

## Render, physics, assets

Decode scene sRGB hex to linear; surfaces stay linear, composite alone encodes sRGB. Bright accents saturate: darken/lower intensity, tune dominant tones, not a global gamma multiplier. Physics is opt-in TumblePlugin; Body/Spring/Joint author it, Tumble.world extends it. Hand-wired joint bodies must spawn non-overlapping to avoid persistent fighting contacts.

Procedural-first, no format-shaped substrate. GltfPlugin converts to mesh/material/VAT/rig data; engine-owned SkinPlugin accepts glTF/physics/procedural poses. Producers compose skin/core surfaces.

## Testing and verify

Unit verdicts need a native adapter; real GPU gates cover compile/raster/readback. Keep permanent tests, temporary labs; derive tolerances (exact ~1e-10, f32 ~1e-6 relative, convergence from order/steps), never tune. Measure GPU timestamps, not FPS.

Verify self-terminates. Install `bun add -d playwright`, `bunx playwright install chromium`. Chromium headless; display uses `--headed`; hardware refusal fails. Build then verify `--dist`. COOP/COEP needs CORS/CORP or local assets; physics is single-threaded.

Without `window.__harness`, verify needs settled nonblank rendering/no page errors. Pin `{ready:false}` immediately, then installHarness from `/harness`; initialize pins, run resolves entities. Verify waits ready, calls run, requires Verdict.ok/no page errors. Queries feed run; JSON preserves checks/extras. Batch `--run k=v` isolates pages/contexts, any failure fails; perf thresholds run separately. Seed storage/exercise restore for persistence. JSDoc owns pose reads/flags; remove temporary verify plugins.
