# Examples

Grep first: `.claude/rules/examples.md`. Manifests: `bunx shallot dev examples/recipes/<name>/` (showcase likewise); visualization: cd there, bun dev.

## Recipes

- scene — `recipes/build-a-scene/` — declarative/code.
- game loop — `recipes/game-loop/` — ECS/plugin.
- input — `recipes/respond-to-input/` — held/edge/mouse.
- first-person — `recipes/first-person/` — controller/platform.
- physics — `recipes/physics-playground/` — bodies/joints; bench pile.

- ragdoll — `recipes/ragdoll/` — LiveSkin, Tumble hatch; tumble ragdoll gold.
- joints — `recipes/joints/` — Spring/Joint; tumble joint golds.
- platform — `recipes/moving-platform/` — kinematic; tumble elevator gold.
- drive a vehicle — `recipes/drive-a-vehicle/` — W/S/A/D, hatch; tumble driving gold.
- breakable joints — `recipes/breakable-joints/` — hatch events; tumble joint-break gold.
- friction — `recipes/surface-friction/` — Body.friction; tumble inclined-plane gold.

- import glTF — `recipes/import-a-model/` — mesh refs/load/place.
- day-night sky — `recipes/day-night-sky/` — procedural sun/time.
- stylize — `recipes/stylize-the-look/` — outlines.
- material — `recipes/custom-material/` — WGSL surface/backdrop.
- compute/readback — `recipes/compute-and-readback/` — slab/Mirror.
- particles — `recipes/gpu-particles/` — compute→vertex buffer.

- annotate — `recipes/annotate-the-world/` — text/lines/arrows.
- sprites — `recipes/billboards-and-sprites/` — modes/meter.
- play sound — `recipes/play-sound/` — spatial listener/sources.
- animate clips — `recipes/animate-with-clips/` — Animator playables.
- overlay/embed UI — `recipes/overlay-ui/` — mountOverlay/run.
- save/restore — `recipes/save-and-restore/` — XML/storage.
- perf — `recipes/measure-performance/` — profiler.

## Bench

Outside `examples/`, at `bench/`: `bun bench --scenario <name>`. Roster: `bench/src/scenarios/index.ts`; cones: `bench/src/scenarios/timeouts.ts`; teaching: recipes.

## Showcase

https://dylanebert.com/shallot/; version-pinned code links, project-owned gates.

- **ascii** — `showcase/ascii/` — the cell-grid cube seeded on the web sink, the same scene `shallot tui` renders in a terminal.
- **collapse** — `showcase/collapse/` — an AVBD rigidbody structure collapsing, profiled.
- **ocean** — `showcase/ocean/` — a full-screen multi-cascade FFT ocean at pinned camera, sun, time, and capture conditions.

- **roads** — `showcase/roads/` — a road network editor across terrain (capture, edit, re-drive corridors) that owns its own Playwright gate.
- **sandbox** — `showcase/sandbox/` — a playable physics sandbox (character + AVBD + modal-synthesis audio).
- **visualization** — `showcase/visualization/` — a multi-canvas gallery of the debug-draw primitives (lines, text, written animations, wireframe).
- **voxel** — `showcase/voxel/` — a voxel editor (carve tools + a greedy mesher) that owns its own Playwright gate.
