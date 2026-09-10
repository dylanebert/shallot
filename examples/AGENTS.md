# Examples

Grep first: `.claude/rules/examples.md`. Manifests: `bunx shallot dev examples/recipes/<name>/` (showcase likewise); gym/visualization: cd there, bun dev.

## Recipes

- scene — `recipes/build-a-scene/` — declarative/code.
- game loop — `recipes/game-loop/` — ECS/plugin.
- input — `recipes/respond-to-input/` — held/edge/mouse.
- orbit — `recipes/orbit-camera/` — framing.
- first-person — `recipes/first-person/` — controller/platform.
- physics — `recipes/physics-playground/` — bodies/joints; gym pile/suspension.

- ragdoll — `recipes/ragdoll/` — LiveSkin, Tumble hatch; gym ragdoll-ragdoll.
- joints — `recipes/joints/` — Spring/Joint; gym joints-suspension/cantilever.
- platform — `recipes/moving-platform/` — kinematic; gym joints-elevator motor.
- drive a vehicle — `recipes/drive-a-vehicle/` — W/S/A/D, hatch; gym joints-driving.
- breakable joints — `recipes/breakable-joints/` — hatch events; gym events-joint-break.
- friction — `recipes/surface-friction/` — Body.friction; gym shapes-inclined-plane.

- import glTF — `recipes/import-a-model/` — mesh refs/load/place.
- day-night sky — `recipes/day-night-sky/` — procedural sun/time.
- fog/light shafts — `recipes/fog-and-light-shafts/` — god rays.
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

## Gym

`gym/`: `bun bench --scenario <name>`; gates/bench/demo. Roster: `gym/src/scenarios/index.ts`; teaching: recipes. Touch: `gym/test/touch.playwright.ts`.

## Flows

`bun run flows` wraps shipped verify.

- `flows/no-walls/` — adopted device/raw WGSL draw, boundary/pixel gate.
- `flows/survive-reload/` — value/warm entity reload/storage/restore.
- `flows/ui-containment/` — invalid fixed UI clipped from host chrome.
- `flows/blank/` — expected pixel failure despite harness ok.

## Showcase

https://dylanebert.com/shallot/; version-pinned code links, project-owned gates.

- **ascii** — `showcase/ascii/` — the cell-grid cube seeded on the web sink.
- **collapse** — `showcase/collapse/` — an AVBD rigidbody structure collapsing, profiled.
- **ocean** — `showcase/ocean/` — a full-screen multi-cascade FFT ocean at pinned camera, sun, time, and capture conditions.

- **roads** — `showcase/roads/` — a road network editor across terrain (capture, edit, re-drive corridors) that owns its own Playwright gate.
- **sandbox** — `showcase/sandbox/` — a playable physics sandbox (character + AVBD + modal-synthesis audio).
- **visualization** — `showcase/visualization/` — a multi-canvas gallery of the debug-draw primitives (lines, text, written animations, wireframe).
- **voxel** — `showcase/voxel/` — a voxel editor (carve tools + a greedy mesher) that owns its own Playwright gate.
