# Examples

The retrieval surface for shallot's examples. Grep it for the problem you have before writing a pattern
from scratch. Four groups: **recipes** (one minimal project per problem, the teaching corpus),
**gym** (machine-verdict atoms), **flows** (standalone-app engine flows), **showcase** (richer capability
exhibits). Each recipe is a manifest project — run it with `shallot dev examples/recipes/<name>/`. The
recipe contract is `.claude/rules/examples.md`.

## Recipes

One minimal project per problem a game developer actually has.

- **build a scene** — `recipes/build-a-scene/` — compose a world declaratively in a `.scene` file (entities, components, vectors, colors, `@` refs), and build more entities from code.
- **code a game loop** — `recipes/game-loop/` — the ECS basics: a custom component, a system over a query, a marker + `not()`, a plugin.
- **respond to input** — `recipes/respond-to-input/` — read held keys and the mouse, edge-triggered presses, drive an entity each frame.
- **orbit camera** — `recipes/orbit-camera/` — the orbit camera: frame a target, tune distance / yaw / pitch.
- **first-person character** — `recipes/first-person/` — the first-person character controller, plus riding a moving platform.
- **physics playground** — `recipes/physics-playground/` — rigidbodies and colliders (`Body`/`Spring`/`Joint`), spawning bodies from code; settling behavior is oracle-verified in the gym physics golds (e.g. `pile`, `joints-suspension`).
- **ragdoll** — `recipes/ragdoll/` — a physics-driven ragdoll posing a live joint palette (`LiveSkin`); the cone/twist joints ride the `Tumble.world` hatch (no substrate-surface equivalent yet — twin: gym `ragdoll-ragdoll`).
- **connect bodies with joints** — `recipes/joints/` — the published substrate `Spring` (a soft distance suspension) and `Joint` (a fixed-weld cantilever); the hertz/damping-tuned versions live in the gym twins `joints-suspension` / `joints-cantilever`.
- **a moving platform** — `recipes/moving-platform/` — a kinematic code-driven lift (`Physics.backend.setKinematic`) carrying crates up and down; the motor-driven prismatic version is the gym twin `joints-elevator`.
- **drive a vehicle** — `recipes/drive-a-vehicle/` — a car on wheel joints (rear spin + front steering) driven with W/S/A/D, via the `Tumble.world` hatch (no substrate-surface equivalent yet — twin: gym `joints-driving`).
- **breakable joints** — `recipes/breakable-joints/` — distance joints that cut under load, read off the tumble joint-event stream (`getJointEvents`), via the `Tumble.world` hatch (no substrate-surface equivalent yet — twin: gym `events-joint-break`).
- **surface friction** — `recipes/surface-friction/` — the substrate `Body.friction` field: a friction ladder of boxes on a tilted ramp, slippery to grippy (twin: gym `shapes-inclined-plane`).
- **import a model (glTF)** — `recipes/import-a-model/` — declarative mesh refs (`mesh: file.glb#0`) and the programmatic `loadGltf` / `placeGltf` / `placeScene`.
- **day-night sky** — `recipes/day-night-sky/` — the procedural sky: sun direction, time of day.
- **fog and light shafts** — `recipes/fog-and-light-shafts/` — volumetric fog and god rays.
- **stylize the look** — `recipes/stylize-the-look/` — the screen-space outline post-effect on selected entities.
- **author a custom material** — `recipes/custom-material/` — register a custom WGSL surface and use it by name in a scene, plus a custom backdrop via `Backgrounds.register`.
- **run a compute pass and read it back** — `recipes/compute-and-readback/` — a custom GPU compute pass over per-entity slab data, results read back to the CPU with `Mirror`.
- **annotate the world** — `recipes/annotate-the-world/` — world-space text labels plus debug lines and arrows (retained components and the immediate `box` / `arrow` API).
- **billboards and sprites** — `recipes/billboards-and-sprites/` — camera-facing sprites, the billboard modes, a radial meter.
- **play sound** — `recipes/play-sound/` — spatial audio: a listener and positioned sound sources.
- **animate with tweens** — `recipes/animate-with-tweens/` — the `Tween` component, easing a transform and color over time.
- **overlay DOM UI / embed in a page** — `recipes/overlay-ui/` — `mountOverlay` for a canvas-bounded HUD, and the `run()` embedding path.
- **save and restore** — `recipes/save-and-restore/` — `serialize` the world to scene XML, persist it to `localStorage`, restore it in place via `parse` → `load` (or `run({ scene })` at app boot).
- **measure performance** — `recipes/measure-performance/` — the profiler overlay and `window.__benchmark` timing.

## Gym

`gym/` — one project, `?scenario=`-selected. Each scenario is a correctness gate + benchmark + live demo
in one, run on a real device via `bun bench --scenario <name>`. The scenario list and the GPU-driven
coverage each carries is the barrel header in `gym/src/scenarios/index.ts`. This is the machine tier, not
a teaching reference — read a recipe first.

## Flows

Standalone-app engine flows — ejected vite apps that exercise engine behavior a `bun test` can't reach,
driven by `bun run flows` (`scripts/flows.ts` over `shallot verify`).

- **survive-reload** — `flows/survive-reload/` — a runtime value + a `warm`-derived entity survive a real
  `location.reload()` through the serialize→sessionStorage→restore path; the restored boot's `window.__harness` asserts it.
- **ui-containment** — `flows/ui-containment/` — a deliberately invalid `config.ui` HUD (`position: fixed`,
  oversized) must stay clipped to the canvas; the flow asserts the host-chrome pixels stay clear.

## Showcase

Richer exhibits — each a self-contained real project that owns its own test gate, run with
`shallot dev examples/showcase/<name>/` (except `visualization`, which owns a vite harness: `cd
examples/showcase/visualization && bun dev`).

- **collapse** — `showcase/collapse/` — an AVBD rigidbody structure collapsing, profiled.
- **fountain** — `showcase/fountain/` — a custom GPU render producer: a particle fountain.
- **sandbox** — `showcase/sandbox/` — a playable physics sandbox (character + AVBD + modal-synthesis audio).
- **visualization** — `showcase/visualization/` — a multi-canvas gallery of the debug-draw primitives (lines, text, tweens, wireframe).
- **voxel** — `showcase/voxel/` — a voxel editor (carve tools + a greedy mesher) that owns its own Playwright gate.
