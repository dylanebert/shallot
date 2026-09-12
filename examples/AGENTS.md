# Examples

From `bun run examples:index` and `examples/*/shallot.json`; edit manifests. Use `bunx shallot dev examples/<name>`.

## Recipes

| name | description | add |
| --- | --- | --- |
| `animate-with-clips` | I want an object to bounce on a loop without writing a per-frame system. | `bunx shallot add animate-with-clips` |
| `ascii` | I want my scene drawn as ASCII characters. | `bunx shallot add ascii` |
| `custom-material` | I want to write my own shader for a material and for the background. | `bunx shallot add custom-material` |
| `day-night-sky` | I want a day and night cycle where the sun moves and the sky follows it. | `bunx shallot add day-night-sky` |
| `drive-a-vehicle` | I want a car I can drive with WASD. | `bunx shallot add drive-a-vehicle` |
| `first-person` | I want to walk around and look with the mouse like a first-person game. | `bunx shallot add first-person` |
| `import-gltf` | I want to load and place one glTF model. | `bunx shallot add import-gltf` |
| `particles` | I want to simulate and draw many particles on the GPU. | `bunx shallot add particles` |
| `play-sound` | I want a sound that comes from a place in the world and pans as it moves. | `bunx shallot add play-sound` |
| `respond-to-input` | I want to move a thing with WASD and react to a click and a key press. | `bunx shallot add respond-to-input` |
| `save-and-restore` | I want to save the game and load it back later. | `bunx shallot add save-and-restore` |
| `svelte-ui` | I want a live-game HUD that updates every frame. | `bunx shallot add svelte-ui` |

## Coverage plan

The builder journey runs from a blank directory to a played URL: create a project and its first scene; define state and persistence; add actions, gameplay and physics; draw the frame and emit confirmed effects; add live UI; then check, build and publish the same project. Coverage is ordered by risk rather than by that journey: physics first, the frame second, then input with gameplay, then engine core with audio, project and CLI. A recipe carries demand; module checks carry reference or invariant claims.

### 1. Physics closeout

No physics recipe is kept or added. At the Box3D resync, regenerate from and pin Erin Catto's upstream Box3D; the current harness fork stops being the referent. Reproduce mechanisms in checks, not sample scenes.

| family | kept or added recipes | retired claims and checks owed | source |
| --- | --- | --- | --- |
| solver parity | — | world hashes and periodic body state at every step, single-thread and every thread count, plus restore/resimulation equality | reference |
| solver scheduling | — | graph coloring, overflow, sleeping, waking and parallel work partition preserve the upstream result | reference |
| contact generation | — | distance, time of impact, geometry, hull, manifold and contact persistence match upstream cases | reference |
| broad phase and queries | — | dynamic tree, pair table, ray casts, shape casts and mover queries retain exact selected results | reference |
| joints and limits | — | `joints`, `breakable-joints` and `ragdoll`: joint creation, limits, motors, break thresholds and events | reference |
| contacts and moving support | — | `moving-platform`, `surface-friction` and `physics-playground`: support velocity, friction/contact response, sleep and stacked mixed-shape behavior | reference |
| shapes and composition | — | compounds, meshes and heightfields compose through the public body/shape seam without changing their upstream physical result | reference |
| public physics seam | — | service-free plugin lifecycle, shared-pool ownership, generic observed poses, snapshot/restore/hash and interpolation | invariant |

### 2. The frame

| family | kept or added recipes | retired claims and checks owed | source |
| --- | --- | --- | --- |
| animation and GPU particles | keep `animate-with-clips`, `particles` | clips remain tweened playables on a loop; compute-to-draw particles remain observable without a per-frame CPU simulation | demand |
| materials, sky and ASCII | keep `custom-material`, `day-night-sky`, `ascii` | custom surface/backdrop, moving-sun sky and character-cell output; CPU structure plus the cheapest rendered witness for each visible claim | demand |
| glTF placement | keep `import-gltf` | load and place one model | demand |
| glTF conformance | — | accessor, sparse-data and node-transform conformance against Khronos assets and Three.js decode | reference |
| world-space UI | add `world-space-ui` from retired `annotate-the-world` and `billboards-and-sprites` | labels, billboards, sprites and meters stay attached and camera-facing through one demand recipe and module checks | demand |
| compute seam | — | retired `compute-and-readback`: dispatched data is readable through the public compute seam | invariant |
| frame pipeline | — | transforms, culling, compaction, lighting, fog, mirrors, post, atlases, skinning and drawing extras; decide T7 at the first rendered claim and persist human captures | invariant |

### 3. Input and gameplay

| family | kept or added recipes | retired claims and checks owed | source |
| --- | --- | --- | --- |
| input response | keep `respond-to-input` | held and edge actions move and trigger the requested thing | demand |
| mapped-action mechanism | — | actions inject above devices on a tick; one Chromium witness proves a real event reaches the same map, following the Godot and Unity precedent | reference |
| vehicle play | keep `drive-a-vehicle` | WASD drives the public vehicle seam; physics mechanisms stay in Physics closeout | demand |
| first-person play | keep `first-person` | walking, grounded movement and mapped mouse look; rebuild the scene after the person chooses or replaces the proposed Valve/Unreal movement-test reference | demand |
| gameplay mechanisms | — | character, player, BVH, orbit and profile run on the stepped clock; fixed gameplay renders through interpolation without refresh-rate judder | invariant |
| observation and replay | — | action injection, query, record and hash share one registered-data contract; confirm the record shape and prove the plugin absent from built games | invariant |

### 4. Engine core, audio, project and CLI

| family | kept or added recipes | retired claims and checks owed | source |
| --- | --- | --- | --- |
| first scene | add `hello-world` in place of retired `game-loop` | a rotating cube and one system establish the smallest blank-project-to-running-scene path | demand |
| state, save and live UI | keep `save-and-restore`, `svelte-ui` | save, change and restore; a HUD reads live game state; scene codec, isolation and snapshot mechanisms stay module checks | demand |
| spatial sound | keep `play-sound` | a moving world source pans and attenuates against its listener through CPU-readable voice state | demand |
| audio mechanism | — | DSP kernels retain their permissive external golds and the worklet gets one integration witness | reference |
| project and CLI | — | create reaches the first scene without a prompt; add, dev, check, packed build, native target and publish/deploy refusal operate from declarations with structured recovery | invariant |
| played URL | — | a packed consumer builds the same checked project and its published URL boots to an observable frame; hosted reports go to files and unsupported seats refuse | demand |
| sandbox showcase | add the sandbox showcase after the area recipes | compose the recipes on public package seams, run only user-project gates, and retain the rule-of-three boundary log | demand |
