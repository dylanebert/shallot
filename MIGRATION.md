# Migrating from 0.9.5 to 0.10

Most 0.9.5 game code carries over. These changes break it.

## `build()` always requires WebGPU

Every app now acquires a GPU device during `build()`, or rejects with the acquisition cause. A composition no longer runs as a CPU-only app. Supply an already-acquired device with `config.device` when the host owns acquisition; otherwise the engine uses `navigator.gpu`.

`Plugin.device` is removed: plugins no longer declare `required` or `optional`, and `deviceTier` / `DeviceTier` are gone. `warmPlugins` also requires a `GPUDevice` and has no CPU-tier argument. `resetCompute`, which only cleared the device for CPU builds, is removed. The device-tier report, generator, checker, its test and fixtures, and the Device tiers section of `CONTRIBUTING.md` are removed with no replacement. The import-time WebGPU enum shim `project/gpu-globals.ts` is removed because engine modules no longer read WebGPU globals during import.

Device acquisition no longer requests adapter-reported maxima wholesale. It requests Shallot's fixed storage-buffer floor and compatibility values where split-stage limits are unsupported; otherwise the device keeps its default limits. Operations that exceed a limit refuse at their own named check.

Tests that build an app use `*.gpu.test.ts`; the root Bun test preload applies Shallot's TypeGPU transform. `bun run test` includes these tests on a device, while hosted jobs without one exclude the suffix and the macOS GPU job selects it with `bun test gpu.test`. Node tests and manual oracles keep their existing tiers. The packed-install headless smoke now runs in the GPU tier: it installs `bun-webgpu` with the packed tarball and steps the world through public engine subpaths. Acquisition-refusal tests stub adapter outcomes so each failure cause is testable without depending on a real adapter.

## Worlds own component and GPU storage

A component field declared with `field(type)` is a schema, not the world's values. Systems resolve its columns once with `const value = state.of(Value)` and read or write through that result; scene load, defaults, reflection and snapshots use the same State-owned columns. The first State to bind a component freezes its object: declare the complete schema up front, and replace the component object to change its schema. Columns grow geometrically as a world's entity ids rise, within that world's fixed entity/GPU reservation. `State.capacity`, entity creation, membership mirrors, transform/render buffers and table growth still use the owning State's configured capacity until the remaining capacity-sized consumers move to tables.

Component registrations, defaults, exclusions and enumeration also belong to the State. Plugin registration during `build()` is unchanged; direct ECS registration and reflection helpers now take the State explicitly, as do scene helpers that interpret registered attrs (`diagnose`, `parseFields`, `formatFields` and `normalizeAttr`). `parse` remains state-free.

Each `State` owns its GPU registries (`buffers`, `textures`, `samplers`, and typed handles) through `state.gpu`, as well as resources created through its tracked device/root. `state.resource(key, create)` is the seam for a module's non-column, per-world state; `state.own(resource)` ties a raw buffer or texture to disposal. Systems and lifecycle hooks can continue to use `Compute` inside their callback; outside one, retain `state.gpu` from the State that owns the resource. Disposing one App releases only its world's GPU allocations and registries.

Build setup is serialized while plugins register and warm. Engine-owned storage and GPU registries are per State, so Apps may coexist on one or separate devices; there is no process-global build lease to release before another App can build. A hot swap with the same component schema reattaches the reloaded handle to that State's existing columns; a changed schema returns `{ ok: false }` so the host rebuilds. No host HMR wiring is added.

## GPU record tables

`slab()`, `sparse()` and `SlabPlugin` are no longer exported from Shallot's public package surface. Declare a table from its owning State with `state.table(name, recordLayout)`. Each table has one struct record layout and dense rows allocated from a free list; a slot stays stable for an entity's lifetime. `table.acquire(eid)` returns the row and `table.release(eid)` makes it reusable. Draws and dispatches consume the table's compact `activeRowsBuffer`, never a sparse entity range. A shader that starts from an eid opts into the `eid + 1`-encoded map with `table.enableEidLookup()` or `table.subscribeMap(...)`; tables that only read dense lists allocate and upload no GPU map.

The table exposes its raw record buffer, typed handle, capacity, generation and upload path. Consumers subscribe once and rebuild bindings only when the corresponding record, map or active-list generation changes. Host writes are bulk byte-range fills followed by `table.markRange(firstRow, count)`. Unchanged records skip upload; `writeBuffer` is the default. Mapped scatter remains explicitly selectable pending the Stage 4 crossover measurements. Growth refuses at `maxStorageBufferBindingSize` with the limit in the error.

Map cost evidence (not an asserted timing): on an Apple M4 Max, Metal 3, macOS 26.7 (Build 25G229), an intentionally diagnostic 100k-row, 100%-population pass reading a 64-byte pose record measured 0.0047 ms by direct eid indexing and 0.0059 ms through the dense eid-to-slot map (+24.3%, about 1.2 µs; 15 alternating timestamp-query samples, each averaging 512 dispatches, all paired samples slower through the map). A full entity-range scan through the map is not an engine workload: draws and dispatches read dense lists, never the entity range. The map is opt-in for point lookups that start from an eid, such as picking or a one-eid query. A consumer that needs a full map-scanned pass is a design finding to report, not a reason to scan eids. The four-byte map row is 6.25% of a 64-byte pose record, or 5% of an 80-byte instance record. At 100k eids, the map buffer is 524,288 bytes for 8,388,608 allocated pose-record bytes; its initial populated upload is 400,000 bytes. Because slots remain stable, only add/remove changes dirty the map; after those changes are uploaded, an unchanged frame uploads zero map bytes.

Record-layout evidence (same adapter, 100k rows; prototypes only, arrays are not an engine table layout): pose records (48 bytes: position, rotation and scale) used one `writeBuffer` call and read all fields in 4,096 ns per dispatch; struct submission / queue cycle was 0.3080 ms / 0.4190 ms, while three per-field arrays were 0.2974 ms / 0.4047 ms, with the same GPU read. Light records (32 bytes: color and parameters) used one `writeBuffer` call and read all fields in 4,096 ns per dispatch; struct submission / queue cycle was 0.2802 ms / 0.4194 ms, while two per-field arrays were 0.2420 ms / 0.3707 ms. The modest upload differences vary by table, and full-record GPU reads tie; this evidence does not show a clear need for per-field arrays. The engine keeps one struct record per row.

`shallot-avbd-physics` drops its `AvbdPlugin`/State facade because it packed Body fields from Slab; the low-level solver core remains. `agentic-engineering` no longer explicitly selects `SlabPlugin` in `hero-engine.ts`; no replacement is added here.

## Plugin GPU state is State-owned

Stage 3 removes plugin GPU resources and render scratch from module scope. Each plugin now resolves a typed resource from its owning `State` (or `worldResource` during a State-bound callback); module-level exports that remain are stateless facades. Buffers and textures allocated through the tracked device or TypeGPU root are owned by that State and released on disposal. Caches of pipelines, bind groups, fallback textures, frame descriptors, and per-world staging data no longer cross App boundaries. The GPU test suite authors content for the exported default, extra and transitional plugins in two live worlds, renders through the headless path, and checks isolation on shared and separate devices while disposing one world.

The removals, by owner:

- `core/rendering`: cluster and light-cull pipelines, bindings, typed views and overflow staging; frame submission, depth-only and view-matrix scratch; image blit pipelines; lighting uniform/backing storage; pending mesh specs and placeholders; view offscreen/scratch targets; Draws, Views, Surfaces and Backgrounds registries; and Render/Frame resource state.
- `standard/rendering`: compiled surface/background and typed bind-group caches; shadow atlas and fallback resources, point/cascade atlas buffers, passes, batches, bundles and regather bindings; the regather pipelines/layouts/device-capacity memo; and forward depth/color/lane targets, pass descriptors, bundle programs and per-view caches.
- `transitional/part`, `transitional/slab`, `transitional/transforms`, and `transitional/cells`: Part culling buffers/pipelines/bindings/scratch; slab membership mirrors, slab bookkeeping and scatter-pipeline cache; transform compose buffers/pipeline/bindings; and cell atlases, samplers, draw/selection pipelines, parameter buffers and per-grid caches.
- `extras/lines`, `extras/fog`, `extras/outline`, `extras/sky`, `extras/sprite`, and `extras/text`: segment and glyph staging/buffers; per-view fog/outline targets and bind groups; sky uniform resources; sprite atlas/sampler/instance buffers and pack scratch; text atlases, glyph buffers, sampler, SDF pipelines and per-font scratch.
- `transitional/glaze` and `transitional/gltf`: composite/pipeline caches and per-camera Glaze buffers/descriptors; glTF assembled geometry, VATs, texture unions, fallback sets, active palette and staged uploads. The glTF decode cache remains process-local because it contains decoded CPU data only; each State owns its GPU assembly. The live-skin substrate is resolved with `liveSkin(state)`. Physics moves its runtime maps and joint caches into `State.resource`, gives each PhysicsPlugin State its own WASM instance, linear memory and worker pool, and scopes State-first calls, snapshots and restores to their explicit State. A `WorldSnapshot` owns detached logical state and copied WASM bytes; it is caller-owned and can restore into a live compatible World, including a fresh one, only when the target kernel has no other live World. `mirror` now takes its owner explicitly (`mirror(state, source)`), so its readback rings and cached snapshots die with that State. Sear's shadow-camera pools and readback diagnostics are State-owned; `cascadeCount`, `cascadeComboEids`, `pointComboCount` and `pointComboEids` take the owning State. Character drive helpers (`move`, `jump`, `pose`, `teleport` and `grounded`) likewise take State before the entity id. `shallot-avbd-physics` scopes its step, Mirrors, stamps and constraint signatures to State as well; its `Avbd.step`, `readBody`, `setKinematic` and `setVelocity` accessors now take State. Direct low-level `World` calls made without an active State retain the standalone kernel used by that API.

The module-level `Compute` surface is only an active-callback façade; the TypeGPU root now belongs to its State even when devices are shared. Resolve the live-skin substrate with `liveSkin(state)` and read profiler data outside a callback with `profile(state)`; their former process-level `LiveSkin` and `Profile` state is no longer shared. Engine-owned weak registries only track device diagnostics and proxy identity. Stage 3 adds no per-frame allocations outside existing Stage 4 Slab sites.

## `shallot recipe` is now `shallot add`

```sh
# 0.9.5
bunx shallot recipe first-person

# 0.10
bunx shallot add first-person
```

## The CLI command set changed

- `shallot list` and `shallot check` are removed. Run the project's cheap tests with `bun test`, named host tiers by file path, and browser tests with `bunx playwright test`.
- `shallot workflow` has no replacement; the workflow file it wrote is yours to edit and is no longer regenerated.
- `shallot <verb>` no longer resolves to `shallot-<verb>` on `PATH`; there is no replacement.
- `shallot run` is now `shallot build` followed by `shallot preview`. Preview never rebuilds.

## `shallot tui` and `shallot verify` are gone

Neither has a replacement command. Tests use Bun's `test()` directly; browser tests use Playwright Test against the project's Vite preview.

## `/render/core`, `/sear/core` and `/utils/core` no longer resolve

Package imports dropped their `/core` suffix, and rendering split in two: `/rendering` for what every renderer shares, and `/standard/rendering` for Shallot's default mesh renderer.

```ts
// 0.9.5
import { FrameGpu } from "@dylanebert/shallot/render/core";
import { engineLayout, registerSurface, surfaceLayout } from "@dylanebert/shallot/sear/core";
import { Xform } from "@dylanebert/shallot/utils/core";

// 0.10
import { FrameGpu, registerSurface, surfaceLayout } from "@dylanebert/shallot/rendering";
import { engineLayout } from "@dylanebert/shallot/standard/rendering";
import { Xform } from "@dylanebert/shallot/utils";
```

Likewise `/ecs/core` is `/ecs`, `/scene/core` is `/scene` and `/physics/core` is `/physics`. The `/src/*` wildcard is gone: only the paths in `package.json` `exports` resolve.

## `Inputs` is now `devices(state)`

Input is plain data on each State: `devices(state)` returns its keys, mouse, touch and viewport. With the default plugins the browser fills it; a test or replay fills the same record with `pressKey`, `pointerMove` and the other producers, with no browser. `setInputEnabled` takes the State, and the canvas size moved from `mouse` to `viewport`.

```ts
// 0.9.5
if (Inputs.isKeyDown("KeyW")) moveForward();
if (Inputs.isKeyPressed("Space")) jump();
const width = Inputs.mouse.canvasWidth;
setInputEnabled(false);

// 0.10
const input = devices(state);
if (input.keys.held.has("KeyW")) moveForward();
if (input.keys.pressed.has("Space")) jump();
const width = input.viewport.get(input.focused)?.cssWidth ?? 0;
setInputEnabled(state, false);

// a test drives the same record
pressKey(state, "KeyW");
```

`keys.pressed` holds a press until the next frame. A `fixed` system reads `keys.tickPressed`, which holds it until the next fixed update.

`isKeyPressedWithin(code, seconds)` measured wall time, so its replacement counts fixed updates: `keys.pressedTick` records the update each key was pressed on.

```ts
const at = input.keys.pressedTick.get("Space");
if (at !== undefined && state.time.fixedTick - at < 6) jump(); // pressed within the last 6 updates
```

## `Tumble` is now `Physics`, and `Physics.backend` is gone

The plugin, its manifest key and its import path are renamed. Bodies are read and driven by functions from `/physics` that take the State.

```json
// 0.9.5
{ "plugins": { "Tumble": true } }

// 0.10
{ "plugins": { "Physics": true } }
```

```ts
// 0.9.5
Physics.backend?.setKinematic(eid, position, rotation);
const body = Tumble.body(eid);

// 0.10
import { body, setKinematic } from "@dylanebert/shallot/physics";
setKinematic(state, eid, position, rotation);
const b = body(state, eid);
```

`Tumble.world` is `physicsWorld(state)`, and `readBody(state, eid)` reads a body's pose.

## `/avbd` is gone

0.10 ships no AVBD solver. Apps that select it use the built-in `Physics` instead.

## `Tween`, `Sequence` and `/tween/core` are gone

0.10 has no animation plugin. Animation is app code.

## `/document` and edit mode are gone

`Document`, `History`, `Session` and `ReadbackSystem` are removed, with no replacement for undo, redo or editor sessions. `State.mode`, the app's `mode` option and `annotations.mode` are removed too: every system always runs. Saving and loading scenes still works:

```ts
import { serialize, stringify } from "@dylanebert/shallot";
const saved = stringify(serialize(state));
```

## Browser tests use Playwright Test

Configure Playwright in the project, run its own Vite preview with `webServer`, and put Chromium launch flags in `playwright.config.ts`.

## TypeGPU below 0.12.6 is too old

```sh
bun add typegpu@~0.12.6
```

For web, `shallot dev`, `shallot build` and `shallot preview` run the project's Vite commands. Native `dev` and `build` add the desktop shell to the Vite server or build; native `preview` launches that build. Add Shallot's Vite plugin to the project config; it includes the TypeGPU transform, so do not register a separate TypeGPU plugin:

```ts
import { shallot } from "@dylanebert/shallot/vite";

export default defineConfig({ plugins: [shallot()] });
```

## `sparse(u8)` values wrap

Single-value `sparse` fields now store their declared type on every write: `sparse(u8)` stores 300 as 44, `sparse(u32)` stores negatives as unsigned, and `sparse(f32)` rounds to 32-bit precision. Declare a type wide enough for the values you store.

## `tsc` reports missing `ImportMeta.env` or Node types

The 0.9.5 scaffold's `tsconfig.json` lists only WebGPU types. Add Node and Vite's:

```sh
bun add -d @types/node@^26.0.0
```

```json
{ "compilerOptions": { "types": ["@webgpu/types", "node", "vite/client"] } }
```

These are types only; no Node code reaches the browser bundle.
