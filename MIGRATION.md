# Migrating from 0.9.5 to 0.10

Most 0.9.5 game code carries over. These changes break it.

## `build()` always requires WebGPU

Every app now acquires a GPU device during `build()`, or rejects with the acquisition cause. A composition no longer runs as a CPU-only app. Supply an already-acquired device with `config.device` when the host owns acquisition; otherwise the engine uses `navigator.gpu`.

`Plugin.device` is removed: plugins no longer declare `required` or `optional`, and `deviceTier` / `DeviceTier` are gone. `warmPlugins` also requires a `GPUDevice` and has no CPU-tier argument. `resetCompute`, which only cleared the device for CPU builds, is removed. The device-tier report, generator, checker, its test and fixtures, and the Device tiers section of `CONTRIBUTING.md` are removed with no replacement. The import-time WebGPU enum shim `project/gpu-globals.ts` is removed because engine modules no longer read WebGPU globals during import.

Device acquisition no longer requests adapter-reported maxima wholesale. It requests Shallot's fixed storage-buffer floor and compatibility values where split-stage limits are unsupported; otherwise the device keeps its default limits. Operations that exceed a limit refuse at their own named check.

Tests that build an app use `*.gpu.test.ts`; the root Bun test preload applies Shallot's TypeGPU transform. `bun run test` includes these tests on a device, while hosted jobs without one exclude the suffix and the macOS GPU job selects it with `bun test gpu.test`. Node tests and manual oracles keep their existing tiers. The packed-install headless smoke now runs in the GPU tier: it installs `bun-webgpu` with the packed tarball and steps the world through public engine subpaths. Acquisition-refusal tests stub adapter outcomes so each failure cause is testable without depending on a real adapter.

## Worlds own component and GPU storage

A component field declared with `field(type)` is a schema, not the world's values. Systems resolve its columns once with `const value = state.of(Value)` and read or write through that result; scene load, defaults, reflection and snapshots use the same State-owned columns. The first State to bind a component freezes its object: declare the complete schema up front, and replace the component object to change its schema. CPU columns grow geometrically as the entity-id high-water rises. `State.capacity` and `Config.capacity` are removed; dense tables grow with membership and refuse at the owning device's buffer limit.

Component registrations, defaults, exclusions and enumeration also belong to the State. Plugin registration during `build()` is unchanged; direct ECS registration and reflection helpers now take the State explicitly, as do scene helpers that interpret registered attrs (`diagnose`, `parseFields`, `formatFields` and `normalizeAttr`). `parse` remains state-free.

Each `State` owns its GPU registries (`buffers`, `textures`, `samplers`, and typed handles) through `state.gpu`, as well as resources created through its tracked device/root. `state.resource(key, create)` is the seam for a module's non-column, per-world state; `state.own(resource)` ties a raw buffer or texture to disposal. Systems and lifecycle hooks can continue to use `Compute` inside their callback; outside one, retain `state.gpu` from the State that owns the resource. Disposing one App releases only its world's GPU allocations and registries.

Build setup is serialized while plugins register and warm. Engine-owned storage and GPU registries are per State, so Apps may coexist on one or separate devices; there is no process-global build lease to release before another App can build. A hot swap with the same component schema reattaches the reloaded handle to that State's existing columns; a changed schema returns `{ ok: false }` so the host rebuilds. No host HMR wiring is added.

## GPU record tables

`slab()`, `sparse()` and `SlabPlugin` are no longer exported from Shallot's public package surface. Declare a table from its owning State with `state.table(name, recordLayout)`. Each table has one struct record layout and dense rows allocated from a free list; a slot stays stable for an entity's lifetime. `table.acquire(eid)` returns the row and `table.release(eid)` makes it reusable. Draws and dispatches consume the table's compact `activeRowsBuffer`, never a sparse entity range. A shader that starts from an eid opts into the `eid + 1`-encoded map with `table.enableEidLookup()` or `table.subscribeMap(...)`; tables that only read dense lists allocate and upload no GPU map.

The table exposes its raw record buffer, typed handle, capacity, generation and upload path. Consumers subscribe once and rebuild bindings only when the corresponding record, map or active-list generation changes. Host writes are bulk byte-range fills followed by `table.markRange(firstRow, count)`. Unchanged records skip upload; `writeBuffer` is the only retained upload path unless later measurements establish a scatter win. Growth refuses at `maxStorageBufferBindingSize` with the limit in the error.

Map cost evidence (not an asserted timing): on an Apple M4 Max, Metal 3, macOS 26.7 (Build 25G229), an intentionally diagnostic 100k-row, 100%-population pass reading a 64-byte pose record was compared through direct eid indexing and the dense eid-to-slot map, with 15 alternating timestamp-query samples averaging 512 dispatches each. Repeated-run medians ranged from -2.4% to +24.3% (about -0.1 to +1.2 µs per pass), so this scan is evidence only, not a performance claim. A full entity-range scan through the map is not an engine workload: draws and dispatches read dense lists, never the entity range. The map is opt-in for point lookups that start from an eid, such as picking or a one-eid query. A consumer that needs a full map-scanned pass is a design finding to report, not a reason to scan eids. The four-byte map row is 6.25% of a 64-byte pose record, or 5% of an 80-byte instance record. At 100k eids, the map buffer is 524,288 bytes for 8,388,608 allocated pose-record bytes; its initial populated upload is 400,000 bytes. Because slots remain stable, only add/remove changes dirty the map; after those changes are uploaded, an unchanged frame uploads zero map bytes.

Upload-path evidence (same adapter; 8-byte record; queue-cycle medians in ms; contiguous changed ranges; staging remap notification excluded from the timed interval): at 1k rows and 0.1% / 10% / 100% changed, `writeBuffer` measured 0.1180 / 0.1132 / 0.1162 versus mapped scatter at 0.2622 / 0.2561 / 0.2526; at 10k, 0.1164 / 0.1258 / 0.1313 versus 0.2433 / 0.2643 / 0.2655; at 100k, 0.1238 / 0.1407 / 0.2877 versus 0.3152 / 0.3055 / 0.4373. Scatter had no crossover; `writeBuffer` was about 1.5–2.5× faster by queue cycle across these cases, so the table drops mapped scatter and keeps no-upload or `writeBuffer`. No changed-fraction threshold remains.

Dense-table memory evidence (same adapter; GPU buffer allocation bytes; pose record 64 B): at entity ranges 1k / 10k / 100k and 1% / 100% populated, record buffers plus active lists used 1,088 / 69,632; 8,704 / 1,114,112; and 69,632 / 8,912,896 bytes, respectively. With the opt-in map enabled, totals were 5,184 / 73,728; 74,240 / 1,179,648; and 593,920 / 9,437,184 bytes. At 100k and 100% populated, the map accounted for 524,288 bytes of 8,388,608 record bytes; at 1% population with ids spread across the range, the map still sized to the entity high-water (524,288 bytes) while records occupied 65,536 bytes. Initial map uploads were 3,604 / 4,000; 39,604 / 40,000; and 399,604 / 400,000 bytes at 1% / 100% population; every subsequent unchanged upload was zero.

Setter evidence (same adapter, 100k writes): `WorldField.set` measured a 0.7399 ms median against 0.1198 ms for direct column writes. The setter includes bounds/growth and dirty-bit work; timings are reported, not asserted.

Record-layout evidence (same adapter, 100k rows; prototypes only, arrays are not an engine table layout): pose records (48 bytes: position, rotation and scale) used one `writeBuffer` call and read all fields in 4,096 ns per dispatch; a follow-up measured struct submission / queue cycle at 0.2779 ms / 0.4083 ms, and three per-field arrays at 0.2867 ms / 0.4004 ms, with the same GPU read. Light records (32 bytes: color and parameters) used one `writeBuffer` call and read all fields in 4,096 ns per dispatch; struct submission / queue cycle was 0.1927 ms / 0.3000 ms, while two per-field arrays were 0.1920 ms / 0.3063 ms. Upload differences are small and change direction by table; full-record GPU reads tie. This evidence does not show a clear need for per-field arrays. The engine keeps one struct record per row.

Stage 4 implementation checkpoint (incomplete): Transform/Body poses, point/spot light inputs, and Part surface/color/material data now use dense struct tables. The Part pack reads its active `(eid,row)` list instead of dispatching over an entity range; light compaction does the same. Part and Sear no longer depend on `SlabPlugin`; Physics `Body`/`Pose` and built-in light fields are `field()` columns. `State.capacity`/`Config.capacity` are gone, and the built-in GPU coverage passes for the table-bound Part/Sear and light inputs. Explicit transitional Slab use remains in Skin/glTF and still needs a capless table migration before Slab internals can leave every build path.

The architect approved a 16-byte draw payload `(eid, transformSlot, encodedPartSlot, shadowCombo)` to remove per-vertex map loads, retaining Part color/material in the surface context. Production still uses compacted eids and per-vertex Transform/Part map lookups; implementation is paused on the measurement below.

`src/standard/rendering/instance-payload.performance.gpu.test.ts` compares paired payload prototypes on Apple M4 Max / Metal 3, macOS 26.7 (25G229). GPU timestamps cover batches of 128 draws or dispatches; medians use seven samples after two warmups. Both compaction prototypes read the same mapped pose/Part inputs and predicate. Regather copies survivors and writes a shadow combo. Vertex prototypes read 48-byte pose and Part records through independently permuted slots; triangles are outside the viewport to isolate vertex work. These are isolated kernels, not the complete production cull/scan/scatter/regather or an end-to-end scene. Sums below add independently measured medians, not a measured frame.

Second-run medians in microseconds, old/new:

| Instances | Compaction | Regather | Vertex, 6 vertices | Vertex, 36 vertices | Vertex, 240 vertices |
| --- | --- | --- | --- | --- | --- |
| 1k | 6.656 / 6.656 | 4.608 / 4.608 | 7.168 / 6.656 | 14.848 / 11.264 | 30.208 / 21.504 |
| 10k | 2.048 / 2.560 | 1.024 / 1.536 | 12.800 / 12.800 | 30.208 / 30.208 | 194.048 / 194.560 |
| 100k | 11.264 / 14.336 | 3.584 / 4.608 | 126.464 / 126.464 | 294.400 / 294.912 | 1924.096 / 1923.584 |

At 100k six-vertex instances, the sum increases from 141.312 to 145.408 us (+2.9%); the preceding run measured 140.800 to 145.408 us (+3.3%). At 10k six-vertex instances, the sum increases from 15.872 to 16.896 us; the preceding run measured 16.384 to 16.896 us. This is a warning for quad-like workloads, not proof of an actual scene regression. The 1k measurements vary substantially between runs. Stage 4 stops for the architect to assess the warning before production migration; no removal is claimed by this probe.

The first-person allocation row is no longer a todo. Its latest run fails the zero-allocation claim: the former `readBody` proxy path was removed and the largest `readBody` / `scopedHandle` allocation sites disappeared, but the sampler still attributes steady allocations to `transitional/character` (`update`, `syncStates`, `sweepEid`) and `transitional/physics` (`StepSystem`, `SyncSystem`, worker pool, manifold store). These are outside the table/upload path changed here and remain a Stage 4 gap, not evidence that the check passes. The mock GPU adapter reports no identity; these allocation figures are not real-hardware timing evidence.

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
