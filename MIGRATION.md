# Migrating from 0.9.5 to 0.10

Most 0.9.5 game code carries over. These changes break it.

## `build()` always requires WebGPU

Every app now acquires a GPU device during `build()`, or rejects with the acquisition cause. CPU-only compositions no longer build. Supply an already-acquired device with `config.device` when the host owns acquisition; otherwise the engine uses `navigator.gpu`.

Remove `device: "required"` or `device: "optional"` from plugins. `Plugin.device`, `deviceTier`, `DeviceTier` and `resetCompute` are removed without replacements. Calls to `warmPlugins` must supply a `GPUDevice`; its CPU-tier argument is gone.

Device acquisition no longer requests all adapter maxima. If your application needs a non-default device limit, acquire a device with that limit and supply it through `config.device`.

## Resolve component storage from the owning State

Declare component fields with `field(type)`. Resolve their values from the State in each system or lifecycle hook:

```ts
const Health = { value: field(f32) };

function HealSystem(state: State) {
    const health = state.of(Health);
    for (const eid of state.query([Health])) {
        health.value.set(eid, health.value.get(eid) + 1);
    }
}
```

Declare the complete component schema before using it: the first State to bind a component freezes its object. To change a schema, replace the component object.

Remove `capacity` from app configuration. `State.capacity` and `Config.capacity` are gone; component columns and GPU tables grow as needed. A GPU table that exceeds its device's buffer limit refuses with the limit in its error.

Direct ECS registration and reflection helpers now take the State explicitly. Scene helpers that interpret registered attributes (`diagnose`, `parseFields`, `formatFields` and `normalizeAttr`) do too. `parse` remains state-free.

Multiple Apps can coexist on one or separate devices. Dispose each App when finished. A schema-compatible hot swap retains that App's columns; a changed schema returns `{ ok: false }`, requiring the host to rebuild the App.

## Replace Slab fields with GPU record tables

`slab()`, `sparse()` and `SlabPlugin` are removed. Component fields no longer expose `.gpu` or declare a storage kind. Use `field(type)` for component columns and declare GPU records through the owning State:

```ts
const Record = d.struct({ value: d.f32 });
const table = state.table("values", Record);
const row = table.acquire(eid);
// Fill table.bytes with records, then mark the written row range.
table.markRange(row, 1);
```

Each table has one struct record layout and dense rows. `table.acquire(eid)` returns a slot that stays stable until `table.release(eid)`. Use the compact `activeRowsBuffer` for dispatches rather than scanning entity ids. If a shader starts from an eid, enable the lookup with `table.enableEidLookup()` or `table.subscribeMap(...)`; map entries encode `slot + 1`, with zero meaning absent.

Fill records in bulk through `table.bytes`, then call `table.markRange(firstRow, count)`. Upload with `table.upload()` before the passes that read the table. Unchanged tables skip upload; changed tables use a range `writeBuffer`.

Tables expose a raw record buffer, typed handle, capacity and generation. Subscribe to record, map or active-list changes and rebuild bind groups when the corresponding buffer changes. Do not retain a buffer across growth without rebinding.

Part and Sear no longer require `SlabPlugin`. Select their plugins directly. Body, Pose and built-in light fields use `field()` columns; do not read their old per-field `.gpu` buffers.

## Instanced surfaces read a row payload, not a list of eids

For a custom typed surface, change the `eids` binding element from `d.u32` to `d.vec4u`. Each instance is `(eid, transformSlot, encodedPartSlot, shadowCombo)`: the Part slot is encoded as `slot + 1`, or zero when absent. Resolve slots while producing the instance list, not in the vertex stage. Shadow regather preserves the first three lanes and writes its combo index in the fourth.

The logical eid still reaches `VsIn.eid` and `ctx.eid`; use those for identity. Part color and material reach the surface context as `ctx.color` and `ctx.material`. The injected `transformRows` and `partRowMap` surface bindings are removed; the vertex stage reads records using the slots in its instance payload.

## glTF, Skin and Cells are removed

0.10 no longer exports `GltfPlugin`, `SkinPlugin`, `CellsPlugin` or their import, animation, live-skin and ASCII-grid helpers. Remove these plugins from your manifest and imports. They return later as separate packages; there is no replacement in 0.10.

The importer-only scene hooks `Preloader`, `Preloads` and `preload` are removed. Load assets in your plugin's `initialize` before the scene is applied.

The importer's per-mesh shader specialization is also removed: `Surface.specialize`, `Specialize` and `Mesh.variant` are gone. Register separate named surfaces instead. Text's glyph-atlas extension exports used by Cells are removed; the `Text` component and `TextPlugin` remain.

## GPU resources and plugin helpers belong to an App

Do not cache one App's buffers, textures, pipelines or bind groups for another App. Keep per-world plugin data in `state.resource(key, create)`. GPU resources created through the State's tracked device or TypeGPU root are released when the App is disposed; use `state.own(resource)` for raw buffers or textures whose disposal the State should own.

Access GPU registries through `state.gpu`. `Compute` remains available inside systems and lifecycle callbacks; outside those callbacks, retain the owning State's `state.gpu` rather than using `Compute`.

These helpers now take the owning State:

- `profile(state)` replaces process-level `Profile` data.
- `mirror(state, source)` replaces `mirror(source)`.
- `cascadeCount`, `cascadeComboEids`, `pointComboCount` and `pointComboEids` take State first.
- Character helpers `move`, `jump`, `pose`, `teleport` and `grounded` take State before the entity id.

Each PhysicsPlugin App has its own physics runtime. State-first physics calls operate on that runtime; standalone low-level `World` calls remain available. A `WorldSnapshot` owns detached state and copied kernel bytes. Restore it only into a compatible World; restoring into a fresh World requires that its kernel have no other live World.

## `shallot recipe` is now `shallot add`

```sh
# 0.9.5
bunx shallot recipe first-person

# 0.10
bunx shallot add first-person
```

## The CLI command set changed

- `shallot list` and `shallot check` are removed. Run the project's tests with Bun and browser tests with Playwright Test.
- `shallot workflow` has no replacement; edit the workflow file it wrote yourself.
- `shallot <verb>` no longer resolves to `shallot-<verb>` on `PATH`; invoke your external command directly.
- `shallot run` is now `shallot build` followed by `shallot preview`. Preview never rebuilds.
- `shallot tui` and `shallot verify` are removed without replacement commands.

## `/render/core`, `/sear/core` and `/utils/core` no longer resolve

Package imports dropped their `/core` suffix, and rendering split in two: `/rendering` for shared renderer capabilities, and `/standard/rendering` for the default mesh renderer.

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

Likewise `/ecs/core` is `/ecs`, `/scene/core` is `/scene` and `/physics/core` is `/physics`. The `/src/*` wildcard is gone: use the paths in `package.json` `exports`.

## `Inputs` is now `devices(state)`

`devices(state)` returns the owning App's keys, mouse, touch and viewport. The default plugins fill it from browser input; a test or replay can fill the same record with `pressKey`, `pointerMove` and the other producers. `setInputEnabled` takes State, and the canvas size moved from `mouse` to `viewport`.

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
pressKey(state, "KeyW");
```

`keys.pressed` holds a press until the next frame. A `fixed` system reads `keys.tickPressed`, which holds it until the next fixed update. Replace `isKeyPressedWithin(code, seconds)` with fixed-tick comparisons:

```ts
const at = input.keys.pressedTick.get("Space");
if (at !== undefined && state.time.fixedTick - at < 6) jump();
```

## `Tumble` is now `Physics`, and `Physics.backend` is gone

Rename the plugin and manifest key from `Tumble` to `Physics`. Read and drive bodies through State-first functions from `/physics`:

```ts
// 0.9.5
Physics.backend?.setKinematic(eid, position, rotation);
const b = Tumble.body(eid);

// 0.10
import { body, setKinematic } from "@dylanebert/shallot/physics";
setKinematic(state, eid, position, rotation);
const b = body(state, eid);
```

`Tumble.world` is `physicsWorld(state)`, and `readBody(state, eid)` reads a body's pose.

## `/avbd` is gone

0.10 ships no AVBD solver. Select the built-in `Physics` plugin instead. The external `shallot-avbd-physics` package no longer provides its `AvbdPlugin`/State facade; its low-level solver remains.

## `Tween`, `Sequence` and `/tween/core` are gone

0.10 has no animation plugin. Implement animation in app code.

## `/document` and edit mode are gone

`Document`, `History`, `Session` and `ReadbackSystem` are removed, with no replacement for undo, redo or editor sessions. Remove `State.mode`, the app's `mode` option and `annotations.mode`: every system always runs. Saving and loading scenes still works:

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

## Component values use their declared numeric type

Replacing `sparse(u8)` with `field(u8)` retains typed writes: 300 becomes 44, `u32` stores negatives as unsigned, and `f32` rounds to 32-bit precision. Declare a type wide enough for your values.

## `tsc` reports missing `ImportMeta.env` or Node types

The 0.9.5 scaffold's `tsconfig.json` lists only WebGPU types. Add Node and Vite's:

```sh
bun add -d @types/node@^26.0.0
```

```json
{ "compilerOptions": { "types": ["@webgpu/types", "node", "vite/client"] } }
```

These are types only; no Node code reaches the browser bundle.
