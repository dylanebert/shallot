# Migrating from 0.9.5 to 0.10

These changes require updates to a 0.9.5 app.

## Request non-default GPU limits explicitly

0.9.5 requested the adapter's maximum limits automatically. The engine now requests default device limits. If your application needs larger buffers or other non-default limits, acquire a device with those limits and pass it through the existing `config.device` option.

## Resolve component storage from the owning State

`slab()` and `sparse()` are removed. Declare fields with `field(type)` and resolve their values from the State:

```ts
// 0.9.5
const Health = { value: sparse(f32) };
Health.value.set(eid, 100);
```

```ts
// 0.10
const Health = { value: field(f32) };
const health = state.of(Health);
health.value.set(eid, 100);
```

Resolve storage once in a system's setup or a lifecycle hook, then retain it for that world. Declare the complete schema before binding it; to change a schema, replace the component object.

Remove `capacity` from `build()` configuration and `new State()` options. The exported global `capacity` is gone; columns and tables grow as needed.

Registration and metadata helpers that used global component registrations now take State: `register`, `getComponent` and `schemas`. Scene helpers `diagnose`, `parseFields`, `formatFields` and `normalizeAttr` also take State first.

## Component fields no longer expose `.gpu`

`Slab`, `SlabPlugin` and `SlabSystem` are removed. Remove them from imports and plugin dependencies. Replace per-field GPU buffers with a record table:

```ts
// 0.9.5
const Heat = { value: slab(f32, "heat") };
const gpuValues = Heat.value.gpu;
```

```ts
// 0.10
const Heat = { value: field(f32) };
const table = state.table("heat", d.struct({ value: d.f32 }));
table.bindComponent(Heat, { value: "value" });
const gpuRows = table.buffer;
```

Change shaders from entity-indexed scalar arrays to struct records addressed by dense row slots. `table.rowIndex(eid)` gives the CPU slot; enable the table's eid lookup when a shader starts from an eid. Rebind when the table's buffer generation changes. Built-in Body, Transform and light fields no longer publish their old per-field `.gpu` buffers either.

## Instanced surfaces read a row payload, not a list of eids

For a custom typed surface, change the `eids` binding element from `d.u32` to `d.vec4u`. Each instance is `(eid, transformSlot, encodedPartSlot, shadowCombo)`: the Part slot is encoded as `slot + 1`, or zero when absent. Resolve slots while producing the instance list, not in the vertex stage. Shadow regather preserves the first three lanes and writes its combo index in the fourth.

The logical eid still reaches `VsIn.eid` and `ctx.eid`; use those for identity.

## glTF and Skin are removed

`GltfPlugin`, `SkinPlugin` and their import, animation and live-skin helpers are no longer exported. Remove these plugins from manifests and imports; there is no replacement in this release line.

The importer scene hooks `Preloader`, `Preloads` and `preload` are removed. Load assets in your plugin's `initialize` before the scene is applied.

The importer-only shader specialization is also removed: `Surface.specialize`, `Specialize` and `Mesh.variant` are gone. Register separate named surfaces instead.

## GPU registries and plugin helpers take State

Outside systems and lifecycle callbacks, replace access to the process-level `Compute` registries with the owning `state.gpu`. `Compute` still resolves the active world's GPU inside callbacks.

These helpers now take the owning State:

- `Profile` data becomes `profile(state)`.
- `cascadeCount()`, `cascadeComboEids()`, `pointComboCount()` and `pointComboEids()` take State first.
- Character helpers `move`, `jump`, `pose`, `teleport` and `grounded` take State before the entity id.

## Replace Mirror with explicit snapshot requests

`Mirror`, `mirror(source)`, `MirrorSystem` and `MirrorPlugin` are removed, with no continuous replacement. Remove Mirror from manifests and plugin dependencies. For a snapshot that an app actually needs, use a one-shot probe:

```ts
// 0.9.5
const observation = mirror(counters);
// MirrorPlugin eventually updates observation.snapshot.bytes.
```

```ts
// 0.10
import { probeBuffer } from "@dylanebert/shallot/runtime";
const observation = await probeBuffer(state, counters);
const count = new Uint32Array(observation.bytes)[0];
```

Existing `probeBuffer(device, source, options)` and `probeTexture(device, source, options)` calls now take the owning State instead of the GPUDevice. The returned bytes remain owned by that result.

Requests accept only buffers and textures owned by their State. Allocate through `state.gpu.device` or `state.gpu.root`; register an external allocation intended solely for this world with `state.own(resource)` before probing it. Do not request another world's resource, even on a shared device. Keep counts that only size GPU work on the GPU instead of replacing the old Mirror with per-frame requests.

## `shallot recipe` is now `shallot add`

```sh
# 0.9.5
bunx shallot recipe first-person

# 0.10
bunx shallot add first-person
```

## `shallot run` and `shallot verify` are removed

Replace `shallot run` with `shallot build` followed by `shallot preview`. Preview never rebuilds. Replace `shallot verify` with the project's own tests.

## Package imports no longer use `/core` suffixes

Rendering split into `/rendering` for shared capabilities and `/standard/rendering` for the default mesh renderer:

```ts
// 0.9.5
import { FrameGpu } from "@dylanebert/shallot/render/core";
import { engineLayout, registerSurface, surfaceLayout } from "@dylanebert/shallot/sear/core";
import { Xform } from "@dylanebert/shallot/utils/core";
```

```ts
// 0.10
import { FrameGpu, registerSurface, surfaceLayout } from "@dylanebert/shallot/rendering";
import { engineLayout } from "@dylanebert/shallot/standard/rendering";
import { Xform } from "@dylanebert/shallot/utils";
```

Likewise `/ecs/core` is `/ecs`, `/scene/core` is `/scene`, `/physics/core` and `/tumble/core` are `/physics`, `/character/core` is `/character` and `/bvh/core` is `/bvh`. The `/src/*` wildcard is gone: use the paths in `package.json` `exports`.

## `Inputs` is now `devices(state)`

The owning App's keys, mouse, touch and viewport replace the process-level `Inputs` facade. `setInputEnabled` takes State, and canvas size moved from `mouse` to `viewport`.

```ts
// 0.9.5
if (Inputs.isKeyDown("KeyW")) moveForward();
if (Inputs.isKeyPressed("Space")) jump();
const width = Inputs.mouse.canvasWidth;
setInputEnabled(false);
```

```ts
// 0.10
const input = devices(state);
if (input.keys.held.has("KeyW")) moveForward();
if (input.keys.pressed.has("Space")) jump();
const width = input.viewport.get(input.focused)?.cssWidth ?? 0;
setInputEnabled(state, false);
```

In `fixed`, use `keys.tickPressed` instead of the frame-level press. Replace `isKeyPressedWithin(code, seconds)` with fixed-tick comparisons:

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
```

```ts
// 0.10
import { body, setKinematic } from "@dylanebert/shallot/physics";
setKinematic(state, eid, position, rotation);
const b = body(state, eid);
```

`Tumble.world` becomes `physicsWorld(state)`.

## `/avbd` is gone

The engine no longer ships its AVBD solver or `AvbdPlugin`. Select the built-in `PhysicsPlugin` instead.

## `Tween`, `Sequence` and `/tween/core` are gone

There is no animation plugin in this release line. Implement animation in app code.

## `/document` and edit mode are gone

`Document`, `History`, `Session` and `ReadbackSystem` are removed, with no replacement for undo, redo or editor sessions. Remove `State.mode`, the app's `mode` option and `annotations.mode`: every system always runs. Saving and loading scenes still works:

```ts
import { serialize, stringify } from "@dylanebert/shallot";
const saved = stringify(serialize(state));
```

## `/harness` helpers are removed

Remove imports of `installHarness`, `HarnessTarget` and `REAL_GPU_LAUNCH`. Drive `build()` and `state.step()` in the project's tests and observe through public ECS and physics reads. Configure the project's browser tests with Playwright Test, its own Vite `webServer` and its own Chromium launch flags.

## Vite configuration and project scripts are now the project's

0.9.5's CLI synthesized a Vite configuration for manifest projects. Create a `vite.config.ts` with Shallot's plugin. For an ejected 0.9.5 config, replace its separate TypeGPU plugin (or `typegpuPlugin()`) with this one. Remove the old `CROSS_ORIGIN_ISOLATION` import; the plugin sets those headers:

```ts
// 0.9.5 ejected config
import { defineConfig } from "vite";
import typegpu from "unplugin-typegpu/vite";
export default defineConfig({ plugins: [typegpu()] });
```

```ts
// 0.10
import { defineConfig } from "vite";
import { shallot } from "@dylanebert/shallot/vite";
export default defineConfig({ plugins: [shallot()] });
```

`shallot dev`, `shallot build` and `shallot preview` now run the project's Vite commands. Add `dev`, `build` and `preview` scripts that run `vite`, `vite build` and `vite preview`; do not make them call the corresponding Shallot command. Native commands add the desktop shell to those project commands.

The peer TypeGPU version has also changed:

```sh
bun add typegpu@~0.12.6
```

## Scalar sparse values now use their declared numeric type

0.9.5's scalar `sparse(u8)` stored a JS number: writing 300 read back 300. Its `field(u8)` replacement stores 44. Likewise, scalar `u32` now wraps negatives as unsigned and `f32` rounds to 32-bit precision. Choose an integer type wide enough for the values and account for f32 rounding.

## Update the 0.9.5 scaffold's TypeScript types

Its `tsconfig.json` listed only WebGPU types. Add Node and Vite's types:

```sh
bun add -d @types/node@^26.0.0
```

```json
{ "compilerOptions": { "types": ["@webgpu/types", "node", "vite/client"] } }
```

These are types only; no Node code reaches the browser bundle.
