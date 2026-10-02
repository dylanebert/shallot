# Migrating from 0.9.5 to 0.10

These changes require updates to a 0.9.5 app.

## Glaze becomes camera grading and effect passes

Remove `Glaze`, `GlazePlugin`, `GlazeSystem`, `Tonemap`, `tonemapWgsl` and `/glaze` imports. `CorePipelinePlugin` presents views and registers `Tonemapping` and `ColorGrading`, exported from `/rendering` and the root. Default plugins already include it. Add `VignettePlugin` from `/vignette` (or the root) for cameras carrying `Vignette`.

| Glaze field | Replacement |
|---|---|
| `tonemap` | `Tonemapping.method`, using `TonemappingMethod`: `Neutral` → `KhronosPbrNeutral`, `None` → `None`, `Aces` → `AcesFitted`, `Reinhard` → `Reinhard`, `ReinhardLuminance` → `ReinhardLuminance`, `Agx` → `AgX`, `SomewhatBoring` → `SomewhatBoringDisplayTransform`. Do not carry numeric indices forward. |
| `exposure` | Positive multiplier `m` → `ColorGrading.exposure: Math.log2(m)` (stops). |
| `saturation` | `ColorGrading.postSaturation`. |
| `slope` | No per-channel CDL equivalent. Retune using sectional `gain` and global `temperature`/`tint`. |
| `offset` | No per-channel CDL equivalent. Retune using sectional `lift`. |
| `power` | No per-channel CDL equivalent. Retune using sectional `gamma` (inverse exponent). |
| `vignette` | `Vignette.intensity`. Now applied to HDR before tonemapping; the look is not identical. |
| `vignetteInner` | No exact equivalent; retune `Vignette.radius`. |
| `vignetteOuter` | No exact equivalent; retune `Vignette.radius` and `smoothness`. |
| `posterize` | Removed; no built-in replacement. Implement an after-tonemapping pass in an external postprocessing package. |
| `dither` | Removed with posterize; no built-in replacement. |

Cameras without settings now use `TonyMcMapface`, not Khronos PBR Neutral. Add `Tonemapping` with `method: TonemappingMethod.KhronosPbrNeutral` to keep the former operator. `None` skips the operator for display-ready **linear** images; grading and the screen encode still apply. `AgX` retains the former analytic Filament/three.js approximation, rather than Bevy's LUT.

`ColorGrading` applies exposure, temperature, tint and hue (in degrees) globally; `postSaturation` applies after tonemapping. The `saturation`, `contrast`, `gamma`, `gain` and `lift` vectors hold shadows, midtones and highlights in x/y/z (w unused). Defaults are identity; `midtonesRange` defaults to `[0.2, 0.7]`.

Register view-specific passes in `world.resource(EffectPasses)` under the camera eid, with `before` and `after` arrays. Each callback receives `(world, eid, view, input, output)` and records on the frame encoder. Before passes operate on linear HDR; after passes operate on encoded display-referred intermediates. With no after pass, tonemapping writes the presented target directly. Remove registrations when their camera or owner leaves.

## Renamed exports

0.10 renames these 0.9.5 names, with no compatibility aliases:

| 0.9.5 | 0.10 |
|---|---|
| `State`, and `state` in examples | `World`, `world` |
| `Compute` | `world.gpu` |
| Physics `World` | `PhysicsWorld` |
| `Single`, `Pair`, `Quad`, field `Type` | `ScalarField`, `Vector2Field`, `Vector4Field`, `FieldType` |
| `Transform.pos`, `.rot` | `translation`, `rotation` |
| `Body.pos`, `.quat` | `position`, `rotation` |
| `Part` | `MeshInstance` |
| `RenderPlugin` | `RenderingPlugin` for the frame/view substrate; add `CorePipelinePlugin` for shared targets and phases (`StandardRenderingPlugin` includes it) |
| `SearPlugin` | `StandardRenderingPlugin` |
| `Sear`, `Depth`, `Tag`, `Backdrop` | `StandardRenderer`, `DepthPrepass`, `PickingPrepass`, `CameraBackground` |
| `TAG_FORMAT`, `TAG_NONE`, `TagFn`, `view.tag` | `PICKING_ID_FORMAT`, `PICKING_ID_NONE`, `PickingIdFn`, `view.pickingId` |
| `BgCtx`, `BgFn`, `BgLayout` | `BackgroundContext`, `BackgroundFn`, `BackgroundLayout` |
| `/render/core` GPU `View` schema and `linearToSrgb` | `/rendering` `ViewUniforms` and `linearToSrgb3` |
| GPU `View.cluster` | `ViewUniforms.projection` (near, far, perspective flag, slot; unchanged byte layout) |
| `mesh`, `image`, `font`, `text` | `registerMesh`, `registerImage`, `registerFont`, `internText` |
| `segment`, `box`, `arrow` | `drawLine`, `drawWireBox`, `drawArrow` |
| `build`, `run`, `Config`, `swap`, `SwapResult` | `createApp`, `runApp`, `AppConfig`, `swapPlugins`, `PluginSwapResult` |
| `compose`, `decompose`, `multiply`, `invert` | `composeMat4`, `decomposeMat4`, `multiplyMat4`, `invertMat4` |
| `quat`, `euler`, `rotate`, `aim` | `eulerToQuat`, `quatToEuler`, `rotateQuatByEuler`, `lookAtRotation` |
| `composeTransform` | `composeGlobalTransform` |
| `state.stamp`, `state.timescale`, `state.swap` | `world.generation`, `world.setTimeScale`, `world.swapSystem` |
| `/sear/core` `PrepassSystem`, `ColorSystem` | `/standard/rendering` `RenderPrepassesSystem`, `RenderMeshColorSystem` |
| `/src/standard/render/cluster.ts` `ClusterSystem`, `LightCullSystem` | Remove direct imports; these systems are now internal to `StandardRenderingPlugin`. |
| `CharacterSweepSystem`, `PlayerControlSystem` | `SweepCharactersSystem`, `UpdatePlayerControlSystem` |
| Physics `StepSystem`, `ConstraintSystem` | `StepPhysicsSystem`, `SyncPhysicsConstraintsSystem` |

The `pixelRatio` constant is removed (set `AppConfig.pixelRatio`). The `/ecs` wrappers `register`, `getExclusions`, `entries` and `clear` are removed; use `world.registry`.

## Lights own their shadow settings

`Spot` and `Shadow` are removed. A `SpotLight` contains its own light values; do not add a `PointLight` to provide them. `Volumetric` is now `VolumetricLight`. These components and `NotShadowCaster` are exported from the root and `/rendering`.

| 0.9.5 component or field | 0.10 replacement |
|---|---|
| `AmbientLight.color`, `.intensity` | Unchanged: hex sRGB and a linear multiplier. |
| `DirectionalLight.color`, `.intensity`, `.direction` | Unchanged: hex sRGB, a linear multiplier and the light's travel direction. |
| `PointLight.color`, `.intensity`, `.range`, `.radius` on a point light | Unchanged. |
| `PointLight.color`, `.intensity`, `.range`, `.radius` on an entity with `Spot` | `SpotLight.color`, `.intensity`, `.range`, `.radius`; copy the values and remove `PointLight`. |
| `Spot.inner`, `.outer` | `SpotLight.innerAngle`, `.outerAngle`, still half-angles in degrees. |
| Presence of `Shadow` | Set the light's `shadowMapsEnabled` to `1`; `0` disables shadow maps. |
| `Shadow.distance` on a directional light | `DirectionalLight.maximumDistance`, still world units. |
| `Shadow.distance` on a point or spot light | Remove it; it was ignored. Shadow coverage still uses the light's `range`. |
| `Shadow.depthBias`, `.normalBias` | The light's `shadowDepthBias`, `.shadowNormalBias`, with the same values and units. |
| `SHADOW_DEFAULTS` | Removed. Light defaults are `shadowMapsEnabled: 0`, `shadowDepthBias: 0.0005`, `shadowNormalBias: 1.8`; directional `maximumDistance` defaults to `50`. |
| `Volumetric` | `VolumetricLight`, still a marker. |

A shadowed spot light is now authored as:

```ts
world.add(lamp, SpotLight, {
    color: 0xffffff, intensity: 4, range: 12,
    innerAngle: 18,
    outerAngle: 28,
    shadowMapsEnabled: 1,
    shadowDepthBias: 0.0005,
    shadowNormalBias: 1.8,
});
world.add(lamp, VolumetricLight);
```

`NotShadowCaster` is new in 0.10. Add it to a mesh entity to keep it visible without casting shadows. Removing it restores casting. `SunShadows` and `PointShadows` retain their settings. Light brightness units have not changed to lux or lumens.

`Camera` remains one component: `mode`, `fov`, `near`, `far`, `size`, `clearColor` and `antialias` are unchanged, including `fov` in degrees and `antialias: 1` for 4× MSAA. `CameraMode` and `Resolution.width`/`.height` are unchanged.

## Component editor metadata and name lookups are removed

Remove the component traits `requires`, `singleton`, `aliases`, `parse`, `format`, `enums`, `inputs` and `annotations`. Registrations keep `defaults` and add-only `requires`; `excludes` is removed and runtime `provides` becomes `requires`. System annotations remain.

The reflection exports `camel`, `find`, `schema`, `schemas`, `FieldInfo`, `FieldKind`, `Schema`, `isSingleton`, `requiredComponents` (formerly `dependencies`), `provides`, `exclusions` and `kebab` are removed. The `getComponent` and `getTraits` wrappers and registry methods are removed too. Query imported component handles with `world.query([Component])`; a game resolves components and enum values by import, not by name.

Remove `Alias`, `laneAlias`, `eulerAlias`, `formatHex` and the input metadata helpers (`Input`, `Unit`, `units`, `angle`, `degrees`, `radians`; also named `FieldInput`, `FieldUnit`, `unitInput`, `angleInput`, `degreeUnit`, `radianUnit` during 0.10 development). Quaternion conversion helpers remain.

## Register components as one list under exact keys

`Plugin.components` is now a list of registrations, not a component map. `Plugin.traits` and the `Traits` type are removed. Put `defaults` and `requires` directly on each registration:

```ts
// 0.9.5
const GamePlugin = {
    name: "Game",
    components: { Health },
    traits: { Health: { defaults: () => ({ value: 100 }) } },
};
```

```ts
// 0.10
import { registration, type Plugin } from "@dylanebert/shallot";
const GamePlugin: Plugin = {
    name: "Game",
    components: [registration("Health", Health, { defaults: () => ({ value: 100 }) })],
};
```

Defaults use declaration field names and vector arrays, as `world.add` does; replace dotted-lane defaults with complete vectors. Remove imports of `globalTransformTraits`, `bodyTraits`, `springTraits`, `jointTraits`, `PartTraits` and `ColorTraits`; options now live on their plugins' registrations.

`inspect`, `snapshot`, `readFields` and `dump` preserve exact registration keys and declaration field names. A component registered as `"GlobalTransform"` is no longer reported as `"global-transform"`; vectors are arrays such as `{ translation: [0, 0, 0, 0] }`, not dotted lanes. Update saved data and consumers of this output.

## Import app plugins in the entry page

`virtual:project` and `shallot.schema.json` are removed. `shallot.json` no longer supplies app plugins or `pixelRatio`. Import the plugins and pass them directly to `runApp`; set `pixelRatio` there if needed:

```ts
import { OrbitPlugin, runApp } from "@dylanebert/shallot";
import { GamePlugin } from "./src/game";

await runApp({ plugins: [OrbitPlugin, GamePlugin], pixelRatio: "auto" });
```

Default plugins remain enabled unless `defaults: false` is set. Remove manifest-only default exports and import a plugin's named export instead.

The manifest's `identifier` override is removed. Native bundles take `com.shallot.<name>`.

## Request non-default GPU limits explicitly

0.9.5 requested the adapter's maximum limits automatically. The engine now requests default device limits. If your application needs larger buffers or other non-default limits, acquire a device with those limits and pass it through the existing `config.device` option.

## Resolve component storage from the owning World

`slab()` and `sparse()` are removed. Declare bare field types and resolve their values from the World:

```ts
// 0.9.5
const Health = { value: sparse(f32) };
Health.value.set(eid, 100);
```

```ts
// 0.10
const Health = { value: f32 };
const health = world.storage(Health);
health.value.set(eid, 100);
```

Resolve storage once in a system's setup or a lifecycle hook, then retain it for that world. Declare the complete schema before binding it; to change a schema, replace the component object.

Remove `capacity` from `createApp()` configuration and `new World()` options. The exported global `capacity` is gone; columns and tables grow as needed.


## Component fields no longer expose `.gpu`

`state.membership` and the `"membership"` buffer shipped in v0.9.5 are removed. Gate GPU work on a table's active rows.

`Slab`, `SlabPlugin` and `SlabSystem` are removed. Remove them from imports and plugin dependencies. Replace per-field GPU buffers with a record table:

```ts
// 0.9.5
const Heat = { value: slab(f32, "heat") };
const gpuValues = Heat.value.gpu;
```

```ts
// 0.10
const Heat = { value: f32 };
const table = world.table("heat", d.struct({ value: d.f32 }));
table.bindComponent(Heat, { value: "value" });
const gpuRows = table.buffer;
```

Change shaders from entity-indexed scalar arrays to struct records addressed by dense row slots. `table.rowIndex(eid)` gives the CPU slot; enable the table's eid lookup when a shader starts from an eid. Rebind when the table's buffer generation changes. Built-in Body, GlobalTransform and light fields no longer publish their old per-field `.gpu` buffers either.

`f16x4`, `srgb8x4` and `FieldType.gpu` are removed; no table packs a field. Declare `vec4` and bind it as a 16-byte `d.vec4f`. For a 4-byte color, store `packColor4(r, g, b, a)` from `/utils` in a `u32` field, bind it as `d.u32` and read it in the shader with `unpackLdrColor`:

```ts
// 0.9.5
const Material = { params: slab(f16x4, "material"), color: slab(srgb8x4, "color") };
```

```ts
// 0.10
const Material = { params: vec4, color: u32 };
const table = world.table("material", d.struct({ params: d.vec4f, color: d.u32 }));
table.bindComponent(Material, { params: "params", color: "color" });
world.storage(Material).color.set(eid, packColor4(1, 0.5, 0.25, 1));
```

## Authored Transform and world GlobalTransform are separate

`Transform` remains authored placement. The engine derives `GlobalTransform` for each `Transform` or physics `Body` entity; do not add `TransformsPlugin`. Each producer uses `requires: [GlobalTransform]` to add it when missing; removing a producer leaves it attached. Component pairs are no longer refused. Physics warns once per entity carrying both `Body` and `Transform`, since both write its `GlobalTransform`. Read world placement through `world.storage(GlobalTransform)`, not `Transform`. `GlobalTransform` is engine-derived, never authored, and has no hierarchy.

The renderer interpolates previous and current fixed-tick `GlobalTransform` into GPU-only `global-transform-interpolated` rows. It records history copies and interpolation in the renderer's frame submission. Without an interpolated-row reader, the composition does no GlobalTransform GPU work.

Custom typed surfaces change their instance binding from `transforms` to `globalTransforms`; the dense instance record still names its `globalTransform` row. Body, camera, light, text, sprite and other world-space consumers read GlobalTransform rather than Transform as world placement.

## Instanced surfaces read a row payload, not a list of eids

For a custom typed surface, change the `eids` binding element from `d.u32` to `d.vec4u`. Each instance is `(eid, globalTransformSlot, encodedMeshInstanceSlot, shadowCombo)`: the MeshInstance slot is encoded as `slot + 1`, or zero when absent. Resolve slots while producing the instance list, not in the vertex stage. Shadow regather preserves the first three lanes and writes its combo index in the fourth.

The logical eid still reaches `VsIn.eid` and `ctx.eid`; use those for identity.

## glTF and Skin are removed

`GltfPlugin`, `SkinPlugin` and their import, animation and live-skin helpers are no longer exported. Remove these plugins from manifests and imports; there is no replacement in this release line.

The importer scene hooks `Preloader`, `Preloads` and `preload` are removed. Load assets in your plugin's `initialize`.

The importer-only shader specialization is also removed: `Surface.specialize`, `Specialize` and `Mesh.variant` are gone. Register separate named surfaces instead.

## GPU registries and plugin helpers use the owning World

Replace the process-level `Compute` registries with the owning `world.gpu`. Helpers needing GPU state receive the owning World explicitly.

These helpers use the owning World:

- `Profile` data is read with `world.resource(Profile)`.
- `cascadeCount()`, `cascadeComboEids()`, `pointComboCount()` and `pointComboEids()` take World first.
- Character helpers `move`, `jump`, `globalTransform`, `teleport` and `grounded` take World before the entity id.

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
const observation = await probeBuffer(world, counters);
const count = new Uint32Array(observation.bytes)[0];
```

Existing `probeBuffer(device, source, options)` and `probeTexture(device, source, options)` calls now take the owning World instead of the GPUDevice. The returned bytes remain owned by that result.

Requests accept only buffers and textures owned by their World. Allocate through `world.gpu.device` or `world.gpu.root`; register an external allocation intended solely for this world with `world.own(resource)` before probing it. Do not request another world's resource, even on a shared device. Keep counts that only size GPU work on the GPU instead of replacing the old Mirror with per-frame requests.

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
import { FrameGpu } from "@dylanebert/shallot/rendering";
import { engineLayout, registerSurface, surfaceLayout } from "@dylanebert/shallot/standard/rendering";
import { Xform } from "@dylanebert/shallot/utils";
```

`MeshInstance` contains only `mesh`. Add anonymous material values with `const id = world.resource(Materials).add(StandardMaterial(values))`, then add `MeshMaterial` with `{ material: id }`. Retain the returned id to share or update a material; there is no material name or name lookup. Meshes without `MeshMaterial` draw with the shared default `StandardMaterial`.

| 0.9.5 name or value | 0.10 replacement |
|---|---|
| Root `Part` | Root or `/mesh` `MeshInstance` |
| `Part.surface` | Material's `surface`, a `Surfaces` id |
| `Color.rgba` | Material's linear `baseColor` |
| `Material.params` `(metallic, roughness, emissiveStrength, occlusion)` | `StandardMaterial({ metallic, perceptualRoughness, emissive: [baseColor[0] * emissiveStrength, baseColor[1] * emissiveStrength, baseColor[2] * emissiveStrength], occlusion })` |
| `Material` component | Root or `/standard/rendering` `MeshMaterial` referencing an added material's id |
| Root `PartPlugin` | Root or `/standard/rendering` `PartPlugin` |
| `/part/core` `Parts` | `/standard/rendering` `MeshDrawBuffers` (`drawArgs` and `packedEids`) |

`StandardMaterial()` defaults to white base colour, metallic 0, perceptual roughness 0.5, black emissive, occlusion 1 and `diffuseWrap` 1. Set `baseColor: [1, 0, 1, 1]` and `perceptualRoughness: 1` to express the former bare mesh values. `diffuseWrap` blends Lambert (0) with Shallot's squared half-Lambert (1); its default preserves the diffuse look. Publish changed fields with `world.resource(Materials).update(id, values)`; omitted fields retain their values. Set `world.storage(MeshMaterial).material` to switch an entity's material. Material ids belong to the World that added them.

Custom surfaces still receive linear `color`; their `material` lanes are now `(metallic, perceptualRoughness, materialId, occlusion)`, not scalar emissive strength. The standard instance table's `MeshInstanceInput` is `{ mesh: u32, material: u32, flags: u32 }` (`flags` bit 0 excludes the mesh from shadow views); colour and shading values live in the `materials` table, bound in `engineLayout`. `Pbr` inputs now include `diffuseWrap`; use 1 to retain the former diffuse lobe.

Mesh data has its own `/mesh` module. Update mesh imports as follows; these names are also exported from the root in 0.10:

| 0.9.5 import | 0.10 import |
|---|---|
| Root `mesh` | Root or `/mesh` `registerMesh` |
| `/render/core` `Mesh`, `MeshBinding`, `MeshIndex`, `MeshStorage`, `QuantStreams` | `/mesh`, same names |
| `/render/core` `Meshes`, `meshBounds`, `packMeshes`, `quantizeMeshes`, `VERTEX_FLOATS`, `VERTEX_STRIDE` | `/mesh`, same names |

Surface, background and draw contracts belong to `/standard/rendering`. Update imports as follows; contracts re-exported by `/sear/core` use the same mappings:

| 0.9.5 import | 0.10 import |
| --- | --- |
| `/render/core` `Binding`, `SurfaceLayout`, `surfaceLayout`, `Surface`, `Surfaces`, `registerSurface`, `SURFACE_GROUP` | `/standard/rendering`, same names |
| `/render/core` `VsIn`, `vsPatchSchema`, `fsCtxSchema`, `VsFn`, `FsFn`, `assertOwnFn` | `/standard/rendering`, same names |
| `/render/core` `TagFn` | `/standard/rendering` `PickingIdFn` |
| `/render/core` `BgLayout`, `BgCtx`, `BgFn` | `/standard/rendering` `BackgroundLayout`, `BackgroundContext`, `BackgroundFn` |
| `/render/core` `backgroundLayout`, `Background`, `Backgrounds`, `registerBackground` | `/standard/rendering`, same names |
| `/render/core` `Draw`, `DrawIndirectBuffer`, `DrawIndexedIndirect`, `Draws` | `/standard/rendering`, same names |
| `/render/core` `ClusterView`, `CLUSTER_COUNT`, `CLUSTER_X`, `CLUSTER_Y`, `CLUSTER_Z`, `Clusters`, `clusterAabb`, `clusterCell`, `clusterCoord`, `clusterIndex`, `clusterView`, `LIGHT_POOL`, `LightCull`, `lightClusters`, `sliceDepth`, `zSlice` | `/standard/rendering`, same names |
| `/render/core` `LIGHTING_UNIFORM_SIZE`, `Lighting`, `LightingGpu`, `lightingWgsl`, `MAX_POINT_LIGHTS`, `PointLightGpu`, `PointLights`, `pointLightsWgsl`, `distanceAttenuation`, `spotFactor`, `spotParams` | `/standard/rendering`, same names |

`InstanceInput` and `MeshInstanceInput` from `/standard/rendering` are new exports in 0.10, not renamed 0.9.5 exports. The former describes the packed per-draw instance; the latter describes a dense mesh-component row. `StandardMaterial`, `Materials` and `MeshPlugin` are also new exports, replacing the former component-only material values and the mesh registration owned by `RenderPlugin`/`PartPlugin`.

Camera prepass markers and attachment constants are imported from `/rendering`:

| 0.9.5 import | 0.10 import |
| --- | --- |
| Root or `/sear/core` `Depth`, `Tag` | `/rendering` `DepthPrepass`, `PickingPrepass` |
| `/sear/core` `DEPTH_FORMAT`, `TAG_FORMAT`, `TAG_NONE` | `/rendering` `DEPTH_FORMAT`, `PICKING_ID_FORMAT`, `PICKING_ID_NONE` |

`CorePipelinePlugin` from `/rendering` registers both prepass markers and owns view targets, clear, resolve and prepass/opaque/transparent phases. `StandardRenderingPlugin` includes it as a dependency. Custom renderers using these phases depend on `CorePipelinePlugin` and register records in `RenderPhases`; records do not end the shared pass. `RenderingPlugin` alone supplies views and frame/presentation anchors without the shared pipeline. Each marker requests its own camera output; neither requires the other.

Custom surface, background and draw producers depend on `StandardRenderingPlugin`; `RenderingPlugin` alone no longer initializes their registries.

`RenderingPlugin` still registers the light components (`AmbientLight`, `DirectionalLight`, `PointLight`, `SpotLight`, `VolumetricLight`), but no longer packs GPU lights or builds clusters. Compositions using those GPU resources need `StandardRenderingPlugin`.

Custom mesh producers depend on `MeshPlugin` from `/mesh`; `RenderingPlugin` alone no longer initializes mesh storage. Standard rendering and Part include this dependency. `MeshPlugin` registers the built-in cube, sphere and capsule.

Likewise `/ecs/core` is `/ecs`, `/physics/core` and `/tumble/core` are `/physics`, `/character/core` is `/character` and `/bvh/core` is `/bvh`. `/scene/core` is removed with the scene format. The `/src/*` wildcard is gone: use the paths in `package.json` `exports`.

## `Inputs` is now `world.resource(Devices)`

The owning App's keys, pointer, touch and viewport replace the process-level `Inputs` facade; `Inputs.mouse` is `pointer`. `setInputEnabled` takes World, and canvas size moved from `mouse` to `viewport`.

```ts
// 0.9.5
if (Inputs.isKeyDown("KeyW")) moveForward();
if (Inputs.isKeyPressed("Space")) jump();
const width = Inputs.mouse.canvasWidth;
setInputEnabled(false);
```

```ts
// 0.10
const input = world.resource(Devices);
if (input.keys.held.has("KeyW")) moveForward();
if (input.keys.pressed.has("Space")) jump();
const width = input.viewport.get(input.focused)?.cssWidth ?? 0;
setInputEnabled(world, false);
```

In `fixed`, use `keys.tickPressed` instead of the frame-level press. Replace `isKeyPressedWithin(code, seconds)` with fixed-tick comparisons:

```ts
const at = input.keys.pressedTick.get("Space");
if (at !== undefined && world.time.fixedTick - at < 6) jump();
```

## `Tumble` is now `Physics`, and `Physics.backend` is gone

Rename the plugin and manifest key from `Tumble` to `Physics`. Read and drive bodies through World-first functions from `/physics`:

```ts
// 0.9.5
Physics.backend?.setKinematic(eid, position, rotation);
const b = Tumble.body(eid);
```

```ts
// 0.10
import { body, setKinematic } from "@dylanebert/shallot/physics";
setKinematic(world, eid, position, rotation);
const b = body(world, eid);
```

`Tumble.world` becomes `physicsWorld(world)`.

## `/avbd` is gone

The engine no longer ships its AVBD solver or `AvbdPlugin`. Select the built-in `PhysicsPlugin` instead.

## `Tween`, `Sequence` and `/tween/core` are gone

There is no animation plugin in this release line. Implement animation in app code.

## `/document` and edit mode are gone

`Document`, `History`, `Session` and `ReadbackSystem` are removed, with no replacement for undo, redo or editor sessions. Remove `State.mode`, the app's `mode` option and `annotations.mode`: every system always runs.

## Scene format and save/restore are removed

The `.scene` format, `parseScene`, `loadScene`, `serializeScene`, `stringifyScene` and `normalizeAttribute` are removed with no replacement. Remove the app's and manifest's `scene` option. A game saves the component values it needs from `snapshot(world)`, using exact registration keys and declaration field names with vector arrays, and restores them itself.

`Identity`, `world.identity`, `refs()` and the `derived` component trait are removed. Keep the eids returned by `world.create()` instead of naming entities. The `entity` field type still stores a plain eid; save and restore references yourself. `GlobalTransform` remains engine-managed through its producers' `requires`, without the trait.

Author worlds in code:

```ts
import { Transform } from "@dylanebert/shallot";
const eid = world.create();
world.add(eid, Transform, { translation: [0, 1, 0, 0] });
```

## `/harness` helpers are removed

Remove imports of `installHarness`, `HarnessTarget` and `REAL_GPU_LAUNCH`. Drive `createApp()` and `world.step()` in the project's tests and observe through public ECS and physics reads. Configure the project's browser tests with Playwright Test, its own Vite `webServer` and its own Chromium launch flags.

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

0.9.5's scalar `sparse(u8)` stored a JS number: writing 300 read back 300. A `u8` field stores 44. Likewise, scalar `u32` now wraps negatives as unsigned and `f32` rounds to 32-bit precision. Choose an integer type wide enough for the values and account for f32 rounding.

## Update the 0.9.5 scaffold's TypeScript types

Its `tsconfig.json` listed only WebGPU types. Add Node and Vite's types:

```sh
bun add -d @types/node@^26.0.0
```

```json
{ "compilerOptions": { "types": ["@webgpu/types", "node", "vite/client"] } }
```

These are types only; no Node code reaches the browser bundle.
