# Migrating from 0.9.5 to 0.10

These changes require updates to a 0.9.5 app.

## Fixed systems read tick time

Inside `fixed`, `world.time.elapsed` is now `fixedTick * Time.FIXED_DT`, not the
virtual frame elapsed time. Each catch-up tick reads its own time. Keep simulation
trajectories on this clock; move frame-time effects to `simulation` or `draw`,
where `elapsed` remains the virtual frame clock.

## System errors end a stepped frame

Outside `runApp`, `world.step()` now throws when a system's setup or update throws.
The Error names the system and carries the thrown value as `cause`; later systems
do not run, and the next step retries the failing system. Checks that provoke an
error must expect it with `expect(() => world.step()).toThrow(...)`. Catch the error
in a custom frame loop. `runApp` still logs, pauses the failing system until a swap
or rebuild, and finishes the frame.

## Camera rays belong to rendering

Import `Ray` from the root instead of `/physics`. Replace `cursorRay(world, camera)`
with `viewportToWorld(world, camera, pointer.x, pointer.y)` from `/rendering`, after
checking the input pointer's hover state. Coordinates are CSS pixels relative to
that camera's bound canvas, not a focused input viewport chosen by the helper.

Replace `forwardRay(world, camera)` with `viewportToWorld` at the bound viewport's
centre (`cssWidth / 2`, `cssHeight / 2`). Unlike `forwardRay`, the perspective ray's
origin is offset from the camera by its near distance; account for that offset
when interpreting hit distances. `viewportToWorld` reads fixed-tick `GlobalTransform`
and returns null without camera placement or a non-empty bound viewport.
`generateRay` and `screenToRay` are no longer physics exports.

## Hull registries belong to worlds

Replace `Hulls.register(...)` and other registry calls with
`world.resource(Hulls).register(...)` (or the corresponding registry method).
Hull ids belong to that world; register geometry in each world that uses it.
Every world's registry starts with the unit cube at `UNIT_CUBE_ID` (0).

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

## Rendering extension exports

These 0.9.5 exports are removed or renamed in 0.10. They shipped through the rendering extension subpaths or the former `/src/*` subpath. Remove direct imports from internal files; they are not replacement public paths.

### `/rendering`

| Removed name | Replacement |
|---|---|
| `Render` | `RenderContext`, the world-owned shared GPU context. |
| `backingSize` | Use `Resolution` to pin a view's render size; viewport sizing is internal. |
| `COLOR_LANES` | Removed; there is no replacement color lane. |
| `frameWgsl` | Resolve the `FrameGpu` TypeGPU schema. |
| `FRUSTUM_FLOATS` | Read `CullVolumes` with `CULL_VOLUME_FLOATS`. |
| `frustumPlanes` | No public replacement; consume the packed `CullVolumes`. |
| `allocArray` | Use `imageArray` for image-array upload. |
| `arrayFromBitmaps` | Use `imageArray` for image-array upload. |
| `commonSize` | No public replacement; image-array sizing is internal. |
| `mipLevels` | No public replacement; mip allocation is internal. |
| `uploadLayer` | Use `imageArray` for image-array upload. |
| `linearToSrgbWgsl` | Resolve `linearToSrgb3` through TypeGPU. |
| `VIEW_BYTES` | Derive layout size from `ViewUniforms` with TypeGPU. |
| `VIEW_STRIDE` | Read the view's own uniform buffer, not staging offsets. |
| `viewWgsl` | Resolve the `ViewUniforms` TypeGPU schema. |

### `/mesh`

| Removed name | Replacement |
|---|---|
| `meshBounds` | Register mesh vertices; `MeshPlugin` computes culling bounds. |
| `MeshStorage` | Infer storage from the registered `Mesh`. |
| `QuantStreams` | No public replacement; mesh quantization is internal. |
| `packMeshes` | Register with `registerMesh`; `MeshPlugin` packs at warm-up. |
| `quantizeMeshes` | Register with `registerMesh`; quantization is internal. |
| `VERTEX_FLOATS` | Author the vertex data accepted by `registerMesh`; packing is internal. |
| `VERTEX_STRIDE` | Read the registered mesh's GPU bindings; packing is internal. |

### `/standard/rendering`

| Removed name | Replacement |
|---|---|
| `PartPlugin` | `MeshRenderPlugin`. |
| `ClusterView` | No public replacement; standard owns light-cluster view packing. |
| `CLUSTER_COUNT` | No public replacement; standard owns light-grid sizing. |
| `CLUSTER_X` | No public replacement; standard owns light-grid sizing. |
| `CLUSTER_Y` | No public replacement; standard owns light-grid sizing. |
| `CLUSTER_Z` | No public replacement; standard owns light-grid sizing. |
| `clusterAabb` | No public replacement; light-grid construction is internal. |
| `clusterCoord` | Use `clusterCell` to address the shared light grid. |
| `clusterIndex` | Use `clusterCell` to address the shared light grid. |
| `clusterView` | No public replacement; standard owns light-cluster view packing. |
| `LIGHT_POOL` | No public replacement; standard owns light-grid capacity. |
| `lightClusters` | Read the `LightCull` resource and `LightClusters` schema. |
| `sliceDepth` | No public replacement; light-grid construction is internal. |
| `zSlice` | Use `clusterCell` to address the shared light grid. |
| `Binding` | Infer bindings from `surfaceLayout` or `backgroundLayout`. |
| `FsFn` | Infer from `Surface` or `registerSurface`. |
| `SurfaceLayout` | Infer from `surfaceLayout`. |
| `VsFn` | Infer from `Surface` or `registerSurface`. |
| `assertOwnFn` | Registration validates shader ownership. |
| `SURFACE_GROUP` | `surfaceLayout` owns the bind-group index. |
| `clusterOf` | Use `clusterCell` for grid addressing or `lit` for surface lighting. |
| `engineScaffoldWgsl` | Use `surfaceLayout`, shader functions and `registerSurface`. |
| `fragCoord` | Use the fragment context supplied to the surface function. |
| `fragWorld` | Use the fragment context supplied to the surface function. |
| `lightFactor` | Use `lit` for the standard lighting response. |
| `litPbr` | Use `StandardMaterial` or `lit`. |
| `pointFactor` | Use `lit` for surface lighting; `distanceAttenuation` for attenuation. |
| `pointScale` | Use `lit` for surface lighting; `distanceAttenuation` for attenuation. |
| `sunVisibility` | Use `lit` or the shared `sampleSunShadow` function. |
| `LIGHTING_UNIFORM_SIZE` | Derive layout size from `LightingGpu` with TypeGPU. |
| `lightingWgsl` | Resolve the `LightingGpu` TypeGPU schema. |
| `lightEvalWgsl` | Resolve `distanceAttenuation`, `spotFactor` and `clusterCell` through TypeGPU. |
| `MAX_POINT_LIGHTS` | No public replacement; standard owns light capacity. |
| `PointLights` | Use the shared `LightClusters` schema for light/grid input. |
| `pointLightsWgsl` | Resolve the `LightClusters` schema through TypeGPU. |
| `spotParams` | Use `spotFactor` with the packed light record. |
| `getCompiledSurface` | No public replacement; compiled-pipeline diagnostics are internal. |
| `DrawIndirectBuffer` | Infer from `Draw.args.indirect`; allocate with `DrawIndexedIndirect`. |
| `casterWgsl` | No public replacement; shadow-caster code is internal. |
| `Pbr` | Use `StandardMaterial` or `lit`. |
| `pointShadowWgsl` | Resolve `pointShadowRef()` through TypeGPU. |
| `SHADOW_PARAMS_BYTES` | Derive layout size from `SunShadow` with TypeGPU. |
| `sunShadowWgsl` | Resolve `sampleSunShadow` through TypeGPU. |
| `sunStructWgsl` | Resolve the `SunShadow` TypeGPU schema. |
| `cascadeCount` | Read the directional light's `DirectionalLight.numCascades`. |
| `pointAtlasSize` | Read `world.resource(PointShadows).atlas`. |
| `pointComboCount` | No public replacement; shadow-camera diagnostics are internal. |

### `/fog`

| Removed name | Replacement |
|---|---|
| `FogSystem` | Depend on `FogPlugin`; order scene effects with core's phase and overlay anchors. |
| `FogScatter` | No public replacement; fog integration is internal. |
| `FogSun` | No public replacement; fog integration is internal. |
| `FOG_BYTES` | No public replacement; fog uniforms are internal. |
| `FOG_FLOATS` | No public replacement; fog uniforms are internal. |
| `FOG_MAX_STEPS` | Use `Fog.steps`; the pass clamps it. |
| `FogGpu` | Author `Fog`; the pass owns its GPU layout. |
| `fogComposite` | No public replacement; fog integration is internal. |
| `fogDensity` | No public replacement; fog integration is internal. |
| `fogInScatter` | No public replacement; fog integration is internal. |
| `fogInScatterWgsl` | No public replacement; fog integration is internal. |
| `fogMarchWgsl` | No public replacement; fog integration is internal. |
| `fogStructWgsl` | No public replacement; fog uniforms are internal. |
| `fogSunInScatter` | No public replacement; fog integration is internal. |
| `fogTransmittance` | No public replacement; fog integration is internal. |
| `heightOpticalDepth` | No public replacement; fog integration is internal. |
| `henyeyGreenstein` | No public replacement; fog integration is internal. |
| `inScatterContribution` | No public replacement; fog integration is internal. |
| `reconstructWorld` | No public replacement; fog integration is internal. |
| `sunInScatter` | No public replacement; fog integration is internal. |
| `WORKGROUP` | No public replacement; fog dispatch sizing is internal. |
| `packFog` | Set `Fog` fields; the pass owns packing. |

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
| Moving `Body` with positive `mass` | Write `type: BodyType.Dynamic`; `mass` is the dynamic body's mass. |
| `Body` with `mass: 0` | Omit `type` for static geometry, or write `type: BodyType.Kinematic` for caller-driven motion. |
| `Part` | `MeshInstance` |
| `RenderPlugin` | `RenderingPlugin` for the frame/view substrate; add `CorePipelinePlugin` for shared targets and phases (`StandardRenderingPlugin` includes it) |
| `SearPlugin` | `StandardRenderingPlugin` |
| `Sear`, `Depth`, `Backdrop` | `StandardRenderer`, `DepthPrepass`, `CameraBackground` |
| `Tag`, `TAG_FORMAT`, `TAG_NONE`, `TagFn`, `view.tag` | Removed; there is no replacement picking lane. |
| `BgCtx`, `BgFn`, `BgLayout` | `BackgroundContext`, `BackgroundFn`, `BackgroundLayout` |
| `/render/core` GPU `View` schema and `linearToSrgb` | `/rendering` `ViewUniforms` and `linearToSrgb3` |
| GPU `View.cluster` | `ViewUniforms.projection` (near, far, perspective flag, slot; unchanged byte layout) |
| `mesh`, `image`, `font`, `text` | `registerMesh`, `registerImage`, `registerFont`, `internText` |
| `segment`, `box`, `arrow` | `drawLine`, `drawWireBox`, `drawArrow` |
| `build`, `run`, `Config`, `swap`, `SwapResult` | `createApp`, `runApp`, `AppConfig`, `swapPlugins`, `PluginSwapResult` |
| `compose`, `decompose`, `multiply`, `invert` | `composeMat4`, `decomposeMat4`, `multiplyMat4`, `invertMat4` |
| `quat`, `euler`, `rotate`, `aim` | `eulerToQuat`, `quatToEuler`, `rotateQuatByEuler`, `lookAtRotation` |
| `composeTransform` | `composeGlobalTransform` |
| `state.stamp`, `state.timescale`, `state.swap` | `world.ref` and `world.resolve`, `world.setTimeScale`, `world.swapSystem` |
| `/sear/core` `PrepassSystem`, `ColorSystem` | `/rendering` `PrepassSystem`, `MainPassSystem` |
| `/src/standard/render/cluster.ts` `ClusterSystem`, `LightCullSystem` | Remove direct imports; these systems are now internal to `StandardRenderingPlugin`. |
| `CharacterSweepSystem` | Order fixed velocity producers before `CharacterPlugin.systems` |
| `PlayerControlSystem` | `UpdatePlayerControlSystem` (look and camera); `DrivePlayerSystem` consumes fixed-tick input |
| Physics `StepSystem` | `StepPhysicsSystem` |
| Physics `ConstraintSystem` | Removed: author the joint-kind components; standard physics syncs them. |

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
| `SunShadows.cascades` | `DirectionalLight.numCascades`, default `4`, clamped to `MAX_CASCADES`. |
| `SunShadows.overlap` | `DirectionalLight.overlapProportion`, default `0.2`. |
| `SunShadows.lambda` | Removed. Set `DirectionalLight.firstCascadeFarBound`, default `10` world units: the first cascade ends there and the rest are spaced exponentially to `maximumDistance`. |
| `SunShadows.resolution` | `world.resource(DirectionalLightShadowMap).size`, default `2048`. |
| `PointShadows.atlas`, `.casters`, `.hysteresis` | The same fields on `world.resource(PointShadows)`. |

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

`NotShadowCaster` is new in 0.10. Add it to a mesh entity to keep it visible without casting shadows. Removing it restores casting. `PointShadows` and `DirectionalLightShadowMap` belong to each World: instead of assigning to an imported object, write `world.resource(PointShadows).atlas = 1024` in `AppConfig.setup` or a plugin's `initialize`. Light brightness units have not changed to lux or lumens.

`Camera` remains one component: `mode`, `fov`, `near`, `far`, `size`, `clearColor` and `antialias` are unchanged, including `fov` in degrees and `antialias: 1` for 4× MSAA. `CameraMode` and `Resolution.width`/`.height` are unchanged.

## Component editor metadata and name lookups are removed

Remove the component traits `requires`, `singleton`, `aliases`, `parse`, `format`, `enums`, `inputs` and `annotations`. Component declarations keep `defaults` and add-only `requires`; `excludes` is removed and runtime `provides` becomes `requires`. System annotations remain.

The reflection exports `camel`, `find`, `schema`, `schemas`, `FieldInfo`, `FieldKind`, `Schema`, `isSingleton`, `dependencies`, `provides`, `exclusions` and `kebab` are removed. The `getComponent` and `getTraits` wrappers and registry methods are removed too. Query imported component handles with `world.query([Component])`; a game resolves components and enum values by import, not by name.

Remove `Alias`, `laneAlias`, `eulerAlias`, `formatHex` and the input metadata helpers (`Input`, `Unit`, `units`, `angle`, `degrees`, `radians`). Quaternion conversion helpers remain.

## Declare components once under exact keys

`Plugin.components` is now a list of declared field records, not a component map. `Plugin.traits` and the `Traits` type are removed. Declare the exact key, `defaults` and `requires` with the fields:

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
import { component, f32, type Plugin } from "@dylanebert/shallot";
const Health = component("Health", { value: f32 }, { defaults: () => ({ value: 100 }) });
const GamePlugin: Plugin = {
    name: "Game",
    components: [Health],
};
```

Defaults use declaration field names and vector arrays, as `world.add` does; replace dotted-lane defaults with complete vectors. Remove imports of `bodyTraits`, `springTraits`, `jointTraits`, `PartTraits` and `ColorTraits`; options now live on component declarations.

Keep each key byte-for-byte; it identifies saved data and hot reload. `component` returns the field record unchanged, so storage and insertion calls stay the same. Undeclared records in `Plugin.components` are refused, naming the plugin and the record's fields.

`inspect`, `snapshot`, `readFields` and `dump`, and the `EntityData` and `FieldValues` types, are removed. Read component values through `world.storage(Component)`. `world.snapshot()` returns `WorldSnapshot`, opaque, world-local recovery state for `world.restore(snapshot)`, not a save format.

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

`Transform` remains authored placement. The engine derives `GlobalTransform` for each `Transform` or physics `Body` entity; do not add `TransformsPlugin`. Each producer uses `requires: [GlobalTransform]` to add it when missing; removing a producer leaves it attached. Component pairs are no longer refused. Physics warns once per entity carrying both `Body` and `Transform`, since both write its `GlobalTransform`. Read world placement through `world.storage(GlobalTransform)`, not `Transform`. `GlobalTransform` is engine-derived, never authored, and has no hierarchy. Physics publishes rigid pose and velocity, not collider-derived scale. Register a sized mesh with `cube([hx, hy, hz])`, `sphere(radius)` or `capsule(halfHeight, radius)` from `/mesh` when the visual should match the collider; these builders default to the built-in sizes.

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
- `cascadeComboEids()` and `pointComboEids()` take World first; `cascadeCount()` and `pointComboCount()` are removed.

## Character movement and player feel

Import `Character`, `CharacterPlugin` and `GroundState` from `@dylanebert/shallot/standard/physics` (also re-exported from the root). `/character` and `/character/core` are removed. `Character` requires a capsule `Body` with `type: BodyType.Kinematic`; its pogo spring floats the lower sphere centre three radii above ground. Retune spawn and camera heights for that float.

- Replace `move` with a fixed-tick write to `world.storage(Character).velocity`, ordered before `CharacterPlugin.systems`. Standard physics resolves that velocity, without gravity or acceleration.
- Replace `jump` with your input policy's velocity write. With `PlayerPlugin`, the Space press edge is buffered by Player; jump tuning belongs to `Player.jumpSpeed`.
- Replace `globalTransform` with `world.storage(GlobalTransform).translation`; test membership with `world.has(eid, GlobalTransform)` when placement may not yet exist.
- Replace `grounded` with `world.storage(Character).groundState.get(eid) === GroundState.OnGround`; steep ground is a separate state.
- Replace `teleport` with `setKinematic(world, eid, position, rotation, true)` from `/standard/physics`, then clear `Character.velocity` and `Character.pogoVelocity` for a stationary respawn.
- `PlayerPlugin` no longer installs `RenderingPlugin`: it authors the linked camera's pose without presentation. Keep rendering in the app's composition when presenting that camera.
- Move gravity and jump tuning to `Player.gravity` (positive downward acceleration) and `Player.jumpSpeed`. Player owns acceleration, friction, sprint, coyote time (0.15 seconds), jump buffering (0.2 seconds) and platform carry. Its default speed is 6 m/s, sprint multiplier 1.5, jump speed 5 m/s and gravity 15 m/s².

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
| Root `PartPlugin` | Root or `/standard/rendering` `MeshRenderPlugin` |
| `/part/core` `Parts` | Removed; mesh-instance packing is internal to `MeshRenderPlugin`. |
| `Draws` names `part:<surface>:<mesh>`, profiler span `part:pack` | `mesh:<surface>:<mesh>`, `mesh:preprocess` |

`StandardMaterial()` defaults to white base colour, metallic 0, perceptual roughness 0.5, black emissive, occlusion 1 and `diffuseWrap` 1. Set `baseColor: [1, 0, 1, 1]` and `perceptualRoughness: 1` to express the former bare mesh values. `diffuseWrap` blends Lambert (0) with Shallot's squared half-Lambert (1); its default preserves the diffuse look. Publish changed fields with `world.resource(Materials).update(id, values)`; omitted fields retain their values. Set `world.storage(MeshMaterial).material` to switch an entity's material. Material ids belong to the World that added them.

Custom surfaces still receive linear `color`; their `material` lanes are now `(metallic, perceptualRoughness, materialId, occlusion)`, not scalar emissive strength. The standard instance table's `MeshInstanceInput` is `{ mesh: u32, material: u32, flags: u32 }` (`flags` bit 0 excludes the mesh from shadow views); colour and shading values live in the `materials` table, bound in `engineLayout`. Use `StandardMaterial.diffuseWrap: 1` to retain the former diffuse lobe.

Mesh data has its own `/mesh` module. Update mesh imports as follows; these names are also exported from the root in 0.10:

| 0.9.5 import | 0.10 import |
|---|---|
| Root `mesh` | Root or `/mesh` `registerMesh` |
| `/render/core` `Mesh`, `MeshBinding`, `MeshIndex` | `/mesh`, same names |
| `/render/core` `Meshes` | `/mesh`, same name |

Mesh packing and layout helpers not listed here are removed; see the rendering extension export table above.

`Mesh.dynamic`, `Mesh.count` and `Mesh.cast` are removed; they had no effect. Delete them, and add `NotShadowCaster` to an entity that should cast no shadow.

Surface, background and draw contracts belong to `/standard/rendering`. Update imports as follows; contracts re-exported by `/sear/core` use the same mappings:

| 0.9.5 import | 0.10 import |
| --- | --- |
| `/render/core` `surfaceLayout`, `Surface`, `Surfaces`, `registerSurface` | `/standard/rendering`, same names |
| `/render/core` `VsIn`, `vsPatchSchema`, `fsCtxSchema` | `/standard/rendering`, same names |
| `/render/core` `TagFn` | Removed; there is no replacement picking lane. |
| `/render/core` `BgCtx` | `/standard/rendering` `BackgroundContext` |
| `/render/core` `BgLayout`, `BgFn` | Removed; infer from `backgroundLayout` and `Background`. |
| `/render/core` `backgroundLayout`, `Background`, `Backgrounds`, `registerBackground` | `/standard/rendering`, same names |
| `/render/core` `Draw`, `DrawIndexedIndirect`, `Draws` | `/standard/rendering`, same names |
| `/render/core` `Clusters`, `clusterCell`, `LightCull` | `/standard/rendering`, same names |
| `/render/core` `Lighting`, `LightingGpu`, `PointLightGpu`, `distanceAttenuation`, `spotFactor` | `/standard/rendering`, same names |

Other implementation helpers in these contracts are removed; see the rendering extension export table above. `MeshInstanceInput` from `/standard/rendering` is new in 0.10 and describes a dense mesh-component row. `StandardMaterial`, `Materials` and `MeshPlugin` are also new exports, replacing the former component-only material values and mesh registration owned by `RenderPlugin`/`PartPlugin`.

Camera prepass markers and attachment constants are imported from `/rendering`:

| 0.9.5 import | 0.10 import |
| --- | --- |
| Root or `/sear/core` `Depth` | `/rendering` `DepthPrepass` |
| `/sear/core` `DEPTH_FORMAT` | `/rendering` `DEPTH_FORMAT` |
| Root or `/sear/core` `Tag`, `/sear/core` `TAG_FORMAT`, `TAG_NONE` | Removed; there is no replacement picking lane. |

`CorePipelinePlugin` from `/rendering` registers `DepthPrepass` and owns view targets, clear, resolve and prepass/opaque/transparent phases. `StandardRenderingPlugin` includes it as a dependency. Custom renderers using these phases depend on `CorePipelinePlugin` and register records in `RenderPhases`; records do not end the shared pass. `RenderingPlugin` alone supplies views and frame/presentation anchors without the shared pipeline. `DepthPrepass` requests the camera's stored depth output.

Custom surface, background and draw producers depend on `StandardRenderingPlugin`; `RenderingPlugin` alone no longer initializes their registries.

`RenderingPlugin` still registers the light components (`AmbientLight`, `DirectionalLight`, `PointLight`, `SpotLight`, `VolumetricLight`), but no longer packs GPU lights or builds clusters. Compositions using those GPU resources need `StandardRenderingPlugin`.

Custom mesh producers depend on `MeshPlugin` from `/mesh`; `RenderingPlugin` alone no longer initializes mesh storage. `StandardRenderingPlugin` and `MeshRenderPlugin` include this dependency. `MeshPlugin` registers the built-in cube, sphere and capsule.

Likewise `/ecs/core` is `/ecs`, `/physics/core` and `/tumble/core` are `/physics`, `/character/core` is removed in favor of `/standard/physics` and `/bvh/core` is `/bvh`. `/scene/core` is removed with the scene format. The `/src/*` wildcard is gone: use the paths in `package.json` `exports`.

## `Inputs` is now `world.resource(Devices)`

The owning App's keys, pointer and touch replace the process-level `Inputs` facade; `Inputs.mouse` is `pointer`. `setInputEnabled` takes World. Canvas CSS size and device-pixel ratio live in the engine's `world.resource(Viewports)`, keyed by canvas index, not in `Devices`. Import `Viewports`, `Viewport` and `resizeViewport` from the root or `/engine`; `resizeViewport(world, index, width, height, dpr)` writes a row.

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
const width = world.resource(Viewports).get(input.focused)?.cssWidth ?? 0;
setInputEnabled(world, false);
```

In `fixed`, use `keys.tickPressed` instead of the frame-level press. Replace `isKeyPressedWithin(code, seconds)` with fixed-tick comparisons:

```ts
const at = input.keys.pressedTick.get("Space");
if (at !== undefined && world.time.fixedTick - at < 6) jump();
```

## Shared physics data and standard simulation

Replace the old `Tumble`/`Physics` simulation plugin with `StandardPhysicsPlugin` from
`@dylanebert/shallot/standard/physics`. It depends on core's `PhysicsPlugin`, which
registers `Body` and the nine joint-kind components with their defaults but installs no solver.
`createApp` includes that dependency automatically.

Import shared components, `ShapeKind`, `Hulls`, `Hull`, `HullFace`, `UNIT_CUBE_ID`,
`BodyState` from `@dylanebert/shallot/physics`. Import
`StandardPhysicsPlugin`, `StepPhysicsSystem`, `PhysicsWorld`, `physicsWorld`,
`readBody`, `setKinematic`, `setVelocity`, `snapshotPhysics`, `restorePhysics`
and `hashPhysics` from `@dylanebert/shallot/standard/physics`. Both subpaths
are also exported by the root barrel. Import `GlobalTransform` from the root,
not `/physics`.

`Spring` and `Joint` are replaced by `DistanceJoint`, `FilterJoint`, `MotorJoint`,
`ParallelJoint`, `PrismaticJoint`, `RevoluteJoint`, `SphericalJoint`, `WeldJoint`
and `WheelJoint`. Each has body entity references `a` and `b`. Rename `rA` and
`rB` to `localAnchorA` and `localAnchorB` (local-frame origins in meters), and
write `localRotationA` and `localRotationB` as normalized `(x, y, z, w)`
quaternions; both default to identity. Enable flags are authored as 0 or 1.

| Replaced authoring | Write instead |
| --- | --- |
| `Joint` with `stiffnessAng: 0` | `SphericalJoint` |
| `Joint` with `stiffnessAng: Infinity` | `WeldJoint`; write frame B's `localRotationB` explicitly to preserve the old relative orientation (`inverse(bodyRotationB) * bodyRotationA` when frame A's rotation is identity). |
| `Joint` with intermediate `stiffnessAng` | `WeldJoint` with `angularHertz` in Hz and an authored `angularDampingRatio`; stiffness is no longer converted. |
| `Spring` with `rest` and `stiffness` | `DistanceJoint` with `enableSpring: 1`, `length` (formerly `rest`), `hertz` in Hz and `dampingRatio` (1 for critical damping); stiffness is no longer converted. |

The constraint definitions, signatures and sync system are internal. Read body
poses with `readBody`.

`raycast`, `RayBody`, `RayHit`, `bodyCandidates`, `grabHit` and `worldToLocal`
are removed. For picking, cast a camera ray through standard physics. Coordinates
are CSS pixels relative to the camera's bound canvas; the ray length is in world units.
A hit body's user data carries the entity id:

```ts
import type { World } from "@dylanebert/shallot";
import { viewportToWorld } from "@dylanebert/shallot/rendering";
import { physicsWorld } from "@dylanebert/shallot/standard/physics";

function pick(world: World, camera: number, x: number, y: number, maxDistance = 100) {
    const ray = viewportToWorld(world, camera, x, y);
    const physics = physicsWorld(world);
    if (!ray || !physics) return null;
    const hit = physics.castRayClosest(
        { x: ray.origin[0], y: ray.origin[1], z: ray.origin[2] },
        { x: ray.dir[0] * maxDistance, y: ray.dir[1] * maxDistance, z: ray.dir[2] * maxDistance },
    );
    if (!hit.hit || !hit.shape) return null;
    return hit.shape.getBody().getUserData() as number;
}
```

`Physics.backend` is gone. Read and drive bodies through World-first functions:

```ts
// 0.9.5
Physics.backend?.setKinematic(eid, position, rotation);
const b = Tumble.body(eid);
```

```ts
// 0.10
import { readBody, setKinematic } from "@dylanebert/shallot/standard/physics";
setKinematic(world, eid, position, rotation);
const b = readBody(world, eid);
```

`Tumble.world` becomes `physicsWorld(world)`. `Tumble.body(eid)` becomes
`physicsWorld(world)!.getBody(eid)`: a live solver handle for joint creation,
or null before the authored body is marshaled. `readBody` returns pose and
velocity instead of a mutable solver body.
Read gravity through `physicsWorld(world)!.getGravity(out)` after
warm-up and use `Time.FIXED_DT` for the step duration. There is no public
substep setting.

`getSensorEvents`, `getContactEvents` and `getJointEvents` now return arrays
that the next step or the next call overwrites. Copy an array to keep it
across a step; the events and handles inside it stay usable.

```ts
const hits = [...physicsWorld(world)!.getContactEvents().hitEvents];
```

## `/avbd` is gone

The engine no longer ships its AVBD solver or `AvbdPlugin`. Select the built-in `StandardPhysicsPlugin` instead.

## `Tween`, `Sequence` and `/tween/core` are gone

There is no animation plugin in this release line. Implement animation in app code.

## `/document` and edit mode are gone

`Document`, `History`, `Session` and `ReadbackSystem` are removed, with no replacement for undo, redo or editor sessions. Remove `State.mode`, the app's `mode` option and `annotations.mode`: every system always runs.

## Scene format and scene save/restore are removed

The `.scene` format is removed with no replacement: the root's `load`, `serialize`, `diagnose`, `parse` and `stringify` and the `Node`, `Attr`, `ParseError` and `Diagnostic` types, and `/scene/core`'s `normalizeAttr`, `parseFields`, `formatFields`, `readComponent`, `setFieldValue`, `findNodeById` and `findParent`. Remove the app's and manifest's `scene` option. A game saves the component values it needs through `world.storage(Component)` and restores them itself. The world snapshot is local recovery state, not a save format.

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
