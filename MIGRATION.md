# Migrating from 0.9.5 to 0.10

These changes require updates to a 0.9.5 app.

## Loading presets

`shallotLight`, `minimalDark` and `minimalLight` are removed. For a custom loading screen, implement `Loading`, as [`examples/loading-screen`](examples/loading-screen) does. `shallotDark` remains the default preset.

## GPU requirements belong to plugins

Replace `Plugin.features` and `Plugin.preferredFeatures` with `Plugin.gpu`:

```ts
const RenderPlugin = {
    name: "Render",
    gpu: {
        features: ["rg11b10ufloat-renderable"],
        preferredFeatures: ["timestamp-query"],
        limits: { maxStorageBuffersPerShaderStage: 10 },
    },
};
```

Every plugin that touches `world.gpu` or the device declares `gpu`, even if it has no named capability beyond the device itself. Declaring `gpu` (even `{}`) opts the composition into a device. Put each feature and limit on the plugin whose GPU work needs it. A supplied `config.device` is still adopted when no plugin declares a GPU need, and it must satisfy any enabled declarations. Compositions with no `gpu` declarations and no supplied device now build and step without WebGPU; a headless app in that shape needs neither the optional `webgpu` peer nor `setupGlobals`.

`requestGPU` remains public, but its feature-array arguments are replaced by named requirements. Calls that only pass a device still work:

```ts
// 0.9.5
requestGPU(device, ["timestamp-query"], ["subgroups"]);

// 0.10
requestGPU(device, [
    {
        plugin: "MyPlugin",
        features: ["timestamp-query"],
        preferredFeatures: ["subgroups"],
    },
]);
```

## Audio clips belong to the World

Import `AudioPlugin`, `Sound`, `Listener`, `registerClip`, and `play` from `/audio` (or the root). `AudioPlugin` owns the stable data and clip registry; add the opt-in `StandardAudioPlugin` from `/standard/audio` to run playback. `loadClip` in `/standard/audio` decodes a URL or `Blob` into the World's registry. The running implementation publishes its `AudioContext` and connection bus through the World's `AudioOutput` resource.

- Replace `sample(pcm, name)` with `registerClip(world, name, pcm, sampleRate)`, or replace URL/`Blob` loading with `await loadClip(world, name, source)`. Clip IDs and decoded PCM now belong to each World.
- `play(world, name, { pos })` becomes `play(world, clip, { position })`; `clip` may be the registered name or its ID. `play` no longer resolves named graph definitions; register a clip for stable playback.
- The old graph-driven `Sound` field becomes `Sound.clip`; `Sound.pitch` and the runtime-owned `Sound.voice` are gone. `Sound.loop` remains, and `Sound.paused` is added.
- `Sound.volume` is now linear gain rather than quadratic. For former sample playback, preserve an old value `v` in `[0, 1]` by setting the new volume to `v * v`.
- Raw `Float32Array` PCM now defaults to 48 kHz; v0.9.5 used the audio context's rate. To keep that behavior, pass `world.resource(AudioOutput).context?.sampleRate ?? 48000` as `registerClip`'s fourth argument. URL/`Blob` decoding via `loadClip` retains the decoded sample rate.
- `AudioPlugin`, `Sound`, and `Listener` move from `/standard` to `/audio`; `play` moves to `/audio` with `position` in place of `pos`.
- The experimental `Instrument`, `InstrumentDef`, `Instruments`, `instrument`, `ModulationDef`, `NodeDef`, `NodeType`, `byId`, `getParamPairs`, and `MAX_INSTRUMENTS` names formerly exported by `/audio/core` are removed from Shallot; the inventory now lives in [shallot-audio-lab](https://github.com/dylanebert/shallot-audio-lab). The stable `play` API accepts registered clips, not graph definitions.
- `sfx` and `SfxPolicy` are deleted; per-name caps and cooldowns are not part of clip playback.

### Released `/audio/core` helpers

These names were also public in v0.9.5 through `/audio/core`; they are not stable clip APIs:

| v0.9.5 names | Replacement |
|---|---|
| `Audio`, `started`, `running` | Use `AudioOutput.context` and `AudioOutput.bus` for the running implementation, and `Devices.audio.context` for `running`, `suspended`, or `closed`. There is no replacement for the internal `started` flag. |
| `initAudio`, `disposeAudio`, `tickAudio` | Compose `StandardAudioPlugin`; it owns initialization, disposal, and per-frame reconciliation. Do not call lifecycle helpers directly. |
| `MAX_VOICES`, `slotOf`, `valid`, `alloc`, `free`, `gate`, `setParam`, `oneShot`, `watchIdle`, `assign` | Voice slots and handles are runtime-owned. Use `play` and author `Sound` data; there is no public voice-handle API. |
| `spatialize`, `polar`, `addSpatial`, `flushSpatial` | Position `Sound` with `Transform` and mark a `Listener`; `StandardAudioPlugin` derives and batches spatial data. |
| `InstrumentDef`, `Instrument`, `ModulationDef`, `NodeDef`, `NodeType`, `Instruments`, `instrument`, `byId`, `getParamPairs`, `MAX_INSTRUMENTS` | Experimental DAG authoring and its registry helpers are removed from Shallot and now live in [shallot-audio-lab](https://github.com/dylanebert/shallot-audio-lab). `noteFreq` has no stable replacement. |
| `Samples`, `MAX_SAMPLES`, `sample`, `getSample`, `whenLoaded` | Use the per-World `Clips` registry and `registerClip`, or await `loadClip` for a URL/`Blob`; decoded PCM and clip IDs are world-owned. Sample upload and readiness helpers are runtime-owned. |

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
in a custom frame loop. `runApp` still logs, pauses the failing system until the page
reloads, and finishes the frame.

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

## Camera depth, fog and perspective range

Perspective cameras now use infinite reverse-Z. `Camera.far` remains the finite
visibility-culling bound and clustered-light range; it no longer clips the perspective
projection or limits its depth precision. Orthographic cameras still use `far` as a
projection plane. The exported `perspective` helper likewise drops its `far` argument:
call `perspective(fov, aspect, near)`.

`DepthPrepass` remains an explicit request for stored camera depth, but it is no longer
required to enable built-in depth consumers: fog and occluded outlines request the shared
lane when needed. The lane matches the camera's antialiasing sample count; GPU consumers
that read it use sample zero. Keep `DepthPrepass` when the game itself explicitly needs the
lane, not as a switch for these plugins.

When the world has a `Fog` component, `FogPlugin` now applies it to every `StandardRenderer`
camera by default. Attach `NoFog` to an individual camera to opt that camera out; this does
not suppress a separate `DepthPrepass` request:

```ts
import { NoFog } from "@dylanebert/shallot";
world.add(camera, NoFog);
```

For perspective cameras, infinite reverse-Z makes depth zero mean no opaque surface, so the
fog pass leaves those background pixels unchanged rather than unprojecting them as a finite
endpoint. Orthographic depth zero still names its far projection plane.

## Presenting-camera capacity grows on demand

Per-view rendering state grows lazily as additional cameras become presenting; it starts with
one slot and does not compile more pipelines or layouts when it grows. The hard bound is
now **533 presenting cameras**. A 585-slot cull-volume uniform reserves 52 slots for
built-in shadow views, leaving 533 presentation slots. The 534th presenting camera is
skipped with a warning; camera creation itself is not refused, and overflow does not
overwrite the accepted cameras' slots.

`MAX_VIEWS` remains exported from `/rendering`, but its value has changed: v0.9.5
exported it from `/render/core` as 8; 0.10 exports it from `/rendering` as 533. Code
that sizes per-view data from this constant should continue to use it rather than retain
the old literal.

## Hull registries belong to worlds

Replace `Hulls.register(...)` and other registry calls with
`world.resource(Hulls).register(...)` (or the corresponding registry method).
Hull ids belong to that world; register geometry in each world that uses it.
Every world's registry starts with the unit cube at `UNIT_CUBE_ID` (0).

## Hulls hold points

A 0.9.5 `Hull` registered `verts`, `faces` and `edges`. A 0.10 `Hull` holds Box3D's hull input:
`points`, flat xyz in shape-local meters as a `Float32Array`, and an optional `maxVertexCount`
(default 128). Standard physics builds the faces and edges itself, and `HullFace` is removed.
The hull is the convex hull of the points: interior points drop, coplanar faces merge, and the
built hull keeps at most `maxVertexCount` vertices.

```ts
// 0.9.5
Hulls.register({ name: "wedge", verts, faces, edges });

// 0.10
world.resource(Hulls).register({ name: "wedge", points: new Float32Array(verts.flat()) });
```

## Meshes hold Box3D's mesh input

In 0.9.5, `createMesh(def)` from `/tumble/core` built `MeshData` for a `TumbleBody` mesh shape.
In 0.10, register the definition as a `PhysicsMesh` in `world.resource(PhysicsMeshes)` and name
its id in a `Shape` of kind `Mesh`; standard physics builds it once per World. `vertices` is flat
xyz as a `Float32Array`, with an optional `stride` in floats, `indices` is a `Uint32Array`,
`materialIndices` a `Uint8Array`, and the build options keep their names, plus `clockwiseWinding`.
`createMesh`, `MeshDef`, `MeshData` and the `create*Mesh` helpers are removed.

```ts
// 0.9.5
const mesh = createMesh({ vertices, indices, identifyEdges: true })!;
Tumble.body(eid)?.createMesh({}, mesh);

// 0.10
const geometry = world.resource(PhysicsMeshes).register({
    name: "terrain",
    vertices: new Float32Array(vertices.flatMap((v) => [v.x, v.y, v.z])),
    indices: new Uint32Array(indices),
    identifyEdges: true,
});
world.add(eid, Shape, { kind: ShapeKind.Mesh, geometry, scale: [1, 1, 1] });
```

## Surface materials come from `/physics`

Import `SurfaceMaterial` from `/physics` instead of `/tumble/core`. Its `tangentVelocity` is
`[x, y, z]` rather than `{ x, y, z }`. `ShapeMaterials` sets hold `SurfaceMaterial` lists.

```ts
// 0.9.5
const material: SurfaceMaterial = { ...base, tangentVelocity: { x: 1, y: 0, z: 0 } };

// 0.10
const material: SurfaceMaterial = { ...base, tangentVelocity: [1, 0, 0] };
```

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

`ColorGrading` applies exposure, temperature, tint and hue (in degrees) globally; `postSaturation` applies after tonemapping. The `saturation`, `contrast`, `gamma`, `gain` and `lift` vectors hold shadows, midtones and highlights as `[x, y, z]`. Defaults are identity; `midtonesRange` defaults to `[0.2, 0.7]`.

Register view-specific passes in `world.resource(EffectPasses)` under the camera eid, with `before` and `after` arrays. Each callback receives `(world, eid, view, input, output)` and records on the frame encoder. Before passes operate on linear HDR; after passes operate on encoded display-referred intermediates. With no after pass, tonemapping writes the presented target directly. Remove registrations when their camera or owner leaves.

## Rendering extension exports

These 0.9.5 exports are removed or renamed in 0.10. The headings identify their v0.9.5 package paths; the old package's `./src/*` wildcard also made the two direct-source paths below importable. Remove direct imports from internal files; they are not replacement public paths.

### `/render/core` → `/rendering`

All names below came from v0.9.5 `/render/core` except `backingSize` (`/src/standard/render/view`) and `COLOR_LANES` (`/src/standard/sear/codegen`).

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

### `/render/core` mesh exports → `/mesh`

| Removed name | Replacement |
|---|---|
| `meshBounds` | Register mesh vertices; `MeshPlugin` computes culling bounds. |
| `MeshStorage` | Infer storage from the registered `Mesh`. |
| `QuantStreams` | No public replacement; mesh quantization is internal. |
| `packMeshes` | Add with `world.resource(Meshes).add`; `MeshPlugin` packs at warm-up. |
| `quantizeMeshes` | Add with `world.resource(Meshes).add`; quantization is internal. |
| `VERTEX_FLOATS` | Author the vertex data accepted by `Meshes.add`; packing is internal. |
| `VERTEX_STRIDE` | Read the registered mesh's GPU bindings; packing is internal. |

### `/render/core`, `/sear/core` and root exports → `/standard/rendering`

In this table, `PartPlugin` was exported at the root. `ClusterView` through `zSlice`, `Binding` through `SURFACE_GROUP`, `LIGHTING_UNIFORM_SIZE` through `spotParams`, and `DrawIndirectBuffer` came from `/render/core`. `/sear/core` also exported the Surface and Background contracts, `lightEvalWgsl`, `clusterOf` through `sunVisibility`, `getCompiledSurface`, and `casterWgsl` through `pointComboCount`.

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
| `Binding` | Removed; describe material resources with `MaterialBinding` and construct `materialLayout`. |
| `FsFn` | `MaterialFragmentFn`, inferred from a `materialType` shader. |
| `SurfaceLayout` | `MaterialLayout`, inferred from `materialLayout`. |
| `VsFn` | `MaterialVertexFn`, inferred from a `materialType` shader. |
| `assertOwnFn` | Internal; registration validates shader ownership. |
| `SURFACE_GROUP` | Removed; material and background group indices are renderer-owned. |
| `clusterOf` | Use `clusterCell` for grid addressing or `lit` for surface lighting. |
| `engineScaffoldWgsl` | Use `materialLayout`, `materialType` and `materialFragmentContext()`. |
| `fragCoord` | Use the material fragment context or a core render-phase renderer. |
| `fragWorld` | Read `.world` from the context returned by `materialFragmentContext()`. |
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

### `/fog/core` → `/fog`

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
| `Single`, `Pair`, `Quad`, field `Type` | `ScalarField`, `Vector2Field`, `Vector4Field`, `FieldType` |
| `Transform.pos`, `.rot` | `translation`, `rotation` |
| `Body.pos`, `.quat` | `position`, `rotation` |
| Moving `Body` with positive `mass` | Write `type: BodyType.Dynamic`; `mass` is the dynamic body's mass. |
| `Body` with `mass: 0` | Omit `type` for static geometry, or write `type: BodyType.Kinematic` for caller-driven motion. |
| Root `Part` | `MeshInstance` |
| Root `RenderPlugin` | `RenderingPlugin` for the frame/view substrate; add `CorePipelinePlugin` for shared targets and phases (`StandardRenderingPlugin` includes it) |
| Root `SearPlugin` | `StandardRenderingPlugin` |
| Root `Sear`, `Depth`, `Backdrop` | `StandardRenderer`, `DepthPrepass`, `CameraBackground` |
| Root `Tag`, `/sear/core` `TAG_FORMAT` and `TAG_NONE`, `/render/core` `TagFn` and `View.tag` | Removed; there is no replacement picking lane. |
| `/render/core` `BgCtx`, `BgFn`, `BgLayout` | `BackgroundContext`, `BackgroundFn`, `BackgroundLayout` |
| `/render/core` GPU `View` schema and `linearToSrgb` | `/rendering` `ViewUniforms` and `linearToSrgb3` |
| `/render/core` `View.cluster` | `ViewUniforms.projection` (near, far, perspective flag, slot; unchanged byte layout) |
| Root `mesh`, `image`, `font`, `text` | `world.resource(Meshes).add`, `registerImage`, `registerFont`, `internText` |
| Root `segment`, `box`, `arrow` | `drawLine`, `drawWireBox`, `drawArrow` |
| `build`, `run`, `Config` | `createApp`, `runApp`, `AppConfig` |
| `swap`, `SwapResult` | Removed; use `bun run dev`, which reloads the page. |
| `compose`, `decompose`, `multiply`, `invert` | `composeMat4`, `decomposeMat4`, `multiplyMat4`, `invertMat4` |
| `quat`, `euler`, `rotate`, `aim` | `eulerToQuat`, `quatToEuler`, `rotateQuatByEuler`, `lookAtRotation`. These and `slerp` return tuples, `[x, y, z, w]` (or `[x, y, z]` from `quatToEuler`), not `{ x, y, z, w }`: pass a quaternion straight to `Transform.rotation` and read `q[0]` where 0.9.5 read `q.x`. |
| `composeTransform` | `composeGlobalTransform` |
| `state.stamp`, `state.timescale` | `world.ref` and `world.resolve`, `world.setTimeScale` |
| `state.swap` | Removed; use `bun run dev`, which reloads the page. |
| `/sear/core` `PrepassSystem`, `ColorSystem` | `/rendering` `PrepassSystem`, `MainPassSystem` |
| `/src/standard/render/cluster.ts` `ClusterSystem`, `LightCullSystem` | Remove direct imports; these systems are now internal to `StandardRenderingPlugin`. |
| `CharacterSweepSystem` | Order fixed velocity producers before `CharacterPlugin.systems` |
| `PlayerControlSystem` | `UpdatePlayerControlSystem` (look and camera); `DrivePlayerSystem` consumes fixed-tick input |
| Physics `StepSystem` | `StepPhysicsSystem` |
| Physics `ConstraintSystem` | Removed: author the joint-kind components; standard physics syncs them. |

The `pixelRatio` constant is removed (set `AppConfig.pixelRatio`). The `/ecs` wrappers `register`, `getExclusions`, `entries` and `clear` are removed; use `world.registry`.

## Text font load failures stop initialization

In v0.9.5, the root `TextPlugin` warned and continued when a registered font failed to load. It now rejects initialization with the load error as `cause`; a 0.9.5 game with an unreachable registered font therefore fails initialization. The default Inter font ships with the package and no longer needs network access.

## Photometric lights follow transforms

In v0.10, authored light values use photometric units and a directional light shines along its entity's transformed local `-Z`. This replaces the v0.9.5 linear multipliers and explicit `DirectionalLight.direction`. Each camera may carry `Exposure` in EV100; its default, 9.7, follows Bevy's Blender calibration. The old names below were checked in the `v0.9.5` source (`packages/shallot/src/standard/render/lighting.ts`); root `Spot` and `Shadow` fields were separate components.

| v0.9.5 component or field | 0.10 replacement |
|---|---|
| Root `AmbientLight.color`, `.intensity` | `GlobalAmbientLight` resource with `color` and `brightness` in cd/m², plus optional per-camera `AmbientLight` override on the camera entity. Both use hex sRGB color. |
| Root `DirectionalLight.color`, `.intensity`, `.direction` | `DirectionalLight.color`, `.illuminance` in lux, and a `Transform` rotation; `direction` is removed. Use a quaternion that rotates local `-Z` onto the old normalized travel direction. |
| Root `PointLight.color`, `.intensity`, `.range`, `.radius` on a point light | `PointLight` with color, `intensity` in lumens, range and radius; the light now requires `Transform`. |
| Root `PointLight.color`, `.intensity`, `.range`, `.radius` on an entity with root `Spot` | `SpotLight.color`, `intensity` in lumens, range and radius; remove `PointLight`. `SpotLight` requires `Transform`. |
| Root `Spot.inner`, `.outer` | `SpotLight.innerAngle`, `.outerAngle`, still half-angles in degrees. |
| Root `Shadow` presence | Set each light's `shadowMapsEnabled` to `1`; `0` disables its shadow request. |
| Root `Shadow.distance` on a directional light | `DirectionalLight.maximumDistance`, still world units. |
| Root `Shadow.distance` on a point or spot light | Remove it; it was ignored. Shadow coverage still uses the light's `range`. |
| Root `Shadow.depthBias`, `.normalBias` | `depthBias` and `shadowNormalBias`; normal bias remains measured in shadow texels, but `depthBias` is now a world-space receiver offset toward the light. Re-author it in world units. |
| `/src/standard/sear/shadows` `SHADOW_DEFAULTS` | Removed. Per-light shadow requests default off; Bevy's directional defaults apply, including `shadowNormalBias: 1.8` and `maximumDistance: 150`. |
| Root `Volumetric` | `VolumetricLight`, still a marker; every marked point, spot and directional light contributes fog in-scatter. |
| `Sky.sunSize`, `Sky.sunGlow` | Add `SunDisk` to each directional light whose disk should be drawn. To preserve the old disk diameter, set `angularSize` to `2 * Math.acos(1 - 0.0005 * oldSunSize)` radians; copy `sunGlow` to `glow`, the Shallot extension to Bevy's `SunDisk`. The default `angularSize` is Bevy's `EARTH` solar size. |
| `Sky.sunColor` | Removed: a disk takes its directional light's `color`. `SunDisk.intensity` controls only its visual brightness, not illumination or shadows. |
| Root `SunShadows.cascades` | `DirectionalLight.numCascades`, default `4`, clamped to `MAX_CASCADES`. |
| Root `SunShadows.overlap` | `DirectionalLight.overlapProportion`, default `0.2`. |
| Root `SunShadows.lambda` | Removed. Set `DirectionalLight.firstCascadeFarBound`, default `10` world units: the first cascade ends there and the rest are spaced exponentially to `maximumDistance`. |
| Root `SunShadows.resolution` | `world.resource(DirectionalLightShadowMap).size`, default `2048`. |
| Root `PointShadows.atlas`, `.casters`, `.hysteresis` | The same fields on `world.resource(PointShadows)`. |

`DirectionalLight`, `PointLight` and `SpotLight` use Bevy's defaults for their photometric values, range, radius, angles and per-kind shadow biases. Re-author scene values explicitly when preserving a previous look.

At EV100 9.7, `exposure = 2 ** -9.7 / 1.2`. To preserve a former linear multiplier `m`, re-author ambient brightness as `m / exposure` cd/m², directional illuminance as `Math.PI * m / exposure` lux, and point/spot luminous flux as `4 * Math.PI ** 2 * m / exposure` lumens. Keep `Exposure` at its EV100 value (the default is 9.7) or re-evaluate those authored values for another exposure. Example directional aim:

```ts
world.add(sun, Transform, { rotation: lookAtRotation(0, 0, 0, -0.4, -0.8, -0.5) });
world.add(sun, DirectionalLight, { color: 0xfff4e0, illuminance: 3449.1713 });
world.add(sun, SunDisk, { angularSize: 0.0529, intensity: 1, glow: 0.5 });
```

The public v0.9.5 `LightingGpu` schema in `packages/shallot/src/standard/render/lighting.ts` also changes shape. Its packed fields are replaced as follows:

| v0.9.5 `LightingGpu` field | 0.10 replacement |
|---|---|
| `ambientColor` | `ViewUniforms.ambientColor` carries the effective linear ambient radiance from the camera's `AmbientLight` or `GlobalAmbientLight`; camera exposure is `ViewUniforms.exposure`. The old packed alpha multiplier is gone. |
| `sunDirection` | `LightingGpu.directionalLights[i].direction`, the normalized travel direction from the light's transformed local `-Z`. `directionalCount` gives the active prefix. |
| `sunColor` | `LightingGpu.directionalLights[i].color` plus `.params.x` illuminance in lux; exposure is camera-local. The old intensity-baked sun color is gone. |

Standard packs up to ten directional lights. At most one enabled directional casts shadows: Shallot selects the enabled light with greatest illuminance, then lower entity id on a tie. It warns once while multiple directional lights request shadows; other enabled requests remain unshadowed. `shadowMapsEnabled` remains per light, so a game may change which one wins without rewriting the other lights.

A shadowed spot light is now authored as:

```ts
world.add(lamp, Transform, { translation: [0, 4, 0] });
world.add(lamp, SpotLight, {
    color: 0xffffff, intensity: 157612.96, range: 12,
    innerAngle: 18,
    outerAngle: 28,
    shadowMapsEnabled: 1,
    depthBias: 0.0005,
    shadowNormalBias: 1.8,
});
world.add(lamp, VolumetricLight);
```

`NotShadowCaster` is new in 0.10. Add it to a mesh entity to keep it visible without casting shadows. Removing it restores casting. `PointShadows` and `DirectionalLightShadowMap` belong to each World: instead of assigning to an imported object, write `world.resource(PointShadows).atlas = 1024` in `AppConfig.setup` or a plugin's `initialize`.

`Camera` remains one component: `mode`, `fov`, `near`, `far`, `size`, `clearColor` and `antialias` are unchanged, including `fov` in degrees and `antialias: 1` for 4× MSAA. `CameraMode` and `Resolution.width`/`.height` are unchanged.

Standard shadows now use Gaussian filtering by default. Add `ShadowFilteringMethod` to a camera with `method: ShadowFilteringMethodMode.Hardware2x2` to use one hardware comparison sample, or `method: ShadowFilteringMethodMode.Gaussian` for filtered shadows. A camera without the component uses Gaussian.

## Indirect light sources use one shared input

The v0.9.5 package had no indirect-light source registration API, so there is no old call to replace. In 0.10, `IndirectLightInput`, `IndirectLightSource`, `indirectLight`, `indirectLightSlot`, `indirectLightSources` and `registerIndirectLightSource` are new `/standard/rendering` exports. Register a TGSL `IndirectLightSource` during plugin initialization with `registerIndirectLightSource(world, name, source)`, before `StandardRenderingPlugin` warms its pipelines. The source receives `IndirectLightInput` (`worldPosition`, `normal`, `view`, `materialOcclusion` and effective `ambientRadiance`) and returns indirect radiance. Standard composes it after ambient light, scales each source by material occlusion, then applies material albedo and camera exposure. The same input is used by standard materials and fog; registration closes at warm.

## Component editor metadata and name lookups are removed

Remove the component traits `requires`, `singleton`, `aliases`, `parse`, `format`, `enums`, `inputs` and `annotations`. Component declarations keep `defaults` and add-only `requires`; `excludes` is removed and runtime `provides` becomes `requires`. System annotations remain.

The component metadata exports `camel`, `find`, `schema`, `schemas`, `FieldInfo`, `FieldKind`, `Schema`, `isSingleton`, `dependencies`, `provides`, `exclusions` and `kebab` are removed. The `getComponent` and `getTraits` wrappers and registry methods are removed too. Query imported component handles with `world.query([Component])`; a game resolves components and enum values by import, not by name.

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

Keep each key byte-for-byte; hot reload rebinds a reloaded record by it. A composition refuses two different records under one key, naming both plugins. `component` returns the field record unchanged, so storage and insertion calls stay the same. Undeclared records in `Plugin.components` are refused, naming the plugin and the record's fields.

`inspect`, `snapshot`, `readFields` and `dump`, and the `EntityData` and `FieldValues` types, are removed. Read component values through `world.storage(Component)`. `world.snapshot()` returns `WorldSnapshot`, opaque, world-local recovery state for `world.restore(snapshot)`, not a save format. To use recovery, plugins with fixed systems declare `recovery: "stateless"` if components hold all their simulation state, or `recovery(world)` returning `{ snapshot(), restore(state) }` for hidden state. Snapshot returns an independent, reusable image; restore runs against the restored ECS and tick clock.

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

Resolve storage once in a system's setup or a lifecycle hook, then retain it for that world. Declare the complete schema before binding it; to change a schema, replace the component object. A component's values are unreadable after `world.remove`; writing to a component the entity does not hold is unsupported and may leave values for a reused eid.

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

`f16x4`, `srgb8x4` and `FieldType.gpu` are removed; no table packs a field. Declare `vec4` and bind it as a 16-byte `d.vec4f`. Declare data that holds only x, y and z as `vec3`: it has `vec4`'s 16-byte layout, binds as `d.vec3f` or `d.vec4f`, and is written `[x, y, z]` in `world.add` and defaults, with the fourth lane 0. For a 4-byte color, store `packColor4(r, g, b, a)` from `/utils` in a `u32` field, bind it as `d.u32` and read it in the shader with `unpackLdrColor`:

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

## Removed app and field metadata

- `App.skipped` is removed, since an incomplete composition fails in `createApp`.
- `World.record(name, ms)` and `World.fenceWait(ms)` are removed; call `world.recordSink?.(name, ms)` or `world.fenceWaitSink?.(ms)`.
- `FieldType.wgsl` is removed, since tables bind TypeGPU records and a custom field type declares `ctor`, `lanes`, `name` and an optional codec.

## Authored Transform and world GlobalTransform are separate

`Transform` remains authored placement. Replace `TransformsPlugin` with `TransformPlugin` from `@dylanebert/shallot/transform` (also exported at the root). Physics and rendering include it as a dependency; add it explicitly to placement-only compositions. It derives `GlobalTransform` for each `Transform` entity; physics publishes a body's placement. Each producer uses `requires: [GlobalTransform]` to add it when missing; removing a producer leaves it attached. Component pairs are no longer refused. Physics warns once per entity carrying both `Body` and `Transform`, since both write its `GlobalTransform`. Read world placement through `world.storage(GlobalTransform)`, not `Transform`. `GlobalTransform` is derived world placement, never authored, and has no hierarchy. Physics publishes rigid pose and velocity, not collider-derived scale. Register a sized mesh with `cube([hx, hy, hz])`, `sphere(radius)` or `capsule(halfHeight, radius)` from `/mesh` when the visual should match the collider; these builders default to the built-in sizes. `Transform.translation` and `.scale` and `GlobalTransform.translation`, `.scale` and `.linearVelocity` are `vec3` fields, written `[x, y, z]`; `rotation` stays a four-value quaternion.

The renderer interpolates previous and current fixed-tick `GlobalTransform` into GPU-only `global-transform-interpolated` rows. It records history copies and interpolation in the renderer's frame submission. Without an interpolated-row reader, the composition does no GlobalTransform GPU work.

Material vertex shaders receive `MaterialVertexInput.xform` and `.world`; fragment shaders read `.world` from the context returned by `materialFragmentContext()`. The dense mesh-instance record carries eid, mesh, material type and type-local material row. Body, camera, light, text, sprite and other world-space consumers read GlobalTransform rather than Transform as world placement.

## Mesh material types own shaders and typed rows

`Surface`, `registerSurface` and `surfaceLayout` are removed. Define a material type with a parameter schema, a layout, and shader functions; register it with `MaterialPlugin`. Each registered type owns CPU-authored rows and a lazily created GPU parameter table for renderer consumption. `MeshMaterial` stores the World-local material-type id, that type's row, and a numeric `alphaMode`/`alphaCutoff` copy of the row's alpha, rewritten from the row after simulation. Preprocessing groups by `(material type, alpha pipeline key, mesh)`, so changing the type, the row, or the row's alpha mode moves the entity to the corresponding draw on the next frame. A missing `MeshMaterial` selects StandardMaterial type 0, row 0 in Opaque mode.

```ts
import {
    MaterialPlugin,
    materialFragmentContext,
    materialLayout,
    materialType,
} from "@dylanebert/shallot/standard/rendering";

const Params = d.struct({ tint: d.vec4f });
const layout = materialLayout(Params, {});
const Context = materialFragmentContext();
const Tint = materialType({
    name: "Tint",
    parameters: Params,
    layout,
    fragment: tgpu.fn([Context], d.vec4f)((ctx) => {
        "use gpu";
        return Params(layout.$.materialParameters[ctx.material]).tint;
    }),
    defaults: { tint: d.vec4f(1) },
});
const TintPlugin = MaterialPlugin(Tint);
```

Add parameter values with `world.resource(Tint).add(...)`; the returned material handle goes directly into `MeshMaterial`. Partial updates use `world.resource(Tint).update(handle, values)`. Material rows and type ids belong to their World. The renderer supports typed mesh attributes declared by `materialLayout` and validates their element schemas against the registered mesh.

Sprite carries image layer, tint, fill and billboard data in one material type's table, with rows indexed by entity id; it does not allocate a material row per tint. `SpriteBlend.Clip` selects `AlphaMode.Mask(0.5)` and retains depth/shadow participation; `SpriteBlend.Alpha` selects `AlphaMode.Blend` on the same type. The former separate `SpriteAlphaMaterialType` is removed. Sprite remains on the standard mesh path. Its `MeshInstance.cullBounds` is a conservative local-space sphere derived from the size and anchor, so camera- and shadow-view culling includes the expanded quad for every billboard mode. Text and lines own their renderer pipelines and use core's `RenderPhases.transparent` seam; their transparency is not selected by mesh-material `AlphaMode` and they do not enter the opaque depth prepass or cast shadows.

## Material alpha modes belong to each material row

`AlphaMode` is exported from the root and `/standard/rendering`; alpha behavior is no longer declared on a material type with `blend: "alpha"` or `blend: "clip"`. StandardMaterial defaults to `Opaque`, but its fragment keeps `baseColor[3]` instead of forcing alpha to 1. Choose a non-opaque mode explicitly when that value should affect compositing. Add or update a row with the alpha option:

```ts
const glass = world.resource(Materials).add(
    StandardMaterial({ baseColor: [0.3, 0.7, 1, 0.35] }),
    { alphaMode: AlphaMode.Blend },
);
world.add(window, MeshMaterial, glass);
world.resource(Materials).update(glass, { baseColor: [0.3, 0.7, 1, 0.6] }, {
    alphaMode: AlphaMode.Blend,
});
```

Custom material types use the same `{ alphaMode }` option on `add`, `update` and `setAt`; their fragment shader must return the intended alpha. `Opaque` writes depth. `Mask(cutoff)` discards fragments below its per-row cutoff and writes depth; `Blend`, `Premultiplied`, `Add` and `Multiply` use transparent color pipelines without depth writes or shadows. `Premultiplied` expects premultiplied shader RGB; `Add` premultiplies RGB and shares its pipeline key, with shader output distinguishing the additive behavior. `AlphaToCoverage` emits a fragment `sample_mask` from alpha for multisampled views because the standard `rg11b10ufloat` HDR target has no alpha channel; it falls back to a 0.5 mask on a single-sample view. Transparent draws join the shared per-view phase and sort by view-space depth across renderers; see the shared `RenderPhases` contract below. This is not OIT and does not promise correct ordering for intersecting geometry.

Do not move Text or Lines onto `AlphaMode`: they own their renderer pipelines and stay in the shared transparent phase. Sprite retains its `SpriteBlend` component API and maps Clip/Alpha to Mask/Blend on its one material type as described above.

## glTF and Skin are removed

`GltfPlugin`, `SkinPlugin` and their import, animation and live-skin helpers are no longer exported. Remove these plugins from manifests and imports; there is no replacement in this release line.

The importer scene hooks `Preloader`, `Preloads` and `preload` are removed. Load assets in your plugin's `initialize`.

The importer-only shader specialization is also removed: `Surface.specialize`, `Specialize` and `Mesh.variant` are gone. Register separate material types when shaders differ.

## GPU registries and plugin helpers use the owning World

Replace the process-level `Compute` registries with the owning `world.gpu`. Helpers needing GPU state receive the owning World explicitly.

These helpers use the owning World:

- `Profile` data is read with `world.resource(Profile)`.
- `cascadeComboEids()` and `pointComboEids()` take World first; `cascadeCount()` and `pointComboCount()` are removed.

## Profile no longer exposes benchmark windows

`ProfilePlugin` still provides its live profiler overlay and `Profile` resource, but no longer installs a measurement-window global or tracks benchmark-only counters.

| v0.9.5 export or field (path) | 0.10 replacement |
|---|---|
| `window.__benchmark` | Removed; read live data through `world.resource(Profile)`. |
| Root and `/extras` `BenchmarkAPI` | Removed; there is no measurement-window API. |
| Root and `/extras` `BenchmarkCompileStats` | Removed; inspect `Profile.compile` and `Profile.compileMs` for live compile data. |
| Root and `/extras` `BenchmarkCpuStats` | Removed; inspect `Profile.cpu` for current-frame timings. |
| Root and `/extras` `BenchmarkFrameStats` | Removed; frame-window summaries are no longer provided. |
| Root and `/extras` `BenchmarkGpuStats` | Removed; inspect `Profile.gpu` for displayed pass timings. |
| Root and `/extras` `BenchmarkMeasurement` | Removed; there is no measurement-window API. |
| Root and `/extras` `BenchmarkMemoryStats` | Removed; inspect `Profile.bufferBytes` and `Profile.textureBytes` for live totals. |
| `Profile.indirectCount` | Removed; indirect-draw windows are no longer tracked. |
| `Profile.indirectFires` | Removed; indirect-draw windows are no longer tracked. |
| `Profile.compiledPipelines` | Removed; pipeline-only counts are no longer tracked. |
| `Profile.pipelineCalls` | Removed; pipeline-constructor counts are no longer tracked. |
| `Profile.submitCount` | Removed; submit counts are no longer tracked. |
| `Profile.allocBytes` | Removed; use the overlay's per-allocation memory details. |
| `Profile.lazyBytes` | Removed; lazy allocations are included in `Profile.bufferBytes` and `Profile.textureBytes`. |
| Root and `/runtime` `LazyAlloc` | Removed; remove `lazy` metadata used only for profiler accounting. |

## Character movement and player feel

Import `Character`, `CharacterPlugin` and `GroundState` from `@dylanebert/shallot/standard/physics` (also re-exported from the root). `/character` and `/character/core` are removed. `Character` requires a capsule `Body` with `type: BodyType.Kinematic`; its pogo spring floats the lower sphere centre three radii above ground. Retune spawn and camera heights for that float.

- Replace `move` with a fixed-tick write to `world.storage(Character).velocity`, ordered before `CharacterPlugin.systems`. Standard physics resolves that velocity, without gravity or acceleration.
- An external producer can add and write `PlayerInput` with held `move` axes, `sprint`, absolute `yaw`/`pitch`, and a cumulative `jumpPresses` count. If absent, `DrivePlayerSystem` adds it seeded from `Player.yaw` and `Player.pitch`; every fixed tick copies the record's look angles to `Player`. It does not sample devices.
- The former keyboard-driven `Player` also needs `LocalPlayer`; its setup producer ensures `PlayerInput` and samples this World's devices before the fixed loop. Move mouse-look `sensitivity` from `Player` to `LocalPlayer.sensitivity`; local view angles start from `Player.yaw` and `Player.pitch`.
- Replace `jump` with your input policy's velocity write. The local producer counts Space presses; `Player` buffers each new count once, and jump tuning belongs to `Player.jumpSpeed`.
- Replace `globalTransform` with `world.storage(GlobalTransform).translation`; test membership with `world.has(eid, GlobalTransform)` when placement may not yet exist.
- Replace `grounded` with `world.storage(Character).groundState.get(eid) === GroundState.OnGround`; steep ground is a separate state.
- Replace the 0.9.5 character helper `teleport(eid, x, y, z)` with `setTransform(world, eid, position, rotation)` from `/standard/physics`, then clear `Character.velocity` and `Character.pogoVelocity` for a stationary respawn. To move an entity without a body, write its `Transform` in a fixed system and call `resetInterpolation(world, eid)` from `/transform`, so rendering does not blend from the old placement.
- `PlayerPlugin` no longer installs `RenderingPlugin`: it authors the linked camera's pose without presentation. Keep rendering in the app's composition when presenting that camera.
- Move gravity and jump tuning to `Player.gravity` (positive downward acceleration) and `Player.jumpSpeed`. Player owns acceleration, friction, sprint, coyote time (0.1 seconds), jump buffering (0.1 seconds) and platform carry. Its defaults follow Minecraft Java's speeds and jump: speed 4.317 m/s, sprint multiplier 1.3, acceleration 20/s, friction 16/s, jump speed 8.85 m/s and gravity 29.5 m/s². `Player.hover` (0.05 m) sets `Character.pogoRest`, whose default of 3 radii is Box3D's mover sample, and Player turns on `Character.steepWalls`, off by default as in that sample.

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

## `shallot recipe` is now `shallot create`

```sh
# 0.9.5
bunx shallot recipe first-person

# 0.10, from the outer folder without an engine install
bun create shallot my-game --template first-person

# 0.10, inside a project where Shallot is installed
bunx shallot create my-game --template first-person
```

## `shallot run` and `shallot verify` are removed

Replace `shallot run` with the project's `bun run build` followed by `bun run preview`; preview never rebuilds. For a desktop app, use `shallot build --target <platform>` followed by `shallot preview --target <platform>`. Replace `shallot verify` with the project's own tests.

## Package imports no longer use `/core` suffixes

Rendering split into `/rendering` for shared capabilities and `/standard/rendering` for the default mesh renderer:

```ts
// 0.9.5
import { FrameGpu } from "@dylanebert/shallot/render/core";
import { engineLayout, Surface, surfaceLayout, registerSurface } from "@dylanebert/shallot/sear/core";
import { Xform } from "@dylanebert/shallot/utils/core";
```

```ts
// 0.10
import { FrameGpu } from "@dylanebert/shallot/rendering";
import { engineLayout, materialLayout, materialType, MaterialPlugin } from "@dylanebert/shallot/standard/rendering";
import { Xform } from "@dylanebert/shallot/utils";
```

`MeshInstance` stores `mesh` and an optional local-space culling sphere; ordinary meshes need only set `mesh`, while Sprite maintains the sphere for its billboard quad. Add StandardMaterial values with `const handle = world.resource(Materials).add(StandardMaterial(values))`, then add `MeshMaterial` with that `{ type, material }` handle. Retain the handle to share or update a material; there is no material name or name lookup. Meshes without `MeshMaterial` draw with StandardMaterial type 0, row 0. `VertexMaterialType`, exported at the root and from `/standard/rendering`, is the built-in per-vertex-lit alternative and accepts the same `StandardMaterial` parameter schema.

| v0.9.5 export or field (path) | 0.10 replacement |
|---|---|
| Root `Part` | Root or `/mesh` `MeshInstance` |
| Root `Part.surface` | A `MeshMaterial` handle selecting a registered material type and its row |
| `/render/core` `Draw.surface` | `Draw.materialType`, the World-local material-type id |
| Root `Color.rgba` | Material's linear `baseColor` |
| Root `Material.params` (lanes `metallic`, `roughness`, `emissive`, `occlusion`; `emissive` is a strength) | `StandardMaterial({ metallic, perceptualRoughness, emissive: [baseColor[0] * emissive, baseColor[1] * emissive, baseColor[2] * emissive], occlusion })` |
| Root `Material` component | Root or `/standard/rendering` `MeshMaterial` referencing an added material handle |
| Root `PartPlugin` | Root or `/standard/rendering` `MeshRenderPlugin` |
| `/part/core` `Parts` | Removed; mesh-instance packing is internal to `MeshRenderPlugin`. |
| `/render/core` `Draws` names `part:<surface>:<mesh>`, profiler span `part:pack` | `mesh:material:<type>:<alpha key>:<mesh>:<handle>`, `mesh:preprocess` |

`StandardMaterial()` defaults to white base colour, metallic 0, perceptual roughness 0.5, black emissive, occlusion 1, `diffuseWrap` 1 and lit shading. Set `unlit: true` to use the base colour without lighting. Add the same parameter record to `world.resource(VertexMaterialType)` to evaluate the standard lighting model per vertex and interpolate its result. Set `baseColor: [1, 0, 1, 1]` and `perceptualRoughness: 1` to express the former bare mesh values. `diffuseWrap` blends Lambert (0) with Shallot's squared half-Lambert (1); its default preserves the diffuse look. Publish changed fields with `world.resource(Materials).update(handle, values)`; omitted fields retain their values. Change both `MeshMaterial.type` and `.material` when switching to another type; changing either field takes effect on the next frame, with the new row's alpha mode. Material handles belong to the World that added them.

Custom material shaders read linear `color` and the type-local `material` row from the context returned by `materialFragmentContext()`. `MeshInstanceInput` is `{ mesh: u32, materialType: u32, material: u32, alphaMode: u32, alphaCutoff: f32, flags: u32, cullBounds: vec4f }`; `flags` bit 0 excludes the mesh from shadow views, and a negative `cullBounds.w` uses the registered mesh sphere instead of the per-instance `(center.xyz, radius)` override. Parameter values live in one typed table per material type. Use `StandardMaterial.diffuseWrap: 1` to retain the former diffuse lobe.

Mesh data has its own `/mesh` module. Add a mesh with `world.resource(Meshes).add`, as materials are added with `world.resource(Materials).add`; it returns a world-local `MeshHandle`. Retain the handle for `MeshInstance` and `Draw.mesh` instead of looking it up by name. Mesh names are display labels, not identities, so equal labels add distinct meshes:

```ts
const mesh = world.resource(Meshes).add({ name: "block", ...cube() });
world.add(eid, MeshInstance, { mesh });
```

Replace `Meshes.id(name)` with the retained handle from `Meshes.add`; names no longer resolve mesh identity. `Meshes.register(mesh)` becomes `Meshes.add(mesh)`: a GPU-resident `Mesh` record is added as-is, while typed arrays are staged and packed. `MeshPlugin` adds only the default cube. To use another built-in geometry constructor, add its mesh explicitly:

```ts
import { Meshes, sphere } from "@dylanebert/shallot/mesh";

const sphereMesh = world.resource(Meshes).add({ name: "sphere", ...sphere() });
```

Update mesh imports as follows; these names are also exported from the root in 0.10:

| 0.9.5 import | 0.10 import |
|---|---|
| Root `mesh` | Root or `/mesh` `Meshes`, then `.add` |
| — | Root or `/mesh` `MeshHandle` (returned by `Meshes.add`) |
| `/render/core` `Mesh`, `MeshBinding`, `MeshIndex` | `/mesh`, same names |
| `/render/core` `Meshes` | `/mesh`, same name |

Mesh packing and layout helpers not listed here are removed; see the rendering extension export table above.

`/render/core` `Mesh.dynamic`, `Mesh.count` and `Mesh.cast` are removed; they had no effect. Delete them, and add `NotShadowCaster` to an entity that should cast no shadow.

Material, background and draw contracts belong to `/standard/rendering`. Update imports as follows; backgrounds and draws remain on that path:

| 0.9.5 import | 0.10 import |
| --- | --- |
| `/render/core` `surfaceLayout`, `Surface`, `Surfaces`, `registerSurface` | Removed; use `materialLayout`, `materialType` and `MaterialPlugin` from `/standard/rendering` |
| `/render/core` `VsIn`, `vsPatchSchema`, `fsCtxSchema` | Removed; use `MaterialVertexInput`, `materialVertexOutput` and `materialFragmentContext()` |
| `/render/core` `TagFn` | Removed; there is no replacement picking lane. |
| `/render/core` `BgCtx` | `/standard/rendering` `BackgroundContext` |
| `/render/core` `BgLayout`, `BgFn` | Removed; infer from `backgroundLayout` and `Background`. |
| `/render/core` `backgroundLayout`, `Background`, `Backgrounds`, `registerBackground` | `/standard/rendering`, same names |
| `/render/core` `Draw`, `DrawIndexedIndirect`, `Draws` | `/standard/rendering`, same names |
| `/render/core` `computeViewProj`, `Frame`, `FrameGpu`, `BeginFrameSystem`, `OverlaySystem`, `CULL_FRUSTUM`, `CULL_VOLUME_FLOATS`, `Views`, `MAX_VIEWS`, `MAX_SLOTS`, `attachCanvas`, `attachView`, `detachCanvas`, `sceneTransform`, `imageArray` | `/rendering`, same names, except `detachCanvas` is `detachView`; `computeViewProj`, `attachCanvas`, `attachView` and `detachView` take World first |
| `/render/core` `Clusters`, `clusterCell`, `LightCull` | `/standard/rendering`, same names |
| `/render/core` `Lighting`, `LightingGpu`, `PointLightGpu`, `distanceAttenuation`, `spotFactor` | `/standard/rendering`, same names |

`sceneTransform` returns one result object per camera, rewritten by the next call for that camera; destructure or copy `read` and `write` before calling it again.

Other implementation helpers in these contracts are removed; see the rendering extension export table above. `MeshInstanceInput` from `/standard/rendering` is new in 0.10 and describes a dense mesh-component row. `StandardMaterial`, `Materials` and `MeshPlugin` are also new exports, replacing the former component-only material values and mesh registration owned by `RenderPlugin`/`PartPlugin`.

Camera prepass markers and attachment constants are imported from `/rendering`:

| 0.9.5 import | 0.10 import |
| --- | --- |
| Root or `/sear/core` `Depth` | `/rendering` `DepthPrepass` |
| `/sear/core` `DEPTH_FORMAT` | `/rendering` `DEPTH_FORMAT` |
| Root or `/sear/core` `Tag`, `/sear/core` `TAG_FORMAT`, `TAG_NONE` | Removed; there is no replacement picking lane. |

`CorePipelinePlugin` from `/rendering` registers `DepthPrepass` and owns view targets, clear, resolve and prepass/opaque/transparent phases. `StandardRenderingPlugin` includes it as a dependency. Custom renderers using these phases depend on `CorePipelinePlugin` and register `PhaseRenderer` records in `RenderPhases`; callbacks record into the shared pass and do not end it. `RenderingPlugin` alone supplies views and frame/presentation anchors without the shared pipeline. `DepthPrepass` requests the camera's stored depth output.

In 0.10, transparent producers contribute one item per independently sortable draw through `transparent(world, eid, view, phase)`, using `phase.add(draw, batch, centerX, centerY, centerZ, stableId, stablePart?)` with a world-space bounds center. Core sorts ascending by view-space Z for each view, then by renderer namespace, stable id and stable part; provide a unique identity per renderer-owned draw, separate from `draw`, which remains the renderer's lookup index. Standard meshes and text use their entity id; each retained line segment uses its line entity id plus its segment index. `batch` identifies compatible state. `prepareTransparent(world, eid, view, phase, renderer)` may stage view-specific data before the pass. `drawTransparent(world, eid, view, phase, start, end, pass, renderer)` issues only the consecutive compatible range supplied by core; it must not move or regroup those items. The `opaque` callback still records into the main pass. The 0.9.5 package did not ship this `RenderPhases` seam, so there is no 0.9.5 transparent callback signature to preserve.

Custom mesh material types register with `MaterialPlugin` and use `MeshRenderPlugin` / `StandardRenderingPlugin` for standard draws. Backgrounds remain on `StandardRenderingPlugin`. Text and lines depend on `CorePipelinePlugin` and contribute to `RenderPhases.transparent`; `RenderingPlugin` alone does not provide that shared pass.

`RenderingPlugin` still registers the light components (`AmbientLight`, `DirectionalLight`, `PointLight`, `SpotLight`, `VolumetricLight`), but no longer packs GPU lights or builds clusters. Compositions using those GPU resources need `StandardRenderingPlugin`.

Custom mesh producers depend on `MeshPlugin` from `/mesh`; `RenderingPlugin` alone no longer initializes mesh storage. `StandardRenderingPlugin` and `MeshRenderPlugin` include this dependency. `MeshPlugin` registers the built-in cube, sphere and capsule.

### `/bvh/core`

Removed. Shallot no longer provides a GPU BVH builder or traversal helpers; there is no replacement subpath. Keep or vendor the acceleration structure in the package that consumes it.

Likewise `/ecs/core` is `/ecs`, `/physics/core` and `/tumble/core` are `/physics`, and `/character/core` is removed in favor of `/standard/physics`. `/scene/core` is removed with the scene format. The `/src/*` wildcard is gone: use the paths in `package.json` `exports`.

## `Inputs` is now `world.resource(Devices)`

The owning App's keys, pointer and touch replace the process-level `Inputs` facade; `Inputs.mouse` is `pointer`. `setInputEnabled` takes World, and `inputEnabled()` is `world.resource(Devices).enabled`. Canvas CSS size and device-pixel ratio live in the engine's `world.resource(Viewports)`, keyed by canvas index, not in `Devices`. Import `Viewports`, `Viewport` and `resizeViewport` from the root or `/engine`; `resizeViewport(world, index, width, height, dpr)` writes a row.

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

Fixed systems do not read `Devices`. Sample devices in a `setup`-group system into a component record on each controlled actor, counting press edges as `PlayerInput.jumpPresses` does. In `fixed`, consume a new count once and keep windows such as the former `isKeyPressedWithin` in actor component state advanced by fixed ticks, as Player's jump buffer does:

```ts
const presses = input.jumpPresses.get(eid);
if (presses !== state.lastPresses.get(eid)) {
    state.lastPresses.set(eid, presses);
    state.bufferTicks.set(eid, 12);
}
```

Advance `bufferTicks` once each fixed tick.

## Physics is authored and read through the ECS world

Replace the 0.9.5 `TumblePlugin`/`AvbdPlugin` backend setup with `StandardPhysicsPlugin` from
`@dylanebert/shallot/standard/physics`. It depends on core's `PhysicsPlugin`, which registers the
shared physics components and installs no solver. `Physics.backend`, `PhysicsBackend`,
`installBackend` and `uninstallBackend` are removed; `StepSystem` is now `StepPhysicsSystem`,
and `ConstraintSystem` and `ComposeSystem` are internal to the plugin.

In 0.9.5, `Tumble.world` exposed a `TumbleWorld` and `Tumble.body(eid)` returned a `TumbleBody`
handle after the body was marshaled. Those handles, `Shape` and `Joint` handles, and `BodyState`
are removed. The ECS `World` is the only physics world: author components and call
functions that take the `World` first. Import `Body`, `Shape`, `ShapeKind`, `BodyType`, `Hulls`, `Hull` and
`UNIT_CUBE_ID` from `/physics`; import `StandardPhysicsPlugin`, `StepPhysicsSystem`, `hashPhysics`,
`PhysicsWorldDefinition`, `PhysicsEvents`, the state-write and query functions, and `drawPhysics`
from `/standard/physics`. Import `GlobalTransform` from the root or `/transform`.

### Bodies and shapes

The 0.9.5 `Body.pos` and `Body.quat` become spawn-only `position` and `rotation`; scene-authored
Euler degrees in `quat` must become an `(x, y, z, w)` quaternion. The new `type` field
replaces the old mass-based motion choice. 0.9.5's default `mass` was 1,
while 0.10 `Body` defaults to static, so author `type` explicitly. The other new Box3D body
fields are `linearVelocity`, `angularVelocity`, `linearDamping`, `angularDamping`, `gravityScale`,
`sleepThreshold`, `motionLocks`, `enableSleep`, `isAwake`, `isBullet`, `isEnabled`,
`allowFastRotation` and `enableContactRecycling`. Initial velocities and `isAwake` are spawn-only;
fields with Box3D setters apply at the next fixed sync.

Move the 0.9.5 `Body.shape`, `halfExtents`, `mass` and `friction` fields to `Shape`: `shape` becomes
`Shape.kind`, `halfExtents.xyz` becomes `Shape.scale` for hulls, `mass` becomes per-shape density,
and `friction` becomes `Shape.friction`. The old `.w` radius becomes `Shape.radius`, shared by spheres and
capsules; a sphere's center is `Shape.sphereCenter` and a capsule's hemisphere centers are
`Shape.capsuleCenter1` and `Shape.capsuleCenter2`; a hull id moves to `Shape.geometry`. `ShapeKind.Box` becomes a scaled
`ShapeKind.Hull`. Box3D computes mass from shape density and geometry, so do not copy total mass
straight into density.

```ts
// 0.9.5
const eid = state.create();
state.add(eid, Body);
Body.shape.set(eid, ShapeKind.Box);
Body.pos.set(eid, 0, 5, 0, 0);
Body.halfExtents.set(eid, 0.5, 0.5, 0.5, 0);
Body.mass.set(eid, 1);
Body.friction.set(eid, 0.5);
```

```ts
// 0.10
const eid = world.create();
world.add(eid, Body, { type: BodyType.Dynamic, position: [0, 5, 0] });
world.add(eid, Shape, {
    kind: ShapeKind.Hull, geometry: UNIT_CUBE_ID, scale: [0.5, 0.5, 0.5],
    density: 1, friction: 0.5,
});
```

`Body.position`, `.linearVelocity` and `.angularVelocity`, `Shape.sphereCenter`, `.capsuleCenter1`, `.capsuleCenter2`, `.scale` and `.tangentVelocity`, the joints' `localAnchorA`/`localAnchorB` and velocities, and `Character`'s `velocity`, `groundNormal`, `groundVelocity` and `up` are `vec3` fields, written `[x, y, z]`. Rotations stay four-value quaternions.

0.9.5 `TumbleBody` could create several shapes directly; author each as a `Shape` entity instead:

```ts
// 0.9.5
const body = Tumble.body(eid);
body?.createSphere({}, { center: { x: 1, y: 0, z: 0 }, radius: 0.25 });
```

```ts
// 0.10
const collider = world.create();
world.add(collider, Shape, { body: eid, kind: ShapeKind.Sphere, sphere: [1, 0, 0, 0.25] });
```

The 0.9.5 `Spring` and `Joint` components become `DistanceJoint`, `FilterJoint`, `MotorJoint`,
`ParallelJoint`, `PrismaticJoint`, `RevoluteJoint`, `SphericalJoint`, `WeldJoint` and `WheelJoint`
components. Their body references are `a` and `b`; rename `rA`/`rB` to `localAnchorA`/`localAnchorB`
and author normalized `localRotationA`/`localRotationB` quaternions. For joints created through
`Tumble.world`, add the corresponding component to a joint entity instead of retaining its handle:

```ts
// 0.9.5
Tumble.world?.createRevoluteJoint(Tumble.body(a)!, Tumble.body(b)!, config);
```

```ts
// 0.10
world.add(jointEid, RevoluteJoint, { a, b });
```

| 0.9.5 authoring | 0.10 component |
| --- | --- |
| `Joint` with `stiffnessAng: 0` | `SphericalJoint` |
| `Joint` with `stiffnessAng: Infinity` | `WeldJoint`; set `localRotationB` explicitly to preserve the old relative orientation (`inverse(bodyRotationB) * bodyRotationA` when frame A's rotation is identity). |
| `Joint` with intermediate `stiffnessAng` | `WeldJoint` with `angularHertz` and `angularDampingRatio`; stiffness is no longer converted. |
| `Spring` with `rest` and `stiffness` | `DistanceJoint` with `enableSpring: 1`, `length`, `hertz` and `dampingRatio`; stiffness is no longer converted. |

### State writes and reads

The 0.9.5 backend's `setKinematic` becomes `setTargetTransform` for motion and `setTransform` for
teleporting. Box3D derives target velocities, so the old optional velocity override is removed.
`setVelocity` becomes `setLinearVelocity`; the other body state writes are functions
over the `World`, each mapped to one Box3D call (`setAngularVelocity`, `setAwake`, force, torque and impulse functions):

```ts
// 0.9.5
Physics.backend?.setKinematic(eid, position, rotation);
Physics.backend?.setKinematic(eid, respawnPosition, rotation, true);
Physics.backend?.setVelocity(eid, vx, vy, vz);
```

```ts
// 0.10
setTargetTransform(world, eid, position, rotation);
setTransform(world, eid, respawnPosition, rotation);
setLinearVelocity(world, eid, { x: vx, y: vy, z: vz });
```

The 0.9.5 backend's `readBody(eid)` returned `pos`, `quat` and `vel`; read them from
`GlobalTransform.translation`, `rotation` and `linearVelocity` columns. `Tumble.world.getBodyEvents()`
is consumed internally to publish pose:

```ts
// 0.9.5
const pose = Physics.backend?.readBody(eid);
const x = pose?.pos[0];
```

```ts
// 0.10
const global = world.storage(GlobalTransform);
const x = global.translation.column[eid * 4];
```

### World settings, events and callbacks

The 0.9.5 `TumblePlugin` used gravity `(0, -10, 0)` and four substeps. `Tumble.world.setGravity`
becomes a write to `PhysicsWorldDefinition`; substeps are now authored there too. Changes apply
at the next fixed sync:

```ts
// 0.9.5
Tumble.world?.setGravity({ x: 0, y: -9.8, z: 0 });
const gravity = Tumble.world?.getGravity();
```

```ts
// 0.10
const definition = world.resource(PhysicsWorldDefinition);
definition.gravity = { x: 0, y: -9.8, z: 0 };
definition.subStepCount = 6;
const gravity = definition.gravity;
```

Replace `Tumble.world.getSensorEvents()`, `getContactEvents()` and `getJointEvents()` with
`world.resource(PhysicsEvents)`. Its `sensorBegin`, `sensorEnd`, `contactBegin`, `contactEnd`,
`contactHit` and `joints` entries hold records and a count for the last fixed step. Where 0.9.5 events carried shape or joint handles,
these records carry generation-aware `EntityRef`s; resolve them with `world.resolve(ref)`:

```ts
// 0.9.5
const begins = Tumble.world?.getContactEvents().beginEvents;
```

```ts
// 0.10
const begins = world.resource(PhysicsEvents).contactBegin;
for (let i = 0; i < begins.count; i++) world.resolve(begins.records[i]!.shapeA);
```

Custom-filter and pre-solve callbacks are new in 0.10; they receive the owning `World` and
shape `EntityRef`s, with no 0.9.5 callback to replace. Move friction and restitution mix callbacks
to `PhysicsWorldDefinition`; their old `(a, idA, b, idB)` arguments stay, with `World` added first.

### Queries and debug drawing

Replace `Tumble.world` methods with functions that take the `World` first. `castRay`, `castRayClosest`, `castShape`,
`castShapeClosest`, `overlapAABB` and `overlapShape` report ECS shape/body ids; closest-hit functions
write to caller-owned output. `draw` becomes `drawPhysics`:

```ts
// 0.9.5
Tumble.world?.castRay(origin, translation, (hit) => hit.fraction);
Tumble.world?.draw(draw);
```

```ts
// 0.10
castRay(world, origin, translation, (hit) => hit.fraction);
drawPhysics(world, draw);
```

The 0.9.5 backend-neutral `raycast`, `RayBody`, `RayHit`, `bodyCandidates`, `grabHit` and
`worldToLocal` helpers are removed. For camera picking, use `viewportToWorld` from `/rendering`, then pass its ray to `castRay` or
`castRayClosest` (with caller-owned output); coordinates are CSS pixels relative to the camera's
bound canvas, and ray length is in world units.

## `/avbd` is gone

The engine no longer ships its AVBD solver or `AvbdPlugin`. Select the built-in `StandardPhysicsPlugin` instead.

## `Tween`, `Sequence` and `/tween/core` are gone

There is no animation plugin in this release line. Implement animation in app code.

## `/document` and edit mode are gone

`Document`, `History`, `Session` and `ReadbackSystem` are removed, with no replacement for undo, redo or editor sessions. Remove `State.mode`, the app's `mode` option and `annotations.mode`: every system always runs.

## Scene format and scene save/restore are removed

The `.scene` format is removed with no replacement: the root's `load`, `serialize`, `diagnose`, `parse` and `stringify` and the `Node`, `Attr`, `ParseError` and `Diagnostic` types, and `/scene/core`'s `normalizeAttr`, `parseFields`, `formatFields`, `readComponent`, `setFieldValue`, `findNodeById` and `findParent`. Remove the app's and manifest's `scene` option. A game saves the component values it needs through `world.storage(Component)` and restores them itself. The world snapshot is local recovery state, not a save format.

`Identity`, `world.identity`, `refs()` and the `derived` component trait are removed. Keep the eids returned by `world.create()` instead of naming entities. The `entity` field type still stores a plain eid; save and restore references yourself. `GlobalTransform` remains attached through its producers' `requires`, without the trait.

Author worlds in code:

```ts
import { Transform } from "@dylanebert/shallot";
const eid = world.create();
world.add(eid, Transform, { translation: [0, 1, 0] });
```

## `/harness` helpers are removed

Remove imports of `installHarness`, `HarnessTarget` and `REAL_GPU_LAUNCH`. Drive `createApp()` and `world.step()` in the project's tests and observe through public ECS and physics reads. Configure the project's browser tests with Playwright Test, its own Vite `webServer` and its own Chromium launch flags.

## `./brand` is removed

Remove imports from `@dylanebert/shallot/brand`. The mark and palette are repository assets, not a package subpath; the default loading screen continues to render the mark internally. There is no public brand-assets replacement.

## `OrbitOverlayPlugin` is removed

Remove `OrbitOverlayPlugin` from plugin imports and registrations. `OrbitPlugin` remains; there is no replacement overlay plugin.

## Vite configuration and project scripts are now the project's

0.9.5's CLI generated a Vite configuration for manifest projects. Create a `vite.config.ts` with Shallot's plugin. For an ejected 0.9.5 config, replace its separate TypeGPU plugin (or `typegpuPlugin()`) with this one. Remove the old `CROSS_ORIGIN_ISOLATION` import; the plugin sets those headers:

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

`shallot dev`, `shallot build` and `shallot preview` no longer run web projects. For web, add `dev`, `build` and `preview` scripts that run `vite`, `vite build` and `vite preview`, then invoke them with `bun run dev`, `bun run build` and `bun run preview`. For desktop, pass `--target mac`, `--target windows` or `--target linux`; native dev and build add the desktop shell, and native preview launches the existing build.

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
