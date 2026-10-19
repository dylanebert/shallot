# Migrating from 0.9.5 to 0.10

Start with the compiler errors in your app. Search this guide for the old symbol or command; each entry goes directly from the published 0.9.5 package to 0.10. For the engine's layers and extension rules, see [CONTRIBUTING.md](CONTRIBUTING.md).

Code fragments assume an owning `world`, an existing entity `eid`, and the imports named in the surrounding text. `state` denotes the old `State`. Values such as URLs and geometry are app data, not engine exports.

## `shallot recipe`, `shallot run` and `shallot verify`

`recipe` becomes `create`. Web development, builds and preview now use the project's Vite scripts. Preview does not build. Replace `verify` with your project's tests.

```sh
# 0.9.5
bunx shallot recipe first-person
bunx shallot run
bunx shallot verify
```

```sh
# 0.10
bun create shallot my-game --template first-person
cd my-game
bun run dev
bun run build
bun run preview
bun test
```

Inside an installed project, `bunx shallot create my-game --template first-person` also works. Native commands require `--target mac`, `--target windows` or `--target linux`: use `shallot build --target <platform>` followed by `shallot preview --target <platform>`.

## `virtual:project`, `shallot.json` and Vite configuration

Import app plugins in the entry page, not through `shallot.json` or `virtual:project`. Remove `shallot.schema.json`, manifest-only default exports and the manifest's `scene` option. The native `identifier` override is removed; bundles use `com.shallot.<name>`.

```ts
// 0.9.5
import project from "virtual:project";
import { run } from "@dylanebert/shallot";
await run(project);
```

```ts
// 0.10
import { OrbitPlugin, runApp } from "@dylanebert/shallot";
await runApp({ plugins: [OrbitPlugin], pixelRatio: "auto" });
```

For a manifest-only 0.9.5 app with no entry page, run `bunx shallot create migration-template --template minimal`. Copy `migration-template/index.html` and `migration-template/vite.config.ts` to the app root, and copy `migration-template/public/startup.js` to `public/startup.js`. If the old `tsconfig.json` extends a repository-relative configuration, replace it with `migration-template/tsconfig.json`, which extends the installed engine's configuration. In `index.html`, replace the template's `Welcome` import and plugin with your app's named plugin imports. Keep the canvas and loading bootstrap. Scene files are no longer loaded: move their entities into plugin initialization as described below.

Add these scripts to the app's `package.json` and install Vite as a dev dependency (`bun add -d vite`):

```json
{
    "scripts": {
        "dev": "vite --host 127.0.0.1",
        "build": "vite build",
        "preview": "vite preview --host 127.0.0.1"
    }
}
```

Default plugins remain enabled unless `defaults: false` is set. `build`, `run` and `Config` are now `createApp`, `runApp` and `AppConfig`; the app's `state` is now `world`. Replace the exported `pixelRatio` constant with `AppConfig.pixelRatio`. Remove `App.skipped`: incomplete compositions fail initialization. `swap`, `SwapResult` and `state.swap` are removed; use Vite's development reload.

Create a `vite.config.ts`. In an ejected config, replace the separate TypeGPU plugin and remove `CROSS_ORIGIN_ISOLATION`; `shallot()` supplies both TypeGPU compilation and the headers.

```ts
// 0.9.5
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

Update the TypeGPU peer and scaffold types:

```sh
# 0.10
bun add typegpu@~0.12.8
bun add -d @types/node@^26.0.0
```

```json
{ "compilerOptions": { "types": ["@webgpu/types", "node", "vite/client"] } }
```

## `State`, `sparse`, `slab` and component traits

`State` becomes `World`. Fields declare types; storage belongs to each World. Replace `Single`, `Pair`, `Quad` and field `Type` with `ScalarField`, `Vector2Field`, `Vector4Field` and `FieldType`.

Declare components with an exact key using `component`, then list them in `Plugin.components`. `Plugin.traits` and `Traits` are removed. Move defaults and runtime `provides` to the declaration's `defaults` and add-only `requires`. Defaults use vector arrays, not dotted lanes.

```ts
// 0.9.5
const Health = { value: sparse(f32) };
const GamePlugin = {
    name: "Game",
    components: { Health },
    traits: { Health: { defaults: () => ({ value: 100 }) } },
};
Health.value.set(eid, 80);
```

```ts
// 0.10
import { component, f32, type Plugin } from "@dylanebert/shallot";
const Health = component("Health", { value: f32 }, { defaults: () => ({ value: 100 }) });
const GamePlugin: Plugin = { name: "Game", components: [Health] };
const health = world.storage(Health);
health.value.set(eid, 80);
```

Resolve storage in setup or initialization and retain it only for that World. A component's values are valid only while the entity holds it. Declare the full schema before binding it; replace the record to change its schema. Preserve exact keys across hot updates; two different component records under one key are refused.

Remove `capacity` from app and World options and imports: storage grows on demand. Scalar fields now use their declared numeric type: `u8` wraps 300 to 44, `u32` wraps negative values, and `f32` rounds to 32-bit precision; scalar `sparse` previously retained JS numbers.

Remove editor traits `singleton`, `aliases`, `parse`, `format`, `enums`, `inputs`, `annotations`, `excludes` and the old editor meaning of `requires`. Remove `System.annotations`. These metadata helpers have no replacement:

- `camel`, `kebab`, `find`, `schema`, `schemas`, `FieldInfo`, `FieldKind`, `Schema`, `isSingleton`, `dependencies`, `provides`, `exclusions`, `getComponent`, `getTraits`
- `Alias`, `laneAlias`, `eulerAlias`, `formatHex`, `Input`, `Unit`, `units`, `angle`, `degrees`, `radians`
- `bodyTraits`, `springTraits`, `jointTraits`, `PartTraits`, `ColorTraits`

Resolve components by import. Replace `/ecs/core` wrappers `register`, `getExclusions`, `entries` and `clear` with `world.registry` operations where needed.

## `readFields`, `inspect`, `snapshot` and entity identity

`readFields(Component, eid)` becomes `world.get(eid, Component)`, returning vectors as arrays and `undefined` for absent components. `inspect`, `dump`, `EntityData` and `FieldValues` are removed. `world.snapshot()` is opaque, world-local recovery state, not the old inspection or scene-save format.

```ts
// 0.9.5
const values = readFields(Transform, eid);
```

```ts
// 0.10
const values = world.get(eid, Transform);
const image = world.snapshot();
world.restore(image);
```

Plugins with fixed systems must declare `recovery: "stateless"` when components hold all simulation inputs, or `recovery(world)` returning `{ snapshot(), restore(state) }` for hidden state. The returned image must be independent and reusable.

`state.stamp` becomes `world.ref`; resolve a reference with `world.resolve`. Replace `state.timescale` writes with `world.setTimeScale`. `Identity`, `world.identity`, `refs()` and the `derived` trait are removed. Keep returned eids instead of naming entities; the `entity` field still stores a plain eid, not a generation-aware reference.

## `Slab`, field `.gpu` and `Compute`

`Slab`, `SlabPlugin`, `SlabSystem`, `slab()` and per-field `.gpu` buffers are removed. Replace entity-indexed scalar buffers with typed record tables and dense row indices. `state.membership` and the named `"membership"` GPU buffer are removed; dispatch over active table rows.

```ts
// 0.9.5
const Heat = { value: slab(f32, "heat") };
const gpuValues = Heat.value.gpu;
```

```ts
// 0.10
import * as d from "typegpu/data";
const Heat = component("Heat", { value: f32 });
const table = world.table("heat", d.struct({ value: d.f32 }));
table.bindComponent(Heat, { value: "value" });
const gpuRows = table.buffer;
```

Use `table.rowIndex(eid)` on the CPU; enable its eid lookup if a shader starts from an eid. Rebind after buffer-generation changes. Built-in body, transform and light fields no longer expose per-field GPU buffers.

`f16x4`, `srgb8x4` and field `Type.gpu`/`Type.wgsl` metadata are removed. Use `vec4` bound as `d.vec4f`; for packed color, use a `u32` field with `packColor4` and shader `unpackLdrColor`. Custom field types declare `ctor`, `lanes`, `name` and an optional codec.

`Compute` becomes `world.gpu`. Its named `buffers`, `textures`, `samplers` and `typed` registries are gone. Keep shared state in a World resource. Custom material and background resources come from `bindings(world)`, keyed by the layout's binding names.

## `Plugin.features`, `preferredFeatures` and `requestGPU`

Move GPU requirements onto `Plugin.gpu`. Every plugin using the GPU declares it, even if it needs only `gpu: {}`. Without declarations or a supplied device, an app can now build and step without WebGPU.

```ts
// 0.9.5
const RenderPlugin = { name: "Render", features: ["indirect-first-instance"] };
requestGPU(device, ["timestamp-query"], ["subgroups"]);
```

```ts
// 0.10
import { requestGPU, type Plugin } from "@dylanebert/shallot";
const RenderPlugin: Plugin = {
    name: "Render",
    gpu: { features: ["indirect-first-instance"], limits: { maxStorageBuffersPerShaderStage: 10 } },
};
await requestGPU(device, [{
    plugin: "MyPlugin", features: ["timestamp-query"], preferredFeatures: ["subgroups"],
}]);
```

0.9.5 requested adapter maxima; 0.10 requests default limits unless a plugin declares more. Declare required limits on their owning plugin. An externally acquired `config.device` still works, but must satisfy every enabled declaration. Headless GPU apps install the optional `webgpu` peer and call `setupGlobals` from `/webgpu` before building.

## `State.step` errors and fixed `time.elapsed`

Inside `fixed`, `world.time.elapsed` is now `fixedTick * Time.FIXED_DT`, not frame elapsed time. Keep simulation on this clock; move frame effects to `simulation` or `draw`.

Outside `runApp`, `world.step()` now throws on a system error, stops later systems and retries the failing system on the next step. Catch errors in custom loops. `runApp` still logs and pauses a failing system until reload or re-registration.

```ts
// 0.9.5
state.step(); // system errors were logged
```

```ts
// 0.10
try {
    world.step();
} catch (error) {
    console.error(error);
}
```

## `Transform.pos`, `Transform.rot` and quaternion helpers

Rename `pos` to `translation`, `rot` to `rotation`, and `TransformsPlugin` to `TransformPlugin` (`/transform`). Placement readers use derived `GlobalTransform`, not authored `Transform`. Physics and rendering include the plugin; placement-only apps add it explicitly.

```ts
// 0.9.5
Transform.pos.set(eid, 1, 2, 3, 0);
Transform.rot.set(eid, ...Object.values(quat(0, 90, 0)));
```

```ts
// 0.10
world.storage(Transform).translation.set(eid, 1, 2, 3);
world.storage(Transform).rotation.set(eid, ...eulerToQuat(0, Math.PI / 2, 0));
const placement = world.storage(GlobalTransform);
```

`quat`, `euler`, `rotate` and `aim` become `eulerToQuat`, `quatToEuler`, `rotateQuatByEuler` and `lookAtRotation`. Quaternion results (including `slerp`) are tuples, not `{ x, y, z, w }`; Euler values now use radians, not degrees. Multiply old angles by `Math.PI / 180`.

`compose`, `decompose`, `multiply`, `invert` and `composeTransform` become `composeMat4`, `decomposeMat4`, `multiplyMat4`, `invertMat4` and `composeGlobalTransform`.

Translation, scale and linear velocity use `vec3`: three-value arrays in `world.add`, three arguments in storage `.set`, and four-lane raw-column stride. Quaternion rotation remains four values. Rendering interpolates fixed-tick placement; write gameplay motion in fixed systems.

## `Inputs` and `setInputEnabled`

Replace process-level `Inputs` with `world.resource(Devices)`; `mouse` becomes `pointer`. `inputEnabled()` becomes `Devices.enabled`. Viewport dimensions live in `Viewports`, not input state.

```ts
// 0.9.5
const held = Inputs.isKeyDown("KeyW");
const pressed = Inputs.isKeyPressed("Space");
const width = Inputs.mouse.canvasWidth;
setInputEnabled(false);
```

```ts
// 0.10
const input = world.resource(Devices);
const held = input.keys.held.has("KeyW");
const pressed = input.keys.pressed.has("Space");
const width = world.resource(Viewports).get(input.focused)?.cssWidth ?? 0;
setInputEnabled(world, false);
```

Fixed systems do not read devices. Sample in the `setup` group into actor components, count press edges, and consume each new count once in `fixed`. Replace `isKeyPressedWithin` with a tick-counted window in actor state. `pointer.buttonEvents` preserves ordered presses and releases with their positions; use it when both can happen between frames.

## `cursorRay`, `forwardRay`, `generateRay` and `screenToRay`

Import `Ray` from the root, not physics. Replace camera-ray helpers with `/rendering` `viewportToWorld`, using CSS pixels relative to the camera's bound canvas (or pixels of its attached texture). For a forward ray, use the viewport centre.

```ts
// 0.9.5
const ray = cursorRay(state, camera);
```

```ts
// 0.10
const pointer = world.resource(Devices).pointer;
const ray = pointer.hover ? viewportToWorld(world, camera, pointer.x, pointer.y) : null;
```

The function reads `GlobalTransform` and returns null without placement or a non-empty target. A perspective ray starts on the transformed near plane, not at the eye; account for that offset in hit distances.

## `RenderPlugin`, `SearPlugin` and `/core` imports

`RenderPlugin` becomes `RenderingPlugin` for views, cameras, lights and frame anchors. `SearPlugin` becomes `StandardRenderingPlugin`, including `CorePipelinePlugin` for shared targets and phases and `MeshPlugin` for mesh storage. Custom renderers sharing the main pass depend on `CorePipelinePlugin`, not just `RenderingPlugin`.

```ts
// 0.9.5
import { RenderPlugin, SearPlugin } from "@dylanebert/shallot";
import { FrameGpu } from "@dylanebert/shallot/render/core";
```

```ts
// 0.10
import { RenderingPlugin, StandardRenderingPlugin } from "@dylanebert/shallot";
import { FrameGpu } from "@dylanebert/shallot/rendering";
```

Use public subpaths, not `/src/*`; the wildcard is removed.

| 0.9.5 import | 0.10 import |
|---|---|
| `/ecs/core` | `/ecs` |
| `/utils/core` | `/utils` (except placement and mesh codecs below) |
| `/render/core` view, frame, camera and lighting components | `/rendering` |
| `/render/core` mesh data, `Meshes`, `MeshBinding`, `MeshIndex` | `/mesh` |
| `/sear/core`, `/render/core` material, draw, background and light-cluster contracts | `/standard/rendering` |
| `/physics/core` | `/physics` for data, `/standard/physics` for simulation |
| `/tumble/core`, `/character/core` | `/standard/physics` functions; see physics entries |
| `/fog/core` | `/extras` for fog authoring and shader reads |
| `/utils/core` `Xform`, `xformQuat`, `xformPoint`, `xformNormal`, `xformMat`, `xformWgsl` | `/rendering` |
| `/utils/core` `MeshQuant`, `meshIdOf`, `decodePos`, `decodeUv`, `encodePos`, `encodeUv`, `posQuantWgsl`, `posQuantPackWgsl` | `/mesh` |

`Sear`, `Depth` and `Backdrop` become `StandardRenderer`, `DepthPrepass` and `CameraBackground`. `PrepassSystem` moves to `/rendering`; `ColorSystem` becomes `MainPassSystem`. Order effects before `OverlaySystem` and presentation after `PresentationSystem`.

`computeViewProj`, `attachCanvas`, `attachView` and `detachView` now take World first (`detachCanvas` is renamed `detachView`). `sceneTransform` returns a reused per-camera result; copy its `read` and `write` before another call for that camera. `MAX_VIEWS` moves from `/render/core` to `/rendering` and changes from 8 to 533; use the constant rather than the old literal.

## `Part`, `Color`, `Material` and mesh registration

`Part` becomes `MeshInstance`; `PartPlugin` becomes `MeshRenderPlugin`. Mesh and material handles belong to a World. Names are labels, not lookup keys; keep the returned handles instead of calling `Meshes.id(name)`.

```ts
// 0.9.5
const meshId = mesh({ name: "block", ...cube() });
state.add(eid, Part);
Part.mesh.set(eid, meshId);
Color.rgba.set(eid, 1, 0, 1, 1);
```

```ts
// 0.10
const mesh = world.resource(Meshes).add({ name: "block", ...cube() });
const material = world.resource(Materials).add(StandardMaterial({
    baseColor: [1, 0, 1, 1], perceptualRoughness: 1,
}));
world.add(eid, MeshInstance, { mesh });
world.add(eid, MeshMaterial, material);
```

Replace `Part.surface` and the `Material` component with a `MeshMaterial` handle. Convert `Material.params` as follows: `metallic` is unchanged, `roughness` becomes `perceptualRoughness`, `occlusion` is unchanged, and emissive strength `e` becomes RGB `baseColor.rgb * e`. `Color.rgba` becomes linear `baseColor`.

`StandardMaterial` defaults to white, metallic 0, roughness 0.5, black emissive and occlusion 1. Use `unlit: true` for unlit color. Meshes without a `MeshMaterial` use the default material. Update rows with `world.resource(Materials).update(handle, values)`; omitted fields retain their values.

`Meshes.register` becomes `world.resource(Meshes).add`. Removing or clearing meshes invalidates retained handles: instances draw nothing until assigned a new handle. `MeshPlugin` adds only the default cube; explicitly add other primitives such as `{ name: "sphere", ...sphere() }`. Remove unused `Mesh.dynamic`, `count` and `cast`; use `NotShadowCaster` on an entity to suppress casting.

`sphere` and `capsule` now reject noninteger or too-small tessellation: segments must be >= 3 and rings >= 2. Capsule rounds odd ring counts down to even. Do not depend on the old degenerate pole-triangle index counts.

## `Surface`, `surfaceLayout` and `registerSurface`

Replace surfaces with material types registered by `MaterialPlugin` from `/standard/rendering`. Parameter rows replace per-entity color and material buffers. Shader functions are plain `"use gpu"` functions with inferred arguments, not prebuilt TypeGPU functions.

```ts
// 0.9.5
const layout = surfaceLayout({});
registerSurface(state, { name: "tint", layout, fs });
```

```ts
// 0.10
import * as d from "typegpu/data";
import { MaterialPlugin, materialType } from "@dylanebert/shallot/standard/rendering";
const Params = d.struct({ tint: d.vec4f });
const Tint = materialType({
    name: "Tint",
    parameters: Params,
    fragment: (ctx) => {
        "use gpu";
        return d.vec4f(ctx.parameters.tint);
    },
    defaults: { tint: d.vec4f(1) },
});
const TintPlugin = MaterialPlugin(Tint);
const tint = world.resource(Tint).add({ tint: d.vec4f(1, 0, 0, 1) });
world.add(eid, MeshMaterial, tint);
```

Use `materialLayout(Params, bindings)` instead of `parameters` when the type owns bindings. Supply those resources through `bindings(world)`. Both shader arguments expose `.parameters`; vertex shaders also expose `.xform` and `.world`, fragments `.world`. Declare interpolated fields with `varyings`. `Surface.specialize`, `Specialize` and `Mesh.variant` are removed: register separate types for different shaders.

For standard shading, use `StandardMaterial` or `pbrInputFromMaterial` with `applyPbrLighting`. `extendedMaterial(Materials, { name, fragment })` shares standard's parameter schema and bindings but owns its rows; it cannot add parameters or bindings.

## Surface `blend` and material transparency

Move surface `blend: "alpha"` or `blend: "clip"` to each material row's alpha option. Standard material alpha is no longer forced to 1; select a non-opaque mode to composite it.

```ts
// 0.9.5
const surface = { blend: "alpha" };
```

```ts
// 0.10
const glass = world.resource(Materials).add(
    StandardMaterial({ baseColor: [0.3, 0.7, 1, 0.35] }),
    { alphaMode: AlphaMode.Blend },
);
world.add(eid, MeshMaterial, glass);
```

Use `AlphaMode.Mask(cutoff)` for clipping. `Opaque` and `Mask` write depth; `Blend`, `Premultiplied`, `Add` and `Multiply` do not write depth or cast shadows. `AlphaToCoverage` requires MSAA or falls back to a 0.5 mask. Custom fragments return the intended alpha. Transparent draws sort back-to-front across renderers, not correctly through intersecting geometry.

## `Backdrop.name` and background registration

`Backdrop` becomes `CameraBackground`; `name` becomes a retained World-local background handle. `Backgrounds.id(name)` is removed. Import background contracts from `/standard/rendering`; `BgCtx` becomes `BackgroundContext`; infer `BackgroundFn` and `BackgroundLayout` from the definition instead of importing `BgFn` or `BgLayout`.

```ts
// 0.9.5
Backdrop.name.set(camera, Backgrounds.id("sky"));
```

```ts
// 0.10
world.add(camera, CameraBackground, { background: skyBackgroundHandle(world) });
```

`registerBackground` now takes the owning World and returns a handle. Equal names create different backgrounds. Supply owned resources through `bindings(world)` instead of publishing them in `Compute`.

## `Glaze`, `Tonemap` and `/glaze`

`CorePipelinePlugin` replaces Glaze presentation. Use camera `Tonemapping`, `ColorGrading` and optional `VignettePlugin`/`Vignette`.

```ts
// 0.9.5
state.add(camera, Glaze);
Glaze.tonemap.set(camera, Tonemap.Neutral);
Glaze.exposure.set(camera, 2);
```

```ts
// 0.10
world.add(camera, Tonemapping, { method: TonemappingMethod.KhronosPbrNeutral });
world.add(camera, ColorGrading, { exposure: Math.log2(2) });
```

| Old Glaze setting | Replacement |
|---|---|
| `tonemap` | `Tonemapping.method`; map names, not numeric indices |
| `Neutral`, `Aces`, `Agx`, `SomewhatBoring` | `KhronosPbrNeutral`, `AcesFitted`, `AgX`, `SomewhatBoringDisplayTransform` |
| `Reinhard`, `ReinhardLuminance`, `None` | Same names on `TonemappingMethod` |
| `exposure` multiplier | `ColorGrading.exposure` in stops: `Math.log2(multiplier)` |
| `saturation` | `ColorGrading.postSaturation` |
| `slope`, `offset`, `power` | No exact per-channel CDL equivalent; retune sectional `gain`, `lift`, `gamma` and global `temperature`/`tint` |
| `vignette`, `vignetteInner`, `vignetteOuter` | `Vignette.intensity`, retuned `radius`/`smoothness`; now before tonemapping |
| `posterize`, `dither` | No posterization replacement; presentation dithers by default |

Cameras now default to `TonyMcMapface`. `TonemappingMethod.None` skips grading, exposure and dither, but still display-encodes linear color; `Linear` keeps grading, exposure and dither with an identity curve. Add camera `DebandDither` with `{ enabled: 0 }` for undithered output.

For custom postprocessing, register per-camera passes in `EffectPasses`. Before passes operate on linear HDR; after passes on encoded display-referred intermediates. Remove registrations when their owner leaves. Remove `GlazePlugin`, `GlazeSystem` and `tonemapWgsl` imports.

## `DirectionalLight.direction`, `Spot`, `Shadow` and light intensity

Lights now use photometric units and transformed placement. Directional light travels along local `-Z`; replace explicit `direction` with rotation. Point and spot lights require `Transform`. Camera `Exposure` defaults to EV100 9.7.

```ts
// 0.9.5
DirectionalLight.direction.set(sun, -0.4, -0.8, -0.5, 0);
DirectionalLight.intensity.set(sun, 1);
state.add(sun, Shadow);
```

```ts
// 0.10
const exposure = 2 ** -9.7 / 1.2;
world.add(sun, Transform, { rotation: lookAtRotation(0, 0, 0, -0.4, -0.8, -0.5) });
world.add(sun, DirectionalLight, {
    illuminance: Math.PI / exposure, shadowMapsEnabled: 1,
});
```

To preserve an old multiplier `m`, at EV100 `ev` use exposure `e = 2 ** -ev / 1.2`: ambient brightness `m / e` cd/m², directional illuminance `Math.PI * m / e` lux, point/spot intensity `4 * Math.PI ** 2 * m / e` lumens. Re-author these explicitly; new defaults differ.

| 0.9.5 | 0.10 |
|---|---|
| Singleton `AmbientLight` | `world.resource(GlobalAmbientLight)`; optional camera `AmbientLight` override; `intensity` becomes `brightness` |
| `PointLight` + `Spot` | `SpotLight`; `inner`/`outer` become `innerAngle`/`outerAngle` in radians |
| `Shadow` presence | Per-light `shadowMapsEnabled: 1` (default off) |
| Directional `Shadow.distance` | `DirectionalLight.maximumDistance` |
| Point/spot `Shadow.distance` | Remove; it was ignored; coverage uses range |
| `Shadow.depthBias`, `normalBias` | Per-light `depthBias` in world units (retune), `shadowNormalBias` in texels |
| `Volumetric` | `VolumetricLight` |
| `SunShadows.cascades`, `overlap` | `DirectionalLight.numCascades`, `overlapProportion` |
| `SunShadows.lambda` | Retune `firstCascadeFarBound`; later cascades are exponentially spaced |
| `SunShadows.resolution` | `world.resource(DirectionalLightShadowMap).size` |
| Global `PointShadows` fields | Same fields on `world.resource(PointShadows)` |

At most one directional casts shadows: highest illuminance wins, then lower eid. Shadows now default to Gaussian filtering; add camera `ShadowFilteringMethod` with `method: ShadowFilteringMethodMode.Hardware2x2` for one comparison sample. Remove `SHADOW_DEFAULTS` imports.

Custom shaders: `LightingGpu.ambientColor` moves to `ViewUniforms.ambientColor`; exposure is camera-local. `sunDirection` and `sunColor` become `directionalLights[i].direction`, `.color` and `.params.x` illuminance, for `directionalCount` entries.

## `Sky.sunSize`, `sunGlow` and `sunColor`

Sky color is now exposed radiance. Set `Sky.brightness: 1.2 * 2 ** ev100` to preserve its former appearance at that camera exposure (the default preserves EV100 9.7).

```ts
// 0.9.5
Sky.sunSize.set(sky, 1);
Sky.sunGlow.set(sky, 0.5);
```

```ts
// 0.10
world.add(sky, Sky, { brightness: 1.2 * 2 ** 9.7 });
world.add(sun, SunDisk, { angularSize: 2 * Math.acos(1 - 0.0005), glow: 0.5 });
```

Add `SunDisk` to each directional light whose disk should appear. `angularSize` replaces `sunSize` using `2 * Math.acos(1 - 0.0005 * oldSunSize)`. `sunColor` is removed: disks use the light's color. `SunDisk.intensity` changes the disk, not illumination.

## `Camera.far`, `perspective` and `Fog`

Perspective depth now uses infinite reverse-Z: `Camera.far` still bounds visibility culling and clustered lights but does not clip the projection. Orthographic `far` remains a projection plane. Drop `perspective`'s far argument.

```ts
// 0.9.5
const projection = perspective(fov, aspect, near, far);
```

```ts
// 0.10
const projection = perspective(fov, aspect, near);
```

A world `Fog` now affects every standard-renderer camera; attach `NoFog` to opt a camera out. Fog and occluded outlines request shared depth automatically: remove `Depth`/`DepthPrepass` if it was only enabling those effects. Keep `DepthPrepass` for your own depth consumers.

`Fog.absorption` and `scattering` are nonnegative medium coefficients, not an absorbed fraction and brightness gain. For old absorption `a` in [0,1], scattering `s` and scatter intensity `i`, keep density, use absorption `a`, scattering `1 - a`, and `lightIntensity: s * i / Math.PI`. Old defaults are `a = 0`, `s = 1`, `i = 1`. Values outside that range need retuning. Rename `anisotropy` to `scatteringAsymmetry`.

```ts
// 0.9.5
state.add(eid, Fog);
Fog.absorption.set(eid, 0.2);
Fog.scattering.set(eid, 1);
Fog.scatterIntensity.set(eid, 2);
Fog.anisotropy.set(eid, 0.3);
Fog.steps.set(eid, 32);
Fog.jitter.set(eid, 1);
```

```ts
// 0.10
world.add(eid, Fog, {
    absorption: 0.2, scattering: 0.8, lightIntensity: 2 / Math.PI,
    scatteringAsymmetry: 0.3, brightness: 1.2 * 2 ** 9.7,
});
```

Remove `steps`, `jitter` and `scatterIntensity`: fog uses a frustum grid. `viewDistance` sets its far depth (default 64 world units); analytic height fog continues beyond it. `maxDistance` is a world-unit ray endpoint (default 5000); nonpositive values disable fog. Backgrounds now receive fog to that endpoint.

`Fog.color` is exposed ambient light in the medium, at `brightness` cd/m²; use `1.2 * 2 ** ev100` for the former fade color at another exposure. Absorption-only fog no longer shows color. Ambient scene in-scatter is now 4π brighter; retune scene lighting if necessary.

Fog composites after opaque surfaces inside the main pass, before transparency. Effects cannot run between opaque shading and fog. Standard transparent materials take fog at their own depth; custom transparent renderers, text and gizmos are unfogged unless they explicitly read fog. Fog authoring and shader reads are exported from `/extras`, not a `/fog` subpath:

```ts
// 0.10
import { Fog, FogPlugin, readFog, fogReadLayout, fogReadBindGroup } from "@dylanebert/shallot/extras";
```

## `Sprite.anchor`, `Text.anchor`, visibility and sprite blending

Move pivots to `Anchor` from `/rendering`. For old pivot `p` in 0..1, use `offset: [p[0] - 0.5, p[1] - 0.5]`. Without `Anchor`, sprites remain centred and text bottom-left anchored; adding it without values centres either.

```ts
// 0.9.5
Sprite.anchor.set(eid, 0.5, 0);
Sprite.visible.set(eid, 0);
```

```ts
// 0.10
world.add(eid, Anchor, { offset: Anchor.BOTTOM_CENTER });
const values = world.get(eid, Sprite);
world.remove(eid, Sprite);
// To show it again, restore its authored values:
if (values) world.add(eid, Sprite, values);
```

`Sprite.visible`, `Text.visible` and `Line.visible` are removed. Remove the drawing component to hide it; save values before removal. For retained lines remove `Gizmo` instead.

`SpriteBillboard` becomes shared `Billboard` from `/rendering`; enum values are unchanged. Text defaults to `Billboard.World`. Rename sprite `fill` to `fillAmount` and `fillMode` to `fillMethod`. Replace `blend: SpriteBlend.Clip` with `alphaMode: SpriteAlphaMode.Mask(0.5)`, and `SpriteBlend.Alpha` with `SpriteAlphaMode.Blend`. Mask keeps depth and shadows; Blend casts no shadow.

## `font`, `image` and `text` registration

Replace `font(url, name)`, `image(source, name)` and `text(content)` with World-local `registerFont`, `registerImage` and `internText`. Remove registration aliases; keep returned ids. Repeated URLs share an id, while Blob registrations get distinct ids. `Text.font` 0 always selects the bundled default font.

```ts
// 0.9.5
const face = font("/font.ttf", "labels");
const picture = image("/image.png", "icon");
const label = text("Hello");
```

```ts
// 0.10
const face = registerFont(world, "/font.ttf");
const picture = registerImage(world, "/image.png", {
    sampler: { mipmapFilter: ImageFilterMode.Nearest },
});
const label = internText(world, "Hello");
```

Image sampling defaults to clamp-to-edge and linear texel and mip filtering. Mesh image samplers formerly used nearest-mip filtering; use `mipmapFilter: ImageFilterMode.Nearest` to keep that behavior. Sprites already used linear mip filtering. Sampler settings participate in URL identity: different effective samplers return distinct ids.

`TextPlugin` now fails initialization if a registered font cannot load, instead of warning and continuing. The bundled default no longer requires a network fetch.

## `Line`, `Arrow`, `segment`, `box` and `arrow`

Replace retained `Line` and `Arrow` with `GizmoAsset` and `Gizmo`. Build local-space geometry, register once, and share the handle among transformed entities. `Line.thickness` becomes pixel `Gizmo.width` (default 2); `Line.offset` becomes the line's local endpoint.

```ts
// 0.9.5
state.add(eid, Line);
Line.offset.set(eid, 2, 0, 0, 0);
Line.thickness.set(eid, 3);
state.add(eid, Arrow);
```

```ts
// 0.10
const asset = new GizmoAsset();
drawArrow(asset, [0, 0, 0], [2, 0, 0], 0xffffff, undefined, 0.4, [0, 1, 0]);
const handle = registerGizmo(world, asset);
world.add(eid, Gizmo, { handle, width: 3 });
```

`Arrow.size` and the old relative arrow size become world-unit `tipLength` (sixth argument). To preserve head length, use `0.2 * shaftLength * oldSize`. Omission or negative values choose a tenth of shaft length; zero omits the head. Reverse endpoints for `Arrow.start`; use a plain line for no heads. Heads are now one filled triangle. The final normal defaults to world up and must be nonzero and perpendicular to the shaft; a vertical arrow needs e.g. `[0, 0, 1]`.

Immediate `segment`, `box` and `arrow` become `drawLine`, `drawWireBox` and `drawArrow`, taking World first. Replace numeric width with an options record, e.g. `{ width: 2 }`. For retained builders pass `undefined` in that slot. Immediate options, `Gizmo` and `Text` accept `depthBias`: 0 is normal testing, -1 draws on top.

## `Outline.color`

Convert the linear RGBA array to an sRGB `0xRRGGBB` number. Alpha was unused.

```ts
// 0.9.5
Outline.color.set(eid, 1, 0.85, 0.2, 1);
```

```ts
// 0.10
world.storage(Outline).color.set(eid, 0xffed7c);
```

## `Hulls.register` and physics geometry

Geometry registries belong to each World. Replace name lookup with retained handles from `add`. Names are labels; duplicate names keep distinct definitions. To change geometry, add a new definition and assign its handle rather than mutating a registered definition.

`Hull` now holds flat `Float32Array` points, not prebuilt `verts`, `faces` and `edges`. The solver builds the convex hull; `HullFace` is removed. `maxVertexCount` defaults to 128.

```ts
// 0.9.5
const hull = Hulls.register({ name: "wedge", verts, faces, edges });
```

```ts
// 0.10
const hull = world.resource(Hulls).add({ name: "wedge", points: new Float32Array(verts.flat()) });
world.add(eid, Shape, { kind: ShapeKind.Hull, geometry: hull });
```

`Hulls`, `PhysicsMeshes`, `HeightFields`, `Compounds` and `ShapeMaterials` use `add`, `get`, `has` and `remove`. Handles belong to that World and are invalidated by removal or clear. `Shape.materialSet` takes a `ShapeMaterials` handle (-1 for none). Each World starts with the unit cube at `UNIT_CUBE_ID` (0). `Shape.scale` defaults to `[1, 1, 1]`.

## `createMesh`, `createHeightField` and `createCompound`

Replace `/tumble/core` built geometry with definitions in the World's registries. Assign their handles to ECS shapes; standard physics builds the geometry.

```ts
// 0.9.5
const mesh = createMesh({ vertices, indices, identifyEdges: true })!;
Tumble.body(eid)?.createMesh({}, mesh);
```

```ts
// 0.10
const geometry = world.resource(PhysicsMeshes).add({
    name: "terrain", vertices: new Float32Array(vertices.flatMap((v) => [v.x, v.y, v.z])),
    indices: new Uint32Array(indices), identifyEdges: true,
});
world.add(eid, Shape, { kind: ShapeKind.Mesh, geometry });
```

Mesh definitions retain build-option names, with optional `clockwiseWinding`. Use `Uint8Array` material indices and a float stride. `MeshDef`, `MeshData` and `create*Mesh` helpers are removed.

```ts
// 0.9.5
const field = createHeightField({ heights, scale: { x: 1, y: 1, z: 1 }, countX: 8, countZ: 8,
    globalMinimumHeight: -10, globalMaximumHeight: 10, materialIndices: null, clockwiseWinding: false });
Tumble.body(eid)?.createHeightField({}, field);
```

```ts
// 0.10
const geometry = world.resource(HeightFields).add({
    name: "terrain", heights: new Float32Array(heights), scale: [1, 1, 1], countX: 8, countZ: 8,
    globalMinimumHeight: -10, globalMaximumHeight: 10,
});
world.add(eid, Shape, { kind: ShapeKind.HeightField, geometry });
```

`createGrid`, `createWave`, `HeightFieldDef`, `HeightFieldData` and `HEIGHT_FIELD_HOLE` are removed; material index 255 still marks a hole.

```ts
// 0.9.5
const compound = createCompound({ hulls: [{ hull: slab, transform: { p: { x: 1, y: 2, z: 0 }, q } }] })!;
Tumble.body(eid)?.createCompound({}, compound);
```

```ts
// 0.10
const geometry = world.resource(Compounds).add({
    name: "stairs",
    hulls: [{ hull: slab, transform: { position: [1, 2, 0], rotation: [0, 0, 0, 1] } }],
});
world.add(eid, Shape, { kind: ShapeKind.Compound, geometry });
```

Compound hull and mesh children use registry handles. Sphere and capsule centers are tuples; radius is separate. Child `material` is optional; mesh `materials` replaces the old list/count pair. `CompoundDef`, the `Compound*Def` types and `CompoundData` are removed.

`SurfaceMaterial` moves from `/tumble/core` to `/physics`; change `tangentVelocity: { x, y, z }` to `[x, y, z]`. `ShapeMaterials` holds lists of these records.

## `TumblePlugin`, `AvbdPlugin`, `Body` and `ShapeKind.Box`

Use `StandardPhysicsPlugin` from `/standard/physics`. Core `PhysicsPlugin` owns data but installs no solver. Remove `Physics.backend`, `PhysicsBackend`, `installBackend` and `uninstallBackend`. The ECS World replaces `Tumble.world`, `TumbleBody`, body/shape/joint handles and `BodyState`.

Move `Body.pos`/`quat` to `Transform.translation`/`rotation`. `Body` now defaults to static, not mass 1: set `type: BodyType.Dynamic` explicitly for moving bodies. Use `BodyType.Kinematic` for caller-driven motion.

```ts
// 0.9.5
state.add(eid, Body);
Body.shape.set(eid, ShapeKind.Box);
Body.pos.set(eid, 0, 5, 0, 0);
Body.halfExtents.set(eid, 0.5, 0.5, 0.5, 0);
Body.mass.set(eid, 1);
Body.friction.set(eid, 0.5);
```

```ts
// 0.10
world.add(eid, Transform, { translation: [0, 5, 0] });
world.add(eid, Body, { type: BodyType.Dynamic });
world.add(eid, Shape, {
    kind: ShapeKind.Hull, geometry: UNIT_CUBE_ID, scale: [1, 1, 1], density: 1, friction: 0.5,
});
```

Move `shape`, `halfExtents`, `friction` and hull id into `Shape.kind`, `scale` (double half extents), `friction` and `geometry`. Box becomes a scaled unit-cube Hull. The old half-extents `.w` radius becomes `radius`; sphere and capsule centers are `sphereCenter`, `capsuleCenter1` and `capsuleCenter2`. Density replaces total mass authoring: compute density from the desired mass and geometry, do not copy arbitrary mass values.

For multiple shapes, add separate shape entities referencing the body eid:

```ts
// 0.9.5
Tumble.body(eid)?.createSphere({}, { center: { x: 1, y: 0, z: 0 }, radius: 0.25 });
```

```ts
// 0.10
const collider = world.create();
world.add(collider, Shape, { body: eid, kind: ShapeKind.Sphere, sphereCenter: [1, 0, 0], radius: 0.25 });
```

Physics publishes pose and linear velocity to `GlobalTransform`. Writing a body's Transform teleports at the next fixed sync without resetting interpolation. Body scale scales both shapes and meshes: spheres/capsules require uniform magnitudes, height fields positive components, all shapes nonzero finite components. Invalid shapes warn and retry when valid. Joint anchors stay unscaled body-frame metres.

## `Spring`, `Joint` and `ConstraintSystem`

Use joint-kind components with body eids `a`, `b`. Rename `rA`/`rB` to `localAnchorA`/`localAnchorB`; local rotations are normalized quaternions. `ConstraintSystem` is removed; standard physics syncs the components.

```ts
// 0.9.5
Tumble.world?.createRevoluteJoint(Tumble.body(a)!, Tumble.body(b)!, config);
```

```ts
// 0.10
world.add(jointEid, RevoluteJoint, { a, b });
```

| 0.9.5 | 0.10 |
|---|---|
| `Joint`, `stiffnessAng: 0` | `SphericalJoint` |
| `Joint`, `stiffnessAng: Infinity` | `WeldJoint`; preserve relative orientation with `localRotationB = inverse(bodyRotationB) * bodyRotationA` when frame A is identity |
| Intermediate angular stiffness | `WeldJoint.angularHertz` and `angularDampingRatio`; retune |
| `Spring.rest`, `stiffness` | `DistanceJoint.length`, `enableSpring: 1`, `hertz`, `dampingRatio`; retune |

Other Tumble joint kinds map to `FilterJoint`, `MotorJoint`, `ParallelJoint`, `PrismaticJoint`, `RevoluteJoint` and `WheelJoint` components.

## `setKinematic`, `setVelocity` and `readBody`

Use `/standard/physics` functions taking World first. `setKinematic` becomes `setTargetTransform` for motion, or `setTransform` for immediate teleport with interpolation reset. The optional velocity override is removed. `setVelocity` becomes `setLinearVelocity`; angular velocity, awake, force, torque and impulse writes also become World-first functions.

```ts
// 0.9.5
Physics.backend?.setKinematic(eid, position, rotation);
Physics.backend?.setKinematic(eid, respawnPosition, rotation, true);
Physics.backend?.setVelocity(eid, vx, vy, vz);
const x = Physics.backend?.readBody(eid)?.pos[0];
```

```ts
// 0.10
setTargetTransform(world, eid, position, rotation);
setTransform(world, eid, respawnPosition, rotation);
setLinearVelocity(world, eid, { x: vx, y: vy, z: vz });
const x = world.storage(GlobalTransform).translation.column[eid * 4];
```

Initial `Body.linearVelocity`, `angularVelocity` and `isAwake` are spawn-only fields; use the setters afterward. Other supported body definition writes apply at the next fixed sync. `StepSystem` becomes `StepPhysicsSystem`.

## `Tumble.world` settings, events and queries

Move gravity, substeps and material-mix callbacks to `PhysicsWorldDefinition` from `/standard/physics`. Changes apply at the next fixed sync. Mix callbacks retain `(a, idA, b, idB)` with World added first.

```ts
// 0.9.5
Tumble.world?.setGravity({ x: 0, y: -9.8, z: 0 });
const begins = Tumble.world?.getContactEvents().beginEvents;
```

```ts
// 0.10
const definition = world.resource(PhysicsWorldDefinition);
definition.gravity = { x: 0, y: -9.8, z: 0 };
definition.subStepCount = 6;
const begins = world.resource(PhysicsEvents).contactBegin;
for (let i = 0; i < begins.count; i++) world.resolve(begins.records[i]!.shapeA);
```

Replace sensor/contact/joint event methods with `PhysicsEvents.sensorBegin`, `sensorEnd`, `contactBegin`, `contactEnd`, `contactHit` and `joints`. Records carry `EntityRef`s, not kernel handles; resolve them because entities may have been removed.

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

`castRayClosest`, `castShape`, `overlapAABB` and `overlapShape` also become World-first functions and report ECS shape/body ids. Closest-hit calls write caller-owned output from `emptyQueryHit`. Remove backend-neutral `raycast`, `RayBody`, `RayHit`, `bodyCandidates`, `grabHit` and `worldToLocal`; camera picking uses `viewportToWorld` followed by a physics query.

## `Character`, `move`, `jump` and `PlayerPlugin`

Import `Character`, `CharacterPlugin` and `GroundState` from `/standard/physics`. The `/character/core` subpath is removed. Character needs a kinematic capsule body. Its default pogo float places the lower sphere centre three radii above ground; retune spawn/camera height.

```ts
// 0.9.5
move(eid, vx, vz);
const onGround = grounded(eid);
```

```ts
// 0.10
world.storage(Character).velocity.set(eid, vx, vy, vz);
const onGround = world.storage(Character).groundState.get(eid) === GroundState.OnGround;
```

Write velocity in `fixed` before `CharacterPlugin.systems`; acceleration, gravity and jumping belong to the game. Replace the character `pose` read with `GlobalTransform.translation`. Replace character `teleport` with `setTransform`; clear `Character.velocity` and `pogoVelocity` for a stationary respawn.

`PlayerPlugin` is no longer shipped. Copy `player.ts`, `motion.ts` and `follow.ts` from `examples/first-person/src` into your game. The copied controller uses `PlayerInput` for tick-addressed movement/look/jump counts, `LocalPlayer` for device sampling and sensitivity, and `Player.gravity`/`jumpSpeed` for tuning. It does not install rendering; retain rendering in the app composition. Replace `CharacterSweepSystem` ordering with `CharacterPlugin.systems`; replace `PlayerControlSystem` references with the local controller's `UpdatePlayerControlSystem` and `DrivePlayerSystem` as appropriate.

## `sample`, `play`, `Sound` and `/audio/core`

Clips now belong to a World. Core `/audio` exports `AudioPlugin`, `Sound`, `Listener`, `registerClip` and `play`; add `StandardAudioPlugin` from `/standard/audio` to run playback. `loadClip` there decodes URLs and shares repeated loads per World.

```ts
// 0.9.5
sample(pcm, "click");
play(state, "click", { pos: [1, 2, 3], volume: 0.5 });
```

```ts
// 0.10
const clip = registerClip(world, pcm, 48000, "click");
play(world, clip, { position: [1, 2, 3], volume: 0.25 });
world.add(eid, Sound, { clip, volume: 0.25, loop: 0 });
```

`play` takes a clip handle, not a graph name; `pos` becomes `position`. `Sound.instrument` becomes `clip`; remove `pitch` and runtime `voice`. `loop` remains; `paused` is new. Volume is linear gain, formerly quadratic: replace an old sample volume `v` with `v * v`.

Raw PCM defaults to 48 kHz, not the audio context's rate. For the former behavior pass `world.resource(AudioOutput).context?.sampleRate ?? 48000` as `registerClip`'s third argument. URL decoding retains the decoded rate; for a Blob, create an object URL and revoke it after `loadClip`.

Remove `sfx` and `SfxPolicy`; caps and cooldowns are game policy. The experimental graph APIs (`Instrument`, `InstrumentDef`, `Instruments`, `instrument`, `ModulationDef`, `NodeDef`, `NodeType`, `byId`, `getParamPairs`, `MAX_INSTRUMENTS`) left Shallot; see [shallot-audio-lab](https://github.com/dylanebert/shallot-audio-lab).

Other `/audio/core` removals:

| Old names | Replacement |
|---|---|
| `Audio`, `started`, `running` | `AudioOutput.context`/`bus`, `Devices.audio.context`; no started flag |
| `initAudio`, `disposeAudio`, `tickAudio` | Compose `StandardAudioPlugin`; do not call lifecycle helpers |
| `MAX_VOICES`, `slotOf`, `valid`, `alloc`, `free`, `gate`, `setParam`, `oneShot`, `watchIdle`, `assign` | Author `Sound` or call `play`; no public voice slots |
| `spatialize`, `polar`, `addSpatial`, `flushSpatial` | Position Sound with Transform and mark a Listener |
| `Samples`, `MAX_SAMPLES`, `getSample`, `whenLoaded` | `registerClip` or awaited `loadClip` |
| `noteFreq` | No stable replacement |

## `Mirror`, `probeBuffer` and `probeTexture`

Remove `Mirror`, `mirror`, `MirrorSystem` and `MirrorPlugin`. Request one-shot observations through `/runtime` probes; keep GPU work counts on the GPU rather than polling each frame.

```ts
// 0.9.5
const observation = mirror(typedArgs);
// MirrorPlugin updates observation.snapshot.bytes.
```

```ts
// 0.10
import type { World } from "@dylanebert/shallot";
import { probeBuffer } from "@dylanebert/shallot/runtime";
import { DrawIndexedIndirect } from "@dylanebert/shallot/standard/rendering";
import type { StorageFlag, TgpuBuffer } from "typegpu";

async function observe(world: World, typedArgs: TgpuBuffer<typeof DrawIndexedIndirect> & StorageFlag) {
    const observation = await probeBuffer(world, world.gpu.root.unwrap(typedArgs));
    return new Uint32Array(observation.bytes)[0];
}
```

`probeBuffer` takes a raw `GPUBuffer`, not a TypeGPU buffer. Pass raw buffers directly; unwrap a TypeGPU buffer with the owning `world.gpu.root.unwrap`, as above. Existing probe calls also change their first argument from GPUDevice to World. Buffers/textures must belong to that World; allocate through `world.gpu.device` or `world.gpu.root`, or register an external allocation with `world.own`. Returned bytes remain owned by the observation.

## `Profile`, `window.__benchmark` and timing sinks

Replace global `Profile` with `world.resource(DiagnosticsStore)`. `ProfilePlugin` records without a HUD; add `DiagnosticsOverlayPlugin` from `/extras` for F3 and `showProfiler(world)`.

```ts
// 0.9.5
const bytes = Profile.bufferBytes;
```

```ts
// 0.10
const bytes = world.resource(DiagnosticsStore).get("memory/bufferBytes")?.value();
```

| Old Profile field | Diagnostic path |
|---|---|
| `cpu`, `gpu` | `render/<name>/elapsed_cpu`, `render/<name>/elapsed_gpu` (ms) |
| `compile`, `compileMs` | `compile/<label>/elapsed`, `compile/elapsed` (ms) |
| `fenceWaitMs` | `frame/fence_wait` (ms) |
| `bufferBytes`, `textureBytes` | `memory/bufferBytes`, `memory/textureBytes` (bytes) |

Diagnostic `value()` is latest, `smoothed()` is the time-weighted exponential average, `measurements()` is bounded history in monotonic seconds. Paths escape label `%` as `%25`, `/` as `%2F`, empty as `%`. Absent spans retain history.

Remove `window.__benchmark`, `BenchmarkAPI`, `BenchmarkCompileStats`, `BenchmarkCpuStats`, `BenchmarkFrameStats`, `BenchmarkGpuStats`, `BenchmarkMeasurement`, `BenchmarkMemoryStats` and Profile's window/lifetime counters (`gpuTime`, `gpuFires`, `indirectCount`, `indirectFires`, `compiledPipelines`, `pipelineCalls`, `submitCount`). `allocBytes`/`lazyBytes` have no direct counter replacement; use the overlay's allocation details. Remove `LazyAlloc` profiler metadata.

`recordSink` now receives a borrowed `[monotonicSeconds, elapsedMilliseconds]` Float64Array, not scalar ms; copy values before retaining them. Use `sampleDiagnosticTime` to stamp a reusable two-cell buffer. `state.record`/`fenceWait` are removed; call the corresponding sinks if needed.

## `OrbitPick`, `OrbitOverlayPlugin` and orthographic fly drags

Remove `OrbitPick.claim`: orbit now runs on every drag. Decide click selection from `pointer.buttonEvents` and a game-owned travel threshold.

```ts
// 0.9.5
OrbitPick.claim = (x, y) => bodyUnderCursor(x, y) !== null;
```

```ts
// 0.10
const events = world.resource(Devices).pointer.buttonEvents;
for (const event of events) {
    // Apply the game's click/drag policy using event.x and event.y.
    console.log(event.x, event.y);
}
```

Remove `OrbitOverlayPlugin` imports/registrations; there is no replacement overlay. Orthographic `Orbit.flyButton` now rotates around the target instead of translating with WASD/QE; scroll changes `Camera.size`, not fly speed. Perspective flythrough is unchanged.

## `shallotLight`, `minimalDark` and `minimalLight`

`shallotLight`, `minimalDark` and `minimalLight` are removed. Use the default `shallotDark` or implement `Loading`, as `examples/loading-screen` does.

```ts
// 0.9.5
await run({ plugins: [], loading: minimalDark });
```

```ts
// 0.10
await runApp({ plugins: [] });
```

## `.scene`, `/document`, glTF, skinning, tween and harness removal

The `.scene` format and editor sessions have no replacement. Author worlds in code and save the component values your game needs. Remove app `scene` and `mode` options, `State.mode` and `annotations.mode`: every system runs.

```ts
// 0.9.5
await load(scene, state);
```

```ts
// 0.10
const eid = world.create();
world.add(eid, Transform, { translation: [0, 1, 0] });
```

Remove these plugins, subpaths and their calls:

| Old API | Porting action |
|---|---|
| `/scene/core`: `load`, `serialize`, `diagnose`, `parse`, `stringify`, `Node`, `Attr`, `ParseError`, `Diagnostic`, `normalizeAttr`, `parseFields`, `formatFields`, `readComponent`, `setFieldValue`, `findNodeById`, `findParent` | Replace with game-owned loading/saving |
| `/document`: `Document`, `History`, `Session`, `ReadbackSystem` | Remove editor sessions, undo/redo integration |
| `GltfPlugin`, `SkinPlugin`, `/gltf/core`, `/skin/core` | No importer, animation or live skinning replacement in this release |
| `Preloader`, `Preloads`, `preload` | Load assets in plugin `initialize` |
| `Tween`, `Sequence`, `/tween/core` | Implement animation in game code |
| `/bvh/core` | Keep acceleration structures in the consuming package |
| `/avbd`, `/avbd/core`, `AvbdPlugin` | Use `StandardPhysicsPlugin` |
| `/harness`, `/harness/pixels`, `/harness/browser`: `installHarness`, `HarnessTarget`, `REAL_GPU_LAUNCH` | Drive `createApp`/`world.step` in tests; own the Playwright configuration |

## Removed rendering shader and packing exports

These public 0.9.5 helpers no longer expose the renderer's packing or generated WGSL. Replace raw WGSL assembly with TypeGPU schemas/functions and material definitions. Do not substitute direct `/src/*` imports.

```ts
// 0.9.5
import { viewWgsl, VIEW_BYTES } from "@dylanebert/shallot/render/core";
```

```ts
// 0.10
import * as d from "typegpu/data";
import { ViewUniforms } from "@dylanebert/shallot/rendering";
const viewBytes = d.sizeOf(ViewUniforms);
```

| Old names | Replacement or action |
|---|---|
| `Render` | `RenderContext` |
| `View`, `viewWgsl`, `VIEW_BYTES`, `VIEW_STRIDE` | `ViewUniforms`; use each view's uniform buffer, not staging offsets |
| `View.cluster` | `ViewUniforms.projection` |
| `frameWgsl` | `FrameGpu` |
| `linearToSrgb`, `linearToSrgbWgsl` | `linearToSrgb3` |
| `FRUSTUM_FLOATS`, `frustumPlanes` | `CullVolumes`, `CULL_VOLUME_FLOATS` |
| `allocArray`, `arrayFromBitmaps`, `uploadLayer` | `imageArray` |
| `commonSize`, `mipLevels`, `backingSize` | Sizing is internal; use camera `Resolution` for explicit size |
| `MeshStorage`, `QuantStreams`, `meshBounds`, `packMeshes`, `quantizeMeshes`, `VERTEX_FLOATS`, `VERTEX_STRIDE` | Add mesh data through `Meshes.add`; packing/bounds are owned by MeshPlugin |
| `Binding`, `SurfaceLayout`, `VsFn`, `FsFn`, `VsIn`, `vsPatchSchema`, `fsCtxSchema`, `assertOwnFn`, `SURFACE_GROUP`, `engineScaffoldWgsl` | `materialLayout`, `MaterialBinding`, inferred material shader arguments |
| `fragCoord`, `fragWorld` | Inferred material fragment context (including `.world`) |
| `litPbr`, `Pbr` | `StandardMaterial` or `PbrInput`/`applyPbrLighting` |
| `lightFactor`, `pointFactor`, `pointScale` | `lit`, `distanceAttenuation` |
| `clusterCoord`, `clusterIndex`, `zSlice`, `clusterOf` | `clusterCell` |
| `lightClusters`, `PointLights`, `pointLightsWgsl` | `LightCull`, `LightClusters` |
| `lightingWgsl`, `LIGHTING_UNIFORM_SIZE` | `LightingGpu` and its TypeGPU size |
| `lightEvalWgsl`, `spotParams` | `distanceAttenuation`, `spotFactor`, `clusterCell` |
| `sunVisibility`, `sunShadowWgsl`, `sunStructWgsl`, `SHADOW_PARAMS_BYTES` | `sampleSunShadow`, `SunShadow` |
| `pointShadowWgsl` | `pointShadowRef()` |
| `cascadeCount`, `pointAtlasSize` | `DirectionalLight.numCascades`, `world.resource(PointShadows).atlas` |
| `DrawIndirectBuffer` | Infer from `Draw.args.indirect`; allocate with `DrawIndexedIndirect` |
| `Tag`, `TagFn`, `View.tag`, `TAG_FORMAT`, `TAG_NONE` | Picking lane removed |
| `Parts`, `getCompiledSurface`, `casterWgsl`, `pointComboCount`, `COLOR_LANES` | Remove; no public replacement |
| `ClusterView`, `CLUSTER_COUNT`, `CLUSTER_X`, `CLUSTER_Y`, `CLUSTER_Z`, `clusterAabb`, `clusterView`, `LIGHT_POOL`, `sliceDepth`, `MAX_POINT_LIGHTS` | Grid construction/capacity are internal |
| `FogSystem`, `FogScatter`, `FogSun`, `FOG_BYTES`, `FOG_FLOATS`, `FOG_MAX_STEPS`, `FogGpu`, `fogComposite`, `fogDensity`, `fogInScatter`, `fogInScatterWgsl`, `fogMarchWgsl`, `fogStructWgsl`, `fogSunInScatter`, `fogTransmittance`, `heightOpticalDepth`, `henyeyGreenstein`, `inScatterContribution`, `reconstructWorld`, `sunInScatter`, `WORKGROUP`, `packFog` | Author Fog and depend on FogPlugin; integration/packing are private |

`lit` moves from `/sear/core` to `/standard/rendering` and now uses standard metallic-0 lighting. Recheck surfaces very close to point/spot lights. Register custom draws with `world.resource(Draws).register`: replace `Draw.surface` with `materialType: materialTypeId(world, YourMaterialType)`, retain the handle from `Meshes.add` for `mesh`, and add `alphaPipelineKey: AlphaPipelineKey.Opaque` for an opaque draw. These names are exported from `/standard/rendering`. Shadow-camera helpers `cascadeComboEids` and `pointComboEids` take World first; their cameras hold `GlobalTransform`, not `Transform`. Remove direct `ClusterSystem`/`LightCullSystem` imports.

For `/utils/core` WGSL helpers, import `pack2x16snorm`, `unpack2x16snorm`, `pack2x16unorm`, `unpack2x16unorm`, `pack4x8unorm` from `typegpu/std` instead of `packSnorm2x16`, `unpackSnorm2x16`, `packUnorm2x16`, `unpackUnorm2x16`, `packUnorm4x8`. `bitcastF32toU32` has the same name there. Raw WGSL chunks call the corresponding WGSL builtins. `spliceNs` is removed; the octahedral and quaternion chunks no longer need namespace splicing.
