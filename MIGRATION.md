# Migrating to 0.10.x

Start with the compiler errors in your app. Search this guide for the old symbol or behavior; each entry goes directly from the published 0.10 package to the latest 0.10 patch. For the engine's layers and extension rules, see [CONTRIBUTING.md](CONTRIBUTING.md).

Code fragments assume an owning `world`, an existing entity `eid`, and the imports named in the surrounding text.

## Captures use the rendered target's own size

`captureFrame` now accepts any canvas backing size instead of refusing sizes other than 1280×720. Remove capture-only resizes or writes to `CAPTURE_CONTRACT`; read the actual size from the returned capture. `CAPTURE_CONTRACT` still declares the fixed viewport size and device scale for fixtures.

`Capture.identity` from both `captureFrame` and `captureTexture` no longer includes `deviceScale`. Use its `width`, `height`, `surface` and `encoding` to identify the rendered target. For page screenshots, set the viewport and device scale in the browser driver; a target's backing size does not declare the page's device scale.

## Readbacks refuse after a GPU error

`captureFrame` on a bound canvas, `captureTexture`, `probeTexture` and `probeBuffer` now reject once the world's device has stopped on a GPU error, naming that first error, instead of returning contents from before the failed frame. Fix the error the rejection names; a check that provokes a GPU error expects the rejection rather than reading the returned bytes.

## Effect stack replaces vignette

`@dylanebert/shallot/vignette` and `VignettePlugin` still work as deprecated aliases and are removed in 0.11. Switch imports to `@dylanebert/shallot/effect-stack` and use `EffectStackPlugin` (also in root and extras imports). Both plugin names refer to the same plugin. `Vignette` keeps its fields and defaults. The plugin now also supports camera components `LensDistortion` and `ChromaticAberration`, fused with vignette into one pass before tonemapping.

## A second camera and the first canvas

In 0.10 a camera without a target auto-bound the document's first canvas only while no other camera held it; a second such camera stayed unbound and drew nothing. Now every camera without a target joins the first canvas, and cameras sharing a canvas or texture draw bottom to top by the new `Camera.order`, the bottom one clearing. `attachCanvas` on a canvas another camera of the World holds joins it the same way, where 0.10 let both cameras present over each other. A camera that drew nothing in 0.10 now draws on top of the first: give it its own target, or remove its `Camera`.

```ts
// 0.10: a second camera without a target drew nothing
world.add(eid, Camera);
```

```ts
// 0.10.1: give it its own target
world.add(eid, Camera);
attachTexture(world, eid, { width: 320, height: 180 });
// or stack it over the first camera on purpose
world.add(eid, Camera, { order: 1 });
```

## GPU light and entity-instance rows

`PointLightGpu` from `/standard/rendering` gains `layers: u32`, a 32-bit render-layer mask. A manually authored light record writes `1` for layer 0, or the intended mask; zero matches no camera. Its stride is now 64 bytes rather than 48. Derive strides and offsets with TypeGPU's `sizeOf` and `memoryLayoutOf` instead of retaining the old byte constants.

The entity table returned by `meshInstanceTable(world)` gains `layers` in existing padding, with bit 1 of `flags` indicating that `RenderLayers` is present; without that bit, the effective mask is `1`. Preserve those fields when publishing alpha or bounds directly. `MeshInstanceInput` remains the producer/material schema, with unchanged stride and field offsets; producer-table rows belong to layer 0. If an explicit material binding used `meshInstanceTable(world).typed` against `MeshInstanceInput`, use `meshInstanceTable(world).buffer` instead: the raw buffer binds the unchanged material-field layout without claiming that the entity and producer schemas are identical. Standard supplies this binding automatically.

## Material binding kinds take options

Texture and sampler entries in `materialLayout` and `backgroundLayout` name one kind, `"texture"` or `"sampler"`, and choose the WGSL type through options. Rename each entry; the layouts and shaders they produce are unchanged.

| 0.10 | 0.10.2 |
|---|---|
| `{ type: "texture-2d" }` | `{ type: "texture" }` |
| `{ type: "texture-2d-array" }` | `{ type: "texture", dimension: "2d-array" }` |
| `{ type: "texture-depth-2d" }` | `{ type: "texture", sampleType: "depth" }` |
| `{ type: "sampler" }` | `{ type: "sampler" }` |
| `{ type: "sampler-comparison" }` | `{ type: "sampler", samplerType: "comparison" }` |

## Fragment contexts gain prepass

A material's fragment also runs in the depth prepass, the normal prepass and shadow maps, where its lighting is stubbed and `fragCoord` is that pass's target. `ctx.prepass` is true there and false in the color pass; put storage-buffer and storage-texture writes under `if (!ctx.prepass)`.

`lit` now returns zero radiance in every depth pass, and `applyPbrLighting` returns emissive alone. An alpha cutoff derived from lit color therefore sees those values there. Derive coverage from material data rather than scene lighting when it must agree across passes. The normal prepass writes the normal of the fragment's last lighting call, or the interpolated normal if there is none.

## Optional mesh streams and material color

`color` is removed from the material vertex input. Fragment `ctx.color` now reads optional mesh RGBA when the material declares `fragmentInputs: { color: true }`; otherwise it remains white. `Meshes.add` accepts `color` (four floats per vertex), `uv1` (two) and `tangent` (four, unit xyz and w ±1) beside the base `vertices`. Materials request the latter with `fragmentInputs.uv1` and `.tangent`; absent values are `ctx.uv` and `vec4f(0)`. Tangents reach fragments in world space, with handedness corrected for mirrored instances. Material vertex functions cannot read or patch these streams. `Materials` and `extendedMaterial(Materials, ...)` declare all three inputs.

GPU producers writing `MeshQuant` change `uvScale` from `vec4f` to `vec2f` and add `streams: u32`, zero when there are no optional streams. Stride stays 48 bytes; position and UV lane offsets are unchanged. Raw writers put `streams` at byte 40 as a u32, leaving byte 44 as padding. For meshes declaring `Mesh.streams`, this is the vec4u index in `vertices` where the family's optional words begin; the layout is documented on `Mesh.streams`.

## Material rows choose their cull mode

A material row takes `cullMode` beside `alphaMode`, as a `GPUCullMode` (`"back"` by default, `"none"` or `"front"`), in `add`, `setAt` and `update`. `MaterialHandle` and `MeshMaterial` gain its `cullMode` code (`CullModeCode`, back `0`). `StandardMaterial({ doubleSided: true })` shades back faces with the flipped normal; give such a row `cullMode: "none"`, as glTF's `doubleSided` does both. Fragment contexts gain `frontFacing`.

```ts
materials.add(StandardMaterial({ doubleSided: true }), { cullMode: "none" });
```

- `MaterialSource.alphaKeys` is renamed `pipelineKeys`. For each row it holds, a source sets bit `1 << pipelineKey(alphaPipelineKey(alphaMode), cullMode)`, where `alphaMode` is the row's `AlphaModeCode` and `cullMode` its `CullModeCode`; both functions come from `@dylanebert/shallot/standard/rendering`. A source holding only back-culled rows keeps its value. An optional `cullMode(row)` reports a row's code for CPU transparent sorting.
- `MeshInstanceInput` gains `cullMode: u32` in existing padding; stride and the other offsets are unchanged. A producer row writes its handle's `cullMode`, `0` for back; the entity table's `layers` moves to the next word.
- `StandardMaterialInput` gains `doubleSided: u32`; `StandardMaterial()` fills it, and a hand-built row writes `0`.
- A hand-built `Draw` may set `cullMode`; omitted, it culls back faces.

## View uniforms use Bevy's matrix names

`ViewUniforms.viewProj` is now `clipFromWorld` and `invViewProj` is `worldFromClip`, as in Bevy's `View`; `computeViewProj` is now `computeClipFromWorld`, which also takes an optional view→clip output. Rename each read in a material, background or extension shader. `ViewUniforms` gains `viewFromWorld` and `clipFromView` after its existing fields, so a shader reads the view and projection matrices instead of uploading its own; existing byte offsets are unchanged, and a raw WGSL struct that declares only the leading matrix still binds.

`RenderContext.viewStaging` now spaces slots by `VIEW_BYTES`, not 256 bytes. Use `VIEW_BYTES` from `/rendering` when indexing slots and TypeGPU's `memoryLayoutOf(ViewUniforms, ...)` for field offsets. The internal `VIEW_STRIDE` constant is removed.

## Adding an attached component replaces its values

In 0.10, `world.add` on an entity that already held the component left its values unchanged and warned. Now it replaces them, as Bevy's `insert` does: supplied fields take their values, omitted fields take the declared defaults, or 0, and every field publishes a change. A call that relied on the old no-op to ensure presence resets the component; guard it with `world.has`, or write single fields through `world.storage`.

```ts
// 0.10: left the material unchanged
world.add(eid, MeshMaterial, handle);
```

```ts
// 0.10.1: the entity draws with the new material
world.add(eid, MeshMaterial, handle);
// to ensure presence without resetting
if (!world.has(eid, MeshMaterial)) world.add(eid, MeshMaterial);
```

## AgX uses a default-contrast LUT

`TonemappingMethod.AgX` now uses a log transform and default-contrast LUT instead of the analytic approximation. Its displayed colors change without an API change. Review AgX scenes and capture expectations; retune exposure or color grading if your look depended on the old approximation.

## Optional Playwright peer requires 1.64.1

If your project installs Playwright for browser checks or publishing verification, update it to `^1.64.1` and run `bunx playwright install chromium` to install its matching browser. Projects without Playwright do not need to add it unless they use a command that requires it.

## Web builds name their files relative to the page

`shallot()` now builds with Vite's `base: "./"` unless your config sets `base`, so the output loads under any path, as itch.io and a subpath host serve it. A game that serves its page at nested routes through the History API sets `base: "/"` in its Vite config to keep root-absolute URLs.
## Renderer bindings and prepass targets

`engineLayout` gains `prepassDepth`, `prepassNormal` and `ambientOcclusion` texture bindings. Code constructing its own group 0 must supply them; prefer the renderer's prepared groups. Material and background layouts now validate complete per-stage WebGPU binding budgets, including engine and shadow bindings; transparent pipelines also count registered fog. Reduce owned bindings or their stage visibility if preparation refuses the layout. Custom material varyings are checked against the device's inter-stage-variable limit; reduce varyings if refused.

A `PhaseRenderer.prepass` receives a `NORMAL_FORMAT` color target when `view.normal` is set. Supply a matching pipeline target and write world normals encoded as `normal * 0.5 + 0.5`, or do not enable normal prepasses for a renderer that cannot supply them. Material types reading `prepassDepth` or `prepassNormal` must declare `readsPrepass.depth` or `readsPrepass.normal` respectively.

## Camera preparation precedes frame setup

Camera producers that previously ran before `BeginFrameSystem` must run before the new `PrepareViewsSystem`, which binds, orders and sizes target cameras before shadow fitting. View consumers run after `PrepareViewsSystem`. A hand-built `View` now supplies `aspect` (projection width/height) and `normal` (`null` without a normal target); `viewport` is optional. Existing cameras without `CameraViewport` keep full-target presentation.

## Physics geometry definitions admit generated forms

`PhysicsMesh` and `HeightField` are now unions including generated geometry. Existing literal definitions still work, but code reading their fields must narrow first, for example with `"vertices" in mesh` or `"heights" in field`. Use `Extract<HeightField, { heights: Float32Array }>` for a grid-only height-field annotation and `Extract<PhysicsMesh, { vertices: Float32Array }>` for an explicit mesh definition. An interface cannot extend either union; use a type intersection or a narrowed variant instead.

## Contact events identify compound children

`PhysicsContactEvent` gains `childIndex`, identifying the compound child within its shape pair (`0` for a non-compound pair). Hand-written event records and fixtures must supply it; event consumers can keep reading their existing fields.

## Physics world definitions name automatic stepping

`PhysicsWorldDefinition` gains `enableAutomaticStep`, defaulting to `true`. A complete hand-written definition supplies `enableAutomaticStep: true` to preserve automatic simulation; partial resource configuration keeps the default. Set it to `false` only when calling `stepPhysics` yourself from a fixed system ordered after `StepPhysicsSystem`. The optional `maximumMemory` keeps the shared kernel's default 1 GiB ceiling unless explicitly set.

## CPU-only ticks clear field marks

A successful CPU-only `World.tick` now clears field-change marks before returning, as `World.step` already does. Read marks in a system during the tick, or record changes in your own state rather than inspecting dirty bits after return. GPU worlds keep marks for the next frame upload.

## Orbit follows interpolated targets

`Orbit.target` now follows interpolated fixed placement at display refresh, snapping the whole follow pose on between-tick teleports. Remove game-side target interpolation that would apply it twice; review camera captures taken between ticks.
