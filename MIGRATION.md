# Migrating from 0.10 to 0.10.1

Start with the compiler errors in your app. Search this guide for the old symbol or behavior; each entry goes directly from the published 0.10 package to 0.10.1. For the engine's layers and extension rules, see [CONTRIBUTING.md](CONTRIBUTING.md).

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

| 0.10 | 0.10.1 |
|---|---|
| `{ type: "texture-2d" }` | `{ type: "texture" }` |
| `{ type: "texture-2d-array" }` | `{ type: "texture", dimension: "2d-array" }` |
| `{ type: "texture-depth-2d" }` | `{ type: "texture", sampleType: "depth" }` |
| `{ type: "sampler" }` | `{ type: "sampler" }` |
| `{ type: "sampler-comparison" }` | `{ type: "sampler", samplerType: "comparison" }` |

## Fragment contexts gain prepass

A material's fragment also runs in the depth prepass, the normal prepass and shadow maps, where its lighting is stubbed and `fragCoord` is that pass's target. `ctx.prepass` is true there and false in the color pass; put storage-buffer and storage-texture writes under `if (!ctx.prepass)`.

## Material shaders lose the constant color

`color` is removed from the material vertex input and fragment context; it was always `vec4f(1)`. Write `1`, or read a mesh attribute: declare `{ type: "attribute", element: d.vec4f }` in `materialLayout`, add the stream under the same name in the mesh's `attributes`, and pass `layout.$.<name>[input.vidx]` from the vertex function to the fragment as a varying.

## Material rows choose their cull mode

A material row takes `cullMode` beside `alphaMode`, as a `GPUCullMode` (`"back"` by default, `"none"` or `"front"`), in `add`, `setAt` and `update`. `MaterialHandle` and `MeshMaterial` gain its `cullMode` code (`CullModeCode`, back `0`). `StandardMaterial({ doubleSided: true })` shades back faces with the flipped normal; give such a row `cullMode: "none"`, as glTF's `doubleSided` does both. Fragment contexts gain `frontFacing`.

```ts
materials.add(StandardMaterial({ doubleSided: true }), { cullMode: "none" });
```

- `MaterialSource.alphaKeys` is renamed `pipelineKeys`: its bit for a row's pair is `1 << (alphaPipelineKey + 6 * cullMode)`, so an existing source that held only back-culled keys keeps its value. An optional `cullMode(row)` reports a row's code for CPU transparent sorting.
- `MeshInstanceInput` gains `cullMode: u32` in existing padding; stride and the other offsets are unchanged. A producer row writes its handle's `cullMode`, `0` for back; the entity table's `layers` moves to the next word.
- `StandardMaterialInput` gains `doubleSided: u32`; `StandardMaterial()` fills it, and a hand-built row writes `0`.
- A hand-built `Draw` may set `cullMode`; omitted, it culls back faces.

## View uniforms use Bevy's matrix names

`ViewUniforms.viewProj` is now `clipFromWorld` and `invViewProj` is `worldFromClip`, as in Bevy's `View`; `computeViewProj` is now `computeClipFromWorld`, which also takes an optional view→clip output. Rename each read in a material, background or extension shader. `ViewUniforms` gains `viewFromWorld` and `clipFromView` after its existing fields, so a shader reads the view and projection matrices instead of uploading its own; existing byte offsets are unchanged, and a raw WGSL struct that declares only the leading matrix still binds.

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
