# Migrating from 0.10 to 0.10.1

Start with the compiler errors in your app. Search this guide for the old symbol or behavior; each entry goes directly from the published 0.10 package to 0.10.1. For the engine's layers and extension rules, see [CONTRIBUTING.md](CONTRIBUTING.md).

Code fragments assume an owning `world`, an existing entity `eid`, and the imports named in the surrounding text.

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
