# Migrating from 0.10.1 to 0.10.2

Start with the compiler errors in your app. Each entry below goes from the published 0.10.1 package to 0.10.2.

## Readbacks refuse after a GPU error

`captureFrame` on a bound canvas, `captureTexture`, `probeTexture` and `probeBuffer` now reject once the world's device has stopped on a GPU error, naming that first error, instead of returning contents from before the failed frame. Fix the error the rejection names; a check that provokes a GPU error expects the rejection rather than reading the returned bytes.

## Material binding kinds take options

Texture and sampler entries in `materialLayout` and `backgroundLayout` name one kind, `"texture"` or `"sampler"`, and choose the WGSL type through options. Rename each entry; the layouts and shaders they produce are unchanged.

| 0.10.1 | 0.10.2 |
|---|---|
| `{ type: "texture-2d" }` | `{ type: "texture" }` |
| `{ type: "texture-2d-array" }` | `{ type: "texture", dimension: "2d-array" }` |
| `{ type: "texture-depth-2d" }` | `{ type: "texture", sampleType: "depth" }` |
| `{ type: "sampler" }` | `{ type: "sampler" }` |
| `{ type: "sampler-comparison" }` | `{ type: "sampler", samplerType: "comparison" }` |

## Fragment contexts gain prepass

A material's fragment also runs in the depth prepass, the normal prepass and shadow maps, where its lighting is stubbed and `fragCoord` is that pass's target. `ctx.prepass` is true there and false in the color pass; put storage-buffer and storage-texture writes under `if (!ctx.prepass)`.

`lit` now returns zero radiance in every depth pass, and `applyPbrLighting` returns emissive alone. An alpha cutoff derived from lit color therefore sees those values there. Derive coverage from material data rather than scene lighting when it must agree across passes. The normal prepass writes the normal of the fragment's last lighting call, or the interpolated normal if there is none.

## Material shaders lose the constant color

`color` is removed from the material vertex input and fragment context; it was always `vec4f(1)`. Write `1`, or read a mesh attribute: declare `{ type: "attribute", element: d.vec4f }` in `materialLayout`, add the stream under the same name in the mesh's `attributes`, and pass `layout.$.<name>[input.vidx]` from the vertex function to the fragment as a varying.

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
