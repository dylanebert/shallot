# Changelog

## 0.10.2

### highlights

- added screen-space ambient occlusion (`ambient-occlusion`)
- added camera rectangles and sampleable camera texture targets (`minimap`)
- added `isometric`, `low-res` and `snow-trails` examples
- the `physics` example now holds every registered box3d sample
- added plain-data resource recovery (`recoverResource`) and public membership observation (`World.observeMembership`)
- added physics contact and sensor reads, mass overrides, manual stepping, body queries, standalone collision and geometry functions, mesh builds and dynamic trees
- added material extension parameters, bindings and vertex functions, more texture and sampler binding options, per-row cull modes and double-sided standard materials
- various bugfixes, including late body-scale writes, query callbacks throwing, orthographic lighting, distant affine inversion, material depth passes and sprite shadows

### changes

- material texture and sampler binding kinds take options instead of names such as `"texture-2d"` and `"sampler-comparison"`
- material shaders lose the constant `color`; fragment contexts gain `prepass` and `frontFacing`, and lighting is stubbed in depth passes
- `MaterialSource.alphaKeys` becomes `pipelineKeys`; material and instance rows gain cull-mode fields
- `ViewUniforms.viewProj` and `invViewProj` become `clipFromWorld` and `worldFromClip`; `computeViewProj` becomes `computeClipFromWorld`
- view uniforms gain view and projection matrices; renderer bindings gain prepass and ambient-occlusion textures, with complete binding-budget checks
- camera producers run before `PrepareViewsSystem`, ahead of `BeginFrameSystem`
- physics mesh and height-field definitions become unions with generated forms
- cpu-only `World.tick` clears field marks on successful return; gpu worlds keep them for the next frame upload
- immediate lines sort as one item; retained gizmos draw after sorted transparency and use interpolated placement
- orbit cameras follow interpolated fixed targets
- readbacks reject after a gpu error, naming the first error that stopped the device

to update a 0.10 app, run `bunx shallot migrate` or read [`MIGRATION.md`](https://github.com/dylanebert/shallot/blob/v0.10.2/MIGRATION.md).

## 0.10.1

### highlights

- added `bunx shallot publish`, which publishes a web game to cloudflare workers or itch.io
- added camera stacking, with render layers choosing what each camera draws (`view-model`)
- added bloom and color grading, in one post-processing stack with tonemapping and vignette (`post-processing`, `bloom`, `color-grading`, `tonemapping`)
- added volumetric fog lit by the sun, spotlights and point lights (`volumetric-fog`)
- added mobile support: touch input, and touch controls in `first-person` and `interact`
- new projects typecheck in their ci
- various bugfixes, including ios rendering, safari startup stalls and desktop builds

### changes

- a camera without a canvas or texture draws on the first canvas, where 0.10 drew nothing; `attachCanvas` joins the same draw order
- `PointLightGpu` and `meshInstanceTable` gain render-layer fields
- `captureFrame` accepts any canvas size; `Capture.identity` no longer has `deviceScale`
- `world.add` on a component the entity already has replaces its values, resetting omitted fields to defaults; 0.10 kept them and warned
- web builds default to `base: "./"`; an app that routes nested urls through the history api sets `base: "/"`
- agx tonemapping colors change
- the optional playwright peer requires `^1.64.1`
- `@dylanebert/shallot/vignette` and `VignettePlugin` are deprecated in favor of `@dylanebert/shallot/effect-stack` and `EffectStackPlugin`, and are removed in 0.11

to update a 0.10 app, run `bunx shallot migrate` or read [`MIGRATION.md`](https://github.com/dylanebert/shallot/blob/v0.10.1/MIGRATION.md).

## 0.10.0

a foundational reset. 0.10 rebuilds shallot so everything can be run and checked from a terminal: no display, software gpu, ci, etc. this makes the engine far more testable, and lays the groundwork for multiplayer support.

### highlights

- **composable layers.** core, standard and optional extras, with cleaner separation between modules
- **headless.** cpu-only worlds, and software webgpu through dawn, for servers, tests and ci
- **independent worlds.** one page or process can run several ecs worlds
- **box3d physics.** deterministic port of box3d, with heavy lifting in multithreaded wasm
- **rendering rework.** masked and blended materials, sorted transparency, more cameras, infinite reverse-z depth, and more
- **simulation control.** fixed ticks apart from frame pacing, input addressed by tick, and snapshot and restore, so a run can be stepped as data
- **discoverable cli.** command-line interface lists every command and example, copies any example as a project, and prints its agent skill and migration guide

### removed

- `.scene` files, document model, editor mode, audio graph, tweens and sequences

### factored out to optional plugins

- gltf loading is [`@dylanebert/shallot-gltf`](https://www.npmjs.com/package/@dylanebert/shallot-gltf) (experimental, incomplete conformance)
- avbd physics (reworked gpu physics coming soon as an optional plugin. box3d physics recommended in most cases)

to port a 0.9.5 app, run `bunx shallot migrate` or read [`MIGRATION.md`](https://github.com/dylanebert/shallot/blob/v0.10.0/MIGRATION.md). earlier history is at [`archive/changelog-pre-0.10`](https://github.com/dylanebert/shallot/tree/archive/changelog-pre-0.10).
