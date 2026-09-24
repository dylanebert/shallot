# Contributing

For modifying the engine. For using Shallot, see the [README](README.md). Each API is documented in the JSDoc beside it. This page covers what the code, the CLI and failing checks don't.

## Architecture

Shallot is a WebGPU game engine for TypeScript, built on an entity component system (ECS). An entity is an id. A component is plain typed data stored per entity. A system is a function the scheduler runs every frame, in ordered groups such as `fixed` for gameplay on a fixed tick and `draw` for presentation. A scene is a file of entities and their components, loaded into the same data.

A plugin is how behavior gets into a game: a named bundle of components, systems and lifecycle hooks (`initialize`, `warm`, `dispose`), plus the plugins it needs. A project lists its plugins in `shallot.json`, and `build()` composes them into an app. Games and checks run the same composed app on the same stepped clock.

## Layout

Shallot is layered like an onion: `engine` at the center, then `core`, `standard` and `extras`, then external packages outside the repo. Code only imports from layers inside its own. Outer layers make more choices for a game, so games are more likely to replace or remove them.

```
src/
  engine/        Shallot itself: app lifecycle, ECS, scenes, runtime, utils.
  core/          One plugin each for rendering, physics, audio and input, with only what every approach needs.
  standard/      Shallot's default approach to each, built on core and extensible.
  extras/        Features most games use. A plugin moves here after a stable release cycle as its own package.
  project/       Build-time project handling: manifest, plan, code generation, the Vite plugin.
  cli/           The commands.
  native/        The desktop shell.
  harness/       The verification protocol a project publishes, and the check framework that runs it.
  types/         Ambient declarations.
crates/          The WASM kernels (audio, physics) and the native window host.
examples/        One folder per example. `examples/AGENTS.md` is generated from their manifests.
assets.json      Every asset except the shipped icon, fetched by URL and sha256 with `bun run assets`.
```

- `engine`, `core`, `standard` and `extras` are the game layers. `project`, `cli`, `native` and `harness` are tooling. Tooling can import any layer; game layers never import tooling.
- Modules in the same layer don't import each other, so a game can use one without the others.
- Each folder is one module. Its `index.ts` is the only entry point and defines its plugin; other files are internal. A layer's `index.ts` only re-exports its modules, except that `standard/index.ts` also defines the default plugin set.
- A module is a plugin only if it registers systems or resources. Otherwise it exports plain data and functions.
- Each public module has one import path, its subpath in `package.json` `exports`. The root re-exports every layer with `export *`, so duplicate names fail `tsc`.
- Each module does one useful thing completely. If it doesn't, fix it, split it, move it out or remove it.

### Core and standard

- Name a module for what it owns, not its technique. Core and standard use the same noun: `core/rendering` at `/rendering`, `standard/rendering` at `/standard/rendering`.
- Core gets the plain names. A standard export adds the `Standard` prefix when core has the same role, like `StandardRenderingPlugin`. Other implementations use their own prefix, like `AvbdPhysicsPlugin`.
- Physics never imports rendering, in any layer.

### Rendering

`core/rendering` doesn't assume how an image is made. Mesh rasterization, texel splatting, Gaussian splatting and generative rendering can each be built on it alone, and draw into the same views. It contains what they all share: cameras and projection, views and their targets, the coordinate system, the GPU layout of shared data, the frame, color space and presentation. It contains no meshes, materials or draw submission.

`standard/rendering` is the extensible mesh pipeline built on it.

Each view reaches the screen through one final pass. The scene image is marked HDR or display-ready; the final pass tonemaps HDR images, then applies grading and encodes for the screen. An effect that only needs its own pixel, like a vignette, runs as a step inside the final pass, before or after tonemapping, and can read its own texture. An effect that needs other pixels, like fog or outlines, runs as its own pass before the final pass. A game can replace the final pass. `standard/rendering` has no post-processing; it publishes data effects need, like depth.

## Commands

```bash
bun run build     # regenerate committed audio WASM, dist/vite.js, physics kernel
bun run check     # structure, population, workflow and static gates; run before every push
bun run test      # every unit test, hermetic, under the 250ms unit limit
bun run test -- --integration --base <ref> --diff <ref>   # tests whose subject changed, plus tests with no subject
bun run test -- --integration --all | --requires <tag> | --subject <prefix>   # combine freely, but not with --base/--diff
bun run test -- --oracle <claim>   # one named oracle, never part of a sweep
bun run list      # what the same selectors would run: claim, size, requirements, budget, file
bun run format    # biome, the scene formatter and the examples index, writing
```

Each run replaces `.artifacts/` with its report and child process output.

## Verification

A module's promises are tested beside the module and through the examples that use it. Test each claim at the cheapest level that can observe it. A check's result depends only on its declared inputs.

- A check declares its claim, size and required host capabilities in `check()`. A host without a required capability refuses with the reason; it never runs a weaker version. Untagged checks are CPU-only, `gpu` requires a WebGPU device, and `display` requires a declared monitor and takes its keyboard and cursor.
- Checks run on the scheduler's stepped clock, never wall time. Simulation state lives in registered components or behind a snapshot, restore and hash hook. Gameplay runs in `fixed` from per-tick actions, presentation runs in `draw`, and `local` components are excluded from the hash. Runs are deterministic within one runtime and engine version; across versions, the hash detects divergence.
- Test a frame at the cheapest level that shows the defect: CPU state, GPU readback, browser pixels, then a person. Capture frames only with `captureFrame`. Add a golden image only for a defect no cheaper level shows, and never update one to make it pass. Screenshots are not results.
- Steady play allocates nothing; the integration check fails on any steady allocation. Sampler allocation sites are diagnostics, not results.
- A memory check creates and disposes its subject, verifies memory returns to baseline, and fails on a deliberately leaking control. Retention is a separate check, taken after GC.
- Timings are measured on real hardware, labeled with it, and reported, never asserted.
- An oracle is a tool the suite can't run. Run it when the check it validates is created or its tool changes, record the result in that commit, and rerun it only for a specific doubt. It never determines a result.
- A known failure stays failing until fixed; it is never skipped.

| Claim | Size | Tool |
|---|---|---|
| Deterministic work and owned counts | unit | Stepped assertions against values derived from the scene; engine counters; `FinalizationRegistry` under `Bun.gc(true)` |
| WASM kernel memory | integration | A counting allocator per crate behind a cargo feature; `memory.buffer.byteLength` |
| Native heap per step | integration (`cargo`) | `dhat` assertions, one profiler per process |
| Steady JavaScript allocation | integration (`node`) | The V8 sampling heap profiler over the composed subject in a Node child process |
| GPU resources released | integration (`gpu`) | A counting wrapper over the real Dawn device |
| Beyond the suite | oracle | Heap-snapshot diffs, CDP tracing, `measureUserAgentSpecificMemory`, WebGPU `timestamp-query` |

## Examples

A recipe is a small project that solves one problem; a showcase is a complete game. Something an example can't do is a gap in the engine.

- A recipe states its problem in its manifest's `problem` field and has a check that fails when the solution breaks. A recipe without such a check is removed.
- A showcase runs only the checks a user's project can run.
- Fix a gap in the module that owns it, not in the example.

## Heavy work

Heavy computation runs in WASM or on the GPU; TypeScript coordinates it and runs lightweight gameplay. TypeScript that needs runtime-specific tricks to meet a performance target belongs in WASM or on the GPU.

## Dependencies and releases

- A pin is the last verified version. Update a pin everywhere it appears in one commit; `check-pins` fails on drift.
- The root links to itself, so examples import the package by name. `@types/node` and `@webgpu/types` are runtime dependencies, because `types` points at source.
- A link doesn't prove what ships; a packed tarball installed in a scratch project does. Changes to the CLI, manifest, dependencies, runtime or native shell require that test.
- `main` may be mid-change. A release is a `v*` tag; its workflow builds the native shells and publishes to npm. Publish only to release. Consumers pin a published version or a full commit SHA.
- To retire a module, example or tool, tag its last commit, add a row to [`ARCHIVE.md`](ARCHIVE.md) and delete it. There is no archive directory.

## Device tiers

Generated from the plugin declarations, which decide whether a composition needs a device.

<!-- device-tiers:start -->
| Plugin set | Composition tier | `required` (GPU) | `optional` (CPU/GPU) | absent declaration (CPU) |
| --- | --- | --- | --- | --- |
| standard | gpu | BVH, Fog, Glaze, Mirror, Render, Sear | Part, Physics, Slab, Transforms | Audio, BrowserInput, Character, Input, Player |
| extras | gpu | Profile | Cells, Gltf, Lines, Outline, Skin, Sky, Sprite, Text | Orbit, OrbitOverlay, PhysicsProfile |
<!-- device-tiers:end -->
