# Contributing

For modifying the engine. For using Shallot, see the [README](README.md). Each API is documented in the JSDoc beside it. This page covers what the code, the CLI and failing checks don't.

## Layout

Shallot is layered like an onion: `engine` at the center, then `core`, `standard` and `extras`, then external packages outside the repo. Code only imports from layers inside its own. Outer layers make more choices for a game, so games are more likely to replace or remove them.

```
src/
  engine/        Shallot itself: app lifecycle, ECS, scenes, runtime, utils.
  core/          One plugin each for rendering, physics, audio and input, with only what every approach needs. Also transforms.
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
- Modules in the same layer don't import each other, so a game can use one without the others. The exception is `core/transforms`, which the others read positions from.
- Each folder is one module. Its `index.ts` is the only entry point and defines its plugin; other files are internal. A layer's `index.ts` only re-exports its modules, except that `standard/index.ts` also defines the default plugin set.
- A module is a plugin only if it registers systems or resources. Otherwise it exports plain data and functions.
- Each public module has one subpath. The root re-exports every layer with `export *`, so duplicate names fail `tsc`.
- Each module does one useful thing completely. If it doesn't, fix it, split it, move it out or remove it.

### Core and standard

A `core` module contains only the data, rules and small mechanisms that every approach needs. Add something to core only when two different approaches need it with the same meaning. Reuse alone isn't enough.

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

Each run replaces `.artifacts/` with its report and the output of its child processes.

## Verification

Test each promise twice: with tests beside the module, and through the examples builders use. Test each claim at the cheapest level that can see it. A check's result depends only on what it declares.

- A check declares its claim, size and required host capabilities in `check()`. A host missing one refuses and says why, instead of running a weaker version. No tag means CPU only. `gpu` needs a real WebGPU device. `display` needs a declared monitor, and uses its keyboard and cursor.
- Checks run on the scheduler's stepped clock, never wall time. Simulation state is kept in registered components or behind a snapshot, restore and hash hook. Gameplay runs in the fixed group from per-tick actions, presentation runs in draw, and `local` components are left out of the hash. Runs are deterministic within one runtime and engine version; across versions, a hash detects divergence.
- Test a frame with a CPU property first, then GPU readback, then browser pixels, then a person looking. `captureFrame` is the only way to capture a frame, so checks, artifacts and people see the same thing. Add a golden image only for a defect nothing cheaper can catch, and never edit one to match. Screenshots are never a pass or fail.
- Steady play allocates nothing. Any steady allocation fails the integration check. The sampler's allocation sites help find the cause but don't decide the result.
- A memory check creates and destroys its subject, confirms memory returns to baseline, and must fail on a control that deliberately leaks. Retention is checked separately, after a GC.
- Timings come from real devices, are labeled with the hardware, and are reported but never pass or fail.
- An oracle is a tool the test suite can't run. Run it once when the check it confirms is created or its tool changes, and note it in that commit. Run it again only when a result is in doubt. It never decides pass or fail.
- A known failure stays failing until it's fixed. It is never skipped or excused.

| Claim | Size | Tool |
|---|---|---|
| Deterministic work and owned counts | unit | Stepped assertions against expected values from the scene; engine counters; `FinalizationRegistry` under `Bun.gc(true)` |
| WASM kernel memory | integration | A counting allocator per crate behind a cargo feature; `memory.buffer.byteLength` |
| Native heap per step | integration (`cargo`) | `dhat` assertions, one profiler per process |
| Steady JavaScript allocation | integration (`node`) | The V8 sampling heap profiler over the composed subject in a Node child process |
| GPU resources released | integration (`gpu`) | A counting wrapper over the real Dawn device |
| What the suite can't see | oracle | Heap-snapshot diffs, CDP tracing, `measureUserAgentSpecificMemory`, WebGPU `timestamp-query` |

## Examples

Examples are where builders meet the engine. What an example can't do shows what the engine is missing.

- A recipe solves one builder problem, stated in its manifest's `problem`, and has a check that fails when the solution breaks. Remove a recipe that no check can prove.
- A showcase is a game a person would want to play, and runs only the checks a user's project can run.
- When an example finds a gap, fix it in the module that owns it. Never work around it in the example.

## Heavy work

Heavy work runs in WASM or on the GPU. TypeScript coordinates and runs light gameplay. If TypeScript only meets a performance target through tricks aimed at the runtime's internals, move that work to WASM or the GPU.

## Dependencies and releases

- A pin records the version last verified. Bump a pin in every doc and fixture in one commit; `check-pins` fails on drift.
- The root links to itself, so examples import the package by name. `@types/node` and `@webgpu/types` are runtime dependencies, because `types` points at source.
- A link doesn't prove what ships; installing a packed tarball into a scratch project does. Changes to the CLI, manifest, dependencies, runtime or native shell need one.
- `main` can be mid-change. A release is a `v*` tag; its workflow builds the native shells and publishes to npm. Never publish to try a change. Consumers pin a published version or a full commit SHA.
- To retire a unit, tag its last commit, add a row to [`ARCHIVE.md`](ARCHIVE.md) and delete it. There is no archive directory.

## Device tiers

Generated from the plugin declarations, which decide whether a composition needs a device.

<!-- device-tiers:start -->
| Plugin set | Composition tier | `required` (GPU) | `optional` (CPU/GPU) | absent declaration (CPU) |
| --- | --- | --- | --- | --- |
| standard | gpu | BVH, Fog, Glaze, Mirror, Render, Sear | Part, Physics, Slab, Transforms | Audio, BrowserInput, Character, Input, Player |
| extras | gpu | Profile | Cells, Gltf, Lines, Outline, Skin, Sky, Sprite, Text | Orbit, OrbitOverlay, PhysicsProfile |
<!-- device-tiers:end -->
