# Contributing

For anyone changing the engine, person or agent. Using Shallot is the [README](README.md). Each API's contract is the JSDoc beside it. This page holds what the tree, the CLI and a failing check do not say.

## Layout

Shallot is an onion: the engine at the center, then `core`, `standard`, `extras`, and outside the repo, external packages for what most games won't use. Code depends only on layers inside its own. Each layer out decides more for a game, so a game is more likely to swap or drop it.

```
src/
  engine/        Shallot itself: app lifecycle, ECS, scenes, runtime, utils. Nothing about rendering, physics, audio or input.
  core/          One plugin each for rendering, physics, audio and input, holding only what every way of doing it needs, plus transforms.
  standard/      Shallot's default way of doing each, built on core and made to extend. A game that wants another way swaps it.
  extras/        Features most games use, included and easy to drop. A plugin moves here from its own package once stable there for a release cycle.
  project/       What a project is at build time: manifest, plan, generation, the Vite plugin.
  cli/           The commands and their dispatcher.
  native/        The desktop shell.
  harness/       The verification protocol a project publishes and the check framework that drives it.
  types/         Ambient declarations.
crates/          The WASM kernels (audio, physics) and the native window host.
examples/        One flat directory per example; `examples/AGENTS.md` is generated from their manifests.
assets.json      Every asset but the shipped icon, fetched by URL and sha256 by `bun run assets`.
```

- `engine`, `core`, `standard` and `extras` are the game layers. `project`, `cli`, `native` and `harness` are tooling: they build, run and verify a game, may import any layer, and no game layer imports them.
- Modules in the same layer never import each other, so a game can take one without the rest. `core/transforms` is the exception: the others read positions from it.
- A folder is one module. Its `index.ts` is its only entry and holds its plugin; every other file is internal. A layer's own `index.ts` is its barrel and holds nothing else, and `standard/index.ts` also holds the default plugin set.
- A module is a plugin only when it registers systems or resources. Plain data and functions stay plain modules.
- Every public module has one subpath. The root re-exports every layer with `export *`, so a duplicate name fails `tsc` and no import needs `as`.
- Every module does one useful thing, fully. One that doesn't is fixed, split, moved out or removed.

### Core and standard

A `core` module holds only the data, rules and small mechanisms every way of doing its job needs, so a game keeps it when it changes approach. Something moves into core only once two different approaches need it to mean the same thing; being reused is not enough. The `standard` module beside it is one opinionated, extensible approach.

- A module is named for what it owns, not its technique. Core takes the plain noun, and standard the same noun: `core/rendering` at `/rendering`, `standard/rendering` at `/standard/rendering`.
- Core owns the plain names. A standard export takes the `Standard` prefix where it plays a role core also names, such as `StandardRenderingPlugin`, and keeps its plain name otherwise. An alternative implementation qualifies its own names, such as `AvbdPhysicsPlugin`.
- Physics never depends on rendering, in any layer.

### Rendering

`core/rendering` makes no assumption about how an image is made. Mesh rasterization, texel splatting, Gaussian splatting and generative rendering each build on it alone, and draw into the same views. It holds what all of them share: cameras and projection, views and their targets, the coordinate system and the GPU layout of shared data, the frame, color space and presentation. Meshes, materials and draw submission are not in it.

`standard/rendering` is the extensible mesh pipeline over it. Replacing it is how a game changes rendering approach; the camera, its views and its presentation carry over.

Each view reaches the screen through one final pass. The scene image says whether it is HDR or already display-ready; the final pass tonemaps HDR, then grades and encodes for the screen. An effect that needs only its own pixel, like a vignette, runs as a step inside that pass, before or after tonemapping, and may read its own texture. An effect that needs other pixels, like fog or an outline, runs as its own pass before it. A game can replace the final pass. `standard/rendering` has no post-processing of its own; it publishes what effects read, such as depth.

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

Each run replaces `.artifacts/` with its report and its children's output.

## Verification

Every promise is checked from both sides: tests beside the module, and the examples builders use. Check each claim at the cheapest level that can see it. A check's result depends only on what it declares.

- A check declares its claim, size and required host capabilities in `check()`. A host without one refuses and says why; it never runs a weaker version. No tag means CPU only, `gpu` a real WebGPU device, and `display` a declared monitor with its keyboard and cursor, because showing is what it measures.
- Checks run on the scheduler's stepped clock, never wall time. Simulation state lives in registered components or behind a snapshot, restore and hash hook: gameplay runs in the fixed group from per-tick actions, presentation in draw, and `local` components stay out of the hash. Determinism holds within one runtime and engine version; across them, a hash detects divergence.
- A frame is proved by a CPU property, then GPU readback, then browser pixels, then a person looking. `captureFrame` is the only capture, so checks, artifacts and people see the same frame. A golden image is added only for a defect nothing lower can see and is never edited to match; a screenshot is never a verdict.
- Steady play allocates nothing. Any steady allocation fails the integration check; the sampler's sites help find the cause but don't decide the result.
- A memory check creates and destroys its subject, confirms memory returns to baseline, and must fail on a control that deliberately leaks. Retention is checked separately, after a GC.
- Timings come from real devices, carry hardware labels and are reported, never gated.
- An oracle is an instrument the suite can't run. Run it once when the check it confirms is created or its instrument changes, note it in that commit, and again only when something is in doubt. It never decides a result.
- A known failure keeps failing until fixed; it is never skipped or excused.

| Claim | Size | Instrument |
|---|---|---|
| Deterministic work and owned counts | unit | Stepped assertions against scene-derived expectations; engine counters; `FinalizationRegistry` under `Bun.gc(true)` |
| WASM kernel memory | integration | A counting allocator per crate behind a cargo feature; `memory.buffer.byteLength` |
| Native heap per step | integration (`cargo`) | `dhat` assertions, one profiler per process |
| Steady JavaScript allocation | integration (`node`) | The V8 sampling heap profiler over the composed subject in a Node child |
| GPU resources released | integration (`gpu`) | A counting wrapper over the real Dawn device |
| What the suite cannot see | oracle | Heap-snapshot diffs, CDP tracing, `measureUserAgentSpecificMemory`, WebGPU `timestamp-query` |

## Examples

Examples are where a builder meets the engine, so what one cannot do names what the engine is missing.

- A recipe answers one builder problem, stated in its manifest's `problem`, and carries a check that fails when its answer breaks. A recipe no check can prove is removed.
- A showcase is a game a person would play, running only the checks a user's project can run.
- A gap an example finds goes to the module that owns it, never worked around in the example.

## Heavy work

Heavy work runs in WASM or on the GPU; TypeScript coordinates and carries light gameplay. TypeScript that meets a performance bar only through tricks aimed at the runtime's internals belongs in WASM or on the GPU instead.

## Dependencies and releases

- A pin records what was last verified. A bump updates every doc and fixture in one commit, and `check-pins` fails on drift.
- The root links to itself, so examples import the package by name. `@types/node` and `@webgpu/types` are runtime dependencies, because `types` points at source.
- A link does not prove what ships; a packed tarball installed into a scratch project does. A change to the CLI, manifest, dependencies, runtime or native shell owes one.
- `main` may be mid-change. A release is a `v*` tag, whose workflow builds the native shells and publishes to npm; publishing is never a way to try a change. Consumers pin a published version or a full commit SHA.
- A retired unit is tagged at its last commit, given a row in [`ARCHIVE.md`](ARCHIVE.md) and deleted. There is no archive directory.

## Device tiers

Generated from the plugin declarations, the one source of whether a composition needs a device.

<!-- device-tiers:start -->
| Plugin set | Composition tier | `required` (GPU) | `optional` (CPU/GPU) | absent declaration (CPU) |
| --- | --- | --- | --- | --- |
| standard | gpu | BVH, Fog, Glaze, Mirror, Render, Sear | Part, Physics, Slab, Transforms | Audio, BrowserInput, Character, Input, Player |
| extras | gpu | Profile | Cells, Gltf, Lines, Outline, Skin, Sky, Sprite, Text | Orbit, OrbitOverlay, PhysicsProfile |
<!-- device-tiers:end -->
