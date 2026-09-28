# Contributing

For modifying the engine. For using Shallot, see the [README](README.md). Each API is documented in the JSDoc beside it. This page covers what the code, the CLI and failing checks don't.

## Architecture

Shallot is a WebGPU game engine for TypeScript, built on an entity component system (ECS). An entity is an id. A component is plain typed data stored per entity. A system is a function the scheduler runs every frame, in ordered groups such as `fixed` for gameplay on a fixed tick and `draw` for presentation. A scene is a file of entities and their components, loaded into the same data.

A plugin is how behavior gets into a game: a named bundle of components, systems and lifecycle hooks (`initialize`, `warm`, `dispose`), plus the plugins it needs. A project lists its plugins in `shallot.json`, and `build()` composes them into an app. Games and tests run the same composed app on the same stepped clock.

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
  harness/       Host-side seams for capture, readback, allocation and observation.
  types/         Ambient declarations.
crates/          The WASM kernels (audio, physics) and the native window host.
examples/        One folder per example. `examples/AGENTS.md` is generated from their manifests.
assets.json      Every asset except the shipped icon, fetched by URL and sha256 with `bun run assets`.
```

- `engine`, `core`, `standard` and `extras` are the game layers. `project`, `cli`, `native` and `harness` are tooling. Tooling can import any layer; game layers never import tooling.
- Modules in the same layer don't import each other, so a game can use one without the others.
- Each folder is one module. Its `index.ts` is the only entry point and defines its plugin; other files are internal. A layer's `index.ts` only re-exports its modules, except that `standard/index.ts` also defines the default plugin set.
- A module is a plugin only if it registers systems or resources. Otherwise it exports plain data and functions.
- Provider-specific observation belongs in optional application integrations, outside Shallot.
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
bun run build                         # regenerate committed audio WASM, dist/vite.js, physics kernel
bun run check                         # static gates; run before every push
bun test                              # cheap tier: *.test.ts
bun test ./src/transitional/mirror/index.gpu.ts  # named GPU tier
bun test ./diagnostics/.../allocation.oracle.ts # named display oracle (manual)
bun run test:browser                  # Playwright Test browser tier, *.e2e.ts
bun test --todo                       # run quarantined test.todo entries, if any
bun run format                        # biome, scene formatter and examples index
```

`check-imports` stays red while modules live in `src/transitional/`; their `// Destination:` lines name the migration owners. The current `core/rendering/view.ts` → `core/input` sibling import is a separate unresolved violation, not a permitted dependency. For unrelated work, compare import reds with main: report unchanged violations and continue, but stop on a new or changed violation. The gate remains red and is never skipped.

## Verification

A module's promises are tested beside the module and through the examples that use it. Each test name states the claim; its timeout is the wall-clock budget.

- Plain `bun test` discovers the cheap `*.test.ts` tier. GPU, Node and oracle tests with a different premise belong in named files and run by path.
- Rust suites run directly with Cargo:

  ```bash
  cargo test -p shallot-audio
  cargo test -p shallot-physics --lib --test stages
  ```
- A missing premise in a named tier is a test failure, never a skip. A quarantined claim stays visible as `test.todo` and runs with `bun test --todo`.
- Browser tests use Playwright Test in `*.e2e.ts`; `playwright.config.ts` starts each subject's own Vite preview and declares Chromium launch flags.
- GPU tests require an in-process WebGPU device. Browser tests request no adapter themselves; browser GPU observations accept software adapters.
- Display-bound measurements require a declared monitor and take its keyboard and cursor.
- Tests that make gameplay assertions step the composed app's clock. Simulation state lives in registered components or behind a snapshot, restore and hash hook. Gameplay runs in `fixed` from per-tick actions, presentation runs in `draw`, and `local` components are excluded from the hash. Runs are deterministic within one runtime and engine version; across versions, the hash detects divergence.
- Test a frame at the cheapest level that shows the defect: CPU state, GPU readback, browser pixels, then a person. Choose the capture by the claim:
  - An engine frame: read back a texture the test owns with `probeTexture`.
  - Page composition, such as overlays, posters and canvas reveal: step the app with `build()` and `state.step(dt)`, take a Playwright page screenshot, and assert semantic regions of it.
  - A running app's canvas: `captureFrame`, which reads during the next frame the loop presents.
- A WebGPU canvas reads as transparent black once its frame is presented, though the page still displays that frame: a canvas read works only inside the presenting frame, and a page screenshot works after it.
- Add a golden image only for a defect no cheaper level shows, and never update one to make it pass.
- Steady play allocates nothing. A memory test creates and disposes its subject, verifies memory returns to baseline, and fails on a deliberately leaking control. Retention is a separate measurement taken after GC; sampler allocation sites are diagnostics, not results.
- Timings are measured on real hardware, labeled with it, and reported, never asserted.
- An oracle is a tool the suite cannot run. Run it when the claim or its tool changes, record the result in that commit, and rerun it only for a specific doubt.
- A known failure stays failing until fixed; it is never skipped.

| Claim | Tier | Tool |
|---|---|---|
| Deterministic work and owned counts | cheap | Stepped assertions against scene-derived values; engine counters; `FinalizationRegistry` under `Bun.gc(true)` |
| WASM kernel memory | Cargo | A counting allocator per crate behind a Cargo feature; `memory.buffer.byteLength` |
| Native heap per step | Cargo | `dhat` assertions, one profiler per process |
| Steady JavaScript allocation | Node | The V8 sampling heap profiler over the composed subject in a Node child process |
| GPU resources released | GPU | A counting wrapper over the real Dawn device |
| Beyond the suite | oracle | Heap-snapshot diffs, CDP tracing, `measureUserAgentSpecificMemory`, WebGPU `timestamp-query` |

### CI coverage

CI runs static gates and the complete cheap tier on GitHub-hosted Ubuntu and macOS, plus the complete GPU, Cargo, Node and browser tiers on hosts with their required tools. The display-bound allocation oracle is run manually on its declared display seat. These jobs do not qualify a Windows runner or native packaging.

## Examples

Shallot is built from both ends, as a double loop. An example, built as a user would build it, is the outer loop: what it can't do is a gap in the engine. The module that owns the gap is the inner loop: the gap is fixed and proved there, never in the example.

An example is an app in the shape of a user's project, under `examples/<name>/`. Its manifest's `problem` field names what its user wants, as a piece of a game.

- An example answers one problem a user meets while making any game. Only `first-person` composes many.
- Every public promise the engine keeps is used by some example whose answer depends on it. A promise no example uses is a gap.
- No example only repeats what another shows. Of two that would show the same promises, the one kept shows the engine at its best.
- An example has a test that fails when its answer breaks. An example without one is removed.
- An example imports only published names; its tests use the same public seams.
- An example lives in the `examples/` of the repository whose promise its answer is about: the engine's or a package's.

## Heavy work

Heavy computation runs in WASM or on the GPU; TypeScript coordinates it and runs lightweight gameplay. TypeScript that needs runtime-specific tricks to meet a performance target belongs in WASM or on the GPU.

## Dependencies and releases

- A pin is the last verified version. Update a pin everywhere it appears in one commit; `check-pins` fails on drift.
- The root links to itself, so examples import the package by name. `@types/node` and `@webgpu/types` are runtime dependencies, because `types` points at source.
- A link doesn't prove what ships; a packed tarball installed in a scratch project does. Changes to the CLI, manifest, dependencies, runtime or native shell require that test.
- `main` may be mid-change. A release is a `v*` tag; its workflow builds the native shells and publishes to npm. Publish only to release. Consumers pin a published version or a full commit SHA.
- A breaking change updates every dependent it breaks in the same change. A removal or reshape strips what a dependent can no longer use rather than rebuilding it, so no dependent keeps a pattern the engine has left. Rebuilding in the new shape, additive features and hardening reach dependents once the engine side is settled, not through each intermediate state.
- To retire a module, example or tool, tag its last commit `archive/<name>`, then delete it in a commit that says why. There is no archive directory.

## Device tiers

Generated from the plugin declarations, which decide whether a composition needs a device.

<!-- device-tiers:start -->
| Plugin set | Composition tier | `required` (GPU) | `optional` (CPU/GPU) | absent declaration (CPU) |
| --- | --- | --- | --- | --- |
| core | gpu | Render | — | BrowserInput, Input |
| standard | gpu | BVH, Glaze, Mirror, Sear | Part, Physics, Slab, Transforms | Audio, Character |
| extras | gpu | Fog, Profile | Cells, Gltf, Lines, Outline, Skin, Sky, Sprite, Text | Orbit, OrbitOverlay, PhysicsProfile, Player |
<!-- device-tiers:end -->
