# Contributing

For modifying the engine. For using Shallot, see the [README](README.md). Each API is documented in the JSDoc beside it. This page covers what the code, the CLI and failing checks don't.

## Architecture

Shallot is a WebGPU game engine for TypeScript, built on an entity component system (ECS). An entity is an id. A component is plain typed data stored per entity. A system is a function the scheduler runs every frame, in ordered groups such as `fixed` for gameplay on a fixed tick and `draw` for presentation. A scene is a file of entities and their components, loaded into the same data.

A plugin is how behavior gets into a game: a named bundle of components, systems and lifecycle hooks (`initialize`, `warm`, `dispose`), plus the plugins it needs. A project lists its plugins in `shallot.json`, and `build()` composes them into an app. Games and tests run the same composed app on the same stepped clock.

## Layout

Shallot is layered like an onion: `engine` at the center, then `core`, `standard` and `extras`, then external packages outside the repo. Dependencies point inward. Outer layers make more choices for a game, so games are more likely to replace or remove them.

```
src/
  engine/        App lifecycle, ECS, scenes, runtime and utilities.
  core/          Shared rendering and input capabilities.
  standard/      Shallot's default plugins and features, built on core.
  extras/        Features most games use. A plugin moves here after a stable release cycle as its own package.
  transitional/  Modules awaiting their declared destination; import checks keep them visible.
  project/       Project manifests, plan resolution, generated virtual module, Vite/Bun plugins and build support.
  cli/           Shallot's commands.
  native/        The desktop shell.
  types/         Ambient declarations.
crates/          The WASM kernels (audio, physics) and the native window host.
diagnostics/     Host-side diagnostics, including first-person allocation sampling.
examples/        One folder per example. Each owns its page and Vite config with `shallot()`; `examples/AGENTS.md` is generated from manifests.
assets.json      Every asset except the shipped icon, fetched by URL and sha256 with `bun scripts/assets.ts`.
```

- `engine`, `core`, `standard` and `extras` are the game layers. `project`, `cli`, `native` and `types` are tooling. Game modules never import tooling; dependencies among game layers point inward. `transitional` contains modules with a declared migration destination, and its import-check reds remain until those modules move.
- Core, standard and extras modules don't import sibling modules in their layer. Physics never imports rendering.
- Each game module's `index.ts` is its public entry point; other source files are internal. A module that registers systems or resources defines a plugin; otherwise it exports plain data and functions. Layer indexes re-export their modules, except `standard/index.ts`, which also defines the default plugin set.
- Provider-specific observation belongs in optional application integrations, outside Shallot.
- `package.json` declares the public package subpaths. The root barrel re-exports every game layer with `export *`, so duplicate names fail `tsc`.
- Each module does one useful thing completely. If it doesn't, fix it, split it, move it out or remove it.

### Core and standard

- Name a module for what it owns, not its technique. Core and standard use the same noun: `core/rendering` at `/rendering`, `standard/rendering` at `/standard/rendering`.
- Export names describe the implementation: core exports `RenderPlugin`, while the standard mesh renderer exports `SearPlugin`.
- Physics never imports rendering, in any layer.

### Rendering

`core/rendering` doesn't assume how an image is made. Mesh rasterization, texel splatting, Gaussian splatting and generative rendering can each be built on it alone, and draw into the same views. It contains what they all share: cameras and projection, views and their targets, the coordinate system, the GPU layout of shared data, the frame, color space and presentation. It contains no meshes, materials or draw submission.

`standard/rendering` is the extensible mesh pipeline built on it.

Each view reaches the screen through one final pass. The scene image is marked HDR or display-ready; the final pass tonemaps HDR images, then applies grading and encodes for the screen. An effect that only needs its own pixel, like a vignette, runs as a step inside the final pass, before or after tonemapping, and can read its own texture. An effect that needs other pixels, like fog or outlines, runs as its own pass before the final pass. A game can replace the final pass. `standard/rendering` has no post-processing; it publishes data effects need, like depth.

## Commands

A `shallot` command earns its place only by doing what only Shallot knows. It never owns a process Vite, Bun or Playwright owns; it may run the project's own commands as a step. Web `dev`, `build` and `preview` run the project's Vite commands. Native `dev` and `build` add the shell to the project's Vite dev server or build; native `preview` launches the existing desktop build.

A `package.json` script is a lifecycle hook the package manager runs, such as `prepare`, or a verb every Shallot project has; any other tool runs by path, `bun scripts/<tool>.ts`.

```bash
bun run build                         # regenerate audio WASM, dist/vite.js, physics kernel
bun run check                         # static gates; run before every push
bun run test                          # *.test.ts, including GPU tests on a device
bun run test --path-ignore-patterns '**/*.gpu.test.ts' # no-device host tier
bun test gpu.test                     # GPU tier
bun test ./diagnostics/first-person-allocation/allocation.oracle.ts # named display oracle (manual)
bun run test:browser                  # wide browser run; every subject config
bun test --todo                       # run quarantined test.todo entries, if any
bun run format                        # biome, scene formatter and examples index
```

`check-imports` stays red while modules live in `src/transitional/`; their `// Destination:` lines name the migration owners. The current `core/rendering/view.ts` → `core/input` sibling import is a separate unresolved violation, not a permitted dependency. For unrelated work, compare import reds with main: report unchanged violations and continue, but stop on a new or changed violation. The gate remains red and is never skipped.

## Verification

A module's promises are tested beside the module and through the examples that use it. Each test name states the claim; its timeout is the wall-clock budget.

| Tier | Per-test timeout ceiling |
|---|---|
| Cheap `*.test.ts` | 250 ms |
| GPU `*.gpu.test.ts` | 1 s |
| Node `*.node.ts` | 20 s |
| Manual `*.oracle.ts` | Unbounded; run by path |

A test may declare a lower timeout, never a higher one. A test that needs more is split or moves to a heavier tier. Each GPU and Node file sets its tier ceiling with `setDefaultTimeout`. `check-timeouts` rejects larger or unresolved timeout declarations and missing tier defaults. Every bounded GPU wait fits inside its test's ceiling.

- `bun run test` discovers `*.test.ts` files, including `*.gpu.test.ts`, with a 250 ms default timeout per test. Individual test timeouts override it. Bare `bun test` discovers the same files with Bun's default timeout. Hosted jobs without a device exclude `*.gpu.test.ts` by pattern; the macOS GPU job runs `bun test gpu.test`. Node tests remain named `*.node.ts` files; the Node tier runs on macOS with Dawn's `webgpu` binding available to child processes. Oracles are manual and run by path.
- Root `bunfig.toml` loads Shallot's Bun plugin so the engine tests receive the same TypeGPU transform as project test preloads.
- Rust suites run directly with Cargo:

  ```bash
  cargo test -p shallot-audio
  cargo test -p shallot-physics
  ```
- A missing premise in a named tier is a test failure, never a skip. A quarantined claim stays visible as `test.todo` and runs with `bun test --todo`.
- Browser tests use Playwright Test in `*.e2e.ts`. Each subject keeps `playwright.config.ts` beside its `vite.config.ts`, imports the shared Chromium flags from `scripts/chromium.ts` and sets a `globalTimeout` just above its measured run.
- Run one subject with `playwright test -c <subject>` (for example, `playwright test -c examples/loading-screen`); run the wide browser tier with `bun run test:browser`.
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
| Steady JavaScript allocation | Node/macOS | The V8 sampling heap profiler over the GPU-backed composed subject in a Node child process with Dawn's `webgpu` binding |
| GPU resources released | GPU | A counting wrapper over the real Dawn device |
| Beyond the suite | oracle | Heap-snapshot diffs, CDP tracing, `measureUserAgentSpecificMemory`, WebGPU `timestamp-query` |

### CI coverage

CI runs static gates and all tests except `*.gpu.test.ts` on GitHub-hosted Ubuntu and macOS, plus the GPU, Cargo, Node and browser tiers on hosts with their required tools. The Node tier runs on `macos-15` so the GPU-backed Node subject can use Dawn's `webgpu` binding; its test command remains unchanged. The browser run discovers subject directories from `vite.config.ts` or `playwright.config.ts` under `examples/` and `scripts/`, requires both files for each discovered subject, and runs each subject; an `*.e2e.ts` file outside a subject fails as an orphan. The display-bound allocation oracle is run manually on its declared display seat. These jobs do not qualify a Windows runner or native packaging.

## Examples

Shallot is built from both ends, as a double loop. An example, built as a user would build it, is the outer loop: what it can't do is a gap in the engine. The module that owns the gap is the inner loop: the gap is fixed and proved there, never in the example.

An example is an app in the shape of a user's project once copied out (`shallot add` writes its `package.json`), under `examples/<name>/`. Its manifest's `problem` field names what its user wants, as a piece of a game.

- An example answers one problem a user meets while making any game. Only `first-person` composes many.
- Every public promise the engine keeps is used by some example whose answer depends on it. A promise no example uses is a gap.
- No example only repeats what another shows. Of two that would show the same promises, the one kept shows the engine at its best.
- An example has a test that fails when its answer breaks. An example without one is removed.
- An example imports only published names; its tests use the same public seams.
- An example lives in the `examples/` of the repository whose promise its answer is about: the engine's or a package's.

## Heavy work

Heavy computation runs in WASM or on the GPU; TypeScript coordinates it and runs lightweight gameplay. TypeScript that needs runtime-specific tricks to meet a performance target belongs in WASM or on the GPU.

## Dependencies and releases

`MIGRATION.md` tells a user of the last stable release what breaks between it and the current tree, and what to write instead, in the user's terms. A change to something that release did not ship is not an entry; an entry is rewritten, not appended to, when the tree moves again. Measurements, design reasons, progress and test internals belong in commits and this guide, not there.

- A project can use Shallot released, staged or live: `bun add @dylanebert/shallot`; `bun pm pack` here, then `bun add --no-save <tarball>` in the project, with `--dev` when the project declares Shallot as a dev dependency, since the overlay otherwise targets `dependencies` and silently changes nothing; or `bun link` here, then `bun link @dylanebert/shallot` in the project. `bun install` returns a staged or linked project to its manifest pin only when that Shallot range resolves on npm. While `0.10.0` is unpublished, `^0.10.0` cannot restore, and it does not match `0.10.0-next.1`; use a published range such as `^0.9.5` for a staged overlay and restore. Under a live link, Vite's `resolve.dedupe` and the Bun preload redirect `typegpu` and its subpaths imported by name to the project's TypeGPU copy; a copy reached under another package name is outside this promise.
- The Bun entry is a factory, not an import side effect. In `tests/preload.ts`, import `plugin` from `bun` and `shallot` from `@dylanebert/shallot/bun`; register `plugin(shallot({ root: import.meta.dir }))`. The plugin walks up from the preload directory to the nearest `package.json`, not from Bun's working directory. Configure `[test] preload = ["./tests/preload.ts"]` in `bunfig.toml`; `shallot add` writes this setup into copied projects.
- A pin is the last verified version. Update a pin everywhere it appears in one commit; `check-pins` fails on drift. Until the release-candidate bump, move a pin only to fix a named defect.
- The package self-references by its name, so its source and examples import it by name without a dependency entry. `@types/node` and `@webgpu/types` are runtime dependencies, because `types` points at source.
- A link doesn't prove what ships; a packed tarball installed in a scratch project does. Changes to the CLI, manifest, dependencies, runtime or native shell require that test.
- `main` is development and carries the next unpublished version; the commit after a publish moves it forward, because Bun keys an installed package by name and version and a tarball with a published version can silently fail to stage or restore. A commit on it is not a release. Publish a version from a commit whose `package.json` contains it, with one `v<version>` tag per published version. Pushing the tag runs `release.yml`, which creates the GitHub Release and attaches the native archives; it does not publish to npm. Run `bun publish` for a stable version (npm `latest`; GitHub Release marked Latest) or `bun publish --tag next` for a prerelease (npm `next`; GitHub Release marked Pre-release). Keep one prerelease identifier per release line: Semver orders identifiers alphabetically, so changing identifiers can make a later prerelease sort below an earlier one.
- A breaking change updates every dependent it breaks in the same change. A removal or reshape strips what a dependent can no longer use rather than rebuilding it, so no dependent keeps a pattern the engine has left. Rebuilding in the new shape, additive features and hardening reach dependents once the engine side is settled, not through each intermediate state.
- To retire a module, example or tool, tag its last commit `archive/<name>`, then delete it in a commit that says why. There is no archive directory.
