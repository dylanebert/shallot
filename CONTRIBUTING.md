# Contributing

For modifying the engine. For using Shallot, see the [README](README.md). Each API is documented in the JSDoc beside it. This page covers what the code, the CLI and failing checks don't.

## Architecture

Shallot is a WebGPU game engine for TypeScript, built on an entity component system (ECS). An entity is an id. A component is plain typed data stored per entity. A system is a function the scheduler runs every frame, in ordered groups such as `fixed` for gameplay on a fixed tick and `draw` for presentation. A scene is a file of entities and their components, loaded into the same data.

A plugin is how behavior gets into a game: a named bundle of components, systems and lifecycle hooks (`initialize`, `warm`, `dispose`), plus the plugins it needs. A project lists its plugins in `shallot.json`, and `build()` composes them into an app. Games and tests run the same composed app on the same stepped clock.

### Device and world ownership

Every built app requires a WebGPU device, even without rendering plugins. Acquisition requests the engine's required features and the active plugins' feature union, grants preferred features where available, and reports missing capabilities with their cause. It does not forward adapter maxima. The current storage-binding requirement is ten per shader stage; the accepted eight-binding target and removal of mandatory BGRA storage still await rendering and presentation changes.

Apps can share a device by passing it through `config.device`; omitting it acquires a device for that build. Each app owns a separate `State`, TypeGPU root, component storage, typed resources, GPU registries, tables and readback pool. Builds serialize registration and warm-up, but completed apps coexist. Disposing one releases its resources, not the shared device or another app's data.

A component declares named typed fields, not storage or GPU residency. Its world owns one eid-indexed typed-array column per field. Columns grow by doubling to cover the entity high-water mark and do not shrink during play. Resolve storage once with `state.of(Component)` and retain the returned accessors, not a column array that growth can replace. Schema-compatible hot reload reuses that world's storage; an incompatible schema requires a rebuild.

Setters and bulk `write(eids, source)` mark changed entities for that field. Every table consuming those marks reads them at the frame's upload point, before they clear; writes after it remain for the next frame. Direct column writes must publish the same marks. There are no per-component change ticks.

Keep plugin state in `state.resource` and use `state.gpu` for the world's GPU context, not module globals. Buffers and textures created through that context or its TypeGPU root are owned automatically; register other raw GPU allocations with `state.own` and non-GPU cleanup with `state.onDispose`. Legacy component accessors and `Compute` still resolve an ambient world during callbacks; they are compatibility syntax, not process-owned storage, and their removal belongs to the authoring-syntax work.

### GPU tables

A GPU consumer declares its rows, one TypeGPU struct record per row, and how the records are filled through `state.table`. Pipelines bind consumer-shaped tables, not individual authoring fields. Group records by access, update frequency and lifetime; a pass needing only a few fields reads a narrower table, not a second engine-wide layout.

- Rows are stable dense slots allocated from a free list, not entity ids. Shaders starting from an eid opt into the uploaded eid-to-row map; zero means absent and other entries encode row + 1. Draws and dispatches iterate a compact active-row or instance list, never the sparse eid range.
- CPU fill uses bound component columns or bulk byte ranges, including WASM memory, not a per-row JavaScript callback. A compute producer can derive a table's records from uploaded inputs. GPU-only tables have no CPU backing or record upload; a GPU pass owns their writes.
- Unchanged records upload nothing. Changed records upload the contiguous range spanning the changed rows with `writeBuffer`; mapped-staging scatter was measured and not retained. Uploads inside an open frame encoder use a staging copy to preserve command order after buffer growth.
- Growth replaces the buffer, preserves its contents and changes its generation. Consumers subscribe or track generations to rebuild bind groups when a buffer changes, not when its contents change. Use runtime-sized shader arrays so capacity growth does not require recompiling the pipeline. Table capacity is bounded by device buffer and storage-binding size limits, with refusal naming the cause.
- Fixed simulation writes precede the head-of-draw upload and GPU passes. Engine GPU work records into the renderer's frame encoder rather than opening a separate steady-play submission.

TypeGPU is the GPU language for the engine and extensions: it describes records, shader types and typed bind-group layouts. Raw WGSL uses its sanctioned escape hatch. Steady updates use raw handles and byte ranges rather than allocating object-form TypeGPU writes. TypeGPU upgrades remain subject to the pin rule.

### Readback

Steady play reads nothing back. GPU values that size later GPU work stay on the GPU as indirect dispatch or draw arguments; diagnostics such as overflow counts clamp safely without CPU observation and are read only on request. Shipped shaders do not use TypeGPU's shader `console.log`, which triggers readback.

`probeBuffer` and `probeTexture` request one owned snapshot of a world-owned resource, optionally encoding the work to capture before its copy. Each request submits its copy and returns independent bytes stamped with the copy-time frame and fixed tick, not the arrival time. Staging is pooled per world and released after `state.readback.maxUnusedFrames` idle frames. No request means no mapping or readback allocation; a request still allocates because WebGPU mapping creates a promise and a mapped buffer that unmapping detaches. There is no continuous-readback mode.

Arrival timing and GPU floating-point results vary across adapters. A deterministic plugin never consumes readback in `fixed`; one that does declares `deterministic: false`. This is a declaration proved by replay, not runtime enforcement of access to retained bytes.

### GlobalTransform

`GlobalTransform` is derived world placement, never scene-authored data. CPU gameplay and physics queries read its fixed-tick columns; rendering reads the engine's interpolated dense rows through `globalTransformTable(state)`. GPU history and interpolation become resident only when requested by a reader. A teleport uses `state.teleport(eid)` to discard interpolation across the discontinuity.

Exactly one producer provides an entity's `GlobalTransform`, declared through the `provides` trait and enforced by producer exclusion: `Transform` for authored placement, a body for simulation, or a domain's skeleton or attachment. Readers never treat `Transform` as the shared world-space result. Producers write through world storage; they do not write the renderer's interpolated output.

Hierarchy is not an engine structure. A domain needing relative placement owns the relation and derives world placement from it. A general attachment relation enters core only when two examples need the same one.

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
bun test ./examples/first-person/src/demo.node.ts  # a Node-tier file your change selects, by path
bun test ./diagnostics/first-person-allocation/allocation.oracle.ts # named display oracle (manual)
bun test --todo                       # run quarantined test.todo entries, if any
bun run format                        # biome, scene formatter and examples index
```

Wide runs confirm a final candidate before it lands or is released, and CI runs them on every push; they are not the loop. While iterating, run the files your change selects:

```bash
bun test $(find ./src ./examples ./scripts ./diagnostics -name '*.node.ts')  # the whole Node tier
bun run test:browser                  # every browser subject
```

`check-imports` stays red while modules live in `src/transitional/`; their `// Destination:` lines name the migration owners. The current `core/rendering/view.ts` → `core/input` sibling import is a separate unresolved violation, not a permitted dependency. For unrelated work, compare import reds with main: report unchanged violations and continue, but stop on a new or changed violation. The gate remains red and is never skipped.

## Verification

A module's promises are tested beside the module and through the examples that use it. Each test name states the claim. Tier ceilings are categorical backstops against hangs and runaway work, not performance targets; design each test to run far below its backstop on the hosted runner.

`scripts/test-tiers.ts` is the one home for bounded tier ceilings and the build/startup backstop.

| Tier | Timeout backstop |
|---|---|
| Cheap `*.test.ts` | `CEILING.cheap` |
| GPU `*.gpu.test.ts` | `CEILING.gpu` |
| Node `*.node.ts` | `CEILING.node` |
| Browser subject | `CEILING.browser` for the whole subject |
| Build and server startup | `CEILING.startup` |
| Manual `*.oracle.ts` | Unbounded; run by path |

A test that needs more is split or moves to a heavier tier. Each GPU and Node file declares its tier once with `setDefaultTimeout(CEILING.gpu)` or `setDefaultTimeout(CEILING.node)`, importing `CEILING` from that module. `check-timeouts` rejects literal timeout declarations, repeated per-test defaults, missing tier headers and per-subject browser tailoring. It also checks the package test command's timeout against `CEILING.cheap`. Every bounded GPU wait fits inside its test's ceiling.

- `bun run test` discovers `*.test.ts` files, including `*.gpu.test.ts`, with the `CEILING.cheap` default timeout per test. Individual test timeouts override it. Bare `bun test` discovers the same files with Bun's default timeout. Hosted jobs without a device exclude `*.gpu.test.ts` by pattern; the macOS GPU job runs `bun test gpu.test` through the compile-reporting oracle. Node tests remain named `*.node.ts` files; the Node tier runs on macOS with Dawn's `webgpu` binding available to child processes. Oracles run by path; hosted qualification also runs the compile-time reporting oracles.
- Root `bunfig.toml` loads Shallot's Bun plugin so the engine tests receive the same TypeGPU transform as project test preloads.
- Rust suites run directly with Cargo:

  ```bash
  cargo test -p shallot-audio
  cargo test -p shallot-physics
  ```
- A missing premise in a named tier is a test failure, never a skip. A quarantined claim stays visible as `test.todo` and runs with `bun test --todo`.
- Browser tests use Playwright Test in `*.e2e.ts`. Each subject keeps `playwright.config.ts` beside its `vite.config.ts` and spreads `BROWSER_CONFIG` from `scripts/chromium.ts`. That shared configuration derives its global backstop from `CEILING.browser` and leaves Playwright's per-test default unchanged. A subject's Vite server spreads `WEB_SERVER_CONFIG`, whose build/startup backstop derives from `CEILING.startup`. Do not restate or tailor these values in a subject.
- Run one subject with `playwright test -c <subject>` (for example, `playwright test -c examples/loading-screen`); run the wide browser tier with `bun run test:browser`.
- The GPU tier runs its files serially on one device, not concurrently. GPU tests hold only small device compositions. Each file acquires its in-process device and warms its independent worlds once in `beforeAll`; each test times only its own work. Do not share compiled pipelines across worlds. Device-loss claims prepare the separate devices they destroy during that file's compile step.
- When touching a built-in composition, add an exact counted claim for the native pipelines it compiles. Compile time is reported, never asserted: `bun test ./scripts/compile.oracle.ts` reports composition durations, and `bun scripts/gpu-compile.oracle.ts` runs the serial GPU files and reports each file's `beforeAll` duration.
- Tests that build the full default renderer or the Physics kernel belong in Node, especially multi-world isolation tests. The Node tier runs on the same hosted macOS device.
- A device-backed check records its adapter classification. Only a hardware claim requires a real adapter; browser GPU observations accept software adapters, and browser tests request no adapter themselves.
- Display-bound measurements require a declared monitor and take its keyboard and cursor.
- Tests that make gameplay assertions step the composed app's clock. Simulation state lives in registered components or behind a snapshot, restore and hash hook. Gameplay runs in `fixed` from per-tick actions, presentation runs in `draw`, and `local` components are excluded from the hash. Runs are deterministic within one runtime and engine version; across versions, the hash detects divergence.
- Test a frame at the cheapest level that shows the defect: CPU state, GPU readback, browser pixels, then a person. Choose the capture by the claim:
  - An engine frame: read back a texture the test owns with `probeTexture`.
  - Page composition, such as overlays, posters and canvas reveal: step the app with `build()` and `state.step(dt)`, take a Playwright page screenshot, and assert semantic regions of it.
  - A running app's canvas: `captureFrame`, which reads during the next frame the loop presents.
- A WebGPU canvas reads as transparent black once its frame is presented, though the page still displays that frame: a canvas read works only inside the presenting frame, and a page screenshot works after it.
- Add a golden image only for a defect no cheaper level shows, and never update one to make it pass.
- Steady play allocates nothing. A memory test creates and disposes its subject, verifies memory returns to baseline, and fails on a deliberately leaking control. Retention is a separate measurement taken after GC; sampler allocation sites are diagnostics, not results.
- Speed and performance are proved by counted work and same-machine ratio oracles. Timings are measured on real hardware, labeled with it, and reported, never asserted.
- An oracle is a separately named measurement, not a bounded suite claim. Run it when the claim or its tool changes, record the result in that commit, and rerun it only for a specific doubt.
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

`MIGRATION.md` tells a user of the last stable release what breaks between it and the current tree, and what to write instead, in the user's terms. A change to something that release did not ship is not an entry; an entry is rewritten, not appended to, when the tree moves again. For the 0.10 line, audit against `v0.9.5:packages/shallot`, the package in the tagged monorepo. Measurements, design reasons, progress and test internals belong in commits and this guide, not there.

- A project can use Shallot released, staged or live: `bun add @dylanebert/shallot`; `bun pm pack` here, then `bun add --no-save <tarball>` in the project, with `--dev` when the project declares Shallot as a dev dependency, since the overlay otherwise targets `dependencies` and silently changes nothing; or `bun link` here, then `bun link @dylanebert/shallot` in the project. `bun install` returns a staged or linked project to its manifest pin only when that Shallot range resolves on npm. While `0.10.0` is unpublished, `^0.10.0` cannot restore, and it does not match `0.10.0-next.1`; use a published range such as `^0.9.5` for a staged overlay and restore. Under a live link, Vite's `resolve.dedupe` and the Bun preload redirect `typegpu` and its subpaths imported by name to the project's TypeGPU copy; a copy reached under another package name is outside this promise.
- The Bun entry is a factory, not an import side effect. In `tests/preload.ts`, import `plugin` from `bun` and `shallot` from `@dylanebert/shallot/bun`; register `plugin(shallot({ root: import.meta.dir }))`. The plugin walks up from the preload directory to the nearest `package.json`, not from Bun's working directory. Configure `[test] preload = ["./tests/preload.ts"]` in `bunfig.toml`; `shallot add` writes this setup into copied projects.
- A pin is the last verified version. Update a pin everywhere it appears in one commit; `check-pins` fails on drift. Until the release-candidate bump, move a pin only to fix a named defect.
- The package self-references by its name, so its source and examples import it by name without a dependency entry. `@types/node` and `@webgpu/types` are runtime dependencies, because `types` points at source.
- A link doesn't prove what ships; a packed tarball installed in a scratch project does. Changes to the CLI, manifest, dependencies, runtime or native shell require that test.
- `main` is development and carries the next unpublished version; the commit after a publish moves it forward, because Bun keys an installed package by name and version and a tarball with a published version can silently fail to stage or restore. A commit on it is not a release. Publish a version from a commit whose `package.json` contains it, with one `v<version>` tag per published version. Pushing the tag runs `release.yml`, which creates the GitHub Release and attaches the native archives; it does not publish to npm. Run `bun publish` for a stable version (npm `latest`; GitHub Release marked Latest) or `bun publish --tag next` for a prerelease (npm `next`; GitHub Release marked Pre-release). Keep one prerelease identifier per release line: Semver orders identifiers alphabetically, so changing identifiers can make a later prerelease sort below an earlier one.
- A breaking change updates every dependent it breaks in the same change. A removal or reshape strips what a dependent can no longer use rather than rebuilding it, so no dependent keeps a pattern the engine has left. Rebuilding in the new shape, additive features and hardening reach dependents once the engine side is settled, not through each intermediate state.
- To retire a module, example or tool, tag its last commit `archive/<name>`, then delete it in a commit that says why. There is no archive directory.
