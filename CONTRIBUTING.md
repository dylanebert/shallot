# Contributing

For modifying the engine; for using it, see the [README](README.md). API contracts live in JSDoc. This guide holds the rules the code, CLI and checks don't explain.

## Architecture

### Device and world ownership

- Every built app requires WebGPU. Acquire the composition's required capabilities, grant preferred features where available, and refuse with the cause; never forward adapter maxima.
- Each build acquires a device unless supplied `config.device`. Apps may share that device; each owns its world storage and allocations. Builds serialize registration and warm-up; completed apps coexist. Disposing one releases its resources, not the device or a sibling's data.
- Components declare fields, not storage or GPU residency. Each world owns eid-indexed typed columns that double to cover the entity high-water mark and never shrink during play. Resolve accessors once with `world.storage(Component)`; growth replaces arrays, not accessors. Compatible hot reload reuses storage; incompatible schemas require rebuilding.
- Field changes are frame-scoped: all consuming tables read them at upload before they clear; later writes reach the next frame. Raw column writes publish only after `markChanged(eid)` on the owning field or lane; setters and bulk writes mark automatically. There are no per-component change ticks.
- Keep plugin state in `world.resource`, not module globals, and use `world.gpu`. Its context and TypeGPU root own buffers and textures automatically; register other GPU allocations with `world.own` and other cleanup with `world.onDispose`. Field, resource and GPU access resolves through the World passed to the callback.

### GPU tables

GPU consumers bind tables, not authoring fields. Each table has one TypeGPU struct per row, shaped by access, update frequency and lifetime. A pass needing fewer fields reads a narrower table, not a second engine-wide layout.

Built-in rendering layouts fit WebGPU's default eight storage-buffer bindings per shader stage. Count declared visibility across all bind groups, including bindings a shader does not read. The standard composition requires `indirect-first-instance` and `rg11b10ufloat-renderable`. Qualification supplies `config.device` requested with only the composition's required features and default limits, and records the resulting capabilities. Layout repairs preserve precision and capacity; measure uploads, passes and dispatches on named hardware.

- Rows are stable dense slots from a free list. Draws and dispatches read compact active-row or instance lists, never the sparse eid range. Eid-based lookups opt into the uploaded map.
- Fill is bulk: bound columns, byte ranges or compute, never a per-row JavaScript callback. GPU-only tables have no CPU record source or upload.
- Unchanged records upload nothing; changed records upload their spanning range. Uploads preserve command order across buffer growth. Growth preserves contents and changes generation; consumers rebuild affected bind groups, not pipelines. Use runtime-sized shader arrays. Device buffer limits bound table capacity; refusal names the cause.
- Fixed writes precede head-of-draw upload and GPU passes. Engine work uses the renderer's frame encoder, not a separate steady-play submission.

TypeGPU is the engine and extension GPU language; raw WGSL uses its escape hatch. Steady updates use raw handles and byte ranges, not allocating object-form writes. Buffer generations, row addressing and upload APIs are documented beside [GpuTable](src/engine/ecs/table.ts).

### Readback

Steady play reads nothing back. GPU work sizes later GPU work through indirect arguments; diagnostics clamp safely without CPU observation and are read only on request. Shipped shaders never use TypeGPU's shader `console.log`.

[Probes](src/engine/runtime/probe.ts) return one independent snapshot of a world-owned resource, stamped with the copy-time frame and fixed tick, not arrival time. Staging is pooled per world and expires after declared idle frames. Requests allocate; without requests, there is no mapping or readback allocation. There is no continuous-readback mode.

A deterministic plugin never consumes readback in `fixed`; one that does declares `deterministic: false`. Replay proves the declaration; runtime guards cannot enforce access to retained bytes.

### GlobalTransform

`GlobalTransform` is derived world placement, never authored. Gameplay and physics queries read its fixed-tick columns; rendering reads the engine's interpolated table, resident only when requested. Teleports discard interpolation across the discontinuity.

A placement producer requires `GlobalTransform`, adding it when missing without removing it on detachment: `Transform` for authored placement, or a domain's body, skeleton or attachment. Physics warns once per entity carrying both `Body` and `Transform`, since both write its `GlobalTransform`. Producers write world storage, never the interpolated output; readers never treat `Transform` as the shared world-space result.

Hierarchy belongs to the domain deriving placement, not the engine. A general attachment relation enters core only when two examples need the same one.

## Layout

Layers run from `engine` outward through `core`, `standard` and `extras`; dependencies point inward. Outer layers make more choices a game can replace. Extras admit features after a stable release cycle as external packages.

- Each module owns one useful responsibility completely; split, fix or remove one that doesn't.
- Game modules never import tooling (`project`, `cli`, `native`, `types`). Core, standard and extras modules never import siblings in their layer. Physics never imports rendering.
- A game module's `index.ts` is public; its other files are internal. Modules registering systems or resources define plugins; others export data and functions. Layer indexes re-export modules, and the root barrel re-exports every game layer. `standard/index.ts` also sets the default plugins. `package.json` declares public subpaths.
- `transitional` modules declare their destination and migration owner.
- Provider-specific observation belongs in optional application integrations, outside Shallot.

### Core and standard

Name modules for what they own, not their technique. Core and standard use the same noun and matching subpaths, such as `core/rendering` at `/rendering` and `standard/rendering` at `/standard/rendering`. Export names describe the implementation.

### Rendering

`core/rendering`'s `RenderingPlugin` owns cameras, shared views, canvas binding, projection, view and frame uniforms, capture, light components and frame ordering anchors. It knows no meshes or materials. Scene effects run before `OverlaySystem`, overlays between it and `PresentationSystem`, and presentation after that anchor.

`CorePipelinePlugin` adds each view's clear, depth and multisampled color targets, resolve, opt-in `DepthPrepass` and `PickingPrepass` lanes, and `RenderPhases`. Core opens the prepass and one main render pass per view; renderers record opaque then transparent work into the main pass with core's formats and sample count. A renderer contributes to an effect's depth or picking input only by recording into that prepass lane. Transparent ordering is by renderer, not by object across renderers: standard submits GPU-driven indirect draws per surface.

`CorePipelinePlugin` also owns the tonemapping pass. `Tonemapping` selects the operator (TonyMcMapface by default); `ColorGrading` grades the image, and presentation encodes it for the screen. `TonemappingMethod.None` skips the operator for display-ready linear images, not grading or encoding. Effects are separate passes. Register per-camera passes in `EffectPasses` before or after tonemapping; before passes read linear HDR, after passes read encoded display-referred intermediates. `CustomPresentation` replaces the built-in presentation for a camera.

`core/mesh` owns mesh data, GPU storage, built-in primitives and `MeshInstance`, independent of rendering. `standard/rendering` builds the clustered forward mesh pipeline over both core modules: registered surfaces and materials, `MeshMaterial`, `StandardMaterial`, light packing, culling, indirect draws, backgrounds and shadow passes. It records into core's phases and includes `CorePipelinePlugin`; effects belong in extras, not standard. `MeshRenderPlugin` packs mesh instances and registers `MeshMaterial`.

A custom standard surface uses `surfaceLayout`, `VsIn`, `vsPatchSchema` and `fsCtxSchema`, and registers with `registerSurface`; `engineLayout` supplies the shared frame/view bindings and `lit` supplies the standard lighting response. A custom background uses `backgroundLayout`, `BackgroundContext` and `registerBackground`. These extend standard's pipeline, not a second standard pipeline.

## Commands

A `shallot` command does only what Shallot knows. It may invoke project commands, but never owns processes Vite, Bun or Playwright own. Web commands delegate to Vite; native dev/build add the shell and native preview launches the existing build.

Package scripts are package-manager lifecycle hooks or verbs every Shallot project has. Other tools run by path.

```bash
bun run build                    # audio WASM, dist/vite.js, physics kernel
bun run check                    # static gates; before every push
bun run test                     # cheap and GPU files; requires a device
bun run test --path-ignore-patterns '**/*.gpu.test.ts' # no-device tier
bun test gpu.test                # GPU tier
bun test ./examples/first-person/src/demo.node.ts # selected Node file
bun node_modules/playwright/cli.js test -c examples/loading-screen # selected browser subject
bun test --todo                  # quarantined claims
bun run format                   # biome
```

Iterate on selected files or subjects. Wide runs confirm the final candidate before landing or release; CI runs them on every push. The wide Node command is in [CI](.github/workflows/test.yml); `bun run test:browser` runs all browser subjects. Manual oracles run by path.

The import gate runs its Bun assertions with `--todo`. Only the existing debts named in those assertions are todos; all other findings fail normally. A passing todo fails until its scope is retired. These scopes leave with their migrations, not as a framework for future exceptions. Future stages preserve import boundaries at each landing.

## Verification

Test module promises beside their owners and through public examples. Test names state claims. [Tier ceilings](scripts/test-tiers.ts) are hang/runaway backstops, not performance targets; design tests far below them. Split work that needs more or move it to a heavier tier.

| Tier | Files / budget |
|---|---|
| Cheap | `*.test.ts`; `CEILING.cheap` |
| GPU | `*.gpu.test.ts`; `CEILING.gpu` |
| Node | `*.node.ts`; `CEILING.node` |
| Browser | `*.e2e.ts`; `CEILING.browser` per subject |
| Build/startup | `CEILING.startup` |
| Manual oracle | `*.oracle.ts`; unbounded, run by path |

GPU and Node files call `setDefaultTimeout` once with `CEILING.gpu` or `CEILING.node`, importing `CEILING` from `scripts/test-tiers.ts`. Bounded GPU waits fit that ceiling. `bun run test` discovers cheap and GPU files with the cheap default; bare `bun test` uses Bun's default. Node and oracle files run explicitly. Root `bunfig.toml` supplies the TypeGPU transform, as project preloads do.

- Missing tier premises fail, never skip. Quarantine uses `test.todo`, runnable with `--todo`. Known failures stay failing until fixed.
- Browser subjects keep Playwright and Vite configs together. Spread `BROWSER_CONFIG` and `WEB_SERVER_CONFIG` from [scripts/chromium.ts](scripts/chromium.ts); never restate or tailor budgets. Playwright's per-test default remains unchanged.
- GPU files run serially on one device, with small compositions. Acquire and warm each file's independent worlds once in `beforeAll`; tests time only their work. Never share compiled pipelines across worlds. Prepare devices a loss claim destroys during that compile step.
- Full default-renderer or Physics-kernel compositions belong in Node, especially multi-world isolation claims. A built-in composition change adds an exact native-pipeline count. Compile durations are reported by `scripts/compile.oracle.ts` and `scripts/gpu-compile.oracle.ts`, never asserted.
- Device-backed checks record adapter classification; hardware claims require a real adapter. Browser observations accept software adapters and request no adapter themselves. Display-bound measurements name a monitor and take its keyboard and cursor.
- Gameplay assertions step the composed app. Simulation lives in registered components or snapshot/restore/hash hooks. `fixed` reads per-tick actions; `draw` presents; `local` components are excluded from hashes. Determinism holds within one runtime and engine version; hashes detect divergence across versions.
- Frame evidence uses the cheapest observable: CPU state, GPU readback, browser pixels, then a person. An engine's final frame uses `attachTexture` and `captureTexture`; other owned textures use `probeTexture`. Page composition uses a stepped app and semantic screenshot regions; a running canvas uses `captureFrame` inside the presenting frame. Canvas readback after presentation is transparent black; page screenshots still work.
- Goldens cover only defects cheaper evidence cannot show; never update them to make a check pass.
- Steady play allocates nothing. Memory claims create and dispose their subject, return to baseline and fail on a deliberately leaking control. Measure retention after GC; sampler sites are diagnostics, not results.
- Performance uses counted work and same-machine ratio oracles. Real-hardware timings name the hardware and are reported, never asserted. Oracles are separate measurements: run when their claim or tool changes, record results in the commit, and repeat only for a named doubt.

| Claim | Tool / tier |
|---|---|
| Deterministic work and owned counts | Stepped world assertions, engine counters, `FinalizationRegistry` under `Bun.gc(true)` / cheap |
| WASM memory | Feature-gated counting allocator and `memory.buffer.byteLength` / Cargo |
| Native heap per step | `dhat`, one profiler per process / Cargo |
| Steady JavaScript allocation | V8 sampling heap profiler over a GPU-backed Node child with Dawn's `webgpu` / Node, macOS |
| GPU release | Counting wrapper over a real Dawn device / GPU |
| Beyond the suite | Heap snapshots, CDP tracing, `measureUserAgentSpecificMemory`, GPU timestamps / oracle |

### CI coverage

[CI](.github/workflows/test.yml) owns host and tier commands. Rust suites run with `cargo test -p shallot-audio` and `cargo test -p shallot-physics`. Display-bound allocation remains manual. This does not qualify Windows or native packaging.

## Code

Names and techniques come from the domain's precedent: Bevy for the ECS and rendering where they carry meaning in Shallot, WebGPU and TypeGPU for the GPU, glTF for assets. Each rendering divergence names its referent or the Bevy structure Shallot lacks in `strategy/shallot/referents.md`'s `bevy` ledger in the harness. A name means one thing across the tree, and a noun beats a participle ([TigerStyle](https://github.com/tigerbeetle/tigerbeetle/blob/main/docs/TIGER_STYLE.md), Naming Things).

One fact has one name, one producer and one path. A second spelling or a forwarding wrapper goes.

JSDoc states the contract the signature cannot: units, ownership, lifetime, frame timing and what is refused. It carries no `@example`; examples live in `examples/`. A `//` comment says why the code is as it is. Neither restates the code or its history ([Google TypeScript Style](https://google.github.io/styleguide/tsguide.html#comments-documentation), Comments and documentation).

## Code-authored worlds

Write a one-off entity straight out: create, then adds. Where content repeats, use a typed table and a function that creates one entity and returns its eid. Keep needed eids as variables or return values, never names. See `route` in [first-person's Demo](examples/first-person/src/demo.ts).

## Examples

Examples are the outer loop: a user's problem exposes an engine gap; the owning module fixes and proves it, never the example. Each lives in its promise's repository, under `examples/<name>`, in the shape of a copied user project. Its `index.html` carries one `<meta name="description">` describing the example.

- One problem per example; only `first-person` composes many.
- Every public promise has an example whose answer depends on it.
- Keep no duplicate demonstrations; retain the one showing the engine at its best.
- Each example has a check that fails when its answer breaks, or is removed.
- Examples and their checks use only published names and public seams.

## Heavy work

Heavy computation runs in WASM or on the GPU; TypeScript coordinates it and runs lightweight gameplay. TypeScript needing runtime-specific tricks to meet a performance target moves to WASM or the GPU.

## Dependencies and releases

`MIGRATION.md` tells users of the last stable release what breaks and what to write instead. Rewrite entries as the tree changes; exclude changes that stable release never shipped. For 0.10, audit `v0.9.5:packages/shallot`. Measurements, design reasons, progress and test internals belong in commits and this guide, not migration entries.

### Using the package

- Released: `bun add @dylanebert/shallot`.
- Staged: `bun pm pack` here, then `bun add --no-save <tarball>` in the project. Add `--dev` for dev dependencies; otherwise the overlay silently targets the wrong section.
- Live: `bun link` here, then `bun link @dylanebert/shallot` in the project. Vite dedupe and Bun preload route named `typegpu` imports and subpaths to the project's copy, not copies under other package names.

`bun install` restores the manifest pin only if its range resolves on npm. An unpublished `^0.10.0` neither restores nor matches `0.10.0-next.1`; use a published range for staged overlays and restore.

The Bun entry is a factory, not an import side effect. In `tests/preload.ts`, import `plugin` from `bun` and `shallot` from `@dylanebert/shallot/bun`, then register `plugin(shallot({ root: import.meta.dir }))`. Configure `[test] preload = ["./tests/preload.ts"]` in `bunfig.toml`. Root resolution walks from that preload to the nearest `package.json`, not from cwd; `shallot add` writes this setup.

### Plugin packages

A plugin package declares `@dylanebert/shallot` as a peer: the range is its compatibility, and `shallot()` shares the project's engine instance only with packages that declare it. It carries the npm keyword `shallot-plugin`, and its README shows the install and the one import and `plugins` entry that enable it.

### Changing and publishing

- A pin is the last verified version. Update every occurrence in one commit; until the release-candidate bump, move it only for a named defect.
- Source and examples self-reference the package name without a dependency entry. Keep `@types/node` and `@webgpu/types` as runtime dependencies because public types point at source.
- A link cannot prove what ships. CLI, manifest, dependency, runtime or native-shell changes require a packed tarball installed in a scratch project.
- Breaking changes update every affected dependent in the same landing. Removals and reshapes strip unusable code; rebuilding and additive/hardening work follow the settled engine, not each intermediate state.
- `main` carries the next unpublished version; the commit after publishing advances it. Bun's name/version identity can silently defeat staging or restoring a tarball with a published version. A commit is not a release.
- Publish from a commit whose `package.json` contains that version, with one `v<version>` tag. The tag runs [release.yml](.github/workflows/release.yml) to create the GitHub Release and native archives, not publish npm. Stable: `bun publish`, npm `latest`, GitHub Latest. Prerelease: `bun publish --tag next`, GitHub Pre-release. Keep one prerelease identifier per release line; identifiers sort alphabetically.
- Retire a module, example or tool by tagging its last commit `archive/<name>`, then deleting it in a commit naming the failure. There is no archive directory.
