# Contributing

For modifying the engine; for using it, see the [README](README.md). API contracts live in JSDoc. This guide holds the rules the code, CLI and checks don't explain.

## Architecture

### Device and world ownership

- Every built app requires WebGPU.
- A build acquires the composition's required capabilities, grants preferred features where available, and refuses with the cause; it never forwards adapter maxima.
- Each build acquires a device unless supplied `config.device`; apps may share that device.
- Each app owns its world storage and allocations; disposing one releases its resources, not the device or a sibling's data.
- Builds serialize registration and warm-up; completed apps coexist.
- Components declare fields, not storage or GPU residency.
- Each world owns eid-indexed typed columns that double to cover the entity high-water mark and never shrink during play.
- Accessors resolve once with `world.storage(Component)`; growth replaces arrays, not accessors.
- Compatible hot reload reuses storage; an incompatible schema requires rebuilding.
- Field changes are frame-scoped: every consuming table reads them at upload before they clear, and later writes reach the next frame.
- A raw column write publishes only after `markChanged(eid)` on its field or lane; setters and bulk writes mark automatically. There are no per-component change ticks.
- Plugin state lives in `world.resource`, never module globals.
- GPU allocations go through `world.gpu`, whose context and TypeGPU root own buffers and textures; other GPU allocations register with `world.own`, other cleanup with `world.onDispose`.
- Field, resource and GPU access resolves through the World passed to the callback.

### GPU tables

- GPU consumers bind tables, not authoring fields.
- A table has one TypeGPU struct per row, shaped by access, update frequency and lifetime; a pass needing fewer fields reads a narrower table, not a second engine-wide layout.
- Rows are stable dense slots from a free list.
- Draws and dispatches read compact active-row or instance lists, never the sparse eid range; eid-based lookups opt into the uploaded map.
- Fill is bulk: bound columns, byte ranges or compute, never a per-row JavaScript callback. GPU-only tables have no CPU record source or upload.
- Unchanged records upload nothing; a changed record uploads its spanning range.
- Uploads preserve command order across buffer growth.
- Growth preserves contents and changes generation; consumers rebuild affected bind groups, not pipelines.
- Shader arrays are runtime-sized. Device buffer limits bound table capacity, and refusal names the cause.
- Fixed writes precede head-of-draw upload and GPU passes.
- Engine work records on the renderer's frame encoder, never a separate steady-play submission.
- TypeGPU is the engine and extension GPU language; raw WGSL uses its escape hatch.
- Steady updates use raw handles and byte ranges, not allocating object-form writes. Buffer generations, row addressing and upload APIs are documented beside [GpuTable](src/engine/ecs/table.ts).

### GPU floor

- Built-in layouts fit WebGPU's default eight storage-buffer bindings per shader stage, counted by declared visibility across all bind groups, including bindings a shader does not read.
- The standard composition requires `indirect-first-instance` and `rg11b10ufloat-renderable`.
- Floor qualification supplies `config.device` requested with only the composition's required features and default limits, and records the resulting capabilities.
- A layout repair preserves precision and capacity, and reports its uploads, passes and dispatches on named hardware.

### Readback

- Steady play reads nothing back.
- GPU work sizes later GPU work through indirect arguments; diagnostics clamp safely without CPU observation and are read only on request.
- Shipped shaders never use TypeGPU's shader `console.log`.
- A [probe](src/engine/runtime/probe.ts) returns one independent snapshot of a world-owned resource, stamped with the copy-time frame and fixed tick, not arrival time.
- Probe staging is pooled per world and expires after declared idle frames; only requests allocate, and there is no continuous-readback mode.
- A deterministic plugin never consumes readback in `fixed`; one that does declares `deterministic: false`, and replay proves the declaration.

### GlobalTransform

- `GlobalTransform` is derived world placement, never authored.
- Gameplay and physics read its fixed-tick columns; rendering reads the engine's interpolated table, resident only when requested.
- A teleport discards interpolation across the discontinuity.
- A placement producer (`Transform`, or a domain's body, skeleton or attachment) adds `GlobalTransform` when missing and never removes it on detachment.
- Standard physics warns once per entity carrying both `Body` and `Transform`, since both write its `GlobalTransform`.
- A body's simulation owns its `GlobalTransform` pose, velocity and collider-derived scale; no other producer writes that body's scale.
- Producers write world storage, never the interpolated output; readers never treat `Transform` as the shared world-space result.
- Hierarchy belongs to the domain deriving placement. A general attachment relation enters core only when two examples need the same one.

## Layout

- Layers run from `engine` outward through `core`, `standard` and `extras`; dependencies point inward, and outer layers make more choices a game can replace.
- Extras admit features after a stable release cycle as external packages.
- Each module owns one useful responsibility completely; split, fix or remove one that doesn't.
- Game modules never import tooling (`project`, `cli`, `native`, `types`).
- Core, standard and extras modules never import siblings in their layer; physics never imports rendering.
- A game module's `index.ts` is public and its other files are internal.
- A module registering systems or resources defines a plugin; others export data and functions.
- A module's extra plugins are its optional parts; a part with its own responsibility is its own module.
- Layer indexes re-export their modules, the root barrel re-exports every game layer, `standard/index.ts` sets the default plugins, and `package.json` declares public subpaths.
- A `transitional` module declares its destination and migration owner.
- Provider-specific observation belongs in optional application integrations, outside Shallot.

### Core and standard

- Modules are named for what they own, not their technique.
- Core and standard use the same noun and matching subpaths, such as `core/rendering` at `/rendering` and `standard/rendering` at `/standard/rendering`.
- Export names describe the implementation.

### Rendering

- `RenderingPlugin` (`core/rendering`) owns cameras, shared views, canvas binding, projection, view and frame uniforms, capture, light components and the frame-ordering anchors; it knows no meshes or materials.
- Scene effects run before `OverlaySystem`, overlays between it and `PresentationSystem`, and presentation after that anchor.
- `CorePipelinePlugin` owns each view's clear, depth and multisampled color targets, resolve, the opt-in `DepthPrepass` and `PickingPrepass` lanes, `RenderPhases` and the tonemapping pass.
- Core opens the prepass and one main render pass per view; renderers record opaque, then transparent, work into it with core's formats and sample count.
- A renderer reaches an effect's depth or picking input only by recording into that prepass lane.
- Transparent work is ordered by renderer, not by object across renderers, since standard submits GPU-driven indirect draws per surface.
- `Tonemapping` selects the operator, TonyMcMapface by default; `TonemappingMethod.None` skips it for display-ready linear images, but not grading or encoding.
- `ColorGrading` grades the image, and presentation encodes it for the screen.
- Effects are separate passes registered per camera in `EffectPasses`: before tonemapping they read linear HDR, after it they read encoded display-referred intermediates.
- `CustomPresentation` replaces the built-in presentation for a camera.
- `core/mesh` owns mesh data, GPU storage, built-in primitives and `MeshInstance`, independent of rendering.
- `standard/rendering` is the one clustered forward mesh pipeline over both core modules: surfaces, materials, light packing, culling, indirect draws, backgrounds and shadow passes. It records into core's phases.
- Standard contains no effects; they belong in extras.
- Custom surfaces and backgrounds extend standard through `registerSurface` and `registerBackground`, never a second standard pipeline.

### Physics

- `PhysicsPlugin` (`core/physics`) registers shared `Body`, `Spring` and `Joint` authoring data with their defaults; it installs no simulation.
- Core owns `ShapeKind`, the world-owned `Hulls` registry and solver-neutral observation of caller-supplied body poses; it knows no solver.
- Neither core nor standard physics imports rendering or input.
- `StandardPhysicsPlugin` (`standard/physics`) depends on `PhysicsPlugin` and owns the whole Box3D-based simulation: body and constraint synchronization, stepping, events and world operations.
- A body belongs to one simulation. A replacement backend consumes core's data and replaces all of standard physics, not individual solver phases.
- Standard physics steps at `Time.FIXED_DT`; gravity belongs to its solver world and the substep count is internal.
- Box3D is the correctness authority: world hashes equal its reference with no tolerance. Two departures stay: every `mass <= 0` body marshals as kinematic, never static, so static geometry is stepped; and `Spring` authors `stiffness`, converted to Box3D's hertz and damping ratio.
- Each standard physics phase has one implementation, in the kernel. The worker count schedules it and is not a code path: with no pool the calling thread runs the same tasks, as Box3D's serial fallback does. The shared and single-thread kernel artifacts build from one source, since a page without cross-origin isolation has no shared memory.
- Camera rays belong to rendering's `viewportToWorld`, not physics; callers supply pointer or viewport coordinates, and `Ray` belongs to engine math.
- Standard physics publishes the mover queries and plane solver, and its optional `CharacterPlugin` resolves a mass-zero capsule's caller-written velocity. It reports walkable or steep ground, its normal and point velocity; upward motion suppresses its pogo spring only relative to that ground velocity.
- `extras/player` owns input, gravity, acceleration, friction, sprint, jump, coyote time, buffering and platform carry over the published standard physics barrel. Games replace that feel without reaching into physics internals.

## Commands

- A `shallot` command does only what Shallot knows: it may invoke project commands, but never owns processes Vite, Bun or Playwright own.
- Web commands delegate to Vite; native dev and build add the shell, and native preview launches the existing build.
- Package scripts are package-manager lifecycle hooks or verbs every Shallot project has; other tools run by path.

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

- Iterate on selected files or subjects.
- Wide runs confirm the final candidate before landing or release; CI runs them on every push. The wide Node command is in [CI](.github/workflows/test.yml), and `bun run test:browser` runs every browser subject.
- Manual oracles run by path.
- The import gate runs its Bun assertions with `--todo`: only the debts named in those assertions are todos, every other finding fails, and a passing todo fails until its scope is retired.
- Import-gate todos leave with their migrations; they are not a mechanism for new exceptions.

## Verification

- Module promises are tested beside their owners and through public examples; test names state claims.
- [Tier ceilings](scripts/test-tiers.ts) are hang and runaway backstops, not performance targets; a test sits far below its ceiling, and work needing more splits or moves to a heavier tier.

| Tier | Files / budget |
|---|---|
| Cheap | `*.test.ts`; `CEILING.cheap` |
| GPU | `*.gpu.test.ts`; `CEILING.gpu` |
| Node | `*.node.ts`; `CEILING.node` |
| Browser | `*.e2e.ts`; `CEILING.browser` per subject |
| Build/startup | `CEILING.startup` |
| Manual oracle | `*.oracle.ts`; unbounded, run by path |

- GPU and Node files call `setDefaultTimeout` once with `CEILING.gpu` or `CEILING.node` from `scripts/test-tiers.ts`; bounded GPU waits fit that ceiling.
- `bun run test` discovers cheap and GPU files with the cheap default; bare `bun test` uses Bun's default; Node and oracle files run explicitly.
- Root `bunfig.toml` supplies the TypeGPU transform, as project preloads do.
- A missing tier premise fails, never skips.
- Quarantine uses `test.todo`, runnable with `--todo`; a known failure stays failing until fixed.
- A browser subject keeps its Playwright and Vite configs together and spreads `BROWSER_CONFIG` and `WEB_SERVER_CONFIG` from [scripts/chromium.ts](scripts/chromium.ts), never restating or tailoring budgets; Playwright's per-test default stays unchanged.
- GPU files run serially on one device with small compositions; each file acquires and warms its independent worlds once in `beforeAll`, and tests time only their work.
- Compiled pipelines are never shared across worlds; a device a loss claim destroys is prepared during that compile step.
- Full default-renderer or physics-kernel compositions, especially multi-world isolation claims, belong in Node.
- A built-in composition change adds an exact native-pipeline count; compile durations are reported by `scripts/compile.oracle.ts` and `scripts/gpu-compile.oracle.ts`, never asserted.
- Device-backed checks record adapter classification, and a hardware claim requires a real adapter.
- Browser observations accept software adapters and request no adapter themselves.
- Display-bound measurements name a monitor and take its keyboard and cursor.
- Gameplay assertions step the composed app; simulation lives in registered components or snapshot, restore and hash hooks.
- `fixed` reads per-tick actions, `draw` presents, and `local` components are excluded from hashes.
- Determinism holds within one runtime and engine version; hashes detect divergence across versions.
- Frame evidence uses the cheapest observable: CPU state, then GPU readback, then browser pixels, then a person.
- An engine's final frame uses `attachTexture` and `captureTexture`; other owned textures use `probeTexture`.
- Page composition uses a stepped app and semantic screenshot regions; a running canvas uses `captureFrame` inside the presenting frame, since canvas readback after presentation is transparent black.
- Goldens cover only defects cheaper evidence cannot show, and are never updated to make a check pass.
- Steady play allocates nothing.
- A memory claim creates and disposes its subject, returns to baseline and fails on a deliberately leaking control; retention is measured after GC, and sampler sites are diagnostics, not results.
- Performance uses counted work and same-machine ratio oracles; real-hardware timings name the hardware and are reported, never asserted.
- An oracle is a separate measurement: run when its claim or tool changes, recorded in the commit, and repeated only for a named doubt.

| Claim | Tool / tier |
|---|---|
| Deterministic work and owned counts | Stepped world assertions, engine counters, `FinalizationRegistry` under `Bun.gc(true)` / cheap |
| WASM memory | Feature-gated counting allocator and `memory.buffer.byteLength` / Cargo |
| Native heap per step | `dhat`, one profiler per process / Cargo |
| Steady JavaScript allocation | V8 sampling heap profiler over a GPU-backed Node child with Dawn's `webgpu` / Node, macOS |
| GPU release | Counting wrapper over a real Dawn device / GPU |
| Beyond the suite | Heap snapshots, CDP tracing, `measureUserAgentSpecificMemory`, GPU timestamps / oracle |

### CI coverage

- [CI](.github/workflows/test.yml) owns host and tier commands.
- Rust suites run with `cargo test -p shallot-audio` and `cargo test -p shallot-physics`.
- Display-bound allocation stays manual, and CI does not qualify Windows or native packaging.

## Code

- Names and techniques come from the domain's precedent: Bevy for the ECS and rendering where they carry meaning in Shallot, WebGPU and TypeGPU for the GPU, glTF for assets.
- Each rendering divergence from Bevy names its referent, or the Bevy structure Shallot lacks, in the `bevy` entry of the harness's `strategy/shallot/referents.md`.
- A name means one thing across the tree, and a noun beats a participle ([TigerStyle](https://github.com/tigerbeetle/tigerbeetle/blob/main/docs/TIGER_STYLE.md), Naming Things).
- One fact has one name, one producer and one path; a second spelling or a forwarding wrapper goes.
- JSDoc states the contract the signature cannot: units, ownership, lifetime, frame timing and what is refused. It carries no `@example`; examples live in `examples/`.
- A `//` comment says why the code is as it is.
- Neither JSDoc nor a comment restates the code or its history ([Google TypeScript Style](https://google.github.io/styleguide/tsguide.html#comments-documentation), Comments and documentation).

## Code-authored worlds

- A one-off entity is written straight out: create, then adds.
- Repeated content uses a typed table and a function that creates one entity and returns its eid.
- Needed eids are kept as variables or return values, never names. See `route` in [first-person's Demo](examples/first-person/src/demo.ts).

## Examples

- Examples are the outer loop: a user's problem exposes an engine gap, and the owning module fixes and proves it, never the example.
- Each example lives in its promise's repository, under `examples/<name>`, in the shape of a copied user project, and its `index.html` carries one `<meta name="description">` describing it.
- One problem per example; only `first-person` composes many.
- Every public promise has an example whose answer depends on it.
- There are no duplicate demonstrations; the one showing the engine at its best stays.
- Each example has a check that fails when its answer breaks, or the example is removed.
- Examples and their checks use only published names and public seams.

## Heavy work

- Heavy computation runs in WASM or on the GPU; TypeScript coordinates it and runs lightweight gameplay.
- TypeScript that needs runtime-specific tricks to meet a performance target moves to WASM or the GPU.

## Dependencies and releases

- `MIGRATION.md` tells users of the last stable release what breaks and what to write instead; for 0.10 that is `v0.9.5:packages/shallot`.
- Migration entries are rewritten as the tree changes and exclude anything the stable release never shipped.
- Measurements, design reasons, progress and test internals belong in commits and this guide, not migration entries.

### Using the package

- Released: `bun add @dylanebert/shallot`.
- Staged: `bun pm pack` here, then `bun add --no-save <tarball>` in the project; add `--dev` for dev dependencies, or the overlay silently targets the wrong section.
- Live: `bun link` here, then `bun link @dylanebert/shallot` in the project; Vite dedupe and Bun preload route named `typegpu` imports and subpaths to the project's copy, not copies under other package names.
- `bun install` restores the manifest pin only if its range resolves on npm: an unpublished `^0.10.0` neither restores nor matches `0.10.0-next.1`, so staged overlays and restores use a published range.
- The Bun entry is a factory, not an import side effect: `tests/preload.ts` imports `plugin` from `bun` and `shallot` from `@dylanebert/shallot/bun` and registers `plugin(shallot({ root: import.meta.dir }))`, with `[test] preload = ["./tests/preload.ts"]` in `bunfig.toml`.
- Root resolution walks from that preload to the nearest `package.json`, not from cwd; `shallot add` writes this setup.

### Plugin packages

- A plugin package declares `@dylanebert/shallot` as a peer; the range is its compatibility, and `shallot()` shares the project's engine instance only with packages that declare it.
- It carries the npm keyword `shallot-plugin`, and its README shows the install and the one import and `plugins` entry that enable it.

### Changing and publishing

- A pin is the last verified version; every occurrence updates in one commit, and until the release-candidate bump it moves only for a named defect.
- Source and examples self-reference the package name without a dependency entry.
- `@types/node` and `@webgpu/types` stay runtime dependencies because public types point at source.
- A link cannot prove what ships: CLI, manifest, dependency, runtime or native-shell changes require a packed tarball installed in a scratch project.
- A breaking change updates every affected dependent in the same landing.
- Removals and reshapes strip unusable code; rebuilding and additive or hardening work follow the settled engine, not each intermediate state.
- `main` carries the next unpublished version; the commit after publishing advances it, and a commit is not a release.
- Bun's name and version identity can silently defeat staging or restoring a tarball with a published version.
- Publishing happens from a commit whose `package.json` contains that version, with one `v<version>` tag; the tag runs [release.yml](.github/workflows/release.yml) to create the GitHub Release and native archives, not to publish npm.
- Stable: `bun publish`, npm `latest`, GitHub Latest. Prerelease: `bun publish --tag next`, GitHub Pre-release.
- A release line keeps one prerelease identifier; identifiers sort alphabetically.
- A module, example or tool retires by tagging its last commit `archive/<name>`, then deleting it in a commit naming the failure; there is no archive directory.
