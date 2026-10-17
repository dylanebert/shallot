# Contributing

For modifying the engine; for using it, see the [README](README.md). API contracts live in JSDoc. This guide holds the rules the code, CLI and checks don't explain.

## Architecture

### Device and world ownership

- A build acquires WebGPU only when an enabled plugin declares `gpu` (including `{}`) or `config.device` is supplied. CPU-only compositions build and step without a WebGPU backend.
- Native GPU apps require an identified hardware adapter before plugin preparation. `SHALLOT_ALLOW_SOFTWARE_GPU=1` at launch permits deliberate software use; browser and headless apps do not impose this requirement.
- Headless Bun and Node GPU apps install the optional `webgpu` peer and call `setupGlobals` from `@dylanebert/shallot/webgpu` before building; browser GPU apps use the host's WebGPU.
- Each app owns its world storage and allocations; disposing one releases its resources, not the device or a sibling's data.
- Builds serialize registration and warm-up; completed apps coexist.
- Components declare fields, not storage or GPU residency.
- A component's fields promise a value only while an entity holds it.
- A component's identity is its record; its key only rebinds a reloaded record, and a composition refuses two records under one key.
- A component's membership bit is never reassigned, including by restore.
- Each world owns eid-indexed typed columns that double to cover a component's holders and never shrink during play. Eid indexing spans its highest holder's eid: even a one-holder component on a high eid costs its row bytes per eid, as the public raw-column contract requires.
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
- Each mesh material type owns one typed GPU table whose row matches its parameter schema; `MeshMaterial` selects that World-local type and row.
- Draws and dispatches read compact active-row or instance lists, never the sparse eid range; eid-based lookups opt into the uploaded map.
- Fill is bulk: bound columns, byte ranges or compute, never a per-row JavaScript callback. GPU-only tables have no CPU record source or upload.
- Uploads preserve command order across buffer growth. Outside draw, queue writes and immediate growth submissions follow program order; during draw, distinct staging ranges preserve encoded observations on the World's lazy frame encoder.
- All draw work shares the World's lazy frame encoder and one submission; a step that encoded nothing submits nothing. That submission owns the fence `runApp` waits on. Upload staging recycles at completion without another fence; its byte bound is per encoded frame, not accumulated ticks. Callback contracts live beside [World.onFrameSubmit](src/engine/ecs/world.ts).
- On a propagated draw error, system draw commands are discarded; only engine copies replay in their own fenced submission.
- Shader arrays are runtime-sized. Device buffer limits bound table capacity, and refusal names the cause.
- Fixed writes precede head-of-draw upload and GPU passes, including writes from exact ticks before a frame.
- TypeGPU is the engine and extension GPU language; raw WGSL uses its escape hatch.
- Buffer generations, row addressing and upload APIs are documented beside [GpuTable](src/engine/ecs/table.ts).

### GPU floor

- Built-in pipelines and shipped renderer compositions fit WebGPU's default eight storage-buffer bindings per shader stage.
- Storage bindings count across all bound groups by stage visibility, unused declarations included. Pipeline creation refuses a pipeline over the granted per-stage limit, naming its plugin, pipeline, stage and count, and refuses a layout it cannot account for.
- Device acquisition uses the enabled plugins' declared requirements, never adapter maxima. The built-in renderer does not raise the default storage-buffer limit; an external plugin may require more and owns that compatibility choice.
- A floor-device proof requests only the composition's required features, with no `requiredLimits`, builds and draws the covered pipeline families and examples, and records the adapter identity, granted features and limits.
- A built-in pipeline lacking a required feature refuses, naming its owner and the feature.
- A layout repair preserves precision and capacity, and reports uploads, passes and dispatches on named hardware.

### Readback

- Steady play reads nothing back automatically. An asynchronous GPU result enters simulation only as an accepted record with an assigned tick, never by completion timing alone.
- GPU work sizes later GPU work through indirect arguments; diagnostics clamp safely without CPU observation and are read only on request.
- Shipped shaders never use TypeGPU's shader `console.log`.
- [Probes](src/engine/runtime/probe.ts) use [world-owned staging](src/engine/runtime/readback.ts) only on request. Results carry copy-time stamps, not arrival time; result and staging contracts live beside those APIs.
- A deterministic plugin never consumes readback in `fixed`; one that does declares `deterministic: false`, and replay proves the declaration.

### Frame schedule

- `runApp` is clocked by `requestAnimationFrame` alone and requests the next callback before GPU work. A step's delta is the difference between rAF timestamps, never `now()`, whose CPU-sampled deltas jitter motion.
- Simulation runs fixed 60 Hz ticks from an accumulator, at most four a step, and rendering interpolates by `fixedAlpha`. The rate is one constant; changing it rewrites tick-count tests, tuning and replays.
- Input is sampled once per callback into tick-addressed records, and look applies per frame. Fixed systems read tick records, never devices, and an instantaneous action stays latched until a tick consumes it.
- Skip a callback only while six frames are pending: Chrome's rAF throttle can engage too late to bound queued GPU work (`81e3375c`), and a web callback cannot block like a native frame-latency cap.
- No callback is skipped for landing early: a skipped refresh shows one image twice.

### GlobalTransform

- `TransformPlugin` (`core/transform`) owns authored `Transform`, derived `GlobalTransform`, fixed-tick derivation and boundaries, discontinuities and placement recovery.
- A composition without a placement reader registers no placement unless it adds `TransformPlugin`.
- `GlobalTransform` is derived world placement, never authored.
- Gameplay and physics read its fixed-tick columns; rendering owns `globalTransformTable(world)` in `core/rendering`, its GPU history and interpolation. Placement without rendering allocates no placement GPU buffers or pipelines.
- A placement producer (`Transform`, or a domain's body, skeleton or attachment) adds `GlobalTransform` when missing and never removes it on detachment.
- A body's `Transform` holds its pose, which physics writes back each step.
- A body's pose and linear velocity reach `GlobalTransform` through physics, not Transform pose derivation; scale still derives from `Transform` and scales the body's shapes and rendered mesh alike.
- Producers write world storage, never the interpolated output; readers never treat `Transform` as the shared world-space result.
- Hierarchy belongs to the domain deriving placement. A general attachment relation enters core only when two examples need the same one.

## Layout

- Layers run from `engine` outward through `core`, `standard` and `extras`; dependencies point inward, and outer layers make more choices a game can replace.
- An extra is a mechanism that carries no game's choices, that the field's engines ship in-engine or as official add-ons, and that a game cannot write well over public seams. A new one first spends a stable release cycle as an external package.
- A game's choices, what the player may do and how it feels, live in the game: an example writes its own gameplay over published mechanisms.
- Each module owns one useful responsibility completely; split, fix or remove one that doesn't.
- Game modules never import tooling (`project`, `cli`, `native`, `types`).
- A core, standard or extras module imports a sibling in its layer only along an edge the import gate declares, and the declared edges form no cycle. Core physics and rendering depend on core transform. Physics never imports rendering or input.
- Engine owns domain-free storage and scheduling; it neither imports nor names placement code.
- A helper named for one domain lives with that domain, not in engine/utils.
- A game module's `index.ts` is public and its other files are internal.
- A module registering systems or resources defines a plugin; others export data and functions.
- A module's extra plugins are its optional parts; a part with its own responsibility is its own module.
- Layer indexes re-export their modules, the root barrel re-exports every game layer, `standard/index.ts` sets the default plugins, and `package.json` declares public subpaths.
- A `transitional` module declares its destination and migration owner.
- Provider-specific observation belongs in optional application integrations, outside Shallot.

### Core and standard

- Modules are named for what they own, not their technique.
- Core and standard use the same noun and matching subpaths, such as `core/rendering` at `/rendering` and `standard/rendering` at `/standard/rendering`.

### Rendering

- `RenderingPlugin` (`core/rendering`) owns cameras, shared views, canvas binding, projection, view and frame uniforms, capture, light components and the frame-ordering anchors; it knows no meshes or materials.
- Rendering captures derived placement after each fixed tick's placement boundary and every ordinary fixed system. History copies and interpolation record through `world.frameEncoder()`, after pre-frame table growth submissions.
- Each scheduler group end holds an ordered boundary set outside ordinary systems. Boundary edges obey the same refusal of contradictory and cyclic order as ordinary systems.
- Scene effects run before `OverlaySystem`, overlays between it and `PresentationSystem`, and presentation after that anchor.
- `CorePipelinePlugin` owns each view's clear, depth and multisampled color targets, resolve, the opt-in `DepthPrepass` lane, `RenderPhases` and the tonemapping pass.
- Plugins register per-camera depth predicates in `world.resource(DepthPrepassRequests)`. Core opens one shared lane when the camera has `DepthPrepass` or any predicate requests it; its sample count matches the main pass, and readers use sample zero. `NoFog` opts a camera out of scene fog only, not an explicit depth request.
- Each presenting view has one main render pass; renderers record opaque, then transparent, work into it with core's formats and sample count.
- A renderer reaches an effect's depth input only by recording into that prepass lane.
- Transparent items are sorted back-to-front by view-space depth across renderers for each view, with deterministic identity ties; renderers may batch only consecutive compatible items in that order.
- `AlphaMode` is per material row. StandardMaterial preserves base-color alpha; `Mask` writes depth, blended modes do not, and `AlphaToCoverage` uses hardware multisample coverage, sharpening sampled base-color alpha; single-sample views mask at the material's cutoff.
- At most one enabled directional casts shadows: greatest illuminance wins, then lower eid. Light units and direction live beside the [lighting components](src/core/rendering/lighting.ts).
- Presenting view state grows without recompiling pipelines or layouts, up to [MAX_VIEWS](src/core/rendering/view.ts); a camera beyond it is skipped with a warning.
- Presentation tonemaps, grades, dithers and encodes each view. `TonemappingMethod.None` bypasses grading, exposure and dither, applying only display encoding; `Linear` keeps them with an identity curve.
- `core/mesh` owns mesh data, GPU storage, built-in primitives and `MeshInstance`, independent of rendering.
- `standard/rendering` is the one clustered forward mesh pipeline over both core modules: surfaces, materials, light packing, culling, indirect draws, backgrounds and shadow passes. It records into core's phases.
- Standard contains no effects; they belong in extras.
- Custom surfaces and backgrounds extend standard through `registerSurface` and `registerBackground`, never a second standard pipeline.

### Physics

- `PhysicsPlugin` (`core/physics`) registers `Body`, `Shape`, the joint kinds and the geometry registries with their defaults; it knows no solver and installs no simulation.
- `StandardPhysicsPlugin` (`standard/physics`) owns the whole Box3D-based simulation at `Time.FIXED_DT`, including `PhysicsWorldDefinition`. A body belongs to one simulation: a replacement backend consumes core's data and replaces all of standard physics, not single phases.
- The ECS `World` is the only public physics world; the kernel world is an internal resource of it.
- Box3D is the correctness authority: world hashes equal its native reference with no tolerance, and authoring takes its definitions whole, with their defaults. Hashes see only simulation state; allocation, published ECS values and events need their own tests.
- A definition field with a Box3D setter applies through that setter at the next fixed sync; a field without one applies at spawn, and its JSDoc says so.
- A marked body `Transform.translation` or `.rotation` compares with the kernel's pose and applies any difference through Box3D's set-transform at the top of the next step, or before character movement consumes its pose.
- Velocity writes and immediate teleports or targets stay calls over the `World`: velocity setters, `setTransform` and `setTargetTransform`; immediate teleports use `setTransform` to reset interpolation.
- Live pose and linear velocity are read from `GlobalTransform`; a read function exists only for state not yet published.
- A marked definition field stays a candidate until physics' next fixed sync, which compares it with the kernel's current value; physics keeps no copy of ECS values.
- Event records and world callbacks name entities by `EntityRef`, which outlives a recycled eid; a query reports the eids of shapes and bodies live when it runs.
- The kernel's free functions are the internal Box3D-shaped API; the parity oracles call them as their native C files call Box3D, call for call.
- Each phase has one implementation, in the kernel, built into the shared and single-thread artifacts from one source, since a page without cross-origin isolation has no shared memory; the worker count schedules it and is not a code path. TypeScript only coordinates: the public API, the ECS and `GlobalTransform` bridge, pool dispatch, and user callbacks at serial points.
- Every host Shallot owns serves cross-origin isolation: `shallot()`'s `dev` and `preview`, the native shell and every publish target. Elsewhere the kernel steps the single-thread artifact after one log naming the missing headers, and no game branches on which.
- Standard physics publishes the mover queries and `CharacterPlugin`; movement feel belongs to the game, as in `examples/first-person/src/player.ts`, over published names only.

### Audio

- `AudioPlugin` (`core/audio`) owns `Sound`, `Listener`, the world's clip registry, `play` and `AudioOutput`; it knows no context, worklet or kernel.
- `StandardAudioPlugin` (`standard/audio`) owns the context lifecycle, one AudioWorklet running the audio kernel, the voice pool and clip decoding. Its promise is clip playback, basic controls and spatialization; synthesis, effects, sequencing and acoustics live outside the engine, in `shallot-audio-lab`, and reach the mix through `AudioOutput`'s bus ahead of the one master limiter.
- Voices are pooled because Web Audio's `AudioBufferSourceNode` is single-use and allocates on every play; control crosses to the worklet as one typed buffer per frame that the worklet returns, not over `SharedArrayBuffer`, which needs cross-origin isolation.
- Audio is presentation: voice handles live in the runtime, keyed by entity, never in ECS, and every frame reconciles them against `Sound`. On restore, a once sound with no live voice is discarded, not replayed.

## Commands

- A `shallot` command does only what Shallot knows: it may invoke project commands, but never owns processes Vite, Bun or Playwright own.
- Web runs through the project's scripts (`bun run dev`, `bun run build`, `bun run preview`); `bun run build` refreshes committed crate artifacts only when their input stamps are stale, while running a crate build script directly forces regeneration. `shallot dev`, `build` and `preview` require a native target. Native dev and build add the shell, and native preview launches the existing build.
- Package scripts are lifecycle hooks, the engine build, or:
  - an app's `dev`, `build` and `preview`, calling Vite;
  - `lint` and `format`, where the project has them;
  - `typecheck`, only for `svelte-check`.
- Every command runs by its native verb: `bun test` for cheap and GPU files, `bun test ./<file>` for Node and oracle files, `bunx playwright test` for browser subjects (after `bunx playwright install chromium` once), `bun check`, `bun <file>`, `bunx shallot`.
- A project that adopts the engine's timeouts, as the engine does, defines one `test` running `bun test` with a preload that calls `setDefaultTimeout(TIMEOUT.cheap)`, run as `bun run test`.
- No package defines `check` or a renamed tier such as `test:node` or `test:browser`.

```bash
bun run build
bun check                       # or bun check <changed paths>
bun run lint
bun run test --path-ignore-patterns '**/*.gpu.test.ts' # cheap, no-device tier
bun test src/core/input/devices.test.ts # selected cheap subject
bun test src/core/mesh/attributes.gpu.test.ts # selected GPU subject
bun test --todo ./scripts/check-*.node.ts # every repository policy
cargo fmt --check
cargo clippy -p shallot-physics --all-targets -- -D warnings
cargo clippy -p shallot-physics --lib --target wasm32-unknown-unknown -- -D warnings
cargo test -p shallot-physics
bunx playwright test -c examples/minimal # selected browser subject
sh scripts/browser.sh           # every browser subject
bun run test --todo             # quarantined claims
bun run format                  # biome
```

- Bun child processes spread [the executable and host identity](scripts/bun-child.ts) into object-form spawn options.
- After a test child exits, remove directories it used with [removeTestDirectory](scripts/test-directory.ts); Windows can release its file locks after the exit is observed.
- Iterate on selected files or subjects.
- Before it lands, a change runs every repository policy.
- The import gate (`bun test --todo ./scripts/check-imports.node.ts`) admits only the debts its todos name, each leaving with its migration; every other finding fails, a passing todo fails until its scope is retired, and no new exception becomes a todo.

## Verification

- Module promises are tested beside their owners and through public examples; test names state claims.
- A change to an example also runs the engine tests that read it, `rg -l 'examples/<name>' src scripts diagnostics`, apart from manual oracles.
- The engine owns its test ceilings and publishes them in [the `./testing` subpath](src/project/testing.js) alongside shared browser configuration; those ceilings are hang and runaway backstops, not performance targets. The same subpath publishes `BUDGET` for local iteration and `TIMEOUT`, which uses `CEILING` when `CI` is nonempty and tier budgets otherwise; startup always uses `CEILING.startup`. Local runs fail over budget; CI warns above half a budget, including a whole browser subject. Work warned on more than one run is made cheaper, split or moved to a heavier tier; a single slow run needs nothing.

| Tier | Files | Hang ceiling | Speed budget |
|---|---|---|---|
| Cheap | `*.test.ts` | `CEILING.cheap`: 2 500 ms | 250 ms |
| GPU | `*.gpu.test.ts` | `CEILING.gpu`: 10 000 ms | 1 000 ms |
| Node | `*.node.ts` | `CEILING.node`: 120 000 ms | 20 000 ms |
| Browser | `*.e2e.ts` | `CEILING.browser`: 300 000 ms per subject | 60 000 ms per subject |
| Build/startup | | `CEILING.startup`: 120 000 ms | — |
| Manual oracle | `*.oracle.ts`; run by path | unbounded | — |

- GPU and Node files call `setDefaultTimeout` once with `TIMEOUT.gpu` or `TIMEOUT.node` from `@dylanebert/shallot/testing`. A GPU wait has no wall clock of its own: validation errors and device loss fail it, and the ceiling catches a hang.
- A missing tier premise fails, never skips.
- Quarantine uses `test.todo`, runnable with `--todo`; a known failure stays failing until fixed.
- Every browser subject keeps its Playwright and Vite configs together and spreads `BROWSER_CONFIG` from `@dylanebert/shallot/testing`; a configured `webServer` spreads `WEB_SERVER_CONFIG`. `scripts/browser.sh` applies `TIMEOUT.browser` and one worker; a copy runs under Playwright's defaults. Browser tests hold the browser tier's page-only claims; Playwright's per-test default stays unchanged. CI retries a browser test once; a fail-then-pass result is flaky, emits a warning and has its trace uploaded. A test that fails both attempts stays red.
- A test requests the fewest solver threads its claim needs. The solver spins at its stage barriers, so threads beyond the host's free cores multiply its time; a CI runner has three or four vCPUs, shared with the test runner and any child.
- GPU files run serially on one device with small compositions; each file acquires and warms its independent worlds once in `beforeAll`, and tests time only their work.
- Compiled pipelines are never shared across worlds; a device a loss claim destroys is prepared during that compile step.
- Full default-renderer or physics-kernel compositions, especially multi-world isolation claims, belong in Node.
- A built-in composition change adds an exact native-pipeline count; compile durations are reported by `scripts/compile.oracle.ts` and `scripts/gpu-compile.oracle.ts`, never asserted.
- Device-backed checks record adapter classification, and a hardware claim requires a real adapter.
- A browser check runs headless and holds only what needs the built page: assets, cross-origin isolation, focused input and presentation into the canvas.
- Browser observations accept software adapters and request no adapter themselves.
- A display run is a manual oracle on the monitor `SHALLOT_DISPLAY_SEAT` declares, and holds only claims whose evidence is presentation on a physical monitor: refresh-paced play, and a real adapter in a browser where headless reaches none. It runs only when the person starts it, and takes that monitor's keyboard and cursor while it runs.
- A virtual compositor is not desktop isolation: X11 pathname sockets and the session bus can remain shared despite a private `XDG_RUNTIME_DIR`. Native-host boundary spikes run in a disposable VM or verified private mount/PID/network namespaces with private `/tmp`, `/run`, HOME/XDG directories and no shared session bus. Wayland-only fixtures disable Xwayland. Verify that boundary before starting a display server or client; virtual testing must not replace a live desktop's socket.
- Gameplay assertions drive the composed world: `tick()` advances one fixed tick; `step(deltaTime)` paces fixed work and runs simulation and draw. Exact ticks preserve changes for the next frame upload and advance no GPU frame or readback.
- Fixed systems read tick time; simulation and draw read the virtual frame clock. Pause and scale pace `step`, never explicit ticks.
- Simulation lives in components and declared recovery participants; derived presentation reconciles after restore. Capture state a plugin owns that any fixed system reads, not only state it writes. [Plugin.recovery](src/engine/app/index.ts) and [World.snapshot/restore](src/engine/ecs/world.ts) hold the capture and ordering contracts.
- Every fixed system has a recovery owner. Registration guards cannot prove that a participant captures every simulation input.
- Core physics recovers its geometry registries, read by fixed body synchronization. Standard physics recovers its solver image and ECS bindings; a physics hash does not prove ECS or gameplay recovery.
- Determinism holds within one runtime and engine version; hashes detect divergence across versions.
- Frame evidence uses the cheapest observable: CPU state, then GPU readback, then browser pixels, then a person.
- An engine's final frame uses `attachTexture` and `captureTexture`; other owned textures use `probeTexture`.
- Page composition uses a stepped app and semantic screenshot regions; a running canvas uses `captureFrame` inside the presenting frame, since canvas readback after presentation is transparent black.
- Goldens cover only defects cheaper evidence cannot show, and are never updated to make a check pass.
- Steady play allocates nothing.
- An allocation gate measures what a shipped game allocates: runtime flags may make code optimize sooner, never differently, and a run lasts until its steady code has finished optimizing.
- A memory claim creates and disposes its subject, returns to baseline and fails on a deliberately leaking control; retention is measured after GC, and sampler sites are diagnostics, not results.
- Performance uses counted work and same-machine ratio oracles; real-hardware timings name the hardware and are reported, never asserted.
- An oracle is a separate measurement: run when its claim or tool changes, its result recorded in its commit message, and repeated only for a named doubt; the tree keeps the tool, never a run's output.
- Rendering performance: [shared transparent-phase oracle](src/core/rendering/shared-transparent.oracle.ts) for CPU work, [render allocation oracle](diagnostics/readback-allocation/render.oracle.ts) for heap, and [first-person allocation oracle](diagnostics/first-person-allocation/allocation.oracle.ts) for the shipped browser loop. Their assertions own the measured scopes and admitted allocations.

| Claim | Tool / tier |
|---|---|
| Deterministic work and owned counts | Stepped world assertions, engine counters, `FinalizationRegistry` under `Bun.gc(true)` / cheap |
| WASM memory | Feature-gated counting allocator and `memory.buffer.byteLength` / Cargo |
| Native heap per step | `dhat`, one profiler per process / Cargo |
| Steady JavaScript allocation | V8 sampling heap profiler over a GPU-backed Node child with Dawn's `webgpu` / Node |
| GPU release | Counting wrapper over a real Dawn device / GPU |
| Beyond the suite | Heap snapshots, CDP tracing, `measureUserAgentSpecificMemory`, GPU timestamps / oracle |

### Rendering iteration

- A rendering change runs every `*.test.ts` and `*.node.ts` under `src/core/rendering`, `src/standard/rendering` and `src/extras`, passed explicitly to `bun test`, with the render allocation oracle:

```sh
bun test $(find ./src/core/rendering ./src/standard/rendering ./src/extras \( -name '*.test.ts' -o -name '*.node.ts' \) -print | sort) ./diagnostics/readback-allocation/render.oracle.ts
```

- A change to what a view draws also runs that set and every browser subject that reads pixels in the Linux software-rendering container, through `scripts/lavapipe.sh`. It uses Mesa llvmpipe for GPU/Node subjects and the shared Linux browser configuration's SwiftShader for Chromium; neither is hardware qualification. The script needs Docker, builds from `scripts/lavapipe.Dockerfile` with the tree's Bun and Node pins, and runs a command in a `git archive` of the given revision. Commit the candidate before running it; uncommitted edits are not copied. Dependency and Playwright volumes persist between runs.

```sh
sh scripts/lavapipe.sh HEAD sh -c 'bun test $(find ./src/core/rendering ./src/standard/rendering ./src/extras \( -name "*.test.ts" -o -name "*.node.ts" \) -print | sort)'
sh scripts/lavapipe.sh HEAD bunx playwright test -c examples/embedded-scene --workers=1
# Or run every browser subject:
sh scripts/lavapipe.sh HEAD sh scripts/browser.sh
```

### Physics iteration

- A change runs, in the loop, the sets of every row it touches. The common set is `bun check`, `bun run lint` and every `*.test.ts` under `src/core/physics` and `src/standard/physics`; the kernel set adds `bun crates/physics/scripts/build-kernel.ts`, `cargo test -p shallot-physics`, `src/standard/physics/solver/step.gold.node.ts` and `diagnostics/box3d-parity/phases.oracle.ts`.

| Change | Loop set |
|---|---|
| Kernel speedup, unchanged behavior | Kernel; `diagnostics/box3d-parity/ab.ts` on the claim |
| Kernel behavior or native boundary | Kernel; the affected `diagnostics/box3d-parity` oracles |
| TypeScript API, ECS bridge, world or pool ownership | Common; the non-allocation `*.node.ts` under `src/core/physics` and `src/standard/physics`; `src/engine/app/{plugins,device}.node.ts`; `src/core/transform/{recovery,composition}.node.ts`; `src/core/rendering/{global-transform,physics-interpolation}.node.ts` |
| Allocation or lifecycle | Common; the affected `*allocation.node.ts` under `src/standard/physics` |
| Character or Player | Common; `src/standard/physics/character-allocation.node.ts`; `examples/first-person/src/*.node.ts` |
| Checks or documentation only | The changed checks; `bun check`; `bun run lint` |

- A speed claim names its scene, thread count and phase before the A/B; only that phase's verdict decides it, and no difference is not equivalence.
- A code change's final candidate runs once every row's set but the A/B, with every `diagnostics/box3d-parity` oracle.

### CI coverage

- [CI](.github/workflows/test.yml) owns host and tier commands and judges compatibility on those hosts; a local run does not.
- A red push run on main closes landing until a revert or fix naming that run lands with the `Fixes-red: <run id>` trailer.
- A dependency or infrastructure failure is named as such, not quarantined as a test.
- For a red only one hosted OS shows, run that OS and tier before landing: `gh workflow run test.yml --ref <branch> -f os=<os> -f tier=<tier>`.
- Windows GPU and Node tiers use WARP. Dawn loads only in a host named `node.exe` ([dawn-host.ts](src/engine/runtime/dawn-host.ts)), so CI runs Bun from a copy with that name.
- `linux-native` checks packed Linux launch, process-group behavior and WebGPU canvas images on composited X11 with software graphics. Canvas readback is not physical-monitor presentation or hardware qualification.
- Release-time qualification stays outside the push sweep: hardware oracles, display runs, the packed release candidate, newest-in-range resolution and the published-pin spike. Mac/Windows native launch and software WebGPU capability also remain outside it.

## Code

Name and shape pillars, in order; an earlier pillar wins when two conflict:

1. **Self-explanatory where read.** At its call site, with its receiver, arguments and type, a name and the shape of what it takes and returns say what it acts on and what it does; a stranger who has not read the declaration guesses right.
2. **Grounded in precedent.** Names, shapes and techniques come from the domain's precedent: Bevy for the ECS and rendering where they carry meaning in Shallot, Box3D and Avian for physics, WebGPU and TypeGPU for the GPU, glTF for assets. A divergence names its reason; a rendering divergence from Bevy names its referent, or the Bevy structure Shallot lacks, in the `bevy` entry of the harness's `strategy/shallot/referents.md`. Structure, meaning identity, ownership and data layout, is data-oriented: an entity is a row, and behavior is systems over tables. A structure Bevy lacks comes from data-oriented precedent. An object-oriented engine's structure, such as a container object that owns its parts' data, needs the reason its problem holds in Shallot, and without data-oriented precedent it goes to the person.
3. **Concise.** A name does not repeat what its receiver, module or type carries, and a shape carries nothing its user does not need.

A choice the pillars leave open is taste, and goes to the person.

- A name means one thing across the tree, one act takes one verb and one shape, and a noun beats a participle ([TigerStyle](https://github.com/tigerbeetle/tigerbeetle/blob/main/docs/TIGER_STYLE.md), Naming Things).
- One fact has one name, one representation, one producer and one path, and one algorithm has one implementation; a second spelling, a second copy or a forwarding wrapper goes.
- JSDoc states the contract the signature cannot: units, ownership, lifetime, frame timing and what is refused. It carries no `@example`; examples live in `examples/`.
- A `//` comment says why the code is as it is, where the code cannot.
- A comment states its fact plainly, all of it and no more. It does not restate the code, walk through it, or tell how it came to be: no history, how it was found, or alternatives rejected ([Google TypeScript Style](https://google.github.io/styleguide/tsguide.html#comments-documentation), Comments and documentation).
- A comment names a precedent only where its reader acts on it: the source the code is read against, or a divergence and its reason. Following the referent goes unsaid; a precedent is not a reason, so the reason itself is written.

## Code-authored worlds

- A one-off entity is written straight out: create, then adds.
- Repeated content uses a typed table and a function that creates one entity and returns its eid.
- Needed eids are kept as variables or return values, never names. See `route` in [first-person's Demo](examples/first-person/src/demo.ts).

## Examples

- Examples are the outer loop: a user's problem exposes a missing mechanism, and the owning module fixes and proves it, never the example.
- Each example lives in its promise's repository, under `examples/<name>`, in the shape of a copied user project, and its `index.html` carries one `<meta name="description">` describing it.
- One problem per example; only `first-person` composes many.
- An example takes the view its problem needs.
- Every public promise is owed an example whose answer depends on it.
- There are no duplicate demonstrations; the one showing the engine at its best stays.
- Each example has a check that fails when its answer breaks, or the example is removed.
- Examples and their checks use only published names and public seams.
- An example's UI shows only what its player needs; a control that explains itself carries no label.
- What is tied to a place in the world is drawn there, in world space; the overlay holds the rest.
- An example that shows a lone subject copies `minimal`'s cube, platform, orbit and light, and is listed in `scripts/check-cube-scenes.node.ts`.
- In every example the subject accent is the brightest, most saturated thing in frame, lit and shaded faces separate, and a cast shadow reads as shade, not as the clear.
- Examples share the `aperture-carbon` content palette: carbon room and ground neutrals, amber, cyan and brand-orange subject accents, and warm pale text and UI ink. Example suns cast shadows; keep warm sun and cool ambient lighting, with `#191816` clear.
- UI keeps CSS pixels, scaled only by device pixel ratio; nothing zooms it to the viewport.
- `diagnostics/default-look/palette.ts`, with `PALETTES["aperture-carbon"]`, is the source of these values and the `ui` tokens: system monospace, 14px text and 24px titles, weight 400, radius 2px, and 50% for a round mark such as a reticle. Copied user projects carry literal values copied from that source, never imports from `diagnostics/`.

## Heavy work

- Heavy computation runs in WASM or on the GPU; TypeScript coordinates it and runs lightweight gameplay.
- TypeScript that needs runtime-specific tricks to meet a performance target moves to WASM or the GPU.
- WASM code takes the shape of the implementation it ports first. A departure for speed names why that shape is slower here and grounds the technique in a WebAssembly entry of the harness's `strategy/shallot/referents.md`.

## Dependencies and releases

- `MIGRATION.md` tells users of the last stable release what breaks and what to write instead; for 0.10 that is `v0.9.5:packages/shallot`.
- Migration entries are rewritten as the tree changes and exclude anything the stable release never shipped.
- Measurements, design reasons, progress and test internals belong in commit history and this guide, not migration entries.

### Using the package

- Released: `bun add @dylanebert/shallot`.
- Staged: `bun pm pack` here, then `bun add --no-save <tarball>` in the project; add `--dev` for dev dependencies, or the overlay silently targets the wrong section.
- Live: run `bun run prepare`, then `bun link` here. In the consumer, use a published Shallot dependency range, run `bun install`, then `bun link @dylanebert/shallot`. A created scaffold may pin an unpublished version; replace it before installing.
- `./testing` and `./vite` source edits are live under a link. Refresh their declarations with `bun run prepare`; `bun test ./scripts/check-tooling.node.ts` checks them.
- `bun install` restores the manifest pin only if its range resolves on npm: an unpublished `^0.10.0` neither restores nor matches `0.10.0-next.1`, so staged overlays and restores use a published range.

### Plugin packages

- A plugin package declares `@dylanebert/shallot` as a peer; the range is its compatibility, and `shallot()` shares the project's engine instance only with packages that declare it.
- It carries the npm keyword `shallot-plugin`, and its README shows the install and the one import and `plugins` entry that enable it.

### Changing and publishing

- A pin is the last verified version; every occurrence updates in one commit, and until the release-candidate bump it moves only for a named defect.
- Source and examples self-reference the package name without a dependency entry.
- `@types/node` and `@webgpu/types` stay runtime dependencies because public types point at source.
- A link cannot prove what ships: CLI, manifest, dependency, runtime or native-shell changes require a packed tarball installed in a scratch project.
- Package-test fixtures declare the tarball as a `file:` dependency and use [the frozen, offline installer](scripts/packed-install-spawn.ts) in independent scratch roots; this fixture setup does not restrict consumer staging with `bun add --no-save`.
- A breaking change updates every affected dependent in the same landing.
- Removals and reshapes strip unusable code; rebuilding and additive or hardening work follow the settled engine, not each intermediate state.
- `main` carries the next unpublished version; the commit after publishing advances it, and a commit is not a release.
- Bun's name and version identity can silently defeat staging or restoring a tarball with a published version.
- Publishing happens from a commit whose `package.json` contains that version, with one `v<version>` tag; the tag runs [release.yml](.github/workflows/release.yml) to create the GitHub Release and native archives, not to publish npm.
- Stable: `bun publish`, npm `latest`, GitHub Latest. Prerelease: `bun publish --tag next`, GitHub Pre-release.
- A release line keeps one prerelease identifier; identifiers sort alphabetically.
- A module, example or tool retires by tagging its last commit `archive/<name>`, then deleting it in a commit naming the failure; there is no archive directory.
