# Contributing

For anyone changing the engine, person or agent. Using Shallot is the [README](README.md). Each API's contract is the JSDoc beside it. This page holds what the tree, the CLI and a failing check do not say.

## Layout

Dependencies point inward: `src/extras` depends on `src/standard`, which depends on `src/engine`, which imports nothing else under `src`. Each module has one barrel file, and the subpaths in `package.json` `exports` are the only public entry points.

| Path | Owns |
|---|---|
| `src/engine` | The core: app lifecycle, ECS, scenes, the runtime (device and platform setup) and utils. |
| `src/standard` | The default plugins. |
| `src/extras` | Opt-in plugins, published at `/extras`. A plugin starts in its own repo and moves here once it has been stable for a release cycle. |
| `src/harness` | The in-page verdict hook a project publishes, the seat policy, the capture contract and the display seat. |
| `src/project` | The manifest, scene and asset generation, host toolchain resolution and the Vite plugin. |
| `src/native` | The desktop shell: prebuilt download, with a source build as fallback. |
| `src/cli`, `bin` | The commands, their dispatcher and the one-line entry. |
| `crates/audio` | The DSP kernel, compiled to WASM. |
| `crates/physics` | The solver kernel, inlined into committed `.wasm.ts` files. |
| `crates/native` | The desktop window host, compiled per project by `build --target`. |
| `examples` | One flat directory per example. `examples/AGENTS.md` is generated from their manifests by `bun run format`. |
| `assets.json` | Every asset but the shipped icon is fetched by URL and sha256 from here, by `bun run assets`. |

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

The run summary names its JUnit report under `.artifacts/`.
Each run replaces the prior run directory.
Child stdout and stderr are retained in `output.log` beside the report.
A selector that matches no test fails.

## Tests

- A test declares its claim, and optionally its size and requirements, in the `check()` call; declarations do not name a workstation, compositor or personal seat. A Cargo suite is an integration test with `requires: ["cargo"]`. It is compiled once, untimed, and only the run counts against its budget.
- A requirement tag names something the host must have. A host without it refuses and says why; it never runs a weaker version instead. A test with no tag is CPU only. `gpu` needs a real in-process WebGPU device. `display` shows in headed Chromium on a monitor the host declares, and takes that monitor, the keyboard and the cursor while it runs, because showing is what it measures. `src/harness/launch.ts` provides Chromium launch configuration, and `src/harness/browser.json` holds its browser arguments; a launch plan proves neither adapter availability nor display placement.
- `captureFrame` from `@dylanebert/shallot/harness/capture` is the only way to capture a frame, so checks, saved artifacts and frames for people all show the same thing. Assertions run in the page on the stepped clock. A failure keeps its evidence and one real frame under `.artifacts/`.
- Tests run on the scheduler's stepped clock, never wall time. Simulation state lives in registered components or behind a snapshot, restore and hash hook. Gameplay runs in the fixed group from per-tick actions; presentation and effects run in draw; `local` components stay out of the hash. Determinism holds within one runtime and engine version. Across them, a hash detects divergence; it is never assumed away.
- Allocations are a binary integration check.
- A claim about a frame is proved at the lowest level that can see it: a CPU property, then GPU readback, then browser pixels through `captureFrame`, then a person looking. A full-frame golden image is added only for a defect no lower level can see, and a golden is never edited to match. A screenshot helps while iterating and is never a verdict.

### Measurement

| Claim | Tier | Requirement tag | Instrument |
|---|---|---|---|
| Deterministic work and owned counts | unit | — | Stepped assertions for draws, dispatches, uploaded bytes and visited entities against scene-derived expectations; engine counters for owned counts; `FinalizationRegistry` under `Bun.gc(true)` for collectability where it fits the unit budget. |
| WASM kernel memory | integration | — | A counting global allocator per crate behind a cargo feature for allocations, frees and live bytes; `memory.buffer.byteLength` for growth. |
| Native heap per step | integration | cargo | `dhat` heap assertions; one profiler per process. |
| Steady JavaScript allocation is zero | integration | node | The V8 sampling heap profiler over the composed subject in a Node child. |
| GPU resources are released | integration | gpu | A harness-only counting wrapper over the real Dawn device; zero live resources after dispose. |
| Presentation | integration | display | `captureFrame` checks presentation. |
| Claims the suite cannot see | oracle | — | Sampler site attribution, memlab heap-snapshot diffs, CDP tracing queried with Perfetto, `measureUserAgentSpecificMemory`, and WebGPU `timestamp-query`. |
| Look and feel | person | — | A person judges the look and feel. |

- Timings come from real devices, carry hardware labels and are reported, never gated.
- Every owned-memory check creates and destroys its subject, returns to baseline, and reds on a deliberately retained control.
- Retention is its own claim and is measured after a GC.
- Run an oracle once when its cross-checked check is created or its instrument changes.
- Record that run in the same commit as the check's positive control.
- Run an oracle again only for a named doubt: a suite red it cannot attribute or a leak seen outside the suite.
- Oracles have no standing cadence.
- Hosted observation is never a CI verdict.

## Allocation

- Steady play allocates nothing.
- Any steady allocation reds the binary integration check.
- A red prints the sampler's sites for diagnosis.
- Sampler sites do not decide the verdict.
- Node's `v8.getHeapStatistics().total_allocated_bytes` replaces the sampler only after a no-op and known-allocation control on the real step.
- Deferred allocation is recorded in its owner's roadmap note, never excused by a ledger.

## Heavy work

Heavy work runs in wasm or on the GPU. TypeScript coordinates and carries lightweight gameplay. The test is shape, not cost: TypeScript that hits a performance bar only through tricks aimed at the runtime's internals, such as a boxed `let` used as a register, a call reshaped to avoid a deoptimization, or `Math.fround` on every operation, belongs on the other side of the line.

## Pins and dependencies

- A pin records what was last verified. A bump updates every doc and fixture in one commit, and `check-pins` fails on drift. Rust is pinned in `rust-toolchain.toml`, Bun in `packageManager` and `.bun-version`, Node in `.node-version`, `binaryen` in `package.json`, and `bun-webgpu` at a `dylanebert/bun-webgpu` commit until kommander/bun-webgpu#10 merges.
- The root links to itself (`"@dylanebert/shallot": "link:."`), so examples import the package by name and declare no engine dependency. `add` writes the version into the copy.
- `@types/node` and `@webgpu/types` are runtime dependencies, because `types` points at source.
- A consumer is in one of three states. Local is an uncommitted `bun link`; a `file:` dependency hardlinks, so an editor write can detach the checkout from the installed copy. Staged pins a full-SHA Git source or an exact tarball with integrity. Published pins a stable range with its lockfile. Leaving local means a frozen install from an empty cache.
- A link does not prove what ships. Installing a `bun pm pack` tarball into a scratch project does, and a change to the CLI, manifest, dependencies, runtime or native shell owes one.

## Releases

`main` is the work branch and may be mid-change. A release is a `v*` tag. Pushing one runs the release workflow, which builds the native shells, attaches them to the GitHub Release and publishes the same version to npm. Publishing is a release, never a way to try a change. Consumers pin a published version or a full commit SHA, never `main`.

To retire a unit, tag the last commit that has it, add a row to [`ARCHIVE.md`](ARCHIVE.md) (name, tag, path at the tag, why, what would bring it back), then delete it. There is never an archive directory.

## Device tiers

The plugin declaration is the one source of truth for whether composition needs a device. This table is generated, and ships in `src/engine/app/device-tiers.generated.ts`.

<!-- device-tiers:start -->
| Plugin set | Composition tier | `required` (GPU) | `optional` (CPU/GPU) | absent declaration (CPU) |
| --- | --- | --- | --- | --- |
| standard | gpu | BVH, Fog, Glaze, Mirror, Render, Sear | Part, Physics, Slab, Transforms | Audio, BrowserInput, Character, Input, Player |
| extras | gpu | Profile | Cells, Gltf, Lines, Outline, Skin, Sky, Sprite, Text | Orbit, OrbitOverlay, PhysicsProfile |
<!-- device-tiers:end -->
