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

The run summary names its JUnit report under `.artifacts/`; child stdout and stderr are retained in `output.log` beside it. A selector that matches no test fails.

## Tests

- A test declares its claim, and optionally its size and requirements, in the `check()` call. A Cargo suite is an integration test with `requires: ["cargo"]`. It is compiled once, untimed, and only the run counts against its budget.
- A requirement tag names something the host must have. A host without it refuses and says why; it never runs a weaker version instead. A test with no tag is CPU only. `gpu` needs a real in-process WebGPU device. `display` shows in headed Chromium on a monitor the host declares, and takes that monitor, the keyboard and the cursor while it runs, because showing is what it measures. `src/harness/launch.ts` resolves its launch from `launch.json`, which holds only facts about the host. `chromium` refuses on every host until the device-seat roadmap item rebuilds it (see [`ARCHIVE.md`](ARCHIVE.md)).
- `captureFrame` from `@dylanebert/shallot/harness/capture` is the only way to capture a frame, so checks, saved artifacts and frames for people all show the same thing. Assertions run in the page on the stepped clock. A failure keeps its evidence and one real frame under `.artifacts/`.
- Tests run on the scheduler's stepped clock, never wall time. Simulation state lives in registered components or behind a snapshot, restore and hash hook. Gameplay runs in the fixed group from per-tick actions; presentation and effects run in draw; `local` components stay out of the hash. Determinism holds within one runtime and engine version. Across them, a hash detects divergence; it is never assumed away.
- Draws, dispatches, bytes uploaded, allocations and entities visited are deterministic, so they are unit tests whose expected value comes from the scene's content, never a hard-coded number. Timings come from real devices, labeled with the hardware, and are reported; no gate fails on wall time.
- A claim about a frame is proved at the lowest level that can see it: a CPU property, then GPU readback, then browser pixels through `captureFrame`, then a person looking. A full-frame golden image is added only for a defect no lower level can see, and a golden is never edited to match. A screenshot helps while iterating and is never a verdict.

## Allocation

Steady play allocates nothing unaccounted for. `sanctions.json` lists each per-frame allocation the platform forces, with a count derived from the frame's structure. `red-circles.json` lists real, unwanted work deferred to a named owner. Neither has a byte budget. Work that could be hoisted, cached, pooled or moved to wasm is never sanctioned. The person approves every row in both files; an agent proposes one and stops.

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
| standard | gpu | BVH, Fog, Glaze, Mirror, Render, Sear | Part, Physics, Slab, Transforms | Audio, Character, Input, Player |
| extras | gpu | Profile | Cells, Gltf, Lines, Outline, Skin, Sky, Sprite, Text | Orbit, OrbitOverlay, PhysicsProfile |
<!-- device-tiers:end -->
