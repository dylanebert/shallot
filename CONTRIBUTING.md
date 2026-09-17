# Contributing

For anyone changing the engine, person or agent. Using Shallot is the [README](README.md), and each API's contract is the JSDoc beside it. This page holds what the tree, the CLI and a red check do not say.

## Layout

Dependencies point inward: `src/extras` on `src/standard` on `src/engine`, which imports nothing else under `src`. Each module has one barrel, and the subpaths in `package.json` `exports` are the only seams.

| Path | Owns |
|---|---|
| `src/engine` | The core: app lifecycle, ECS, scenes, the runtime (device and platform admission) and utils. |
| `src/standard` | The default plugins. |
| `src/extras` | Opt-in plugins, published at `/extras`. A plugin starts as a satellite repo and is promoted once it has proven stable across a cycle. |
| `src/harness` | The in-page verdict hook a project publishes, the seat policy and capture contract, and the browser driver. |
| `src/project` | The manifest, scene and asset generation, host toolchain resolution and the Vite plugin. |
| `src/native` | Desktop shell resolution: the prebuilt download and the source-build fallback. |
| `src/cli`, `bin` | The verbs, their dispatcher and the one-line entry. |
| `crates/audio` | The DSP kernel, compiled to WASM. |
| `crates/physics` | The solver kernel, inlined into committed `.wasm.ts` files. |
| `crates/native` | The desktop window host, compiled per project by `build --target`. |
| `examples` | One flat directory per example; `examples/AGENTS.md` is generated from their manifests by `bun run format`. |
| `assets.json` | Everything but the shipped icon is fetched by URL and sha256 from here, by `bun run assets`. |

## Commands

```bash
bun run build     # regenerate committed audio WASM, dist/vite.js, physics kernel
bun run check     # structure, population, workflow and static gates; the gate before every push
bun run test      # every unit row, hermetic, under the 250ms unit ceiling
bun run test -- --integration --base <ref> --diff <ref>   # changed-subject rows plus subjectless rows
bun run test -- --integration --all | --requires <tag> | --subject <prefix>   # composable with each other, never with --base/--diff
bun run test -- --oracle <claim>   # one named oracle row, outside every sweep
bun run list      # the population the same selectors choose: claim, size, requirements, budget, file
bun run format    # biome, the scene formatter and the examples index, writing
```

A selector matching no row refuses.

## Tests

A check declares its claim, and optionally its size and requirements, in the runner call:

```ts
check("the body settles", { claim: "Body settles on the floor" }, body);
```

Cargo suites are integration rows with `requires: ["cargo"]`; requirement resolution compiles them once, untimed, and the row budget covers only the compiled run.

## Seats and captures

A requirement tag names a capability, and a host that lacks it refuses with its reason rather than passing on a weaker one. A row with no tag is CPU only. `gpu` is a real in-process WebGPU device. `chromium` is Chromium on a positively identified real adapter that takes nothing from the person's desktop: headless where the host's evidence proves headless, and where only headed is proven, headed under the window class `kex-gate`, read back from the compositor on a special workspace without focus and refused by name when no rule hid it. `display` is the only seat that presents on a monitor: the host declares a monitor in `SHALLOT_DISPLAY_SEAT`, and a run takes that monitor, the keyboard and the cursor while it lasts, because presenting on a real display is the measurement. It needs about 134 Hz and Hyprland, the only placement driver today. `src/harness/launch.ts` derives each launch mode from the seat and the host's evidence in `launch.json`, which holds only that evidence. A row's `host` names one host or a list; `SHALLOT_HOST` names this one, and otherwise macOS is `mac`, a Linux session under Hyprland is `omarchy`, and anything else is `other`.

`captureFrame` from `@dylanebert/shallot/harness/capture` is the one capture, so semantic checks, artifacts and human frames describe the same frame; never write a local `toDataURL` pipeline. Assertions stay in the page on the stepped clock. A failure retains its evidence and one actual frame under `.artifacts/`.

## Determinism

Checks run on the scheduler's stepped clock, never wall time. Simulation state lives in registered components or behind a snapshot, restore and hash hook; gameplay runs in the fixed group from per-tick actions, presentation and effects run in draw, and `local` components stay out of the hash. Determinism holds within one runtime and engine version; across them, divergence is detected by hash, never assumed away.

## Allocation and counts

Draws, dispatches, bytes uploaded, allocations and entities visited are deterministic, so they are unit rows whose expected value comes from the scene's content, never a blessed literal. Times come from real devices, stamped with hardware, and report; no gate carries a wall-time floor.

Steady play allocates nothing unaccounted. `sanctions.json` holds each per-frame allocation the platform forces, with a count derived from the frame's structure; `red-circles.json` holds real, unwanted work deferred to a named owner. Neither carries a byte budget, and work that could be hoisted, cached, pooled or moved to wasm is never sanctioned. The person approves every row in both files; an agent proposes one and stops.

## Frame claims

A frame claim is proved on the lowest rung that can see it: a CPU property, then GPU readback, then semantic browser pixels through `captureFrame`, then a person's look. A full-frame golden waits for an escaped defect no lower rung can see, and a golden is never edited to match. A screenshot helps whoever iterates and is never a verdict.

## Heavy work

Heavy work runs in wasm or on the GPU; TypeScript coordinates and carries lightweight gameplay. The test is shape, not cost: TypeScript that reaches a performance bar only through runtime-internals tricks, such as a boxed-`let` register, a call reshaped to dodge a deoptimisation, or a `Math.fround` discipline over every operation, belongs across the line.

## Pins

A pin records what was last verified, and a bump touches every doc and fixture site in one commit; `check-pins` reds on drift. Rust is pinned in `rust-toolchain.toml`, Bun in `packageManager` and `.bun-version`, Node in `.node-version`, `binaryen` in `package.json`, and `bun-webgpu` at a `dylanebert/bun-webgpu` commit until kommander/bun-webgpu#10 merges.

## Dependencies

- The root links itself (`"@dylanebert/shallot": "link:."`), so examples import the package by name and declare no engine dependency; `add` writes the version into the copy.
- `@types/node` and `@webgpu/types` are runtime dependencies, because `types` points at source.
- A consumer is in one of three states. Local is an uncommitted `bun link` override; a `file:` dependency hardlinks, so editor writes can detach the checkout from the installed copy. Staged pins a full-SHA Git source or an exact tarball with integrity. Published pins a stable range with its lock. Leaving local is a frozen install from an empty cache.
- A link does not prove the published shape: a `bun pm pack` installed into a scratch project does, and CLI, manifest, dependency, runtime and native changes owe one.

## Releases

`main` is the work branch and may be mid-change. A release is a `v*` tag: pushing one runs the release workflow, which builds the native shells and attaches them to the GitHub Release, and the same version is published to npm. Publishing is a release, never a way to see a change. Consumers pin a published version or a full commit SHA, never `main`.

Retire a unit by tagging the last commit that has it, adding a row to [`ARCHIVE.md`](ARCHIVE.md) (name, tag, path at tag, why, what would rebuild it), then deleting it. There is never an archive directory.

## Device tiers

The plugin declaration is the one source of truth for whether composition needs a device. This table is generated, and ships in `src/engine/app/device-tiers.generated.ts`.

<!-- device-tiers:start -->
| Plugin set | Composition tier | `required` (GPU) | `optional` (CPU/GPU) | absent declaration (CPU) |
| --- | --- | --- | --- | --- |
| standard | gpu | BVH, Fog, Glaze, Mirror, Render, Sear | Part, Physics, Slab, Transforms | Audio, Character, Input, Player |
| extras | gpu | Profile | Cells, Gltf, Lines, Outline, Skin, Sky, Sprite, Text | Orbit, OrbitOverlay, PhysicsProfile |
<!-- device-tiers:end -->
