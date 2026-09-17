# Contributing

This page is for anyone changing the engine itself, person or agent. Using Shallot is covered in the [README](README.md); each API's contract is the JSDoc beside it.

## Layout

| Path | Owns |
|---|---|
| `src/engine` | The core: app lifecycle, ECS, scenes, the runtime (device and platform admission) and utils. Imports nothing else under `src`. |
| `src/standard` | The default plugins: render, sear (shading), glaze (post), part, physics, character, player, audio, input, transforms, loading, mirror, slab, bvh, fog. |
| `src/extras` | Opt-in plugins (cells, gltf, lines, orbit, outline, profile, skin, sky, sprite, text), published at `/extras`. A plugin starts as a satellite repo and is promoted into extras once it has proven stable across a cycle. |
| `src/harness` | The in-page verdict hook a project publishes, the seat policy and capture contract, and the browser driver. |
| `src/project` | The manifest, scene and asset generation, host toolchain resolution and the Vite plugin. |
| `src/native` | Desktop shell resolution: the prebuilt download and the source-build fallback. |
| `src/cli` | The verbs and their dispatcher. |
| `bin` | `bin/shallot.ts`, the one-line CLI entry. |
| `crates/audio` | The DSP kernel, compiled to WASM. |
| `crates/physics` | The solver kernel, inlined into committed `.wasm.ts` files. |
| `crates/native` | The desktop window host, compiled per project by `build --target`. |
| `examples` | One flat directory per example; `examples/AGENTS.md` is generated from their manifests. |
| `scripts` | Build, check, format, pack and install tooling. |
| `assets`, `assets.json` | The shipped icon. Everything else is fetched by URL and sha256 from `assets.json`. |

## Commands

```bash
bun run build          # regenerate committed audio WASM, dist/vite.js, physics kernel
bun run check          # structure, population, workflow and static gates
bun run test           # every unit row, whole, inside the sweep ceiling
bun run test -- --integration --base <ref> --diff <ref> # changed-subject rows plus subjectless rows
bun run test -- --integration --all                  # every non-oracle integration row
bun run test -- --integration --requires chromium    # integration rows carrying a requirement
bun run test -- --integration --subject <prefix>     # integration rows under a subject prefix
bun run test -- --oracle <claim>                     # one named oracle row
bun run list            # the population selected by the same selectors
bun run workflow        # render the hosted workflow emitted from the population
bun run format         # biome and the scene formatter, writing
bun run assets         # fetch assets.json entries, linking each into the examples that declare it
bun run prepack        # compile the Node-reachable tooling
```

The same `list`, `check`, `test` and `workflow` verbs are installed by
`shallot`. They operate from the caller's project root and do not assume Rust, a package name,
or a branch. A project with another Bun preload composes it with the carrier preload instead of
replacing it. In `bunfig.toml`, compose the entries as
`preload = ["./src/project-preload.ts", "@dylanebert/shallot/harness/preload"]`.

A root `shallot.json` may admit one check entry or an array of `{ "file": "..." }` entries.
When its `check` field is present, that list is the complete project population: every visible
`.test.ts` and `.oracle.ts` file must appear exactly once, and an omitted file refuses.
Every admitted file must use `check()`; malformed, duplicate, missing and moved entries refuse.
Without a root `check` field, project-rooted discovery remains in force, including nested recipe
manifests. `.oracle.ts` files are named evidence and stay out of ordinary `test` and
subject-selected integration sweeps; run one explicitly with `--oracle <claim>`.

`bun run examples:index` regenerates `examples/AGENTS.md`; `check` reds when it drifts. `check` is the gate before every push.

## Tests

- `bun run test` runs unit rows only, hermetically, under the 250ms unit ceiling, and prints the suite wall time last.
- `bun run test -- --integration --base <ref> --diff <ref>` selects changed-subject integration rows plus subjectless rows; `--all`, `--requires <tag>`, and `--subject <prefix>` are composable direct selectors and refuse an empty match.
- `bun run test -- --oracle <claim>` runs one named oracle row outside the ordinary sweeps.
- `bun run check` validates structure, the discovered population and generated workflow drift.
- Cargo suites are declared integration rows with `requires: ["cargo"]`; requirement resolution compiles them once, untimed, and the row budget covers only the compiled run.

A check declares its claim, optional size and requirements in the runner call:

```ts
check("the body settles", { claim: "Body settles on the floor" }, body);
```

Use `bun run list` with the same selectors to inspect claim, size, requirements, budget and file. Integration selectors compose with each other but refuse with `--base`/`--diff`; a selector matching no row refuses. Retire a unit by tag, add its row to [`ARCHIVE.md`](ARCHIVE.md), then delete it.

## Seats and captures

A requirement tag names a capability, and a host that lacks it refuses rather than passing on a weaker one. No
requirement is CPU only. `gpu` is a real in-process WebGPU device. `chromium` is headless Chromium on a
positively identified real adapter at the capture contract; a software adapter, an adapter with no identity
and an undeclared host each refuse with their own reason, and its launch is always headless — a headed
launch never grants it, because a windowed run proves a different seat. `display` is that seat, and the only
one that launches headed: it refuses unless the host declares a display in `SHALLOT_DISPLAY_SEAT` and a headed
Chromium on the shared floor there classifies a positively identified real adapter. The page is opened on the
monitor that declaration names and read back from the compositor for where it actually landed, because a
window placed by focus presents at whichever monitor's rate happened to be in front of the person. A display
run therefore takes that monitor while it lasts, along with the keyboard and cursor the page under test asks
for. That is the seat, not a defect — presenting on a real display is the measurement. The seat also needs a
display fast enough for the row's frame budget — about 134 Hz sampled under the allocation oracle's
constants — so a 60 or 120 Hz display refuses honestly and always rather than measuring a page it cannot
step in time. Placement is per compositor, and Hyprland is the only driver today: a host on any other
compositor refuses by name rather than letting focus decide which monitor the page presents on. Launch mode
is policy keyed by seat in `src/harness/launch.ts`; `launch.json` holds only each host's headless and headed
evidence.

`captureFrame` from `@dylanebert/shallot/harness/capture` is the one capture: it fixes the viewport, device
scale, surface, presentation boundary and tightly packed RGBA semantics, and refuses a surface at any other
geometry. Read pixels through it rather than writing a local `toDataURL` pipeline, so semantic checks,
artifacts and human frames all describe the same frame. Assertions stay in the page on the stepped clock;
the driver fixes the seat and carries the reproduction record, and a failure retains bounded page, GPU,
server and sub-check evidence with one actual frame under `.artifacts/`.

## Determinism

Checks run on the scheduler's stepped clock, never wall time. Simulation state lives in registered components or behind a snapshot, restore and hash hook; gameplay runs in the fixed group from per-tick actions, presentation and effects run in draw, and `local` components stay out of the hash. Determinism holds within one runtime and engine version; across them, divergence is detected by hash, never assumed away.

## Allocation and counts

Draws, dispatches, bytes uploaded, allocations and entities visited are deterministic, so they are unit rows whose expected value comes from the scene's content, never a blessed literal. Times come from real devices, stamped with hardware, and report; no gate carries a wall-time floor.

Steady play allocates nothing unaccounted. `sanctions.json` holds each per-frame allocation the platform forces, with a count derived from the frame's structure; `red-circles.json` holds real, unwanted work deferred to a named owner. Neither carries a byte budget, and work that could be hoisted, cached, pooled or moved to wasm is never sanctioned. The person approves every row in both files; an agent proposes one and stops.

## Frame claims

A frame claim is proved on the lowest rung that can see it: a CPU property, then GPU readback, then semantic browser pixels through `captureFrame`, then a person's look. A full-frame golden waits for an escaped defect no lower rung can see, and a golden is never edited to match. A screenshot helps whoever iterates and is never a verdict.

## Heavy work

Heavy work runs in wasm or on the GPU; TypeScript coordinates and carries lightweight gameplay. The test is shape, not cost: TypeScript that reaches a performance bar only through runtime-internals tricks, such as a boxed-`let` register, a call reshaped to dodge a deoptimisation, module state read by hoisted callbacks to avoid a closure context, or a `Math.fround` discipline over every operation, belongs across the line.

## Pins and freshness

A pin records what was last verified, and every freshness pass bumps it:

- Rust in `rust-toolchain.toml`.
- Bun in `packageManager` and `.bun-version`.
- Node in `.node-version`, the exact version `node` rows resolve.
- `binaryen` in `package.json`.
- `bun-webgpu` at a `dylanebert/bun-webgpu` commit, until kommander/bun-webgpu#10 merges and it returns to upstream.

A bump touches every doc and fixture site in one commit; `check-pins` reds on drift.

## Dependencies

- The root links itself (`"@dylanebert/shallot": "link:."`), so examples import the package by name.
- Examples declare no engine dependency; `add` writes the version into the copy.
- Satellite consumers stay on a published package range and iterate locally with `bun link`; a `file:` directory dependency uses hardlinks, so editor writes can detach the checkout from the installed copy. That doesn't prove the published shape: a `bun pm pack` installed into a scratch project does, and CLI, manifest, dependency, runtime and native changes owe one.
- Local, staged and published are separate states: a local link is an uncommitted override, a stage pins a full-SHA Git source or an exact tarball with integrity, and published use pins a stable range with its lock. Publishing is a release, never a way to see a change. Cold proof installs frozen from an empty cache.
- `@types/node` and `@webgpu/types` are runtime dependencies, because `types` points at source.

## Branches and releases

`main` is the work branch and may be mid-change. A release is a `v*` tag: pushing one runs the release workflow, which builds the native shells and attaches them to the GitHub Release, and the same version is published to npm. Consumers pin a published version or a full commit SHA, never `main`.

## Archive

Retire a unit by tagging the last commit that has it, adding a row to [`ARCHIVE.md`](ARCHIVE.md) (name, tag, path at tag, why, what would rebuild it), then deleting it. There's never an archive directory.

## Device tiers

The plugin declaration is the one source of truth for whether composition needs a device. The generated context below is also shipped in `src/engine/app/device-tiers.generated.ts`, so packed consumers and contributors read the same classification.

<!-- device-tiers:start -->
| Plugin set | Composition tier | `required` (GPU) | `optional` (CPU/GPU) | absent declaration (CPU) |
| --- | --- | --- | --- | --- |
| standard | gpu | BVH, Fog, Glaze, Mirror, Render, Sear | Part, Physics, Slab, Transforms | Audio, Character, Input, Player |
| extras | gpu | Profile | Cells, Gltf, Lines, Outline, Skin, Sky, Sprite, Text | Orbit, OrbitOverlay, PhysicsProfile |
<!-- device-tiers:end -->

## Engine shape

- The ECS mutates immediately with no deferred command buffer, so the next system reads exactly what the last one wrote.
- Bevy is the structural reference: take its data-first ECS and plugins, skip anything that hides behavior or taxes authoring.
- The core owns mechanisms and extensions own content, so a new feature is a plugin filling a seam, not a branch in the core.
- Each module has one barrel, and the subpaths in `package.json` `exports` are the extension seams; nothing deep-imports `src`.
- Dependencies point inward, extras on standard on engine, so the core stays testable without the layers above it.
- It's data-oriented: components are typed arrays and systems are functions over them, because that's what the GPU consumes.

## Input migration

Device reads are State-scoped: replace the removed `Inputs` singleton with `devices(state)` and read its `keys`, `mouse`, `touch`, `focused`, `viewport`, or `audio` rows.

- `isKeyPressedWithin` was removed; use `devices(state).keys.pressedTick` with a fixed-tick window such as Character's jump/coyote timers.
- `Mouse.canvasWidth` and `Mouse.canvasHeight` were removed; use `devices(state).viewport.get(devices(state).focused)` for the focused canvas's CSS size.
- `setInputEnabled` now takes the State explicitly: `setInputEnabled(state, on)`.
