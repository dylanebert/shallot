# Contributing

This page is for anyone changing the engine itself, person or agent. Using Shallot is covered in the [README](README.md); each API's contract is the JSDoc beside it.

## Layout

| Path | Owns |
|---|---|
| `src/engine` | The core: app lifecycle, ECS, scenes, the runtime (device and platform admission) and utils. Imports nothing else under `src`. |
| `src/standard` | The default plugins: render, sear (shading), glaze (post), part, physics, character, player, audio, input, transforms, loading, mirror, slab, bvh, fog. |
| `src/extras` | Opt-in plugins (animation, cells, gltf, lines, orbit, outline, profile, skin, sky, sprite, text), published at `/extras`. |
| `src/harness` | The in-page verdict hook a project publishes, and the browser driver's data. |
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
bun run build          # audio WASM, dist/vite.js, physics kernel
bun run check          # tsc, biome, every scripts/check-*.ts, examples index, scene format, cargo fmt
bun run test           # cargo test over the workspace, then bun test over src and scripts
bun run format         # biome and the scene formatter, writing
bun run assets         # fetch assets.json entries, linking each into the examples that declare it
bun run prepack        # compile the Node-reachable tooling
```

`bun run examples:index` regenerates `examples/AGENTS.md`; `check` reds when it drifts. `check` is the gate before every push.

## Tests

- `bun run test` runs unit rows only, hermetically, under the 250ms unit ceiling, and prints the suite wall time last.
- `bun run check` validates structure, the discovered population and generated workflow drift.
- `bun run test:changed` runs the unit and integration populations, then runs Cargo when the diff touches `crates/**`, `Cargo.*` or solver fixtures.

A check declares its claim, optional size and requirements in the runner call:

```ts
check("the body settles", { claim: "Body settles on the floor" }, body);
```

Use `bun scripts/surface.ts --list` to inspect claim, size, requirements, budget and file. Rust suites use Cargo with `#[test]` as their declaration and stay outside this list. Retire a unit by tag, add its row to [`ARCHIVE.md`](ARCHIVE.md), then delete it.

## Pins and freshness

A pin records what was last verified, and every freshness pass bumps it:

- Rust in `rust-toolchain.toml`.
- Bun in `packageManager` and `.bun-version`.
- `binaryen` in `package.json`.
- `bun-webgpu` at a `dylanebert/bun-webgpu` commit, until kommander/bun-webgpu#10 merges and it returns to upstream.

A bump touches every doc and fixture site in one commit; `check-docs` reds on drift.

## Dependencies

- The root links itself (`"@dylanebert/shallot": "link:."`), so examples import the package by name.
- Examples declare no engine dependency; `add` writes the version into the copy.
- An outside project on a local engine uses `bun link`. That doesn't prove the published shape: a `bun pm pack` installed into a scratch project does, and CLI, manifest, dependency, runtime and native changes owe one.
- `@types/node` and `@webgpu/types` are runtime dependencies, because `types` points at source.

## Archive

Retire a unit by tagging the last commit that has it, adding a row to [`ARCHIVE.md`](ARCHIVE.md) (name, tag, path at tag, why, what would rebuild it), then deleting it. There's never an archive directory.

## Engine shape

- The ECS mutates immediately with no deferred command buffer, so the next system reads exactly what the last one wrote.
- Bevy is the structural reference: take its data-first ECS and plugins, skip anything that hides behavior or taxes authoring.
- The core owns mechanisms and extensions own content, so a new feature is a plugin filling a seam, not a branch in the core.
- Each module has one barrel, and the subpaths in `package.json` `exports` are the extension seams; nothing deep-imports `src`.
- Dependencies point inward, extras on standard on engine, so the core stays testable without the layers above it.
- It's data-oriented: components are typed arrays and systems are functions over them, because that's what the GPU consumes.
