# Shallot

webgpu game engine

- fast by default
- instant iteration
- runs anywhere

## live demos

Five demos are built and served at [dylanebert.com/shallot](https://dylanebert.com/shallot/). Each links to its source at the version it was built from.

| demo | play | code |
|---|---|---|
| Collapse | [play](https://dylanebert.com/shallot/collapse/) | [code](https://github.com/dylanebert/shallot/tree/v0.9.4/examples/showcase/collapse) |
| Roads | [play](https://dylanebert.com/shallot/roads/) | [code](https://github.com/dylanebert/shallot/tree/v0.9.4/examples/showcase/roads) |
| Sandbox | [play](https://dylanebert.com/shallot/sandbox/) | [code](https://github.com/dylanebert/shallot/tree/v0.9.4/examples/showcase/sandbox) |
| Visualization | [play](https://dylanebert.com/shallot/visualization/) | [code](https://github.com/dylanebert/shallot/tree/v0.9.4/examples/showcase/visualization) |
| Voxel | [play](https://dylanebert.com/shallot/voxel/) | [code](https://github.com/dylanebert/shallot/tree/v0.9.4/examples/showcase/voxel) |

## quick start

All you need is [bun](https://bun.sh):

```bash
bun create shallot my-game
cd my-game
bun install
bunx shallot dev
```

`bunx shallot dev` runs the project with hot reload, and `bunx shallot build` ships it as a web bundle. `bunx shallot build --target windows|mac|linux --release` downloads a prebuilt shell for that version from GitHub Releases, so no Rust toolchain is needed on a hit. A debug build, or any miss (404, offline, checksum mismatch, a source checkout), silently falls back to compiling the Rust window host from source, which needs the Rust toolchain plus that target's system dependencies (see [from source](#from-source)).

A project is plain data plus code: a `shallot.json` manifest, a `.scene` file, and TypeScript plugins you edit in your IDE.

`bunx shallot verify` boots the project in a headless browser and exits 0 or nonzero, a check you, an agent, or CI can run to catch a project that no longer boots or renders. It drives a real browser through the optional playwright peer, so install that once per project: `bun add -d playwright && bunx playwright install chromium`.

## the repo is the docs

The source is the reference: every public export carries a JSDoc contract. There's no docs site to drift from it, and two files carry the consumer surface:

- [`packages/shallot/AGENTS.md`](packages/shallot/AGENTS.md) — the consumer contract: commands, the ECS and plugin conventions, the GPU, render, physics, and testing rules. Ships in the npm package.
- [`examples/AGENTS.md`](examples/AGENTS.md) — the examples index: one line per entry, so you grep for the problem you have. The recipes section ships in the npm package as well.

Written for coding agents first, readable by hand. Both files move in the same commit as the code they describe, so there's no generated layer to fall behind.

## examples

Examples live under `examples/`, indexed by [`examples/AGENTS.md`](examples/AGENTS.md):

- `recipes/` — one minimal project per problem: first-person character, physics playground, import a model, day-night sky, and more.
- `showcase/` — full projects rather than one concept each, several under real-device gates: `collapse`, `roads`, `sandbox`, `visualization`, `voxel`.
- `gym/` — machine-verdict scenarios: the real-device test and benchmark tier.
- `flows/` — ejected standalone apps for behavior a unit test can't reach (`survive-reload`, `ui-containment`, `no-walls`), driven by `bun run flows`.

Run a recipe standalone:

```bash
bunx shallot dev examples/recipes/orbit-camera
```

A new project starts from `bun create shallot <name>` — the scaffold is the single source, so there's no in-repo starter copy.

## links

- [discord](https://discord.gg/eEY75Nqk3C)
- [npm](https://www.npmjs.com/package/@dylanebert/shallot)

## from source

Working on the engine itself needs the full toolchain:

- [bun](https://bun.sh)
- [rust](https://rustup.rs) with the `wasm32-unknown-unknown` target (`rustup target add wasm32-unknown-unknown`)
- `wasm-opt` from [binaryen](https://github.com/WebAssembly/binaryen), optional: the build falls back to copying the unoptimized wasm

```bash
git clone https://github.com/dylanebert/shallot
cd shallot
bun install
bun run build
```

`build` compiles the audio wasm kernel (`packages/shallot/rust/audio`) and the native window host (`packages/shallot/rust/window`). The tumble physics kernel is a committed wasm artifact: rebuild it with `bun run --cwd packages/shallot scripts/build-tumble-kernel.ts` after touching `rust/tumble`.

### native build prerequisites

`shallot build --target <platform>` compiles the Rust window host from the crate source shipped in the npm package. You need [Rust](https://rustup.rs) plus per-target system dependencies:

| target | system webview | portable (CEF) |
|---|---|---|
| mac | Xcode Command Line Tools | same, plus a CEF runtime download on first build (or set `CEF_PATH`) |
| linux | WebKitGTK dev headers (no usable WebGPU; use `--portable`) | `libx11-dev` (X11 dev headers to link the CEF shell), plus CEF runtime download on first build (or `CEF_PATH`) |
| windows | cross-compiled via cargo-xwin (`cargo install cargo-xwin`; no local Windows toolchain needed) | Visual Studio with the C++ workload incl. ATL, from WSL only (the build bridges to the Windows host) |

Portable builds bundle the Chromium runtime (CEF) instead of the system webview. The CEF runtime auto-downloads on first build unless `CEF_PATH` points to a local copy. Release builds download a prebuilt shell when one exists for the installed version; debug builds and any release miss always compile from source.

### layout

- `packages/shallot/` — the engine. published as `@dylanebert/shallot`
- `packages/create-shallot/` — `bun create shallot` scaffold
- `packages/vscode-shallot/` — VS Code extension
- `examples/` — example projects against the engine

### commands

run from the repo root.

```bash
bun run test       # unit tests over packages/shallot (bun-webgpu)
bun bench          # GPU benchmarks
bun check          # format (writes) + tsc + biome + eslint + repo checks
bun run format     # biome + scene formatter
bun run build      # rust artifacts
```

Engine-internal layout, the full command table, and the rules index are in [`AGENTS.md`](AGENTS.md); the conventions themselves are path-scoped under [`.claude/rules/`](.claude/rules/).

`bun check` and `bun run test` are the gate before pushing. The by-path slow suites, the invitation-only PR policy, and where to file an issue are in [CONTRIBUTING.md](CONTRIBUTING.md).

## license

MIT
