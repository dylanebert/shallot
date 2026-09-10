# Shallot

webgpu game engine

- fast by default
- instant iteration
- runs in any WebGPU browser, or native

## live demos

One demo is built and served at [dylanebert.com/shallot](https://dylanebert.com/shallot/) from [dylanebert/shallot-site](https://github.com/dylanebert/shallot-site). It links to its source at the version it was built from, and a staging build against `main` runs at [shallot-staging.pages.dev](https://shallot-staging.pages.dev/).

| demo | play | code |
|---|---|---|
| Visualization | [play](https://dylanebert.com/shallot/visualization/) | [code](https://github.com/dylanebert/shallot/tree/v0.10.0/examples/showcase/visualization) |

## quick start

All you need is [bun](https://bun.sh):

```bash
bun create shallot my-game
cd my-game
bun install
bunx shallot dev
```

`bunx shallot dev` runs the project with hot reload, and `bunx shallot build` ships it as a web bundle. `bunx shallot build --target windows|mac|linux --release` downloads a prebuilt shell for that version from GitHub Releases, so no Rust toolchain is needed on a hit. A debug build, or any miss (404, offline, checksum mismatch, a source checkout), silently falls back to compiling the Rust native host from source, which needs the Rust toolchain plus that target's system dependencies (see [from source](#from-source)).

A project is plain data plus code: a `shallot.json` manifest, a `.scene` file, and TypeScript plugins you edit in your IDE.

`shallot verify` is gone in this version, and `shallot check` will replace it. Until then, `bunx shallot check` says it isn't available and exits 2. Any other verb runs `shallot-<verb>` from your PATH or your project's installed bins, the way Cargo and git do it. The packed-install probe is `bun test --timeout 120000 ./scripts/install-test.probes.ts`.

## add to an existing project

```bash
bun add @dylanebert/shallot typegpu@~0.12.5
bun add -d unplugin-typegpu@~0.12.3
```

TypeGPU is a required peer, and TGSL needs exactly one TypeGPU transform in your bundler. A `shallot.json` project gets that from the CLI. An ejected Vite app adds `typegpu()` from `unplugin-typegpu/vite` plus `optimizeDeps: { exclude: ["@dylanebert/shallot", "typegpu"] }`:

```ts
// vite.config.ts
import { defineConfig } from "vite";
import typegpu from "unplugin-typegpu/vite";
import { projectPlugin } from "@dylanebert/shallot/vite";

export default defineConfig({
    plugins: [typegpu(), projectPlugin(".")],
    optimizeDeps: { exclude: ["@dylanebert/shallot", "typegpu"] },
});
```

`bunx shallot add <name> [dir]` copies a recipe out as a runnable, version-matched project.

## the repo is the docs

The source is the reference: every public export carries a JSDoc contract. There's no docs site to drift from it, and two files carry the agent surface:

- [`AGENTS.md`](AGENTS.md) — the repo's agent contract: commands, pins, the ECS and plugin conventions, the GPU, render, physics, and testing rules.
- [`examples/AGENTS.md`](examples/AGENTS.md) — the examples index: one line per entry, so you grep for the problem you have. The recipes themselves ship in the npm package.

Written for coding agents first, readable by hand. Both files move in the same commit as the code they describe, so there's no generated layer to fall behind.

## examples

Examples live under `examples/`, indexed by [`examples/AGENTS.md`](examples/AGENTS.md):

- `recipes/` — one minimal project per problem: first-person character, physics playground, import a model, day-night sky, and more.
- `showcase/` — full projects rather than one concept each, under real-device gates: `ascii`, `visualization`. Retired units are Git tags indexed in [`ARCHIVE.md`](ARCHIVE.md).

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

`build` compiles the audio wasm kernel (`crates/audio`), the `dist/` tooling and the physics kernel (`crates/physics`, inlined into committed `.wasm.ts` files; the multithreaded artifact needs a nightly toolchain and is kept as committed without one). The native window host (`crates/native`) is built per project by `shallot build --target`.

### native build prerequisites

`shallot build --target <platform>` compiles the Rust window host from the crate source shipped in the npm package. You need [Rust](https://rustup.rs) plus per-target system dependencies:

| target | system webview | portable (CEF) |
|---|---|---|
| mac | Xcode Command Line Tools | same, plus a CEF runtime download on first build (or set `CEF_PATH`) |
| linux | WebKitGTK dev headers (no usable WebGPU; use `--portable`) | `libx11-dev` (X11 dev headers to link the CEF shell), plus CEF runtime download on first build (or `CEF_PATH`) |
| windows | cross-compiled via cargo-xwin (`cargo install cargo-xwin`; no local Windows toolchain needed) | a Windows host with Visual Studio and the C++ workload incl. ATL — cargo-xwin's clang-cl cannot build CEF's `libcef_dll_wrapper`, so the portable target needs the real MSVC toolchain |

Portable builds bundle the Chromium runtime (CEF) instead of the system webview. The CEF runtime auto-downloads on first build unless `CEF_PATH` points to a local copy. Release builds download a prebuilt shell when one exists for the installed version; debug builds and any release miss always compile from source.

### layout

- `` — public engine-and-tools distribution, `@dylanebert/shallot`
- `examples/` — example projects against the engine

### commands

run from the repo root. The `test` script in [`package.json`](package.json) defines the default test paths.

```bash
bun run check      # read-only: tsc, biome, every scripts/check-*.ts, scene format
bun run test       # empty until tests are re-admitted by declaration
bun run format     # biome + scene formatter
bun run build      # rust artifacts
```

The full command table, the toolchain pins and the conventions are in [`AGENTS.md`](AGENTS.md).

`bun run check` is the gate before pushing. The old tests live at the `archive/tests-pre-slice` tag (see [ARCHIVE.md](ARCHIVE.md)) and come back one declared check at a time. File issues at <https://github.com/dylanebert/shallot/issues>.

## license

MIT
