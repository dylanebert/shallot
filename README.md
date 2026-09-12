[![test-surface](https://github.com/dylanebert/shallot/actions/workflows/test-surface.yml/badge.svg?branch=main)](https://github.com/dylanebert/shallot/actions/workflows/test-surface.yml)

# Shallot

Shallot is a WebGPU game engine for TypeScript. You describe a game as data, a `shallot.json` manifest and a `.scene` file, and its behavior as plugins: components hold data, systems do the work, and each plugin declares where its systems run. It runs in any browser with WebGPU, and the same project builds into a native desktop app.

## Install

Start a new project with the scaffold. All you need is [Bun](https://bun.sh):

```bash
bun create shallot my-game
cd my-game
bun install
bunx shallot dev
```

Or add it to an existing project. TypeGPU is a required peer, and your bundler needs exactly one TypeGPU transform:

```bash
bun add @dylanebert/shallot typegpu@~0.12.5
bun add -d unplugin-typegpu@~0.12.3
```

A `shallot.json` project gets the transform from the CLI. An ejected Vite app wires it by hand:

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

## The CLI

Five verbs, run as `bunx shallot <verb> [dir]`:

- `create` starts a project. It's `bun create shallot <name>`, served by the scaffold.
- `dev` runs the project with hot reload.
- `build` ships a web bundle, or a desktop app with `--target windows|mac|linux`.
- `run` builds and runs.
- `add` copies a recipe out as a runnable, version-matched project. Bare `add` lists them.

Any other verb runs `shallot-<verb>` from your PATH or your project's installed bins, the way Cargo and Git do it. `bunx shallot --help` lists every option.

### Desktop builds

`bunx shallot build --target <platform> --release` downloads a prebuilt shell for your installed version from GitHub Releases, so a hit needs no Rust toolchain. A debug build, or any miss (404, offline, checksum mismatch, a source checkout), compiles the Rust window host from the crate source shipped in the package. That needs [Rust](https://rustup.rs) plus the target's system dependencies:

| target | system webview | portable (CEF) |
|---|---|---|
| mac | Xcode Command Line Tools | same, plus a CEF download on first build |
| linux | not supported: WebKitGTK has no usable WebGPU | `libx11-dev`, plus a CEF download on first build |
| windows | cross-compiled with cargo-xwin (`cargo install cargo-xwin`) | a Windows host with Visual Studio, the C++ workload and ATL |

`--portable` bundles the Chromium runtime (CEF) instead of the system webview. It's larger but runs anywhere, and Linux needs it. Set `CEF_PATH` to skip the download.

## Recipes

A recipe is one small project per problem: a first-person character, importing a model, a day-night sky, or driving a vehicle. [`examples/AGENTS.md`](examples/AGENTS.md) indexes every example in one line each, generated from their manifests, so grep it for the problem you have. Then copy one out:

```bash
bunx shallot add first-person
```

Recipes ship in the npm package.

## Extensions

Anything that isn't the engine lives in its own repository and consumes the published package:

- [shallot-avbd-physics](https://github.com/dylanebert/shallot-avbd-physics): an AVBD solver that plugs into the physics seam.
- [shallot-site](https://github.com/dylanebert/shallot-site): the demos at [dylanebert.com/shallot](https://dylanebert.com/shallot/), each built from a release tag and linked to its source.
- [shallot-bench](https://github.com/dylanebert/shallot-bench): an agent benchmark that installs the engine at a pinned tag.
- [create-shallot](https://github.com/dylanebert/create-shallot): the scaffold behind `bun create shallot`.

## Reference

The source is the reference. Every public export carries a JSDoc contract, and there's no separate docs site to drift from it. Questions go to [Discord](https://discord.gg/eEY75Nqk3C), bugs to [issues](https://github.com/dylanebert/shallot/issues), releases to [npm](https://www.npmjs.com/package/@dylanebert/shallot).

## Working on the engine

The engine pins Bun 1.4.2, Rust 1.98.1 with the `wasm32-unknown-unknown` target, and TypeScript 7; `wasm-opt` from binaryen is optional.

```bash
git clone https://github.com/dylanebert/shallot
cd shallot
bun install
bun run build
bun run check
```

[`CONTRIBUTING.md`](CONTRIBUTING.md) covers the layout, commands, tests and conventions. Retired units live in Git tags rather than the tree, and [`ARCHIVE.md`](ARCHIVE.md) lists each one with its tag.

## License

MIT, see [`LICENSE`](LICENSE).
