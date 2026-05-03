# Shallot

webgpu game engine

> in development. not yet ready for use

- fast by default
- instant iteration
- runs anywhere

## prerequisites

- [bun](https://bun.sh)
- [rust](https://rustup.rs) with the `wasm32-unknown-unknown` target (`rustup target add wasm32-unknown-unknown`)
- [wasm-pack](https://rustwasm.github.io/wasm-pack/)
- `wasm-opt` from [binaryen](https://github.com/WebAssembly/binaryen) — optional, build falls back to a copy

## quick start

```bash
git clone https://github.com/dylanebert/shallot
cd shallot
bun install
bun run build
```

`build` compiles the rust crates (transforms wasm, audio wasm, native window host) and the docs site.

run an example:

```bash
cd examples/hello-cube
bun dev
```

other examples: `gym`, `raytracing`, `react-cube`, `svelte-cube`, `visualization`, `workstation`.

## layout

- `packages/shallot/` — the engine. published as `@dylanebert/shallot`
- `packages/shallot/editor/` — Svelte editor app
- `packages/create-shallot/` — `bun create shallot` scaffold
- `packages/vscode-shallot/` — VS Code extension
- `examples/` — standalone vite projects against the engine
- `docs/` — guide, engine, standard, extras, editor. Reference tables generated from JSDoc by `bun run build`

## commands

run from the repo root.

```bash
bun test           # unit tests (bun-webgpu)
bun bench          # gpu benchmarks
bun check          # format + type check
bun run format     # biome + scene formatter
bun run build      # rust artifacts + docs
```

`bun check` and `bun test` should pass before pushing. `bun bench` after gpu changes.

## editor

```bash
bunx shallot examples/hello-cube
```

opens the editor on the example scene.

## consumer install

if you just want to use shallot in a project of your own:

```bash
bun create shallot my-game
```

## license

MIT
