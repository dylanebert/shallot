# Shallot

webgpu game engine

- fast by default
- instant iteration
- runs anywhere

## quick start

All you need is [bun](https://bun.sh):

```bash
bun create shallot my-game
cd my-game
bun install
bunx shallot dev
```

`bunx shallot dev` runs the project with hot reload, and `bunx shallot build` ships it as a web bundle. Native builds (`--target windows|mac|linux`) run from a checkout of this repo, not an installed package: the rust crate they need isn't published to npm.

A project is plain data: a `shallot.json` manifest, a scene file, and TypeScript plugins you edit in your IDE. `bunx shallot verify` boots the project in a headless browser and exits 0 or nonzero — a check you, an agent, or CI can run to prove the game still works.

## the repo is the docs

There's no docs site. The source is the reference — every public export carries a JSDoc contract — and two files carry the rest:

- [`packages/shallot/AGENTS.md`](packages/shallot/AGENTS.md) — the consumer contract: commands, the ECS and plugin conventions, the GPU and render rules. Ships with the npm package.
- [`examples/AGENTS.md`](examples/AGENTS.md) — the examples index: one line per entry, so you grep for the problem you have.

Written for coding agents first — point Claude at the repo and it can build a game — and just as readable by hand.

## examples

Examples live under `examples/`, indexed by [`examples/AGENTS.md`](examples/AGENTS.md):

- `recipes/` — one minimal project per problem: first-person character, physics playground, import a model, day-night sky, and more.
- `showcase/` — richer exhibits: `collapse`, `sandbox`, `fountain`, `voxel`, `visualization`.
- `gym/` — machine-verdict scenarios: the real-device test and benchmark tier.

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
- [wasm-pack](https://github.com/wasm-bindgen/wasm-pack)
- `wasm-opt` from [binaryen](https://github.com/WebAssembly/binaryen) — optional, build falls back to a copy

```bash
git clone https://github.com/dylanebert/shallot
cd shallot
bun install
bun run build
```

`build` compiles the rust crates (transforms wasm, audio wasm, native window host).

### layout

- `packages/shallot/` — the engine. published as `@dylanebert/shallot`
- `packages/create-shallot/` — `bun create shallot` scaffold
- `packages/vscode-shallot/` — VS Code extension
- `examples/` — example projects against the engine

### commands

run from the repo root.

```bash
bun test           # unit tests (bun-webgpu)
bun bench          # gpu benchmarks
bun check          # format + type check
bun run format     # biome + scene formatter
bun run build      # rust artifacts
```

`bun check` and `bun test` should pass before pushing. `bun bench` after gpu changes.

Issues are open; pull requests are by invitation — see [CONTRIBUTING.md](CONTRIBUTING.md).

## license

MIT
