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

A project is plain data plus code: a `shallot.json` manifest, a `.scene` file, and TypeScript plugins you edit in your IDE.

`bunx shallot verify` boots the project in a headless browser and exits 0 or nonzero, a check you, an agent, or CI can run to catch a project that no longer boots or renders. It drives a real browser through the optional playwright peer, so install that once per project: `bun add -d playwright && bunx playwright install chromium`.

## the repo is the docs

There's no docs site. The source is the reference — every public export carries a JSDoc contract — and two files carry the rest:

- [`packages/shallot/AGENTS.md`](packages/shallot/AGENTS.md) — the consumer contract: commands, the ECS and plugin conventions, the GPU and render rules. Ships with the npm package.
- [`examples/AGENTS.md`](examples/AGENTS.md) — the examples index: one line per entry, so you grep for the problem you have.

Written for coding agents first — point Claude at the repo and it can build a game — and just as readable by hand.

## examples

Examples live under `examples/`, indexed by [`examples/AGENTS.md`](examples/AGENTS.md):

- `recipes/` — one minimal project per problem: first-person character, physics playground, import a model, day-night sky, and more.
- `showcase/` — full projects rather than one concept each, several under real-device gates: `collapse`, `sandbox`, `fountain`, `voxel`, `visualization`.
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

### layout

- `packages/shallot/` — the engine. published as `@dylanebert/shallot`
- `packages/create-shallot/` — `bun create shallot` scaffold
- `packages/vscode-shallot/` — VS Code extension
- `examples/` — example projects against the engine

### commands

run from the repo root.

```bash
bun test           # unit tests (bun-webgpu)
bun bench          # GPU benchmarks
bun check          # format (writes) + tsc + biome + eslint + repo checks
bun run format     # biome + scene formatter
bun run build      # rust artifacts
```

Engine-internal layout, the full command table, and the rules index are in [`AGENTS.md`](AGENTS.md); the conventions themselves are path-scoped under [`.claude/rules/`](.claude/rules/).

`bun check` and `bun test` are the gate before pushing. The by-path slow suites, the invitation-only PR policy, and where to file an issue are in [CONTRIBUTING.md](CONTRIBUTING.md).

## license

MIT
