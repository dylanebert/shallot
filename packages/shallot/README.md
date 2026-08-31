# @dylanebert/shallot

webgpu game engine

- fast by default
- instant iteration
- runs in any WebGPU browser, or native

## start a project

```bash
bun create shallot my-game
cd my-game
bun install
bunx shallot dev    # run it, with hot reload
```

`bunx shallot build` ships it as a web bundle. `bunx shallot verify` boots the project in a headless browser and exits 0 or nonzero; it needs the optional playwright peer (`bun add -d playwright && bunx playwright install chromium`). `bunx shallot build --target windows|mac|linux --release` downloads a prebuilt shell for that version from GitHub Releases, so no Rust toolchain is needed on a hit. A debug build, or any miss (404, offline, checksum mismatch, a source checkout), compiles the Rust window host from source, which needs the Rust toolchain plus that target's system dependencies.

## add to an existing project

```bash
bun add @dylanebert/shallot typegpu@~0.12.4
bun add -d unplugin-typegpu@~0.12.3
```

TypeGPU is a required peer, and TGSL needs exactly one TypeGPU transform in your bundler. A `shallot.json` project gets that from the CLI. An ejected Vite app adds `typegpu()` from `unplugin-typegpu/vite` plus `optimizeDeps: { exclude: ["@dylanebert/shallot", "typegpu"] }` — [MIGRATION.md](https://github.com/dylanebert/shallot/blob/main/packages/shallot/MIGRATION.md) has the full setup.

## docs

The docs ship with the package. [`AGENTS.md`](https://github.com/dylanebert/shallot/blob/main/packages/shallot/AGENTS.md) is the consumer contract: commands, the ECS and plugin conventions, the GPU and render rules. [`examples/AGENTS.md`](https://github.com/dylanebert/shallot/blob/main/examples/AGENTS.md) indexes one recipe per problem, to read in place; `bunx shallot recipe <name> [dir]` copies one out as a runnable, version-matched project.

Porting from 0.8? [MIGRATION.md](https://github.com/dylanebert/shallot/blob/main/packages/shallot/MIGRATION.md) is the GPU-consumer port.

## links

- [agents.md](https://github.com/dylanebert/shallot/blob/main/packages/shallot/AGENTS.md)
- [github](https://github.com/dylanebert/shallot)
- [discord](https://discord.gg/eEY75Nqk3C)

## license

MIT
