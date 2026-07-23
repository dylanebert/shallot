# @dylanebert/shallot

webgpu game engine

- fast by default
- instant iteration
- runs anywhere

## start a project

```bash
bun create shallot my-game
cd my-game
bun install
bunx shallot dev    # run it, with hot reload
```

`bunx shallot build` ships it — web by default, or `--target windows|mac|linux` for native. `bunx shallot verify` boots the project in a headless browser and exits 0 or nonzero.

## add to an existing project

```bash
bun install @dylanebert/shallot
```

## docs

The repo is the documentation. [`AGENTS.md`](AGENTS.md) is the consumer contract — commands, the ECS and plugin conventions, the GPU and render rules — and [`examples/AGENTS.md`](examples/AGENTS.md) indexes a runnable project per problem. `bunx shallot recipe <name>` copies one out as a runnable project.

## links

- [agents.md](AGENTS.md)
- [github](https://github.com/dylanebert/shallot)
- [discord](https://discord.gg/eEY75Nqk3C)

## license

MIT
