[![test](https://github.com/dylanebert/shallot/actions/workflows/test.yml/badge.svg?branch=main)](https://github.com/dylanebert/shallot/actions/workflows/test.yml)

# shallot

webgpu game engine. fast by default, instant iteration, runs where webgpu does.

## quick start

```bash
# install bun
curl -fsSL https://bun.sh/install | bash

# new project
bun create shallot my-game
cd my-game
bun install

# run it
bunx shallot dev
```

The project owns its `index.html` and Vite config, with `plugins: [shallot()]`.

For web, `shallot dev`, `shallot build` and `shallot preview` run the project's Vite commands. Native `dev` and `build` add the desktop shell; native `preview` launches that build.

Use `bun test` for Bun checks and `bunx playwright test` for browser checks. `bunx shallot --help` lists every command.

## more

- demos: [dylanebert.com/shallot](https://dylanebert.com/shallot/)
- examples: `bunx shallot add`, or [`examples/AGENTS.md`](examples/AGENTS.md)
- questions: [discord](https://discord.gg/eEY75Nqk3C). bugs: [issues](https://github.com/dylanebert/shallot/issues). releases: [npm](https://www.npmjs.com/package/@dylanebert/shallot)
- changing the engine: [`CONTRIBUTING.md`](CONTRIBUTING.md)

mit, see [`LICENSE`](LICENSE).
