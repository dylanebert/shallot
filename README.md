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

# list commands
bunx shallot --help

# run the game
bun run dev
```

open the url in a [supported browser](https://caniuse.com/webgpu).

## examples

get started by exploring [examples](https://dylanebert.com/shallot/). start from one:

```bash
# list them
bun create shallot --help

# copy one
bun create shallot first-person --template first-person
```

## more

- questions: [discord](https://discord.gg/eEY75Nqk3C). bugs: [issues](https://github.com/dylanebert/shallot/issues). releases: [npm](https://www.npmjs.com/package/@dylanebert/shallot)
- upgrading from 0.10: [`MIGRATION.md`](MIGRATION.md)
- changing the engine: [`CONTRIBUTING.md`](CONTRIBUTING.md)

mit, see [`LICENSE`](LICENSE).
