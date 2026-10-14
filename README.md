[![test](https://github.com/dylanebert/shallot/actions/workflows/test.yml/badge.svg?branch=main)](https://github.com/dylanebert/shallot/actions/workflows/test.yml)

# shallot

webgpu game engine. fast by default, instant iteration, runs where webgpu does.

## quick start

Install [Bun](https://bun.sh/docs/installation). Create a project and start its development server:

```bash
bun create shallot my-game
cd my-game
bun install
bun run dev
```

Open the URL printed in the terminal in a WebGPU-capable browser, such as Chrome or Edge. Edit the scene in `src/welcome/index.ts`; changes reload the page.

## next steps

[Demos](https://dylanebert.com/shallot/). Copy the first-person example with `bun create shallot first-person --template first-person`, then install and run it as above.

In the project, `bunx shallot --help` lists commands.

## more

- questions: [discord](https://discord.gg/eEY75Nqk3C). bugs: [issues](https://github.com/dylanebert/shallot/issues). releases: [npm](https://www.npmjs.com/package/@dylanebert/shallot)
- changing the engine: [`CONTRIBUTING.md`](CONTRIBUTING.md)

mit, see [`LICENSE`](LICENSE).
