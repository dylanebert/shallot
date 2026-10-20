---
name: shallot
description: Learn a Shallot project's API from the installed package and check its work.
---

# Shallot project

Explore this project's entry, plugins and checks first; reuse its existing setup.

- Learn the API from the installed package, not memory: `node_modules/@dylanebert/shallot/examples/` holds working projects and their checks; `node_modules/@dylanebert/shallot/src/` holds the contracts in JSDoc.
- Run `bunx shallot create --help` to find the nearest example by its description. Read its implementation and checks, then the source for the APIs you use before writing code.
- Check gameplay by driving the composed world through `World.tick` / `World.step` and asserting state, as in `examples/first-person/tests/ascent.node.ts` inside the installed package.
- Use the cheapest evidence: state first, then `captureFrame` from `@dylanebert/shallot/rendering` as in `examples/hud/tests/stepped.html` inside the installed package, then browser pixels when needed.
- Put each plugin in `src/<plugin>/index.ts`, with module checks beside it; put whole-app checks in `tests/`.
- `bun test` discovers cheap `*.test.ts` and GPU `*.gpu.test.ts` checks, not Node or browser checks.
- Run Node `*.node.ts` checks by path: `bun test ./<file>`. Full default-renderer or physics-kernel compositions belong in `tests/<goal>.node.ts`, not `*.gpu.test.ts`.
