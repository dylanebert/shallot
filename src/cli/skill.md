---
name: shallot
description: Learn a Shallot project's API from the installed package and check its work.
---

# Shallot project

Explore this project's entry, plugins and checks first; reuse its setup.

- Learn the API from the installed package, not memory. `bunx shallot create --help` lists the examples in `node_modules/@dylanebert/shallot/examples/`; start from the nearest one, its implementation and its checks, then the JSDoc in `node_modules/@dylanebert/shallot/src/` for the APIs it uses.
- Check your work as that example's checks do, with the cheapest evidence that answers: state, then a captured frame, then the browser.
- `bun test` runs `*.test.ts` and `*.gpu.test.ts` only; run a `*.node.ts` check by path with `bun test ./<file>`. A full renderer or physics composition is a `tests/<goal>.node.ts` check, not a `*.gpu.test.ts`.
