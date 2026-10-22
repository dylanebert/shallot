---
name: shallot
description: Learn a Shallot project's API from the installed package and check its work.
---

# Shallot project

Explore this project's entry, plugins and checks first; reuse its setup.

- Learn the API from the installed package, not memory. `bunx shallot create --help` lists the examples in `node_modules/@dylanebert/shallot/examples/`; start from the nearest one, its implementation and its checks, then the JSDoc in `node_modules/@dylanebert/shallot/src/` for the APIs it uses.
- Check your work as that example's checks do, with the cheapest evidence that answers: state, then a captured frame, then the browser.
- While you iterate, run only the check that answers your change: `bun test ./<file>` or `bunx playwright test <file>`. Before you finish, run `bun test` once; CI, `.github/workflows/test.yml`, runs every tier.
- `bun test` runs `*.test.ts` and `*.gpu.test.ts` only; run a `*.node.ts` check by path with `bun test ./<file>`. A full renderer or physics composition is a `tests/<goal>.node.ts` check, not a `*.gpu.test.ts`.
- `bunx shallot publish` builds, publishes to Cloudflare, or to itch.io with `--target itch`, and verifies the public game; read `bunx shallot publish --help` first. Commit its `package.json` destination and dependency changes. A first-link name collision refuses rather than overwrites; do not write `shallot.publish` by hand unless updating that Worker is explicitly intended. An itch.io page is created and set up by a person on itch.io; publish names the settings it cannot verify.
- Publishing without interaction needs the provider's credentials (`CLOUDFLARE_API_TOKEN`, plus `CLOUDFLARE_ACCOUNT_ID` when ambiguous; `BUTLER_API_KEY` for itch.io). Never put them in `package.json` or Vite client environment values: client values are public.
