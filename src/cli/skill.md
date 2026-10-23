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
- `bunx shallot publish` builds, publishes to Cloudflare, or to itch.io with `--target itch`, and verifies the public game; read `bunx shallot publish --help` first. Commit its `package.json` destination and dependency changes. A first-link name collision refuses rather than overwrites; do not write `shallot.publish` by hand unless updating that Worker is explicitly intended. An itch.io page is created and set up by a person on itch.io; publish names the settings it cannot verify. A processed itch.io build awaiting a page view succeeds without interaction but is reported unverified; open the named page, then publish again to verify it.
- Cleanup, rollback and listing stay the provider's; `bunx shallot publish --help` names each command from the project's link. Confirm with the person before deleting a Worker or taking a page down: Wrangler's delete asks nothing without a TTY or in CI.
- Publishing without interaction needs the provider's credentials (`CLOUDFLARE_API_TOKEN`, plus `CLOUDFLARE_ACCOUNT_ID` when ambiguous; butler's stored credentials or `BUTLER_API_KEY` for itch.io). CI's least Cloudflare token is custom, with Account → Workers Scripts → Edit on the one account; the "Edit Cloudflare Workers" template is broader. butler's login key has `wharf` scope, managing builds on every project the identity can reach, not one game; an account-settings API key can be broader still. Never put credentials in `package.json`. Publish adds no credentials to the build; Vite's client values, public files and source maps can still disclose what the project puts there. Never put secrets in them.
