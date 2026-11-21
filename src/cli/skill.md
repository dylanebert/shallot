---
name: shallot
description: Learn a Shallot project's API from the installed package and check its work.
---

# Shallot project

Explore this project's entry, plugins and checks first; reuse its setup.

- Learn the API from the installed package, not memory. `bunx shallot create --help` lists the examples in `node_modules/@dylanebert/shallot/examples/`; start from the nearest one, its implementation and its checks, then the JSDoc in `node_modules/@dylanebert/shallot/src/` for the APIs it uses.
- Check your work as that example's checks do, with the cheapest evidence that answers: state, then a captured frame, then the browser.
- While you iterate, run only the check that answers your change: `bun test ./<file>` or `bunx playwright test <file>`. Before you finish, run `bun check` and `bun test` once; CI, `.github/workflows/test.yml`, runs every tier.
- `bun test` runs `*.test.ts` and `*.gpu.test.ts` only; run a `*.node.ts` check by path with `bun test ./<file>`. A full renderer or physics composition is a `tests/<goal>.node.ts` check, not a `*.gpu.test.ts`.
- `bunx shallot publish` builds, publishes to Cloudflare, or to itch.io with `--target itch`, and verifies the public game; read `bunx shallot publish --help` first. Commit its `package.json` destination and dependency changes. A first-link name collision refuses rather than overwrites; do not write `shallot.publish` by hand unless updating that Worker is explicitly intended. An itch.io page is created and set up by a person on itch.io; publish names the settings it cannot verify. A processed itch.io build awaiting a page view succeeds without interaction but is reported unverified; open the named page, then publish again to verify it.
- Request a server with `bunx shallot publish --server fly`; its app name defaults from the project, is asked in a terminal and reported otherwise. The saved server link carries the choice after that. Publish refuses an existing app on a first link; a complete hand-written link authorizes updating it. Without `--server` or a server link, `server.ts` does not opt into hosting
- On Fly, store server records under `process.env.SHALLOT_DATA_DIR`. Publish mounts one 1 GB volume on its single Machine and refuses to retrofit existing apps outside that layout. Save dirty records on graceful shutdown; Fly publish selects `SIGTERM`.
- Use publish's quoted Fly CLI path as `<flyctl>`: logs are `"<flyctl>" logs --app <app>`. For server rollback, check out an earlier commit, then publish. Confirm with the person before server deletion: `"<flyctl>" apps destroy <app>`
- Cleanup, rollback and listing stay the provider's; `bunx shallot publish --help` names each command from the project's link. Confirm with the person before deleting a Worker or taking a page down: Wrangler's delete asks nothing without a TTY or in CI
- Publishing without interaction needs the provider's credentials (`CLOUDFLARE_API_TOKEN`, plus `CLOUDFLARE_ACCOUNT_ID` when ambiguous; butler's stored credentials or `BUTLER_API_KEY` for itch.io; a saved Fly login or `FLY_API_TOKEN` for Fly). CI's least Cloudflare token is custom, with Account → Workers Scripts → Edit on the one account; the "Edit Cloudflare Workers" template is broader. butler's login key has `wharf` scope, managing builds on every project the identity can reach, not one game; an account-settings API key can be broader still. Never put credentials in `package.json`. Publish adds no credentials to the build; Vite's client values, public files and source maps can still disclose what the project puts there. Never put secrets in them.
