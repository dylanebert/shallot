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

## publish

```bash
bunx shallot publish
# or explicitly choose the default target
bunx shallot publish --target cloudflare
```

builds with your Vite config, checks Cloudflare Workers Static Assets' free limits
(20,000 files, 25 MiB per file), deploys to a lasting `workers.dev` link, then
checks that link anonymously with your project's Playwright Chromium. Chromium is
installed inline when missing. stdout is the play link; progress, the build's link,
verification and provider commands go to stderr. A failed verification exits nonzero,
but the uploaded game's link is still printed.

first publish adds pinned Wrangler to devDependencies. It requires Node.js 22 or
newer on PATH. Wrangler owns login and
credential storage: a terminal can log in inline; a remote terminal uses its device
flow. Without a TTY, in CI or for an agent, no prompt runs: set
`CLOUDFLARE_API_TOKEN`, and `CLOUDFLARE_ACCOUNT_ID` when more than one account is
accessible. Defaults taken are printed.

`package.json` keeps the destination; commit it and the dependency changes:

```json
{
    "shallot": {
        "publish": {
            "target": "cloudflare",
            "name": "my-game"
        }
    }
}
```

`accountId` is also saved when the login reaches multiple accounts. A first-link
name collision prompts for another name in a terminal and refuses otherwise.
Writing a complete link by hand deliberately authorizes updating that Worker;
later publishes update the saved destination without a collision check. Credentials
never belong in this link.

publish writes only the link and dependency changes to your project. It builds in
a temporary directory, generates `_headers` there, and passes its own assets-only
configuration to Wrangler: no autoconfig or Cloudflare Vite plugin. Missing assets
return 404. Pages use the same isolation headers as dev and preview; hashed
`/assets/*` are immutable and `index.html` revalidates. `_headers`, `_redirects`,
and `.assetsignore` in a build are refused rather than silently replaced.

verification checks each built asset's bytes and content type, a missing path's
404, `crossOriginIsolated`, engine startup and console errors. If runtime startup
reports physics, it also requires threaded physics. A software adapter is named;
its startup and loading checks are not hardware-play qualification.

**Vite client environment values are public. Never put secrets in them.**

## more

- questions: [discord](https://discord.gg/eEY75Nqk3C). bugs: [issues](https://github.com/dylanebert/shallot/issues). releases: [npm](https://www.npmjs.com/package/@dylanebert/shallot)
- upgrading from 0.10: [`MIGRATION.md`](MIGRATION.md)
- changing the engine: [contributing](https://github.com/dylanebert/shallot/blob/main/CONTRIBUTING.md)

mit, see [`LICENSE`](LICENSE).
