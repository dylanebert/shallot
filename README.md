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
accessible. CI's least token is a custom token with Account → Workers Scripts → Edit
on the one account; the "Edit Cloudflare Workers" template is broader. Defaults taken are printed.

`package.json` keeps the destination; commit it and the dependency changes:

```json
{
    "shallot": {
        "publish": {
            "target": "cloudflare",
            "name": "my-game",
            "accountId": "0123456789abcdef0123456789abcdef"
        }
    }
}
```

The link always binds the account id and Worker name; the id is not a credential.
A different resolved account refuses, even when it is the login's only account.
After building, a first publish checks the name right before deploying and saves
the link only after a successful deploy. A collision prompts for another name in
a terminal and refuses otherwise.
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

### itch.io

```bash
bunx shallot publish --target itch
```

pushes the same build to the page's `html5` channel with itch.io's butler, fetched
at its pinned version into Shallot's cache; butler owns login and credentials
(`butler login` in a terminal, `BUTLER_API_KEY` otherwise). butler's login key has
`wharf` scope: it manages builds on every project that identity can reach, not one
game. An account-settings API key can be broader still. itch.io allows 1,000
files, 200 MB per file and 500 MB in all, and serves the build with its own headers.
publish waits until itch.io has processed the build, then plays the document the
page's embed frames on `html.itch.zone`: itch.io puts its pages behind a bot check
for automated browsers. itch.io serves a new build only once its page is next
viewed; until then publish exits nonzero naming the page to open, and the next run
verifies it.

itch.io's web UI owns the page: create it with Kind of project HTML and name it in
`package.json` as `{"target": "itch", "name": "<user>/<game>"}`; publish refuses a
project butler cannot find. On its Edit page, tick "This file will be played in the
browser" and "SharedArrayBuffer support", and set Visibility to Public or
Restricted, not Draft. Verification cannot see these settings; publish names them.

Publish adds no credentials to the build. Vite's client values, public files and
source maps can still disclose what the project puts there. Never put secrets in them.

## more

- questions: [discord](https://discord.gg/eEY75Nqk3C). bugs: [issues](https://github.com/dylanebert/shallot/issues). releases: [npm](https://www.npmjs.com/package/@dylanebert/shallot)
- upgrading from 0.10: [`MIGRATION.md`](MIGRATION.md)
- changing the engine: [contributing](https://github.com/dylanebert/shallot/blob/main/CONTRIBUTING.md)

mit, see [`LICENSE`](LICENSE).
