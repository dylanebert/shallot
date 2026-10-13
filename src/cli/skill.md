---
name: shallot
description: Operate a Shallot project: develop, check, and build it with the app's file layout and verification tiers.
---

# Shallot project

From the project root, run `bun install` before using these commands.

- `bun run dev` starts Vite; `bun run build` builds the web app; `bun run preview` serves that build.
- The root `index.html` imports the app's plugins; `vite.config.ts` enables `shallot()`.
- `bun test` runs cheap `*.test.ts` and GPU `*.gpu.test.ts` files; it does not discover Node or browser checks.
- Run Node `*.node.ts` checks by path with `bun test ./<file>`.
- Run browser `*.e2e.ts` checks with `bunx playwright test`; keep the app's config in root `playwright.config.ts`.
- Put each plugin in `src/<plugin>/index.ts`; keep its module checks beside it.
- Put whole-app checks in `tests/`; the directory sets scope, the suffix sets tier.
- Put full default-renderer or physics-kernel compositions in `tests/<goal>.node.ts`, not `*.gpu.test.ts`; run them by path.
- `bunfig.toml` preloads `tests/preload.ts`, which registers Shallot's Bun plugin for the TypeGPU transform.
- Package scripts are `dev`, `build` and `preview`; checks run by the verbs above, with no `test`, `check` or renamed tier script.
