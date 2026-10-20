---
name: shallot
description: Operate a Shallot project: learn its API, develop, check, and build it.
---

# Shallot project

Explore this project's entry, plugins and checks first. Your training may predate this Shallot version; learn its API from the installed package, not memory.

Run `bun install` from the project root before using these commands or reading the installed package.

- Use `node_modules/@dylanebert/shallot/`: `examples/` holds working projects and their checks; `src/` holds the API contracts in JSDoc; `package.json` lists public exports.
- Run `bunx shallot create --help` to find the nearest example by its description. Read its implementation and checks, then the source for the APIs you use before writing code. Import public exports, not package-internal files.
- Check gameplay by driving the composed world through `World.tick` / `World.step` and asserting state, as in `examples/first-person/tests/ascent.node.ts`. Drive inputs and observe results through the public APIs those checks use.
- Use the cheapest evidence: state first, then `captureFrame` from `@dylanebert/shallot/rendering` as in `examples/hud/tests/stepped.html`, then browser pixels when needed. Run the checks covering your change; a build alone does not prove gameplay or presentation.

## Loop and layout

- `bun run dev` starts Vite; `bun run build` builds the web app; `bun run preview` serves that build. `bunx shallot --help` lists commands.
- The root `index.html` imports the app's plugins; `vite.config.ts` enables `shallot()`.
- Put each plugin in `src/<plugin>/index.ts`, with module checks beside it; put whole-app checks in `tests/`. The directory sets scope, the suffix sets tier.
- `bun test` discovers cheap `*.test.ts` and GPU `*.gpu.test.ts` checks, not Node or browser checks.
- Run Node `*.node.ts` checks by path: `bun test ./<file>`. Full default-renderer or physics-kernel compositions belong in `tests/<goal>.node.ts`, not `*.gpu.test.ts`.
- Run `bunx playwright install chromium` once, then `bunx playwright test` for browser `*.e2e.ts` checks; the project's `playwright.config.ts` starts its server.
- `bunfig.toml` preloads `tests/preload.ts`, which registers Shallot's Bun plugin for the TypeGPU transform.
- For projects with Svelte components, `bun run typecheck` runs `svelte-check`.
