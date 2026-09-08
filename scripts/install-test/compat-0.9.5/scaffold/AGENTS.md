# compat-app

A WebGPU game built on `@dylanebert/shallot`.

## Layout

- `shallot.json` — the manifest: which scene to open + which plugins to enable
- `public/scenes/*.scene` — the world as declarative XML (each `<a>` is an entity, each attribute a component)
- `src/*.ts` — your plugins (a plugin is data: components + systems)

## Build, run, verify

```bash
bunx tsc --noEmit                                # typecheck — run after every change
bunx shallot dev                                 # run with hot reload while you work
bunx shallot build                               # ship it (web bundle to dist/)
bunx shallot verify                              # prove it: boot headless, check it renders, exit 0/nonzero
```

`shallot verify` is the verification step — a self-terminating gate that boots the project in a real
headless browser, waits for a settled frame, and exits nonzero on failure. Nothing left running. `--json`
emits the full result; `--screenshot <path>` saves a frame. It drives Playwright, an optional one-time
install: `bun add -d playwright && bunx playwright install chromium` (exit code 3 names this command if
it's missing). By default it checks the scene rendered; to assert your own pass/fail (entity poses,
physics state) install `window.__harness` via `installHarness` from `@dylanebert/shallot/harness` — in a
manifest project like this one, from a plugin's `initialize(state)` hook (the engine's AGENTS.md has the
worked example).

## Engine reference

The engine is the documentation. Read `node_modules/@dylanebert/shallot/AGENTS.md` for the full
contract (ECS, plugins, scenes, GPU, UI, and the `shallot verify` harness), and every public export
carries JSDoc. The examples index lives at `node_modules/@dylanebert/shallot/examples/AGENTS.md` — grep
it for the problem you have, then read that recipe's source, before writing a pattern from scratch.
`bunx shallot recipe <name> [dir]` copies a recipe out of the installed package into a runnable project
(bare: lists them).

## Conventions

Data-oriented, ECS, declarative. Add components and systems, not methods — a `Jump` marker plus a
system, never `player.jump()`. Scenes declare; code transforms. One source of truth: every value has
one authoritative home; derive, don't duplicate.
