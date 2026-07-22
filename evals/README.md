# evals

An agent-agnostic eval suite: can a fresh coding agent, given only what ships on npm, build, observe,
and verify a shallot project? Each task is a self-contained problem an end user would ask for. The
harness sets up an isolated project, an agent works in it, and a withheld gate grades the result by
driving the running app — no cooperation from the agent's code required.

This is the acceptance gate and the iteration driver for making the shipped package the agent surface.
The measurement that matters: the delta in task completion with vs without version-matched context in
the tarball.

## The contract

- **Agents never see the gate.** `tasks/<task>/gate.ts` and `NOTES.md` stay in the repo. Setup copies
  only `PROMPT.md` into the project. An agent that could read the gate could game it.
- **Isolation is an out-of-tree temp dir**, not a worktree. A worktree isolates writes, not reads — the
  agent could still read the engine source and cheat. Setup installs a packed tarball into
  `os.tmpdir()`, so the agent sees only `node_modules/@dylanebert/shallot`: exactly what an npm user gets.
- **Gates assert positive behavior**, never the mere absence of errors. Each one drives the canvas the
  engine draws into — pixels, synthetic input — and checks the task's claim actually holds.

## Tasks

| task | problem | gate observes |
|------|---------|---------------|
| `red-box` | a static red cube on a dark background | centre is red, distinct from background, and holds still |
| `falling-box` | a box drops under gravity and lands | blue-pixel centroid moves down, then settles |
| `orbit-on-drag` | orbit the camera by dragging | idle view is stable; a drag changes it (causal) |
| `color-on-key` | spacebar turns a cube green | centre reads white before the press, green after |
| `persist-color` | number keys paint the cube; the colour survives a reload | key paints the target hue; a fresh load still reads it, no input |
| `striped-material` | a cube with a moving procedural pattern | face brightness oscillates (bands, not flat); two frames ~1s apart differ |

## Run one task

```bash
# 1. set up an isolated project (prints its path on the last line)
bun run eval:setup red-box
#    → /tmp/shallot-eval-red-box-XXXX/app

# 2. an agent works in that dir with only PROMPT.md + the installed engine

# 3. grade it — typecheck, build, then drive the withheld gate
bun run eval:grade red-box /tmp/shallot-eval-red-box-XXXX/app
```

`grade` prints per-check marks, the gate's assertions, a PASS/FAIL/INCOMPLETE verdict, and a
machine-readable JSON result. `--json` emits only the JSON.

`setup` takes `--bare`: it removes the shipped `examples/` corpus from the installed package and strips
the scaffold's pointer to it, leaving only the code, its JSDoc, and the product workflow. That's the
without-context arm — running the same task with and without `--bare` measures the shipped-context delta.

## Layout

- `setup.ts` — pack engine → scaffold via `create-shallot` → install → drop `PROMPT.md`. Emits the dir.
- `grade.ts` — typecheck + build + boot + drive the gate. Uses `harness/` for the browser path.
- `harness/lib.ts` — the shared gate driver: boot, screenshot, pixel/region/diff/centroid helpers.
- `harness/{server,playwright,wsl}.ts` — the self-contained browser path: server boot, `playwright test` runner, WSL→Windows staging.
- `harness/gate.config.ts`, `harness/package.json` — the Playwright config + deps staged to run a gate.
- `tasks/<task>/` — `PROMPT.md` (shown), `gate.ts` + `NOTES.md` (withheld).

## Notes

- The browser gate is **display-gated** like the rest of the harness (WSL → Windows Chrome; auto-skips
  with no display). `typecheck` and `build` always run; the gate reports `skipped` without a display.
- Physics and raster need a real GPU, so those gates only mean anything where a display is present.
- The result schema carries `verification` fields. This script fills the mechanical ones (`booted`,
  `rendered`); the judgment ones (did the agent *claim* it verified, and was that honest) are filled by
  whatever spawns the agent, from its transcript.
