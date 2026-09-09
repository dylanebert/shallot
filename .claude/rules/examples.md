---
paths:
    - "examples/**/*.ts"
    - "examples/**/*.scene"
    - "examples/AGENTS.md"
    - "bench/**/*.ts"
---

# Examples

Two kinds: a recipe answers one problem with one check; a showcase is a project a person would play, composing many and running only gates a user's project could. What fits neither is a test or a bench, living with the code it claims. Admission: `strategy/shallot.md`.

Corpus/JSDoc drift/bloat: bugs. `examples/AGENTS.md`: problem/path/demo. Add/rename/delete entry + index together. Update in place, never -v2; delete unserved problems. Awkward recipes indict the API.

## Recipes

One problem, not a module. Minimal real defaults, no unrelated entities or float garbage. Teach reusable API, comment only why. Demonstrate on open or hint interaction in world-space text; no console-only gestures.

Self-contained shallot.json/public/scenes/src plugins; no HTML/Vite or cross-recipe imports. Root tsc exercises API in a manifest-declared plugin. Dynamics need src/smoke.ts assertions + a `CHECKS` row; compilation isn't behavior.

A row with no check declares why: `static` never moves, `bootOnly` moves with nothing asserting it yet. Both gate on boot plus a nonblank render; neither keeps a smoke.

Physics recipes cite the tumble gold twins (`packages/shallot-tumble/tests`). Teach published substrate; beyond it, name the twin. If none exists, use hatch + twin, never a lesser substitute.

## Showcase

Rich self-contained capabilities, own/dogfood published-surface gates, never repo scripts. Check tier before quality. Site builds published-package ejected consumers.

Showcase perf: ProfilePlugin frame percentiles/GPU spans, real input for interaction costs. Gate structural mechanisms/in-run ratios; absolute timing ungated. Attribute remote reports locally first; no local red: name dominant cost + ungated reading.

RUM intake drives above/below 50ms, asserting the duration-vital call AND the intercepted wire; fulfill matching requests locally, never forward.

## Bench

Not an example: the scenario tier, `bench/`. `gym.ts` owns params/build/optional assert; params drive URL and `--param`, with no live UI. Add file + barrel import + a `covers` row (`timeouts.ts`), the table `--for` and the roster read.

Mirror readback/profiler timing, no CPU timer; deterministic start. Asserts own bounded advancement after the profiler window (`testing.md`), never its duration. GPU physics and recipe-bound atoms hold here; `bench/` is not their home.

## Conventions

CLI runs manifests. Bench/visualization: HTML/Vite + bun dev. All ship public/icon.svg; HTML owners link it, CLI supplies the manifest favicon. Native icon.png overrides. Dispose State on HMR/unmount.

Package AGENTS UI rules apply; only ejected never-embedded bench/visualization may fix the viewport. Complex UI: own-package svelte/@sveltejs/vite-plugin-svelte, svelte.config.js, mount/unmount.
