---
paths:
    - "examples/**/*.ts"
    - "examples/**/*.scene"
    - "examples/AGENTS.md"
---

# Examples

`examples/recipes/` is the teaching corpus: one minimal project per problem a game developer actually has, named by the problem, indexed in `examples/AGENTS.md`. The corpus and the source JSDoc are the documentation — there is no docs site, so an entry that drifts or bloats is a documentation bug. This rule is the corpus contract: the recipe contract first, then the other tiers (gym, flows, showcase) and the corpus-wide conventions.

## What a recipe must be

- **Named by the problem, not the module.** `first-person`, `physics-playground`, `import-a-model` — the name is what a developer would search, never an engine module name. One problem per entry; a second concept is a second entry or a cut.
- **Minimal.** The one concept, real defaults, no unrelated scaffolding. No entity the entry doesn't teach, no float-garbage values in scenes. It's read by a beginner and by an agent grounding a pattern, so it earns the same readability bar as `standard/`.
- **Self-contained.** The canonical project shape: `shallot.json` (scene + plugin enablement) + `public/scenes/*.scene` + plugin modules under `src/`, run via `bunx shallot dev examples/recipes/<entry>/`. No `index.html`, no vite config. Recipes never import from each other or from repo `scripts/` — a small shared shape (a ragdoll build, a ground plane) duplicates rather than coupling entries.
- **Compile-gated.** The root `tsc` (in `bun check`) is the drift gate. An entry earns it by **using the module's API in code, not only in scene attributes** — a scene is untyped, so a declarative-only entry wouldn't break when an export is renamed. Reference the component in a small plugin module the manifest declares, so a rename fails `tsc`.
- **Behavior-gated when the concept moves.** Compiling proves the API is named, not that it runs. A recipe whose concept is a runtime observable (physics motion, an event firing, an interaction responding) also ships a `src/smoke.ts` harness plugin that asserts that observable through `window.__harness`, plus an entry in `scripts/recipes.ts`, and `bun run recipes` is its standing gate. Compile-gating alone is insufficient for a dynamic concept: an entry that only typechecks can ship dead — the platform never slides, the joint never breaks — while the typecheck stays green.
- **Demonstrates on open.** The concept is observable the moment the entry opens, or a world-space `text` label names the interaction that reveals it. Console-only output, an unhinted key, or a gesture-gated effect with no on-screen hint is a silent recipe — a documentation bug.
- **Teaches the feature, not the example.** Show how a user would use the API in any project, not how this entry is wired. Prose lives in the code as ordinary comments held to the comment rule (`style.md`) — a short *why* where the code can't say it, never narration.

## The index

`examples/AGENTS.md` is the retrieval surface — one line per entry: the problem, the path, what it shows. Every recipe has a line; a recipe without one is invisible to agents. Adding, renaming, or deleting an entry updates the index in the same commit. Keep lines greppable: lead with the problem phrasing a developer would use.

## Maintenance

- **A physics recipe cites its gym gold twin.** The recipe is documentation on the published substrate surface; its gym twin (`examples/gym`, an oracle-gated gold) is the verified home of the full behavior. Where a concept's mechanism lives past the published surface (a backend escape-hatch joint), the recipe teaches what the surface *can* express and names the twin; a recipe whose concept has no published-surface expression yet rides the hatch, says so, and cites the twin — never a faked lesser published-surface version.
- A new engine capability that changes how a problem is solved updates the recipe that owns the problem — in place, never a `-v2` sibling.
- A recipe whose problem the engine no longer serves is deleted with its index line, same commit.
- The API is the docs: if a recipe reads awkwardly, first suspect the public surface (an export or field decision is a documentation decision — `exports.md`), only then the recipe.

## The other tiers

- `gym/` — the **testing-harness** tier: param-driven atoms that are a correctness gate + benchmark + live demo at once (the triple-duty bar — one scene, no branching; `pile` is the model), one per subsystem path; the code is a harness, not a teaching reference. One project, `?scenario=`-selected. It's the **targeted real-device tier** — `bun bench --scenario X`, one atom run after its domain changes, opposite `bun test`'s hardware-invariant run-all (the split is `testing.md`). A lab scenario migrates here case-by-case, **folded** into the atom whose subsystem it exercises (a new param/mode), never mechanically file-moved. Keeps the engine honest about itself.
- `flows/` — the **standalone-app engine flows**: ejected apps that exercise engine behavior a `bun test` can't reach — `survive-reload` (a real reload through serialize/restore) and `ui-containment` (`config.ui` overlay clipping). Driven by `bun run flows` (`scripts/flows.ts` over `shallot verify`).
- `showcase/` — the **capability** tier: richer exhibits showing how to do something interesting (the voxel editor), one project per subdir. Each is a self-contained real project that **owns and dogfoods its own testing** — a gate written against the published `@dylanebert/shallot` surface + the project's own driver (bring-your-own Playwright, or `shallot verify`; `voxel`'s `src/gate.ts` + `test/voxel.spec.ts` are the worked example), never reaching into repo `scripts/`. Standalone "wow".

## Gym scenario contract

The gym contract is `examples/gym/src/gym.ts`: a `Scenario` is `params` + `build` + optional `assert` + `live`. `params` is the single source of truth for the scenario's tunables — the URL parses them, `bun bench --param key=value` sets them (they ride `shallot verify --query`), and the live top-right control panel auto-renders from them (a `rebuild` knob reloads, a live knob mutates in place). Adding a scenario is a new file + a `scenarios/index.ts` import. **Assert readback is `Mirror`** (not the legacy `compute/readback`); **timing is the profiler** (`window.__benchmark`, GPU passes incl. `part:pack` / `bvh:sort` / `bvh:build` / `bvh:trace` — the source of truth, don't hand-roll a CPU measure). A scenario has **no environment awareness** — the same page runs headless or in a tab; a scene carries an orbit camera you drive live, while headless leaves it at its deterministic start so the `assert` verifies (e.g. by varying `Camera.far`, orbit-independent). F3 toggles the stats panel. The scenarios and the GPU-driven coverage each carries are enumerated in the `examples/gym/src/scenarios/index.ts` barrel header — the single home for that list. Gym installs the published `window.__harness` (`ready` + `run`); `gym.ts` translates a scenario's internal verdict to the published wire `Verdict` (`ok` + checks, metrics riding through as an extra) that `shallot verify` reads.

## Corpus-wide conventions

- `gym` is a single project with `?scenario=`-selected scenarios; `recipes` + `flows` + `showcase` hold one project per subdir. A manifest project is pure data (`shallot.json` + plugin modules + `public/`) run through the CLI: `bunx shallot dev examples/recipes/<entry>/` — there is no per-project `bun dev`. The ejected exceptions own their own `index.html` (+ vite): `gym` and `showcase/visualization` run with `cd examples/<project> && bun dev`; the `flows/` apps are verify-only (`shallot verify` boots them, no standalone `bun dev`).
- Every example must include `public/icon.svg` (the shallot icon); the examples that own an `index.html` (gym, `showcase/visualization`, the `flows/` apps) add `<link rel="icon" type="image/svg+xml" href="/icon.svg" />` there, while every manifest project has none — the `shallot dev` server and the synthesized `shallot build` entry supply it. For native builds, `public/icon.png` becomes the window icon; if absent, the default shallot icon is used. Always `dispose()` State on HMR/unmount — without it, each hot-reload stacks another State + RAF loop.
- **UI.** App UI mounts into one engine-provided, canvas-bounded, sandboxed container — the full contract (`config.ui` / `mountOverlay` / never `position: fixed` / containment) lives in `packages/shallot/AGENTS.md` "UI". Examples follow it. **The one exemption:** an *ejected* example that owns its full page and is never embedded (`gym`, `showcase/visualization` — they own `index.html` + their own vite) may own the viewport directly with `position: fixed`. Complex example UI owns its page with Svelte — add `svelte` + `@sveltejs/vite-plugin-svelte` + a `svelte.config.js` to that example's own package, mount via `svelte`'s `mount()`/`unmount()`.
