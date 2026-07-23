---
paths:
    - "examples/**/*.ts"
    - "examples/**/*.scene"
    - "examples/AGENTS.md"
---

# Examples

`examples/recipes/` is the teaching corpus: one minimal project per problem a game developer actually has, named by the problem, indexed in `examples/AGENTS.md`. The corpus and the source JSDoc are the documentation — there is no docs site, so an entry that drifts or bloats is a documentation bug. The other groups (gym = testing harness, flows = standalone-app engine flows, showcase = capability exhibits) are described in `CLAUDE.md`; this rule is the recipe contract.

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
