# Style

How shallot code is shaped: naming, the shape of a function, and when to comment — across all code in this repo: engine source (`src/engine/`, `src/standard/`, `src/extras/`), tests, `bin/`, `scripts/`, `evals/`, and examples alike. The data-over-methods philosophy lives in `packages/shallot/AGENTS.md`; choosing a component, system, or singleton primitive lives in `ecs.md`. This file covers the rest: what to call things, how a function reads, and what's worth a comment.

## Imitate the existing code

`src/engine/` and `src/standard/` are the reference for both. Before writing a new function, system, or plugin, read a sibling in the same directory and follow its shape. The names and structure already there are the spec. Two minutes grepping neighbors for the verb that fits beats inventing one, and keeps the surface consistent.

## Naming

The shortest word that's clear in context, a single verb where one fits: `mesh`, `pack`, `warm`, `sparse`, `slab`, `attachCanvas`. Module scope is the context: a function doesn't repeat the name of the file or type it lives in.

- Add a qualifier only to distinguish two real things: `composeTransform` (one entity) vs `composeTransforms` (the batch). Never to describe what the body already shows. A function that builds a mesh from vertices is `mesh`, not `createMeshGeometryFromVertices`. (anti-pattern)
- A multi-word name is usually a function doing several things. Split it, or the name is covering for a call chain.
- PascalCase for components, plugins, and singletons (`Transform`, `RenderPlugin`, `Compute`); camelCase for functions and locals.

## A function is a transform; a system is a loop

Logic is data in, data out. Orchestration is a flat sequence or a query loop, not a stack of private helpers calling helpers. The dominant shape is a system that queries entities and acts on each:

```ts
// standard/sear/index.ts — query, guard, act. Flat.
const ColorSystem: System = {
    group: "draw",
    after: [PrepassSystem],
    update(state) {
        if (!Render.encoder) return;
        for (const eid of state.query([Camera, Sear])) {
            const view = Views.get(eid);
            if (!view?.framebuffer) continue;
            renderColor(eid, view, _frameDraws);
        }
    },
};
```

Guards are early returns, not nested branches. The work it hands off (`renderColor`) is one named transform, not a `prepareX` then `buildY` then `applyZ` chain of helpers calling each other. Extract a step into its own function when it's pure and a test can call it in isolation; inline a step that only runs from one place. A plugin is the same idea, as data: a plain object of `name`, `components`, `systems`, `dependencies`, and lifecycle hooks (`initialize` / `warm`), not a class. See `SearPlugin` and `PartPlugin` in `standard/`. (anti-pattern)

## Comments earn their place

The comment rule is universal: default to none, earn one only with a public export's JSDoc contract or the *why* behind a non-obvious line. One thing is shallot-specific: shallot code is minimal enough that the bar sits higher than elsewhere — `sear/` and `slab/` are the reference for how much to say, and when in doubt, say less.

**A comment states what is true now; how the code got that way is the commit message's job.** Never restate the line, and never narrate the edit — "now we…", "changed to…", "no longer…", "previously…", "used to…" are the greppable surface forms. A `History:` section, a refuted-alternative record, or a workflow-stage chronology in a module header is the same defect at essay length: rewrite it as the invariant that holds today, or delete it. (Algorithm step labels like `// Stage 1:` in a ported kernel are fine — they name the algorithm's own stages, not this repo's workflow.)

**A comment anchored to something outside this repo rots invisibly.** Never cite a workflow stage ID, a private planning path, or a symbol you are deleting; write the invariant instead, so the comment stays checkable by a reader who has only this repo. A stale anchor reads as authoritative for years — one sweep found ~90 such sites, and the last of them had to be caught by name because no regex spelled its surface form.

- **The entry-doc chain from repo root down to the working directory stays under the Codex 32768 B budget** — `wc -c` the AGENTS.md chain (root + `packages/shallot` + `examples`) before committing any entry-doc addition, and fold detail down into this path-scoped rule rather than paying for it in the entry doc; past the budget Codex silently drops the deepest file and its whole contract vanishes.

```ts
// good — says why; survives the next edit
// the shadow pass reads positions only; the attributes stream stays bound for the color pass
bindMesh(state, view.positions);

// bad — narrates the change and restates the code
// now we bind positions instead of the whole mesh like before
bindMesh(state, view.positions); // bind the positions
```
