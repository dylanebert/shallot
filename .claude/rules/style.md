# Style

How shallot code is shaped: naming, the shape of a function, and when to comment, across `src/engine/`, `src/standard/`, and `src/extras/`. The data-over-methods philosophy lives in `packages/shallot/AGENTS.md`; choosing a component, system, or singleton primitive lives in `ecs.md`. This file covers the rest: what to call things, how a function reads, and what's worth a comment.

## Imitate the existing code

`src/engine/` and `src/standard/` are the reference for both. Before writing a new function, system, or plugin, read a sibling in the same directory and follow its shape. The names and structure already there are the spec. Two minutes grepping neighbors for the verb that fits beats inventing one, and keeps the surface consistent.

## Naming

The shortest word that's clear in context, a single verb where one fits: `mesh`, `pack`, `warm`, `sparse`, `slab`, `attachCanvas`. Module scope is the context: a function doesn't repeat the name of the file or type it lives in.

- Add a qualifier only to distinguish two real things: `composeTransform` (one entity) vs `composeTransforms` (the batch). Never to describe what the body already shows. A function that builds a mesh from vertices is `mesh`, not `createMeshGeometryFromVertices`.
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

Guards are early returns, not nested branches. The work it hands off (`renderColor`) is one named transform, not a `prepareX` then `buildY` then `applyZ` chain of helpers calling each other. Extract a step into its own function when it's pure and a test can call it in isolation; inline a step that only runs from one place. A plugin is the same idea, as data: a plain object of `name`, `components`, `systems`, `dependencies`, and lifecycle hooks (`initialize` / `warm`), not a class. See `SearPlugin` and `PartPlugin` in `standard/`.

## Comments earn their place

The comment rule is universal: `kex/.claude/rules/coding.md` (Comments) is its home — default to none, earn one only with a public export's JSDoc contract or the *why* behind a non-obvious line. Two things are shallot-specific. JSDoc is the usage contract the Reference renders from; its conventions (lowercase, `@example`, `@expand`, no JSDoc = hidden) live in `docs.md`, the single home. And shallot code is minimal enough that the bar sits higher than elsewhere — `sear/` and `slab/` are the reference for how much to say, and when in doubt, say less.

```ts
// good — says why; survives the next edit
// the shadow pass reads positions only; the attributes stream stays bound for the color pass
bindMesh(state, view.positions);

// bad — narrates the change and restates the code
// now we bind positions instead of the whole mesh like before
bindMesh(state, view.positions); // bind the positions
```
