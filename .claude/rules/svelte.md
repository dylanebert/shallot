---
paths:
    - "packages/shallot/editor/**/*.svelte"
    - "examples/**/*.svelte"
---

# Svelte

Svelte 5 with runes, never Svelte 4 syntax; `bun run check:svelte` is the gate. This file covers the editor's two Svelte-specific structures.

## Reflection-driven views are pure transforms

The schema→UI derivation — inspector sections from a component's `schema` + traits, outliner rows from the document tree — is a pure function in a `.ts` file (`lib/sections.ts` for inspector sections, `lib/rows.ts` for outliner rows), and the component renders its output through `$derived`. No schema-walking, trait resolution, or field formatting inside the template. The reflection content path is where boot-clean bugs hide (a derived field crashing on select, a mistitled node); keeping it pure makes it `bun test`-testable against synthetic components and leaves the `.svelte` file a thin render. See `testing.md` "Editor tiers".

## Proxy identity

Svelte 5 deep-proxies `$state`, breaking identity comparisons (`===`, `Set.has`, `Map.get`) for stored engine objects: Nodes, ECS components, plugins. Use `$state.raw` for those. The Document is a plain object, so its mutations don't trigger reactivity; a version counter bridges that gap. Full pattern: `docs/editor/editor.md`.
