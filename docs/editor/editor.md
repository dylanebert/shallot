---
title: Editor
description: editor interface, state management
source: editor/document
icon: diamond
order: 0
---

# Editor

<!-- tabs -->
<!-- tab: Code -->

The editor library is the scene document model plus the engine bridge: the pieces you build editor tooling on, independent of the Svelte UI.

A `Document` holds the parsed node tree with undo/redo. Every mutation (add, remove, set attribute, reorder, reparent) records a reversible command, so `undo`/`redo` replay without re-deriving state. Wrap a drag or a multi-field scrub in `begin`/`commit` to coalesce its writes into one undo step.

A `Session` bridges the document to a live `State`: it maps nodes to entity ids and replays each command onto the running engine, so an edit shows up in both the tree and the viewport. `ReadbackSystem` closes the loop the other way. It reflects live field values (a gizmo drag, a running system) back onto the attributes each node already authors.

<!-- API:editor/document -->

<!-- tab: Internals -->

The editor app wraps the engine in Svelte 5 as a UI harness: the inspector, panels, and controls are Svelte; the ECS is not.

The core challenge is **proxy identity**. Svelte 5's `$state` deep-proxies objects, which breaks identity comparisons (`===`, `Set.has`, `Map.get`). It's a problem for anything storing engine objects: Nodes, ECS components, Map keys, plugin arrays. Use `$state.raw` for state that stores objects used in identity comparisons, and split a variable when some fields need reactivity and others need identity:

```typescript
let plugins = $state.raw(defaultPlugins);  // identity-sensitive
let panelSize = $state(300);               // reactive, no identity concern
```

The engine's Document is a plain object, so mutations don't trigger Svelte reactivity. A version counter bridges the gap: `Document.version` bumps after any mutation, and children create a dependency by reading it. For in-place array mutations `{#each}` skips blocks whose item reference is unchanged, so derive a flat view-model that pre-computes mutable state into fresh objects per version bump. Never read mutable Node properties directly in templates.

One naming rule: ECS State is always `ecs`, never `state` (it collides with the `$state` rune), never `engine` or `shallot`.

<!-- /tabs -->
