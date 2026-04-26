---
title: Editor
description: editor interface, state management
source: editor
icon: diamond
---

# Editor

<!-- tabs -->
<!-- tab: UI -->

coming soon

<!-- tab: Code -->

Shallot's editor uses Svelte 5 as a UI harness around the engine. Svelte handles the inspector, panels, and controls — not ECS code.

The core challenge is **proxy identity**. Svelte 5's `$state` deep-proxies objects, which breaks identity comparisons (`===`, `Set.has`, `Map.get`). This matters for anything that stores engine objects: Nodes, ECS components, Map keys, plugin arrays.

The engine's Document is a plain object — mutations don't trigger Svelte reactivity. A version counter bridges the gap: bump it after any Document mutation, and children create a dependency via the version prop.

## Examples

### Identity-safe state

Use `$state.raw` for any state that stores objects used in identity comparisons:

```typescript
let plugins = $state.raw(defaultPlugins);  // identity-sensitive
let panelSize = $state(300);               // reactive, no identity concern
```

When some fields need reactivity and others need identity, split into two variables:

```typescript
let popoverCtx: { node: Node } | null = null;            // identity
let popover: { x: number; y: number } | null = $state(null);  // reactive
```

### Reactivity bridge

For in-place array mutations, `{#each}` skips re-rendering blocks whose item reference hasn't changed. Derive a flat view-model that pre-computes mutable state into fresh objects per version bump. Never read mutable Node properties directly in templates.

### Naming

ECS State is always `ecs` — never `state` (conflicts with `$state` rune), never `engine` or `shallot`.

<!-- /tabs -->
