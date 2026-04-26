---
paths:
    - "packages/shallot/editor/**/*.svelte"
    - "examples/**/*.svelte"
---

# Svelte

Reference: `docs/editor/editor.md` for proxy identity, reactivity bridge, and naming conventions.

## Anti-patterns

- Computing values in `$effect` instead of `$derived`
- Svelte 4 patterns (`createEventDispatcher`, `<slot>`, `export let`, `$:`)
- Over-componentizing — use `{#snippet}` for local reuse, extract to component only when used across files
- `<label>` for visual-only field labels — use `<span>`. Reserve `<label>` for actual form controls with stable IDs
