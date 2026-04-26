---
paths:
    - "examples/**/*.html"
    - "examples/**/*.css"
    - "examples/**/*.svelte"
    - "examples/**/*.vue"
    - "examples/**/*.tsx"
    - "packages/shallot/editor/**/*.svelte"
    - "packages/shallot/editor/**/*.css"
---

# Editor UI

Palette, typography, motion, and surface principles are in `visual-identity.md`. This file covers editor-specific layout and interaction patterns.

## Principles

1. **Canvas supremacy** — viewport is the product. Chrome is subordinate, collapsible, minimal
2. **Reduce to earn** — every border, divider, shadow must earn its place. Use spacing and surface shifts first

## CSS

- **`--ease-out`** — all transitions and animation timing functions use `var(--ease-out)`. Never bare `ease-out` or inline cubic-bezier. Defined on `.editor`
- **`:active` feedback** — buttons get `background: rgba(212, 149, 96, 0.08)` + `transform: scale(0.95)`. List rows (outliner, etc.) get the background only, no scale

## Layout

Single viewport with right sidebar. Outliner top, inspector bottom. Entire sidebar collapses with one shortcut. Future panels: tabs within the sidebar, not new docked regions.
