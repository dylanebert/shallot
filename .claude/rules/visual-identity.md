---
paths:
    - "packages/shallot/editor/**/*.svelte"
    - "packages/shallot/editor/**/*.css"
    - "examples/**/*.html"
    - "examples/**/*.css"
---

# Visual Identity

Shallot's visual language across all surfaces: editor, docs site, examples.

## Palette

| Role | Value |
|------|-------|
| bg | `#0e0d0c` |
| surface-1 | `#161514` |
| surface-2 | `#1f1e1d` |
| surface-3 | `#2b2a29` |
| surface-4 | `#363534` |
| fg | `#f0ece8` |
| fg-secondary | `#cdc5bc` |
| muted | `#a09890` |
| accent | `#d49560` |
| accent-hover | `#e8a86b` |
| border | `rgba(255,255,255,0.09)` |

## Typography

- **Display/headings:** Outfit (weight 700 for display, 600 for headings)
- **Monospace:** JetBrains Mono
- **Body:** Outfit 400

## Motion

- **Easing:** `cubic-bezier(0.34,0,0,1)` — assigned to `--ease-out`, used for all transitions
- **Duration:** 150ms default. 100ms for small interactive feedback (active states)
- **Active feedback:** `transform: scale(0.95)` on buttons, `background: rgba(212, 149, 96, 0.08)` on interactive rows

## Principles

- **Warm, not cool.** Backgrounds have brown undertones, not blue/grey. Text is warm white, not pure white.
- **Reduce to earn.** Every border, divider, shadow must earn its place. Use spacing and surface color shifts first.
- **Surface hierarchy over borders.** Distinguish regions with background color steps, not lines. Borders are a last resort.
