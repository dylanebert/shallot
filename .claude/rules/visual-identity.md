---
paths:
    - "examples/**/*.html"
---

# Visual Identity

Shipped UI: examples, overlays, profiler HUD.

## Palette

Source: `DARK`/`LIGHT` in `src/standard/loading/mark.ts`. Dark: gold `#d49560`, dim `#7a5a3a`, ink `#f0e6d6`, ground `#141210`. Light: same gold, dim `#e6c6a4`, ink `#2a231e`, ground `#f7f3ec`. Tints are `color-mix(in srgb, <base> N%, transparent)`, not raw rgba.

## Typography

JetBrains Mono for headings, code, terminals; IBM Plex Sans for running text (400, emphasis 600); the bitmap wordmark from `mark.ts`, never a text setting. No display face joins them; the runtime names its stacks, never fetching them.

## Mark

The 12×14 pixel mark is the only logo. `scripts/brand-assets.ts --write` regenerates every icon from it; `scripts/brand-assets.test.ts` reds on drift, skipping by content a project's own icon.

## Motion

`cubic-bezier(0.34,0,0,1)`, 150ms; 100ms for active feedback (`scale(0.95)` plus an 8% accent wash).

## Principles

- **Warm, not cool.** Brown undertones, not blue/grey.
- **Reduce to earn.** Spacing and surface steps before borders, dividers, shadows.
- **Keep the name off failure surfaces.** An error names the diagnostic, never the brand.
