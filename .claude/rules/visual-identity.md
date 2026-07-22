---
paths:
    - "examples/**/*.html"
    - "examples/**/*.css"
---

# Visual Identity

Shallot's visual language across shipped UI surfaces: examples, overlays, the profiler HUD.

## Palette

The accent is always shallot gold (`#d49560`). Surfaces are warm-neutral, lifted off pure black so the UI isn't a black hole. Derived tints are `color-mix(in srgb, <base> N%, transparent)`, not hand-picked rgba values.

## Typography

- **Display/headings:** Outfit (weight 700 for display, 600 for headings)
- **Monospace:** JetBrains Mono
- **Body:** Outfit 400

## Motion

- **Easing:** `cubic-bezier(0.34,0,0,1)` for all transitions
- **Duration:** 150ms default. 100ms for small interactive feedback (active states)
- **Active feedback:** `transform: scale(0.95)` + an 8% accent wash

## Principles

- **Warm, not cool.** Backgrounds have brown undertones, not blue/grey. Text is warm white, not pure white.
- **Reduce to earn.** Every border, divider, shadow must earn its place. Use spacing and surface color shifts first.
- **Surface hierarchy over borders.** Distinguish regions with background color steps, not lines. Borders are a last resort.
- **Keep the name off failure surfaces.** An error message points at the diagnostic, never the brand.
