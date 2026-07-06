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

The accent is always shallot gold (`#d49560`). Surfaces are warm-neutral, lifted off pure black so the
UI isn't a black hole. The viewport background isn't a palette token — it mirrors the scene camera's
clear color so the editor reads WYSIWYG against play mode.

Every theme's full token ramp lives in `theme.ts` (the single source — see "Editor theming" below); the
default is **Shallot**. Reference `var(--token)` in editor surfaces, never a copied hex.

## Editor theming (single source)

The editor's color is a theme system, not a fixed palette. **The sole source is
`packages/shallot/editor/src/lib/theme.ts`** — one `Palette` per theme (Shallot default, Dark, Light,
Neutral), the live one in `current.palette`. Adding a theme is one object in `THEMES`; the menu and prefs
key off it. Enforce the single source:

- **No color literal outside `theme.ts`.** Components reference tokens (`var(--accent)`, `var(--text)`);
  derived tints are `color-mix(in srgb, var(--token) N%, transparent)`, never a raw `rgba(r,g,b,a)`. GPU /
  canvas colors come from the palette via `packed` / `rgb`. The only exceptions are pure-black shadows
  (`rgba(0,0,0,*)`) and the functional grid axes (red X / blue Z).
- **CSS tokens are applied at runtime** by `setTheme` (custom properties on the `.editor` root), so the
  static `<style>` holds only non-color tokens (`--ease-out`, `--header-h`). The themed viewport overlays
  (grid, outline) are read from `current.palette` each frame, so a switch retints them live; the clear
  color is the scene camera's, not the palette's.
- The Light theme doubles as a coverage check: an untokenized literal stays dark on a light surface.

## Typography

- **Display/headings:** Outfit (weight 700 for display, 600 for headings)
- **Monospace:** JetBrains Mono
- **Body:** Outfit 400

## Motion

- **Easing:** `cubic-bezier(0.34,0,0,1)` — assigned to `--ease-out`, used for all transitions
- **Duration:** 150ms default. 100ms for small interactive feedback (active states)
- **Active feedback:** `transform: scale(0.95)` + a `color-mix(in srgb, var(--accent) 8%, transparent)` wash (where each lands: `editor-ui.md`)

## Principles

- **Warm, not cool.** Backgrounds have brown undertones, not blue/grey. Text is warm white, not pure white.
- **Reduce to earn.** Every border, divider, shadow must earn its place. Use spacing and surface color shifts first.
- **Surface hierarchy over borders.** Distinguish regions with background color steps, not lines. Borders are a last resort.
