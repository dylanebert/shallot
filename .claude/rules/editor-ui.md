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

Palette, typography, motion, and surface principles: `visual-identity.md`. This file covers the editor's layout and interaction posture.

## Posture: the canvas, not the cockpit

The viewport is the product. The editor is a *canvas* (Figma, Linear): the scene fills the screen, chrome is contextual and minimal. Not a *cockpit* (classic Unity, Blender) where every panel is docked and loud at once. The cockpit's failure isn't having panels. It's that they all claim attention equally and continuously, so nothing is foreground. Attention is the scarce resource, not screen space.

**Play is a faithful preview.** Editor chrome — grid, gizmos, selection outline, the edit camera — is **edit-only Scene-View machinery** (Unity's Scene-View/Game-View split). Play builds the app's plugins verbatim and shows the scene exactly as it ships, so new viewport chrome is edit-only and absent from the play viewport. The composition that enforces this is `compose(mode, app)` (`editor/src/plugins.ts`): play = app set, edit = app ∪ the editor foundation via the `mode`/`layer` axes.

## The gates

Every UI decision clears these.

1. **Earn its place.** Monitored continuously → persistent. Acted on occasionally → summoned (command palette, contextual popover), never docked.
2. **Quiet when silent.** Nothing relevant to show → empty, dimmed, or collapsed, not loud. A persistent *location* is fine; loud-while-irrelevant *contents* are not.
3. **On the object first.** If it can be manipulated in the viewport (gizmo, handle), it belongs there, not in a panel. A panel control justifies itself only when the data has no spatial form (enum, flag, name, raw number). The inspector is the fallback for the un-spatial, not the default editing surface.
4. **Low floor, high ceiling.** A newcomer sees less; an expert reaches it faster. Progressive disclosure for the floor, keyboard/palette for the ceiling. Capability is summoned, not displayed.
5. **Instant and reversible.** Edits show immediately, no apply step, and undo cleanly.

A game scene carries more state than a Figma document: a node has many components, each with many fields. So the inspector earns its place where a pure canvas tool might not need one. That's the one spot the model bends for game dev. Keep the inspector, but hold it to gates 2 and 3 — contextual contents, and only the fields with no in-viewport form.

## Layout

Three roles, no more:

- **Outliner** — primary navigation, persistent, **left**. Always present: you must always know what exists and what's selected, and be able to select, deselect, and delete without relying on the viewport. Its *contents* may swap when the context isn't the 3D scene (gate 2). Its `+` is the one entity-creation surface — a summoned Add menu of **bundles** (named component clusters with authored defaults, in `lib/bundles.ts`) that drop ready-to-use visible entities. A new primitive is a `BUNDLES` entry, never a second create surface elsewhere (the inspector edits an existing selection; it can't create).
- **Inspector** — contextual properties, **right**, selection-driven. The selected entity's editable fields; empty state when nothing's selected.
- **Viewport tools** — direct manipulation (gizmos, handles) in the scene itself. The default editing surface (gate 3).

Outliner left, inspector right, viewport center and dominant: the navigate → manipulate → tune flow, and the convention shared by Figma, Unity, and Godot (a low floor for arrivals). Splitting the two also stops them competing for one column's height and attention. The whole sidebar collapses with one shortcut. A new surface is a tab within an existing region or a summoned palette, never a new docked region.

## Feedback surfaces

Signals split by what the user can act on. Editor-vocabulary events (a node, asset, or file to click) surface in the editor band; the runtime firehose (engine warnings, GPU validation, script throws) stays in the browser console, where the detail belongs. No docked log panel — that's the cockpit's continuous-readout failure (gate 1).

The viewport bar splits by kind: action on the left and center (transform tools; the isolated play/stop transport), ambient status + view controls on the right. Status never shares an action cluster — the cross-tool convention isolates the transport and keeps ambient state in a corner or status bar (Figma, PlayCanvas, Unity), never beside the play button. Gizmos anchors the right edge so a status item appearing or clearing doesn't shift it.

- **Toast** — transient action feedback (load-failed, a save-blocked action) + one throttled pointer for an uncaught runtime error. Floating over the viewport, auto-dismiss.
- **Banner** — a persistent blocking state (build failed, device lost, autosave failing), below the viewport bar, keyed by id (a re-raise replaces, never stacks), cleared when resolved.
- **Issues** — a summoned popover over the live `diagnose(doc)` derivation, not a collected stream; badge hidden at zero (gate 2), each row selects its node; in the bar's right status cluster, opening leftward.
- **Save status** — saving is silent (an ambient `•` in the tab title while unsaved, cleared on flush), the Figma/PlayCanvas autosave convention; a failed autosave raises the keyed `save` banner and retries. Ephemeral mode (`?save=off`, used by capture + engine-dev — never a real file session) shows a quiet `save-off` glyph (lucide-static through `Icon`, no label) in the right status cluster, an exception indicator, not a labeled badge.

## CSS

- **`--ease-out`** — all transitions and animation timing functions use `var(--ease-out)`. Never bare `ease-out` or inline cubic-bezier. Defined on `.editor`
- **`:active` feedback** — buttons get `background: color-mix(in srgb, var(--accent) 8%, transparent)` + `transform: scale(0.95)`. List rows (outliner, etc.) get the background only, no scale
- **Opaque floating surfaces** — menus, dropdowns, pickers, popovers, toasts, and banners use a solid surface (`--surface-3-solid` for elevated menus/popovers, `--surface-2-solid` for toasts/banners), never a translucent fill + `backdrop-filter` blur. A see-through panel over the live viewport hurts legibility; elevation comes from the border + shadow, not a glass effect
- **Summoned panels fit the viewport via `place.ts`** — a panel anchored to a trigger or cursor (color picker, context menu, menu dropdown) positions through the shared `fit` action, never inline `left`/`top` from a raw `getBoundingClientRect()`. `fit` flips to the opposite side when the preferred one would clip and clamps into the viewport, so a panel opened low or near an edge is never cut off. New floating surfaces use it
