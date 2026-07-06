import { arrow, box, type Plugin, type State, type System } from "@dylanebert/shallot";

// #doc:intro
// Debug lines drawn straight into the 3D scene (wireframe boxes, arrows, axis segments) for seeing
// what your code is doing. Two ways in: the retained `Line` / `Arrow` components for annotations you
// author in the editor, and an immediate `segment` / `box` / `arrow` API you call each frame from a
// system.

// #doc:code source:lines/public/scenes/lines.scene
// Put a `line` on an entity and it draws from the entity's position along a world-rotated `offset`; add
// `arrow` for a fletched head. These three share the origin to make an RGB axis gizmo. Both components
// are retained: the editor authors them, and they ride their entity's transform.

// #doc:code
// ### Drawing from code
//
// `segment` / `box` / `arrow` append to the current frame and clear the next, so a system calls them
// every frame. Positions are world-space, `color` is hex sRGB, and `width` is a constant pixel size, so
// the line stays the same thickness on screen as the camera zooms.
// #region immediate
export const gizmos = {
    name: "gizmos",
    group: "simulation",
    update(state: State) {
        const t = state.time.elapsed;
        // a wireframe box breathing around the object
        const s = 1.1 + 0.25 * Math.sin(t * 2);
        box([-s, -s, -s], [s, s, s], 0x44ff88);
        // an arrow sweeping like a clock hand
        arrow([0, 0, 0], [Math.cos(t) * 2.4, 0, Math.sin(t) * 2.4], 0xffcc00, 3);
    },
} satisfies System;
// #endregion

export const Gizmos = { name: "Gizmos", systems: [gizmos] } satisfies Plugin;

export default Gizmos;
