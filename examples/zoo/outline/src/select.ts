import { Outline, type Plugin, type State, type System } from "@dylanebert/shallot";

// #doc:intro
// A drop-in highlight: add the `Outline` component to an object and a colored band hugs its silhouette.
// Use it for hover feedback, selection, or a grab highlight. The band is screen-space and uniform-width,
// so it reads the same thickness at any distance and never fattens on close geometry.

// #doc:code source:outline/public/scenes/outline.scene
// Put an `outline` on an object and a band appears around it: `color` is the band color, `width` its
// thickness in pixels. The highlight rides the object's transform, so it tracks the object as it moves.

// #doc:code
// ### Highlight from code
//
// Presence is the switch — `state.add(eid, Outline)` highlights an object, `state.remove` clears it, the
// same call a hover or selection makes. Set `color` and `width` per entity, so different highlights coexist
// in one pass. This cursor sweeps the highlight across the boxes marked `pick`.
// #region select
export const Pick = {};

export const cursor = {
    name: "cursor",
    group: "simulation",
    update(state: State) {
        const boxes = [...state.query([Pick])];
        if (boxes.length === 0) return;
        const active = Math.floor(state.time.elapsed * 1.5) % boxes.length;
        boxes.forEach((eid, i) => {
            if (i !== active) {
                state.remove(eid, Outline);
                return;
            }
            state.add(eid, Outline);
            Outline.color.set(eid, 0.3, 0.8, 1, 1);
            Outline.width.set(eid, 5);
        });
    },
} satisfies System;
// #endregion

// #doc:code
// By default the band draws on top of everything; set `occlude` to hide it where the object is behind other
// geometry (the camera then needs sear's `Depth`).

export const Selector = {
    name: "Selector",
    components: { Pick },
    systems: [cursor],
    traits: { Pick: { defaults: () => ({}) } },
} satisfies Plugin;

export default Selector;
