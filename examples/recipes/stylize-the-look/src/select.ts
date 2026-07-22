import { Outline, type Plugin, type State, type System } from "@dylanebert/shallot";

// presence is the switch: `state.add(eid, Outline)` highlights an object, `state.remove` clears it, the
// same call a hover or selection makes. This cursor sweeps the highlight across the boxes marked `pick`
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

// by default the band draws on top of everything; set `occlude` to hide it where the object is behind
// other geometry (the camera then needs sear's `Depth`)
export const Selector = {
    name: "Selector",
    components: { Pick },
    systems: [cursor],
    traits: { Pick: { defaults: () => ({}) } },
} satisfies Plugin;

export default Selector;
