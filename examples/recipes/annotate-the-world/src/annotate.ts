import {
    arrow,
    box,
    type Plugin,
    type State,
    type System,
    segment,
    Text,
    text,
} from "@dylanebert/shallot";

// A no-field marker selects the one label this plugin drives live.
export const Counter = {};

// Set `Text.content` to an id from `text()` to change what a world-space label shows. `text()` interns
// the string (identical strings share one id), so calling it each frame with the same value is free;
// the glyph buffer rebuilds only when the resolved content changes.
export const tick = {
    name: "tick",
    group: "simulation",
    update(state: State) {
        const seconds = Math.floor(state.time.elapsed);
        for (const eid of state.query([Counter, Text]))
            Text.content.set(eid, text(String(seconds)));
    },
} satisfies System;

// Immediate-mode debug drawing: `segment` / `box` / `arrow` append to the current frame and clear the
// next, so a system calls them every frame. Positions are world-space, `color` is hex sRGB, and `width`
// is a constant pixel size, so the line keeps its on-screen thickness as the camera zooms.
export const gizmos = {
    name: "gizmos",
    group: "simulation",
    update(state: State) {
        const t = state.time.elapsed;
        const s = 1.1 + 0.25 * Math.sin(t * 2);
        const x = Math.cos(t) * 2.4;
        const z = Math.sin(t) * 2.4;
        box([-s, -s, -s], [s, s, s], 0x44ff88);
        arrow([0, 0, 0], [x, 0, z], 0xffcc00, 3);
        segment([x, 0, z], [x, -0.85, z], 0x9fb2cc);
    },
} satisfies System;

export const Annotate = {
    name: "Annotate",
    components: { Counter },
    systems: [tick, gizmos],
    traits: { Counter: { defaults: () => ({}) } },
} satisfies Plugin;

export default Annotate;
