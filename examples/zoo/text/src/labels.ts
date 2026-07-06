import { type Plugin, type State, type System, Text, text } from "@dylanebert/shallot";

// #doc:intro
// World-space text labels anchored to an entity: a title on a signboard, a name tag over a character, a
// live score above a goal. Glyphs lay out once and ride the entity's transform, so moving a label costs
// nothing.

// #doc:code source:text/public/scenes/text.scene
// Give an entity a `text` and it draws SDF glyphs at the entity's transform: `content` is the string,
// `font-size` the world height of one line, `anchor` the pivot, and `color` a hex tint. A label is a 3D
// object, not a screen overlay: it holds its place and orientation in the world as the camera orbits. Text
// renders in the built-in Inter face by default; register your own with `font()`.

// #doc:code
// `Counter` marks the one label this plugin drives. A no-field marker component is the idiomatic way to
// select a subset of entities from a system.
// #region marker
export const Counter = {};
// #endregion

// #doc:code
// ### Changing a label at runtime
//
// Set `Text.content` to a string id from `text()` to change what a label shows. `text()` interns the
// string (identical strings share one id), so calling it every frame with the same value is free; the
// glyph buffer rebuilds only when the resolved content actually changes.
// #region drive
export const tick = {
    name: "tick",
    group: "simulation",
    update(state: State) {
        const seconds = Math.floor(state.time.elapsed);
        for (const eid of state.query([Counter, Text]))
            Text.content.set(eid, text(String(seconds)));
    },
} satisfies System;
// #endregion

export const Labels = {
    name: "Labels",
    components: { Counter },
    systems: [tick],
    traits: { Counter: { defaults: () => ({}) } },
} satisfies Plugin;

export default Labels;
