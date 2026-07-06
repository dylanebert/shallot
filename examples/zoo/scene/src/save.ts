import {
    diagnose,
    Inputs,
    type Plugin,
    type State,
    type System,
    serialize,
    stringify,
} from "@dylanebert/shallot";

// #doc:intro
// A scene file describes the world as XML: each `<a>` is an entity, each attribute a component. It's the
// declarative source the editor reads and writes, and what `run()` loads on boot. This specimen is a scene
// plus the code that saves it back out, the same round-trip the editor's save and survive-reload use.

// #doc:code source:scene/public/scenes/scene.scene
// ### The scene file
//
// A bare attribute adds a component with its defaults (`part`, `camera`); `field: value` pairs set fields,
// separated by semicolons. A vector takes all its components (`pos: 0 -1 0`) or one broadcast value
// (`scale: 2`); colors are hex (`0xd0dcec`) or `rgba` floats, and `@hero` references another entity by its
// `id`. Field names are kebab-case here, camelCase in code; scenes are flat, so cross-entity links use
// `@name` refs.

// #doc:code
// ### Save it from code
//
// `serialize(state)` reads the live world back to a node tree (the authored entities only, so
// `warm`-derived ones never double), and `stringify` renders that to scene XML. That's the whole save path:
// stash the string (a download, `localStorage`), then `load(parse(text), state)` on a fresh State restores
// it. Here pressing P logs the current scene; `diagnose` reports any unregistered component or unmet
// requirement first, the same check `run()` runs on load.
// #region save
const Save = {
    name: "Save",
    group: "simulation",
    update(state: State) {
        if (!Inputs.isKeyPressed("KeyP")) return;
        const nodes = serialize(state);
        for (const issue of diagnose(nodes)) console.warn(`[scene] ${issue.message}`);
        console.log(stringify(nodes));
    },
} satisfies System;
// #endregion

export default { name: "Save", systems: [Save] } satisfies Plugin;
