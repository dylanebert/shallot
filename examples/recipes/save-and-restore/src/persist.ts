import {
    diagnose,
    Inputs,
    load,
    type Plugin,
    parse,
    type State,
    type System,
    serialize,
    stringify,
} from "@dylanebert/shallot";

const KEY = "shallot:save-and-restore";

// Save the live world. `serialize(state)` reads it back to a node tree — the authored entities only, so
// `warm`-derived ones never double — and `stringify` renders that tree to scene XML. Stash the string
// anywhere: a download, a server, or `localStorage` as here. `diagnose` reports any unregistered
// component or unmet requirement first, the same check `load` runs. Press S to save.
const save = {
    name: "save",
    group: "simulation",
    update(state: State) {
        if (!Inputs.isKeyPressed("KeyS")) return;
        const nodes = serialize(state);
        for (const issue of diagnose(nodes)) console.warn(`[save] ${issue.message}`);
        localStorage.setItem(KEY, stringify(nodes));
        console.log(`saved ${nodes.length} entities to localStorage["${KEY}"]`);
    },
} satisfies System;

// Restore replaces the authored world in place: destroy what `load` authored (`state.identity.authored`
// tracks exactly that set), then run the saved string through the same `parse` → `load` path a fresh
// scene takes — entity refs like the camera's `target: @hero` re-resolve by scene id. A scene with
// assets awaits `preload(nodes, state)` between the two. Orbit the camera, save, orbit again, then
// press L to snap back. An app that owns its boot restores at startup instead:
// `run({ ...config, scene: saved })`.
const restore = {
    name: "restore",
    group: "simulation",
    update(state: State) {
        if (!Inputs.isKeyPressed("KeyL")) return;
        const saved = localStorage.getItem(KEY);
        if (!saved) return;
        const nodes = parse(saved);
        for (const eid of [...state.identity.authored]) state.destroy(eid);
        load(nodes, state);
        console.log(`restored ${nodes.length} entities`);
    },
} satisfies System;

const Persist = { name: "Persist", systems: [save, restore] } satisfies Plugin;
export default Persist;
