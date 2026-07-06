import {
    type Config,
    OrbitPlugin,
    type Plugin,
    type State,
    type System,
    sparse,
    u32,
} from "@dylanebert/shallot";

// Standalone run({survive}) fixture for the survive-reload capture flow (scripts/capture, Stage 7) —
// the real production path, not the editor. `Counter.n` climbs every frame: it's the runtime value
// survive-reload must bring back (a fresh build starts at 0). The `hero` counter is scene-authored, so
// it's in load's authored set and `serialize` captures it; `Sprout` is warm-derived (absent from the
// authored set), so a restore re-derives exactly one and never doubles it.
export const Counter = { n: sparse(u32) };
export const Sprout = {};

const TickSystem: System = {
    name: "survive-tick",
    group: "simulation",
    update: (state: State) => {
        for (const eid of state.query([Counter])) Counter.n.set(eid, Counter.n.get(eid) + 1);
    },
};

const SurvivePlugin: Plugin = {
    name: "survive",
    components: { Counter, Sprout },
    traits: { Counter: { defaults: () => ({ n: 0 }) } },
    systems: [TickSystem],
    // one Sprout per authored Counter, re-derived on every build — the not-doubled check
    warm: (state) => {
        for (const _ of state.query([Counter])) {
            const sprout = state.create();
            state.add(sprout, Sprout);
        }
    },
};

export const config: Config = {
    plugins: [OrbitPlugin, SurvivePlugin],
    scene: "scenes/scene.scene",
};
