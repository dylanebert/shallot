import {
    f32,
    not,
    type Plugin,
    quat,
    type State,
    type System,
    sparse,
    Transform,
} from "@dylanebert/shallot";

// A component is a named set of fields, each declaring how it's stored. `sparse(f32)` holds one CPU
// float per entity, the common case. `Spin` gives an entity a turn rate and a starting offset.
export const Spin = {
    speed: sparse(f32),
    phase: sparse(f32),
};

// State that isn't per-entity lives in a module-level singleton whose type name matches its value
// name. Import it directly wherever you need it. Here one rate scales every spin at once.
export interface SpinControl {
    rate: number;
}
export const SpinControl: SpinControl = { rate: 1 };

// A system runs each frame over the entities a query selects. `state.query([Spin, Transform])` returns
// every entity that has both, and the loop turns each by its own `speed`. Derive the angle from
// `state.time.elapsed`, not a stored accumulator, so it stays correct after a hot reload.
export const spin = {
    name: "spin",
    group: "simulation",
    update(state: State) {
        for (const eid of state.query([Spin, Transform])) {
            const deg =
                state.time.elapsed * Spin.speed.get(eid) * SpinControl.rate * 60 +
                Spin.phase.get(eid);
            const q = quat(0, deg, 0);
            Transform.rot.set(eid, q.x, q.y, q.z, q.w);
        }
    },
} satisfies System;

// A marker is a component with no fields — its presence is the whole message. Query for entities with
// `Spin` but `not(Ready)`, act, then add `Ready` so the same entity is never picked again.
export const Ready = {};

export const stagger = {
    name: "stagger",
    group: "simulation",
    update(state: State) {
        let i = 0;
        for (const eid of state.query([Spin, not(Ready)])) {
            Spin.phase.set(eid, i * 120);
            state.add(eid, Ready);
            i++;
        }
    },
} satisfies System;

// A plugin is plain data: the components to register, the systems to run, and `traits` (parse-time
// metadata). `defaults` seeds a new `Spin` so a scene can omit fields; `requires: [Transform]` means a
// `spin` entity must also carry a `Transform`.
export const SpinPlugin = {
    name: "Spin",
    components: { Spin, Ready },
    systems: [spin, stagger],
    traits: {
        Spin: { defaults: () => ({ speed: 1, phase: 0 }), requires: [Transform] },
    },
} satisfies Plugin;

export default SpinPlugin;
