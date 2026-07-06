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

// #doc:intro
// Everything in a Shallot scene is an entity, a component, or a system. An entity is a number; a component
// is data attached to it; a system is code that runs each frame over the entities a query selects. Every
// built-in (rendering, physics, audio) is built from these three, and so is your own gameplay. This
// specimen adds a custom `Spin` component and the system that turns it.

// #doc:code source:ecs/public/scenes/ecs.scene
// ### Components are data
//
// A component is a named set of fields, each declaring how it's stored. `sparse(f32)` holds one CPU float
// per entity, the common case (the reference lists the other field types). `Spin` gives an entity a turn
// rate and a starting offset, and a scene sets `speed` per box.
// #region component
export const Spin = {
    speed: sparse(f32),
    phase: sparse(f32),
};
// #endregion

// #doc:code
// ### Shared state is a singleton
//
// State that isn't per-entity (a global setting, a service handle) lives in a module-level object whose
// type name matches its value name. Import it directly wherever you need it; a plugin usually fills its
// fields in `initialize`. Here one rate scales every spin at once.
// #region singleton
export interface SpinControl {
    rate: number;
}
export const SpinControl: SpinControl = { rate: 1 };
// #endregion

// #doc:code
// ### Systems run over a query
//
// `state.query([Spin, Transform])` returns every entity that has both components, and the loop turns each
// one by its own `speed`. Derive the angle from `state.time.elapsed` instead of a stored accumulator, so it
// stays correct after a hot reload. `group: "simulation"` runs it in play mode, not while editing.
// #region system
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
// #endregion

// #doc:code
// ### Do something once with a marker
//
// A marker is a component with no fields, so its presence is the whole message. Query for entities that have
// `Spin` but `not(Ready)`, act, then add `Ready` so the same entity is never picked again. This stamps each
// box a phase from its spawn order, staggering ones that share a speed.
// #region marker
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
// #endregion

// #doc:code
// ### A plugin bundles it
//
// A plugin is plain data: the components to register, the systems to run, and `traits` (parse-time
// metadata). `defaults` seeds a new `Spin` so a scene can omit fields, and `requires: [Transform]` means a
// `spin` entity must also carry a `Transform`, which the editor enforces when you add the component.
// #region plugin
export const SpinPlugin = {
    name: "Spin",
    components: { Spin, Ready },
    systems: [spin, stagger],
    traits: {
        Spin: { defaults: () => ({ speed: 1, phase: 0 }), requires: [Transform] },
    },
} satisfies Plugin;
// #endregion

export default SpinPlugin;
