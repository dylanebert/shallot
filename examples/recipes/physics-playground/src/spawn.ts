import { Body, Part, type Plugin, ShapeKind, type State, type System } from "@dylanebert/shallot";

// build a body at runtime: create an entity, add `Body`, set its shape, pose, and mass, then add `Part`
// to render it. This rains a cube onto the pile on a timer while the scene is playing. The scene itself
// (scenes/physics-playground.scene) shows the rest of the substrate surface declaratively — a static
// ground, a `Spring`, and a `Joint`. The settling behavior is oracle-gated in the gym `pile` gold.
const MAX = 12;
let count = 0;
let next = 0;

const spawn = {
    name: "spawn",
    group: "simulation",
    update(state: State) {
        if (count >= MAX || state.time.elapsed < next) return;
        next = state.time.elapsed + 0.8;
        const eid = state.create();
        state.add(eid, Body);
        Body.shape.set(eid, ShapeKind.Box);
        Body.pos.set(eid, ((count % 3) - 1) * 0.7, 9, 0, 0);
        Body.halfExtents.set(eid, 0.4, 0.4, 0.4, 0);
        Body.mass.set(eid, 1);
        state.add(eid, Part);
        count++;
    },
} satisfies System;

export const Spawn = {
    name: "Spawn",
    // module-scoped counters are runtime state, so reset them each build (ecs.md "reload-safety")
    warm() {
        count = 0;
        next = 0;
    },
    systems: [spawn],
} satisfies Plugin;

export default Spawn;
