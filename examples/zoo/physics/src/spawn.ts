import { Body, Part, type Plugin, ShapeKind, type State, type System } from "@dylanebert/shallot";

// #doc:intro
// Rigid-body physics: bodies fall under gravity and collide, springs and joints link them. Physics is
// opt-in, so add `PhysicsPlugin` to enable it.

// #doc:code source:physics/public/scenes/physics.scene
// Give an entity a `body` and the solver simulates it: `half-extents` sizes the box collider, and `mass: 0`
// pins it in place (the ground, a wall, a fixed anchor). Pair `body` with `part` to draw it. A `spring` pulls
// two bodies toward a `rest` length and a `joint` pins them rigidly, each its own entity referencing its
// bodies by `@name`.

// #doc:code
// ### Spawn bodies in code
//
// Build a body at runtime: create an entity, add `Body`, set its shape, pose, and mass, then add `Part`
// to render it. `Body.shape` takes a `ShapeKind`: `Box` here. This rains a cube onto the pile on a timer
// while the scene is playing.
// #region spawn
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
// #endregion

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
