import {
    Color,
    Inputs,
    type Plugin,
    type State,
    type System,
    Transform,
} from "@dylanebert/shallot";

// Input is on in the default plugins — any system reads the `Inputs` singleton directly, nothing to enable.
export const Controlled = {};

// `Inputs.isKeyDown(code)` is true every frame a key is held, the path for movement. Codes are
// `KeyboardEvent.code` strings. Scale by `state.time.deltaTime` so speed is framerate-independent.
const SPEED = 5;

export const move = {
    name: "move",
    group: "simulation",
    update(state: State) {
        let x = 0;
        let z = 0;
        if (Inputs.isKeyDown("KeyW")) z -= 1;
        if (Inputs.isKeyDown("KeyS")) z += 1;
        if (Inputs.isKeyDown("KeyA")) x -= 1;
        if (Inputs.isKeyDown("KeyD")) x += 1;
        const step = SPEED * state.time.deltaTime;
        for (const eid of state.query([Controlled, Transform])) {
            const px = Transform.pos.x.get(eid);
            const py = Transform.pos.y.get(eid);
            const pz = Transform.pos.z.get(eid);
            Transform.pos.set(eid, px + x * step, py, pz + z * step, 0);
        }
    },
} satisfies System;

// `Inputs.mouse` carries buttons + canvas-relative position; `isKeyPressed` fires once on the frame a key
// goes down, the edge versus `isKeyDown`'s held state.
export const react = {
    name: "react",
    group: "simulation",
    update(state: State) {
        for (const eid of state.query([Controlled, Color])) {
            if (Inputs.mouse.left) Color.rgba.set(eid, 0.95, 0.4, 0.35, 1);
            else Color.rgba.set(eid, 0.4, 0.7, 0.9, 1);
            if (Inputs.isKeyPressed("Space")) Transform.pos.set(eid, 0, 0.5, 0, 0);
        }
    },
} satisfies System;

export const Drive = {
    name: "Drive",
    components: { Controlled },
    systems: [move, react],
    traits: { Controlled: { defaults: () => ({}) } },
} satisfies Plugin;

export default Drive;
