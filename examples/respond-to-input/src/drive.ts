import {
    Color,
    devices,
    type Plugin,
    type State,
    type System,
    Transform,
} from "@dylanebert/shallot";

// Input is on in the default plugins; systems read the State-scoped device record directly.
export const Controlled = {};

// `devices(state).keys.held` is true every frame a key is held, the path for movement. Codes are
// `KeyboardEvent.code` strings. Scale by `state.time.deltaTime` so speed is framerate-independent.
const SPEED = 5;

export const move = {
    name: "move",
    group: "simulation",
    update(state: State) {
        const input = devices(state);
        let x = 0;
        let z = 0;
        if (input.keys.held.has("KeyW")) z -= 1;
        if (input.keys.held.has("KeyS")) z += 1;
        if (input.keys.held.has("KeyA")) x -= 1;
        if (input.keys.held.has("KeyD")) x += 1;
        const step = SPEED * state.time.deltaTime;
        for (const eid of state.query([Controlled, Transform])) {
            const px = Transform.pos.x.get(eid);
            const py = Transform.pos.y.get(eid);
            const pz = Transform.pos.z.get(eid);
            Transform.pos.set(eid, px + x * step, py, pz + z * step, 0);
        }
    },
} satisfies System;

// `devices(state).mouse` carries buttons + canvas-relative position; `keys.pressed` fires once on the
// frame a key goes down, the edge versus `keys.held`'s held state.
export const react = {
    name: "react",
    group: "simulation",
    update(state: State) {
        const input = devices(state);
        for (const eid of state.query([Controlled, Color])) {
            if (input.mouse.left) Color.rgba.set(eid, 0.95, 0.4, 0.35, 1);
            else Color.rgba.set(eid, 0.4, 0.7, 0.9, 1);
            if (input.keys.pressed.has("Space")) Transform.pos.set(eid, 0, 0.5, 0, 0);
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
