import {
    Color,
    Inputs,
    type Plugin,
    type State,
    type System,
    Transform,
} from "@dylanebert/shallot";

// #doc:intro
// Read the keyboard and mouse from any system through the `Inputs` singleton. Input is on in the default
// plugins, so a system reads it directly — nothing to enable.

export const Controlled = {};

// #doc:code
// ### Move with held keys
//
// `Inputs.isKeyDown(code)` is true every frame a key is held — the path for movement. Codes are
// `KeyboardEvent.code` strings (`"KeyW"`, `"ArrowUp"`). Read the four direction keys and write the box's
// `Transform` each frame, scaled by `state.time.deltaTime` so speed is framerate-independent:
// #region move
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
// #endregion

// #doc:code
// ### Read the mouse and press edges
//
// `Inputs.mouse` carries the buttons and canvas-relative position; `Inputs.isKeyPressed(code)` fires for the
// single frame a key goes down — the edge, versus `isKeyDown`'s held state. Redden the box while the left
// button is held, and snap it home on a Space press:
// #region react
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
// #endregion

export const Drive = {
    name: "Drive",
    components: { Controlled },
    systems: [move, react],
    traits: { Controlled: { defaults: () => ({}) } },
} satisfies Plugin;

export default Drive;
