import {
    Color,
    lerp,
    type Plugin,
    quat,
    type State,
    type System,
    Transform,
    unpackColor,
} from "@dylanebert/shallot";

// #doc:intro
// Math and color helpers, as plain functions over numbers and `Float32Array`s — no vector or matrix
// classes. `lerp`, `quat`, and `unpackColor` take and return plain numbers, so they slot into systems that
// read and write ECS fields directly. The Reference lists every helper; this page shows the common shapes.

// a bare marker so the system animates one box, not the ground
export const Eased = {};

// two endpoint colors, hex decoded to linear rgb once; the system blends between them each frame
const WARM = unpackColor(0xff9040);
const COOL = unpackColor(0x40a0ff);

// #doc:code source:utils/public/scenes/utils.scene
// The scene places one box over a ground. Its pose and color are only a starting point — the system
// rewrites them every frame.

// #doc:code
// ### Ease a value, a rotation, and a color
//
// A `simulation` system drives the box from `state.time.elapsed`, so it stays correct after a reload
// with no stored accumulator. `lerp(a, b, t)` blends the height and each color channel; `quat` builds the
// spin quaternion from angles in degrees. The blended color writes straight into `Color.rgba`.
// #region ease
const ease = {
    name: "ease",
    group: "simulation",
    update(state: State) {
        const t = (Math.sin(state.time.elapsed) + 1) / 2;
        const q = quat(0, state.time.elapsed * 40, 0);
        for (const eid of state.query([Eased, Transform, Color])) {
            Transform.pos.set(eid, 0, lerp(0.6, 2, t), 0, 0);
            Transform.rot.set(eid, q.x, q.y, q.z, q.w);
            const r = lerp(WARM.r, COOL.r, t);
            const g = lerp(WARM.g, COOL.g, t);
            const b = lerp(WARM.b, COOL.b, t);
            Color.rgba.set(eid, r, g, b, 1);
        }
    },
} satisfies System;
// #endregion

export const Ease = {
    name: "Ease",
    components: { Eased },
    systems: [ease],
    traits: { Eased: { defaults: () => ({}) } },
} satisfies Plugin;

export default Ease;
