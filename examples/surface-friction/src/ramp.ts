import { Body, Color, Part, type Plugin, type State } from "@dylanebert/shallot";

// surface friction — the substrate `Body.friction` field, the coulomb coefficient that decides whether a
// body slides or grips. five boxes are released across a tilted ramp with friction ramping low → high: the
// slippery box on the left slides off, the grippy box on the right holds. friction is authored right on the
// `Body` (no escape hatch — it is one of the substrate's own fields, alongside shape/pos/mass), so this whole
// scene is ordinary substrate physics. The gym twin is `shapes-inclined-plane`, the oracle-gated gold.

const TILT = (40 * Math.PI) / 180;
const COUNT = 5;

function body(
    state: State,
    x: number,
    y: number,
    z: number,
    hx: number,
    hy: number,
    hz: number,
    mass: number,
    friction: number,
    color: [number, number, number],
    tilt = false,
): void {
    const eid = state.create();
    state.add(eid, Body);
    Body.pos.set(eid, x, y, z, 0);
    Body.halfExtents.set(eid, hx, hy, hz, 0);
    Body.mass.set(eid, mass);
    Body.friction.set(eid, friction);
    // a positive rotation about world X so the ramp faces up-slope; a quaternion is (sin(θ/2)·axis, cos(θ/2))
    if (tilt) Body.quat.set(eid, Math.sin(TILT / 2), 0, 0, Math.cos(TILT / 2));
    state.add(eid, Part);
    state.add(eid, Color);
    Color.rgba.set(eid, color[0], color[1], color[2], 1);
}

function build(state: State): void {
    // a flat floor to catch whatever slides off, plus the tilted ramp (its own friction is a middling 1)
    body(state, 0, 0, 0, 22, 0.5, 22, 0, 0.6, [0.4, 0.42, 0.46]);
    body(state, 0, 8.5, -5, 16, 0.5, 10, 0, 1, [0.46, 0.48, 0.52], true);

    // boxes released across the ramp with friction rising (i+1)²·0.04 — a slippery-to-grippy ladder
    for (let i = 0; i < COUNT; i++) {
        const friction = (i + 1) * (i + 1) * 0.04;
        const shade = 0.3 + 0.14 * i;
        body(state, -10 + 5 * i, 16.75, -10.6, 1, 1, 1, 1, friction, [0.9, shade, 0.3]);
    }
}

export const Ramp = {
    name: "Ramp",
    warm: build,
} satisfies Plugin;

export default Ramp;
