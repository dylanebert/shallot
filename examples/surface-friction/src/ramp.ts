import { Body, Color, Part, type Plugin, type State, Time } from "@dylanebert/shallot";
import { installHarness } from "@dylanebert/shallot/harness";

// surface friction — the substrate `Body.friction` field, the coulomb coefficient that decides whether a
// body slides or grips. five boxes are released across a tilted ramp with friction ramping low → high: the
// slippery box on the left slides off, the grippy box on the right holds. friction is authored right on the
// `Body` (no escape hatch — it is one of the substrate's own fields, alongside shape/pos/mass), so this whole
// scene is ordinary substrate physics.

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
): number {
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
    return eid;
}

export function build(state: State): void {
    // a flat floor to catch whatever slides off, plus the tilted ramp (its own friction is a middling 1)
    body(state, 0, 0, 0, 22, 0.5, 22, 0, 0.6, [0.4, 0.42, 0.46]);
    body(state, 0, 8.5, -5, 16, 0.5, 10, 0, 1, [0.46, 0.48, 0.52], true);

    // boxes released across the ramp with friction rising (i+1)²·0.04 — a slippery-to-grippy ladder
    const boxes: number[] = [];
    for (let i = 0; i < COUNT; i++) {
        const friction = (i + 1) * (i + 1) * 0.04;
        const shade = 0.3 + 0.14 * i;
        boxes.push(body(state, -10 + 5 * i, 16.75, -10.6, 1, 1, 1, 1, friction, [0.9, shade, 0.3]));
    }

    const harness = installHarness(state);
    harness.run = async () => {
        for (let i = 0; i < 180; i++) state.step(Time.FIXED_DT);
        const low = harness.read!(boxes[0]);
        const high = harness.read!(boxes[boxes.length - 1]);
        const lowY = low?.pos[1] ?? Number.NaN;
        const highY = high?.pos[1] ?? Number.NaN;
        const lowLeaves = Number.isFinite(lowY) && lowY < 8;
        const highHolds = Number.isFinite(highY) && highY > 9;
        return {
            ok: lowLeaves && highHolds,
            checks: [
                { name: "low-friction box leaves the ramp", ok: lowLeaves },
                { name: "high-friction box holds", ok: highHolds },
            ],
            data: { lowY, highY },
        };
    };
}

export const Ramp = {
    name: "Ramp",
    warm: build,
} satisfies Plugin;

export default Ramp;
