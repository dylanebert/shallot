import {
    Body,
    Color,
    Part,
    Physics,
    type Plugin,
    type State,
    type System,
} from "@dylanebert/shallot";
import { StepSystem } from "@dylanebert/shallot/physics/core";

// a moving platform, on the published substrate surface: a KINEMATIC body driven from code. A `mass: 0`
// body is normally static, but `Physics.backend.setKinematic(eid, pos, quat)` moves it each fixed tick and
// derives its velocity from the pose delta — so dynamic bodies resting on it get carried, pushed up as it
// rises and riding it back down. This is the same mechanism the character controller's moving-platform
// carry is built on. The lift oscillates on a sine so the motion is smooth and reload-safe: the height is
// derived from `state.time.elapsed`, never a module-level accumulator (ecs.md "reload-safety").
//
// A motor on a prismatic joint (with hard translation limits and a live speed target) is the richer
// version, past the substrate on the `Tumble.world` escape hatch — verified in the gym twin `joints-elevator`.

const BASE_Y = 3; // the platform's mid-travel height
const AMP = 1.5; // metres above/below mid — a 3 m peak-to-peak stroke
const PERIOD = 4; // seconds per full up-and-down cycle

// runtime identity is re-created each build (a State is rebuilt on scene switch / play-stop — ecs.md), so
// hold the platform eid in module scope and reset it in `warm`, never across a reload.
let platformEid = -1;

function body(
    state: State,
    x: number,
    y: number,
    z: number,
    hx: number,
    hy: number,
    hz: number,
    mass: number,
    color: [number, number, number],
): number {
    const eid = state.create();
    state.add(eid, Body);
    Body.pos.set(eid, x, y, z, 0);
    Body.halfExtents.set(eid, hx, hy, hz, 0);
    Body.mass.set(eid, mass);
    state.add(eid, Part);
    state.add(eid, Color);
    Color.rgba.set(eid, color[0], color[1], color[2], 1);
    return eid;
}

function build(state: State): void {
    body(state, 0, 0, 0, 12, 0.5, 12, 0, [0.4, 0.42, 0.46]);
    platformEid = body(state, 0, BASE_Y, 0, 1.5, 0.2, 1.5, 0, [0.5, 0.55, 0.85]);
    for (let i = 0; i < 3; i++) {
        body(state, -0.6 + 0.6 * i, BASE_Y + 1 + i * 0.7, 0, 0.35, 0.35, 0.35, 1, [0.85, 0.6, 0.4]);
    }
    state.addSystem(driver);
}

// drive the platform to its sine height each fixed tick, before the solve steps, so the derived velocity is
// this tick's and the resting crates are carried. `setKinematic` with `teleport` unset lets the backend read
// the velocity off the pose delta; passing an identity quaternion keeps the platform level.
const IDENT: [number, number, number, number] = [0, 0, 0, 1];
const driver: System = {
    name: "elevator-driver",
    group: "fixed",
    before: [StepSystem],
    update(state: State) {
        const backend = Physics.backend;
        if (!backend || platformEid < 0) return;
        const y = BASE_Y + AMP * Math.sin((2 * Math.PI * state.time.elapsed) / PERIOD);
        backend.setKinematic(platformEid, [0, y, 0], IDENT);
    },
};

export const Elevator = {
    name: "Elevator",
    warm(state: State) {
        platformEid = -1;
        build(state);
    },
} satisfies Plugin;

export default Elevator;
