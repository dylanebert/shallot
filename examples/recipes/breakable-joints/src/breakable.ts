import {
    Body,
    Color,
    Part,
    type Plugin,
    type State,
    type System,
    Tumble,
} from "@dylanebert/shallot";

// breakable connections — a row of boxes hung from the ceiling by distance joints, each joint given a higher
// force threshold than the last. a rising downward load drives the joint reactions up; when a joint's
// reaction crosses its threshold, tumble reports a joint event and this recipe cuts the joint, dropping the
// box. the joints break left to right as the load climbs. the boxes are substrate `Body` entities; the
// joints, their break thresholds, and the joint-event stream all ride `Tumble.world` — the escape hatch for
// tumble physics past the substrate's `Spring`/`Joint`.
//
// break-on-threshold + the joint-event stream have no published substrate-surface equivalent yet, so this
// recipe rides the escape hatch; the gym twin `events-joint-break` is the oracle-gated gold.

const IDENT = { v: { x: 0, y: 0, z: 0 }, s: 1 };
const COUNT = 6;
const REST = 3; // the joint's rest length; boxes hang taut so the static load loads it from step one
const MAX_LOAD = 1900;

type Joint = ReturnType<NonNullable<typeof Tumble.world>["createDistanceJoint"]>;
type Hung = { joint: Joint; eid: number; broken: boolean };

const CEILING_Y = 11.5;
const ANCHOR_Y = 11; // the hang point above each box; the joint spans REST down from here

let ceilingEid = -1;
let boxes: number[] = [];
let hung: Hung[] = [];
let load = 0;
let wired = false;

function box(
    state: State,
    x: number,
    y: number,
    hx: number,
    hy: number,
    hz: number,
    mass: number,
    color: [number, number, number],
): number {
    const eid = state.create();
    state.add(eid, Body);
    Body.pos.set(eid, x, y, 0, 0);
    Body.halfExtents.set(eid, hx, hy, hz, 0);
    Body.mass.set(eid, mass);
    state.add(eid, Part);
    state.add(eid, Color);
    Color.rgba.set(eid, color[0], color[1], color[2], 1);
    return eid;
}

function build(state: State): void {
    ceilingEid = box(state, 0, CEILING_Y, 12, 0.3, 0.5, 0, [0.5, 0.52, 0.56]);
    boxes = [];
    for (let i = 0; i < COUNT; i++) {
        boxes.push(box(state, -10 + 4 * i, ANCHOR_Y - REST, 0.5, 0.5, 0.5, 1, [0.85, 0.6, 0.4]));
    }
    state.addSystem(driver);
}

// wire each box to the ceiling once the bodies have marshaled. `forceThreshold` is the reaction force at
// which the joint reports a break event; `userData` tags the event so the driver knows which joint crossed.
function wire(): void {
    const world = Tumble.world;
    if (!world || wired) return;
    const ceiling = Tumble.body(ceilingEid);
    const bodies = boxes.map((e) => Tumble.body(e));
    if (!ceiling || bodies.some((b) => !b)) return;

    hung = [];
    for (let i = 0; i < bodies.length; i++) {
        const body = bodies[i]!;
        const x = -10 + 4 * i;
        const joint = world.createDistanceJoint(ceiling, body, {
            localFrameA: { p: { x, y: ANCHOR_Y - CEILING_Y, z: 0 }, q: IDENT },
            localFrameB: { p: { x: 0, y: REST, z: 0 }, q: IDENT },
            length: REST,
            forceThreshold: 300 + 300 * i,
            userData: i,
        });
        hung.push({ joint, eid: boxes[i], broken: false });
    }
    wired = true;
}

// ramp the load and apply it to every still-hanging box, then cut any joint that reported crossing its
// threshold last step. `getJointEvents` returns the joints that broke this step (keyed by the `userData` we
// tagged); `destroy` releases the joint so its box falls.
const driver: System = {
    name: "breakable-driver",
    group: "simulation",
    update() {
        if (!wired) {
            wire();
            return;
        }
        const world = Tumble.world;
        if (!world) return;
        load = Math.min(MAX_LOAD, load + 16);
        for (const h of hung) {
            if (h.broken) continue;
            Tumble.body(h.eid)?.applyForceToCenter({ x: 0, y: -load, z: 0 }, true);
        }
        for (const e of world.getJointEvents()) {
            const h = hung[e.userData as number];
            if (h && !h.broken && h.joint.isValid()) {
                h.joint.destroy();
                h.broken = true;
            }
        }
    },
};

export const Breakable = {
    name: "Breakable",
    warm(state: State) {
        ceilingEid = -1;
        boxes = [];
        hung = [];
        load = 0;
        wired = false;
        build(state);
    },
} satisfies Plugin;

export default Breakable;
