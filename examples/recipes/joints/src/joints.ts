import { Body, Color, Joint, Part, type Plugin, Spring, type State } from "@dylanebert/shallot";

// connect two bodies with the published substrate constraints — `Spring` and `Joint`. Each is its OWN
// entity holding two body references (`a`/`b`) and local anchors (`rA`/`rB`), uploaded to the physics
// backend for you. A `Spring` is a soft distance spring: it pulls its anchors toward a rest length, so a
// platform hung from four corner springs sags under a dropped load and settles. A `Joint` is a hard pin;
// with `stiffnessAng: fixed` it also locks orientation, welding a bar rigidly to a wall so it cantilevers
// out and holds level. Both connect ordinary `Body` entities, so the bodies render and collide for free.
//
// This is the substrate's own joint surface. Motors, limits, and tumble's richer joint types (prismatic,
// wheel, cone/twist) live past it on the `Tumble.world` escape hatch — verified in the gym twins
// `joints-suspension` (the hertz/damping-tuned suspension) and `joints-cantilever` (the multi-link weld).

const FIXED = Number.POSITIVE_INFINITY; // `Joint.stiffnessAng`: ∞ locks orientation (a rigid weld)

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

// a soft spring between two bodies, anchored at the given local offsets. `rest` is the length it pulls
// toward; `stiffness` (N/m) is how hard it pulls.
function spring(
    state: State,
    a: number,
    b: number,
    rA: [number, number, number],
    rB: [number, number, number],
    rest: number,
    stiffness: number,
): void {
    const eid = state.create();
    state.add(eid, Spring);
    Spring.a.set(eid, a);
    Spring.b.set(eid, b);
    Spring.rA.set(eid, rA[0], rA[1], rA[2], 0);
    Spring.rB.set(eid, rB[0], rB[1], rB[2], 0);
    Spring.rest.set(eid, rest);
    Spring.stiffness.set(eid, stiffness);
}

// build the mechanism in code (not the scene) because a constraint references its bodies by eid. suspension
// sits at x = −4, cantilever at x = +4. Constraint fields are set here in `warm`, before the substrate's
// `ConstraintSystem` first uploads them.
function build(state: State): void {
    // suspension: a static frame with a dynamic platform hung from its four corners by springs. the springs
    // are anchored 1.8 m out from each body's centre and pull toward a 3 m rest length, so the platform
    // hangs ~3 m below the frame and sags a little when the two crates land on it.
    const frame = body(state, -4, 8, 0, 2.2, 0.08, 2.2, 0, [0.55, 0.57, 0.6]);
    const platform = body(state, -4, 5, 0, 2, 0.2, 2, 8, [0.5, 0.55, 0.85]);
    for (const [cx, cz] of [
        [-1.8, -1.8],
        [1.8, -1.8],
        [-1.8, 1.8],
        [1.8, 1.8],
    ]) {
        spring(state, frame, platform, [cx, 0, cz], [cx, 0, cz], 3, 400);
    }
    body(state, -4.6, 6.6, 0, 0.4, 0.4, 0.4, 1, [0.85, 0.6, 0.4]);
    body(state, -3.4, 7.2, 0, 0.4, 0.4, 0.4, 1, [0.85, 0.6, 0.4]);

    // cantilever: a bar welded to a static wall. the wall's outer face and the bar's inner end meet at
    // world x = 4, so the joint anchors are coincident there. `stiffnessAng: fixed` locks the bar's
    // orientation to the wall's, so it holds level; the default (0) is a spherical joint that swings free.
    const wall = body(state, 3.5, 6, 0, 0.5, 1.2, 1.2, 0, [0.5, 0.52, 0.56]);
    const bar = body(state, 5.5, 6, 0, 1.5, 0.25, 0.6, 2, [0.9, 0.55, 0.4]);
    const weld = state.create();
    state.add(weld, Joint);
    Joint.a.set(weld, wall);
    Joint.b.set(weld, bar);
    Joint.rA.set(weld, 0.5, 0, 0, 0); // wall-local: its outer +x face
    Joint.rB.set(weld, -1.5, 0, 0, 0); // bar-local: its inner −x end
    Joint.stiffnessAng.set(weld, FIXED);
}

export const Joints = {
    name: "Joints",
    warm: build,
} satisfies Plugin;

export default Joints;
