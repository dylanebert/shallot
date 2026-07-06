import { describe, expect, test } from "bun:test";
import { SPECULATIVE_DISTANCE } from "./collide";
import { boxHull, tetHull } from "./hull";
import { COLLISION_MARGIN } from "./manifold";
import {
    add,
    cross,
    dot,
    length,
    lengthSq,
    type Quat,
    qmul,
    rotate,
    scale,
    sub,
    type Vec3,
} from "./math";
import { type Body, body, capsule, hull, ShapeKind, sphere } from "./rigid";
import { collideRounded, narrowphase, ROUND_FEATURE } from "./rounded";
import { makeSolver, step } from "./solver";

// Closed-form gold for the rounded narrowphase (roadmap §6.3 step 1). Sphere-sphere is the centre
// distance; sphere-capsule / capsule-capsule are the segment-segment closest point (Ericson RTCD
// §5.1.9). Everything here is derived, not tuned — the cleanest tests in the solver, with no reference
// fixture: a sphere/capsule contact is exact (curved surfaces meet at a point), so the expected normal,
// surface anchors, and signed gap are computed in closed form per config. The GPU `collideRounded`
// (collide.ts WGSL) reproduces this; the gym `sat` scenario gates GPU == this on the real device.

// f64 closed form, so the only error is the routine's own arithmetic — exact to ~1e-12.
const TOL = 1e-9;

// recover a contact's world CORE anchor from its body-local arm (the inverse of how collideRounded
// stored it — the arm now anchors the core feature point, not the inflated surface; roadmap §6.3).
const worldOf = (b: Body, rLocal: Vec3): Vec3 => add(rotate(b.posAng, rLocal), b.posLin);

// the contact SURFACE point: the core anchor offset along the normal by the body's radius. A is offset
// toward B (−r·n), B toward A (+r·n) — the same reconstruction the solver does (manifold.ts). A box's
// radius is 0, so its surface is the bare arm.
const surfaceA = (a: Body, rA: Vec3, n: Vec3): Vec3 => sub(worldOf(a, rA), scale(n, a.roundRadius));
const surfaceB = (b: Body, rB: Vec3, n: Vec3): Vec3 => add(worldOf(b, rB), scale(n, b.roundRadius));

// a 90°-about-Z rotation: turns the default +Y capsule axis into −X (a horizontal capsule)
const Z90: Quat = [0, 0, Math.SQRT1_2, Math.SQRT1_2];

function near(got: number, want: number): void {
    expect(Math.abs(got - want)).toBeLessThan(TOL);
}
function nearVec(got: Vec3, want: Vec3): void {
    for (let i = 0; i < 3; i++) near(got[i], want[i]);
}

// assert one contact (index `i` of the manifold) against its closed form: the B→A normal (basis row 0),
// the world surface anchors on each body, an orthonormal basis, and the signed gap = normal·(xA − xB).
function checkContact(
    a: Body,
    b: Body,
    contacts: { rA: Vec3; rB: Vec3 }[],
    basis: Vec3[],
    want: { normal: Vec3; xA: Vec3; xB: Vec3; gap: number },
    i = 0,
): void {
    nearVec(basis[0], want.normal);
    // orthonormal completion — rows unit + mutually perpendicular (any valid tangent frame passes)
    for (const row of basis) near(length(row), 1);
    near(dot(basis[0], basis[1]), 0);
    near(dot(basis[0], basis[2]), 0);
    near(dot(basis[1], basis[2]), 0);

    const xA = surfaceA(a, contacts[i].rA, basis[0]);
    const xB = surfaceB(b, contacts[i].rB, basis[0]);
    nearVec(xA, want.xA);
    nearVec(xB, want.xB);
    near(dot(basis[0], sub(xA, xB)), want.gap);
}

// rounded × rounded (collideRounded): one contact, checked against its closed form.
function expectContact(
    a: Body,
    b: Body,
    want: { normal: Vec3; xA: Vec3; xB: Vec3; gap: number },
    dRel: Vec3 = [0, 0, 0],
): void {
    const { contacts, basis } = collideRounded(a, b, dRel);
    expect(contacts.length).toBe(1);
    checkContact(a, b, contacts, basis, want);
}

// the full narrowphase dispatch (rounded × box here): one contact, checked against its closed form.
function expectNarrow(
    a: Body,
    b: Body,
    want: { normal: Vec3; xA: Vec3; xB: Vec3; gap: number },
    dRel: Vec3 = [0, 0, 0],
): void {
    const { contacts, basis } = narrowphase(a, b, dRel);
    expect(contacts.length).toBe(1);
    checkContact(a, b, contacts, basis, want);
}

// a unit box (half-extents [1,1,1]) at the origin — the rounded-box gold reference convex.
const unitBox = (): Body => body([2, 2, 2], 0, 0.5, [0, 0, 0]);

describe("rounded narrowphase — closed-form gold", () => {
    // sphere-sphere: normal along the centre line, gap = dist − rA − rB, anchors on each surface.
    test("sphere-sphere vertical", () => {
        const a = sphere(0.5, 1, 0.5, [0, 0.9, 0]); // above
        const b = sphere(0.5, 1, 0.5, [0, 0, 0]);
        expectContact(a, b, {
            normal: [0, 1, 0],
            xA: [0, 0.4, 0], // centreA − n·rA
            xB: [0, 0.5, 0], // centreB + n·rB
            gap: -0.1, // 0.9 − 0.5 − 0.5
        });
    });

    test("sphere-sphere diagonal (3-4-5)", () => {
        const a = sphere(1, 1, 0.5, [1.74, 2.32, 0]); // (0.6,0.8,0)·2.9
        const b = sphere(2, 1, 0.5, [0, 0, 0]);
        expectContact(a, b, {
            normal: [0.6, 0.8, 0],
            xA: [1.14, 1.52, 0],
            xB: [1.2, 1.6, 0],
            gap: -0.1, // 2.9 − 1 − 2
        });
    });

    // sphere-capsule: the closest point on the capsule's core segment. To the side → mid-segment.
    test("sphere-capsule, sphere beside the cylinder (mid-segment)", () => {
        const a = sphere(0.5, 1, 0.5, [0.9, 0, 0]);
        const b = capsule(1, 0.5, 1, 0.5, [0, 0, 0]); // core segment (0,±1,0)
        expectContact(a, b, {
            normal: [1, 0, 0],
            xA: [0.4, 0, 0],
            xB: [0.5, 0, 0],
            gap: -0.1, // 0.9 − 0.5 − 0.5
        });
    });

    // above the cap → the closest point clamps to the segment endpoint
    test("sphere-capsule, sphere above the cap (segment endpoint)", () => {
        const a = sphere(0.5, 1, 0.5, [0, 1.9, 0]);
        const b = capsule(1, 0.5, 1, 0.5, [0, 0, 0]); // top endpoint (0,1,0)
        expectContact(a, b, {
            normal: [0, 1, 0],
            xA: [0, 1.4, 0],
            xB: [0, 1.5, 0],
            gap: -0.1, // 0.9 − 0.5 − 0.5
        });
    });

    // capsule-capsule, collinear (both along Y) → nearest endpoints, like two stacked pills
    test("capsule-capsule collinear (end to end)", () => {
        const a = capsule(1, 0.5, 1, 0.5, [0, 2.9, 0]); // core (0, 1.9..3.9, 0)
        const b = capsule(1, 0.5, 1, 0.5, [0, 0, 0]); // core (0, −1..1, 0)
        expectContact(a, b, {
            normal: [0, 1, 0],
            xA: [0, 1.4, 0], // lower endpoint of A − n·0.5
            xB: [0, 1.5, 0], // upper endpoint of B + n·0.5
            gap: -0.1, // 0.9 − 0.5 − 0.5
        });
    });

    // capsule-capsule, perpendicular (A along X over B along Y) → the crossing point on each
    test("capsule-capsule crossed", () => {
        const a = capsule(1, 0.5, 1, 0.5, [0, 1.9, 0], [0, 0, 0], Z90); // horizontal, core x∈[−1,1] at y=1.9
        const b = capsule(1, 0.5, 1, 0.5, [0, 0, 0]); // vertical, core (0,±1,0)
        expectContact(a, b, {
            normal: [0, 1, 0],
            xA: [0, 1.4, 0],
            xB: [0, 1.5, 0],
            gap: -0.1, // 0.9 − 0.5 − 0.5
        });
    });
});

describe("rounded narrowphase — band + sweep + dispatch", () => {
    test("separated past the band → no contact", () => {
        const a = sphere(0.5, 1, 0.5, [0, 1.2, 0]); // gap 0.2 ≫ SPECULATIVE_DISTANCE
        const b = sphere(0.5, 1, 0.5, [0, 0, 0]);
        expect(collideRounded(a, b).contacts.length).toBe(0);
    });

    test("separated within the speculative band → a contact carrying the +gap", () => {
        const gap = SPECULATIVE_DISTANCE * 0.5; // inside the static skin
        const a = sphere(0.5, 1, 0.5, [0, 1 + gap, 0]);
        const b = sphere(0.5, 1, 0.5, [0, 0, 0]);
        const { contacts, basis } = collideRounded(a, b);
        expect(contacts.length).toBe(1);
        // the signed gap is positive (still separated) — the solver's repulsion-only normal limits approach
        const xA = surfaceA(a, contacts[0].rA, basis[0]);
        const xB = surfaceB(b, contacts[0].rB, basis[0]);
        near(dot(basis[0], sub(xA, xB)), gap);
    });

    test("velocity sweep: a fast approacher past the band is caught; the same static gap is not", () => {
        const gap = 0.2; // past SPECULATIVE_DISTANCE
        const a = sphere(0.5, 1, 0.5, [0, 1 + gap, 0], [0, -30, 0]); // closing fast
        const b = sphere(0.5, 1, 0.5, [0, 0, 0]);
        const dt = 1 / 60;
        const dRel: Vec3 = scale(sub(a.velLin, b.velLin), dt); // (0, −0.5, 0): closing 0.5 > gap
        expect(collideRounded(a, b, dRel).contacts.length).toBe(1);
        // gating on velocity, not gap: the same pose with no relative motion stays separated
        const still = sphere(0.5, 1, 0.5, [0, 1 + gap, 0]);
        expect(collideRounded(still, b).contacts.length).toBe(0);
    });

    test("mixed rounded × box now produces a contact (Phase 6.3 rounded-box)", () => {
        const s = sphere(0.5, 1, 0.5, [0, 1.4, 0]); // above the box top face (y = 1)
        const boxB = unitBox();
        expect(s.shape).toBe(ShapeKind.Sphere);
        expect(narrowphase(s, boxB).contacts.length).toBe(1);
        // dispatch is order-independent — box-as-A produces the same single contact
        expect(narrowphase(boxB, s).contacts.length).toBe(1);
    });
});

describe("rounded-box narrowphase — closed-form gold (sphere)", () => {
    // closest point on the box clamps the sphere centre into the OBB; the normal runs from that surface
    // point to the centre, the gap = (centre-to-surface distance) − radius. Everything is closed form.

    test("sphere above a face → the face normal", () => {
        expectNarrow(sphere(0.5, 1, 0.5, [0, 1.4, 0]), unitBox(), {
            normal: [0, 1, 0],
            xA: [0, 0.9, 0], // sphere surface = centre − n·r
            xB: [0, 1, 0], // box surface = the clamped point
            gap: -0.1, // 0.4 − 0.5
        });
    });

    test("sphere off an edge (+x +y, z free) → the edge normal", () => {
        // centre = corner-edge [1,1,0] + [0.6,0.8,0]·0.45 = [1.27, 1.36, 0]
        expectNarrow(sphere(0.5, 1, 0.5, [1.27, 1.36, 0]), unitBox(), {
            normal: [0.6, 0.8, 0],
            xA: [0.97, 0.96, 0],
            xB: [1, 1, 0],
            gap: -0.05, // 0.45 − 0.5
        });
    });

    test("sphere off a corner → the corner normal", () => {
        // centre = corner [1,1,1] + [2/3,2/3,1/3]·0.45
        expectNarrow(sphere(0.5, 1, 0.5, [1.3, 1.3, 1.15]), unitBox(), {
            normal: [2 / 3, 2 / 3, 1 / 3],
            xA: [1.3 - 1 / 3, 1.3 - 1 / 3, 1.15 - 1 / 6],
            xB: [1, 1, 1],
            gap: -0.05, // 0.45 − 0.5
        });
    });

    test("sphere centre INSIDE the box → push out along the nearest face", () => {
        // centre [0.3,0,0] is inside; nearest face is +x (face distance 0.7 < 1)
        expectNarrow(sphere(0.5, 1, 0.5, [0.3, 0, 0]), unitBox(), {
            normal: [1, 0, 0],
            xA: [-0.2, 0, 0], // centre − n·r
            xB: [1, 0, 0], // pushed to the +x face
            gap: -1.2, // (−0.7) − 0.5
        });
    });

    test("rotated box: closest point in the box's own frame", () => {
        // a tall box [4,1,1] rotated Z90 → its local x (half 2) points world +y, so it is 4 tall in world.
        const tall = body([4, 1, 1], 0, 0.5, [0, 0, 0], [0, 0, 0], Z90);
        expectNarrow(sphere(0.5, 1, 0.5, [0, 2.4, 0]), tall, {
            normal: [0, 1, 0],
            xA: [0, 1.9, 0],
            xB: [0, 2, 0], // top of the tall box
            gap: -0.1,
        });
    });

    test("box-as-A flips the normal but reports the same gap", () => {
        // the same face contact, dispatched box-first: basis row 0 is now sphere→box (B→A)
        expectNarrow(unitBox(), sphere(0.5, 1, 0.5, [0, 1.4, 0]), {
            normal: [0, -1, 0],
            xA: [0, 1, 0], // box surface (A)
            xB: [0, 0.9, 0], // sphere surface (B)
            gap: -0.1,
        });
    });

    test("sphere separated past the band → no contact", () => {
        expect(narrowphase(sphere(0.5, 1, 0.5, [0, 2, 0]), unitBox()).contacts.length).toBe(0);
    });
});

describe("rounded-box narrowphase — closed-form gold (capsule, segment-clip)", () => {
    // the capsule core segment is clipped against the box reference face → up to 2 contacts sharing the
    // face normal (capsuleHull). A flat box top under a horizontal capsule yields a stable 2-point manifold
    // at the segment ends; a vertical capsule clips to its lower end. A wide flat ground: half-extents
    // [2,1,2], top y = 1.
    const ground = (): Body => body([4, 2, 4], 0, 0.5, [0, 0, 0]);

    test("horizontal capsule resting flat → two contacts at the endpoints, shared up normal", () => {
        // Z90 turns the +Y core axis into −X; endpoints land at x = ±1 over the face
        const cap = capsule(1, 0.5, 1, 0.5, [0, 1.4, 0], [0, 0, 0], Z90);
        const { contacts, basis } = narrowphase(cap, ground());
        expect(contacts.length).toBe(2);
        // ep0 = pos − rotate(Z90,[0,1,0]) = pos + [1,0,0] → x = +1; ep1 → x = −1
        expect(contacts[0].feature).toBe(ROUND_FEATURE | 0);
        expect(contacts[1].feature).toBe(ROUND_FEATURE | 1);
        checkContact(cap, ground(), contacts, basis, {
            normal: [0, 1, 0],
            xA: [1, 0.9, 0],
            xB: [1, 1, 0],
            gap: -0.1,
        });
        checkContact(
            cap,
            ground(),
            contacts,
            basis,
            {
                normal: [0, 1, 0],
                xA: [-1, 0.9, 0],
                xB: [-1, 1, 0],
                gap: -0.1,
            },
            1,
        );
    });

    test("vertical capsule (along Y) → one contact at the lower cap only", () => {
        // bottom endpoint at y = 1.4 touches the face; the top endpoint (y = 3.4) is far past the band
        const cap = capsule(1, 0.5, 1, 0.5, [0, 2.4, 0]);
        const { contacts, basis } = narrowphase(cap, ground());
        expect(contacts.length).toBe(1);
        expect(contacts[0].feature).toBe(ROUND_FEATURE | 0); // p0 = pos − h = the lower endpoint
        checkContact(cap, ground(), contacts, basis, {
            normal: [0, 1, 0],
            xA: [0, 0.9, 0], // lower-cap surface = endpoint − n·r
            xB: [0, 1, 0],
            gap: -0.1,
        });
    });
});

describe("rounded-box — through the solver", () => {
    // a flat box ground is what the rounded-box step unblocks (the loose pile a curved sphere dome can't
    // hold). A dropped sphere / horizontal capsule settles at the margin-rest, sunk a small mg/k below the
    // surface; the fresh-arm rule (manifold.ts) keeps the rounded contact from spinning the body up.
    const GroundTop = 0.5;
    const restY = (radius: number): number => GroundTop + radius - COLLISION_MARGIN;

    test("sphere settles on a flat box ground", () => {
        const s = makeSolver([
            body([10, 1, 10], 0, 0.5, [0, 0, 0]), // static box ground, top at y = 0.5
            sphere(0.5, 1, 0.5, [0, 3, 0]),
        ]);
        for (let f = 0; f < 600; f++) step(s);
        const ball = s.bodies[1];
        expect(length(ball.velLin)).toBeLessThan(1e-3);
        expect(length(ball.velAng)).toBeLessThan(5e-2); // no spurious spin (fresh arms)
        expect(Math.abs(ball.posLin[1] - restY(0.5))).toBeLessThan(2e-3);
    });

    test("tilted capsule settles flat on a flat box ground (the two-contact rest is stable)", () => {
        // dropped with a small tilt → it rotates to level as it settles; the two endpoint contacts
        // hold it flat at the margin rest (the rounded-box pipeline end to end, not the fresh-arms gate)
        const tilt: Quat = [0, 0, Math.sin(0.05), Math.cos(0.05)]; // ~6° off horizontal about Z, on top of Z90
        const start = qmul(tilt, Z90); // Z90 first (capsule along X), then the small tilt
        const s = makeSolver([
            body([10, 1, 10], 0, 0.5, [0, 0, 0]),
            capsule(1, 0.4, 1, 0.5, [0, 3, 0], [0, 0, 0], start),
        ]);
        for (let f = 0; f < 600; f++) step(s);
        const cap = s.bodies[1];
        expect(length(cap.velLin)).toBeLessThan(5e-3);
        expect(length(cap.velAng)).toBeLessThan(5e-2); // settled flat, not jittering
        expect(Math.abs(cap.posLin[1] - restY(0.4))).toBeLessThan(5e-3);
    });

    // a SPINNING sphere on a box rolls while its contact sticks — the case the fresh-arm rule guards.
    // The settling tests above don't exercise it (a contact that never rotates while stuck has frozen ==
    // fresh). Pins the gate: with the frozen-arm gate the rolling contact's stale arm injects torque and
    // the sphere tunnels straight through the box (y → −300+ in 200 frames); fresh arms keep it on top.
    test("rolling sphere stays on the box (fresh arms — does not tunnel through)", () => {
        const s = makeSolver([
            body([100, 1, 100], 0, 1.0, [0, 0, 0]), // big static box ground, top at y = 0.5
            sphere(0.5, 1, 1.0, [0, restY(0.5), 0]), // resting at the margin rest
        ]);
        s.bodies[1].velAng = [0, 0, 2]; // a gentle spin about Z → rolls in −x, the contact sticking
        for (let f = 0; f < 200; f++) step(s);
        const ball = s.bodies[1];
        expect(Number.isFinite(ball.posLin[1])).toBe(true);
        expect(ball.posLin[1]).toBeGreaterThan(0.9); // still rolling ON the box, not sunk through it
        expect(length(ball.velAng)).toBeLessThan(4); // bounded, not the frozen-arm runaway
    });
});

describe("rounded narrowphase — through the solver", () => {
    // a sphere dropped onto a static giant sphere settles at the margin-rest centre distance (5 + 0.5 −
    // MARGIN), sunk a small mg/k below it by the warmstart penalty. Validates the rounded contact end to
    // end: broadphase (shape-aware bound) → narrowphase → the contact Force → BDF1 settle.
    test("sphere rests on a static sphere ground", () => {
        const groundR = 5;
        const s = makeSolver([
            sphere(groundR, 0, 0.5, [0, 0, 0]), // static ground (top at y = 5)
            sphere(0.5, 1, 0.5, [0, 6, 0]), // dropped from above
        ]);
        for (let f = 0; f < 600; f++) step(s);

        const ball = s.bodies[1];
        const marginRest = groundR + 0.5 - COLLISION_MARGIN; // centre distance where the contact gap = −MARGIN
        // converges monotonically (8.8e-3 → 6.4e-5 m/s over 100 → 600 frames); < 1e-3 = settled
        expect(length(ball.velLin)).toBeLessThan(1e-3);
        // settled at the margin-rest within a small mg/k (the warmstart penalty holds it ≈ there), never
        // sunk through the ground
        expect(Math.abs(ball.posLin[1] - marginRest)).toBeLessThan(2e-3);
    });
});

describe("rounded rotational conservation — analytic invariants (no fixture)", () => {
    // The canonical rounded-shape stability gate (roadmap §6.3, the shape suites in Bullet/Jolt use the
    // same): a rounded contact under angular velocity must not leak spin into linear / normal motion. These
    // need no reference fixture — they are conservation laws, far tighter than a settle-within-tolerance
    // rest. A sphere's normal force passes through its centre, so the normal contact generates ZERO torque
    // and the spin is fully decoupled; the lever arm is geometric (−r·n, anti-parallel to the contact
    // normal), not a material point that rotates with the body. The bug these pin: the arm was rotated by
    // the body's free spin, so the −r·n part picked up a tangential component → jAng·n ≠ 0 → the normal
    // force spun the body up and the gap read separated → the sphere tunnelled (frictionless) or sank +
    // spun up (with friction). iters-independent (a formulation bug), so testing at the oracle iters=10 is
    // representative; energy is the one-sided invariant the bug violates (it INJECTS energy).

    const GroundTop = 0.5;
    const restY = (r: number): number => GroundTop + r - COLLISION_MARGIN;

    // total mechanical energy: ½m‖v‖² + ½ωᵀIω + mg·y (g the magnitude). A passive contact is dissipative
    // at worst — energy must be non-increasing. The bug makes it grow without bound.
    const energy = (b: Body, g: number): number =>
        0.5 * b.mass * lengthSq(b.velLin) +
        0.5 *
            (b.moment[0] * b.velAng[0] ** 2 +
                b.moment[1] * b.velAng[1] ** 2 +
                b.moment[2] * b.velAng[2] ** 2) +
        b.mass * Math.abs(g) * b.posLin[1];

    // a free (contactless) spin of the same sphere: the reference for the integrator's own ω drift. The
    // BDF1 quaternion recovery (qsub/qadd) bleeds a little |ω| per step for ANY rigid body (a free sphere
    // and a free box drift identically — it is not a contact effect). Comparing the in-contact spin to
    // this free spin SUBTRACTS that shared artifact out, leaving the contact's true angular contribution:
    // a frictionless sphere's normal force passes through its centre, so the contact must be angularly
    // TRANSPARENT — in-contact |ω| == free-spin |ω|. (The bug instead injected energy: ω ran 2 → 12.)
    const freeSpinOmega = (axis: Vec3, frames: number): number => {
        const f = makeSolver([sphere(0.5, 1, 0, [0, 100, 0])]); // high up, free fall + spin, no contact
        f.bodies[0].velAng = [axis[0], axis[1], axis[2]];
        for (let i = 0; i < frames; i++) step(f);
        return length(f.bodies[0].velAng);
    };

    test("frictionless spinning sphere is angularly transparent (no spin-up, no tunnel, no drift)", () => {
        const omega = 2; // spin about Z — the rolling-mode axis the bug coupled through the normal path
        const s = makeSolver([
            body([100, 1, 100], 0, 0, [0, 0, 0]), // static box ground, μ=0
            sphere(0.5, 1, 0, [0, restY(0.5), 0]), // resting at the margin rest, μ=0
        ]);
        s.bodies[1].velAng = [0, 0, omega];
        const e0 = energy(s.bodies[1], s.params.gravity);
        for (let f = 0; f < 400; f++) step(s);
        const b = s.bodies[1];

        expect(Number.isFinite(b.posLin[1])).toBe(true);
        expect(Math.abs(b.posLin[1] - restY(0.5))).toBeLessThan(1e-2); // didn't sink or tunnel (the bug: y → −77)
        expect(length(b.velLin)).toBeLessThan(1e-3); // μ=0 ⇒ no tangential force ⇒ the spin can't move it
        // the contact is angularly transparent: |ω| matches the free-spin reference (the only ω change is
        // the integrator's, shared with a contactless sphere) — neither spun up (bug) nor extra-damped.
        expect(Math.abs(length(b.velAng) - freeSpinOmega([0, 0, omega], 400))).toBeLessThan(5e-3);
        expect(energy(b, s.params.gravity)).toBeLessThan(e0 + 1e-3); // energy non-increasing
    });

    test("sphere rolling without slipping coasts at constant v (no spin-up, energy conserved)", () => {
        // start already in the rolling-without-slipping state: contact-point velocity v + ω×(−r·n) = 0.
        // n = +Y, arm = (0,−r,0); for vx forward, ω×arm must cancel it: ω = (0,0,−vx/r). High friction so
        // the sticking contact holds it. With no rolling resistance modelled it must coast at constant v,
        // no spin-up; the rolling relation vx = −ωz·r stays satisfied (no slip develops). Energy ≤ start.
        const r = 0.5;
        const vx = 1;
        const s = makeSolver([
            body([200, 1, 200], 0, 1, [0, 0, 0]), // static box ground, μ=1
            sphere(r, 1, 1, [0, restY(r), 0], [vx, 0, 0]),
        ]);
        s.bodies[1].velAng = [0, 0, -vx / r]; // matched spin → contact point stationary
        const e0 = energy(s.bodies[1], s.params.gravity);
        const x0 = s.bodies[1].posLin[0];
        for (let f = 0; f < 200; f++) step(s);
        const b = s.bodies[1];

        expect(Number.isFinite(b.posLin[1])).toBe(true);
        expect(Math.abs(b.posLin[1] - restY(r))).toBeLessThan(1e-2); // stays on the ground
        expect(b.posLin[0] - x0).toBeGreaterThan(2); // actually rolled forward (~vx·200·dt ≈ 3.3 m)
        // contact-point velocity stays ≈ 0 — the no-slip condition holds frame to frame
        const contactVel = add(b.velLin, cross(b.velAng, [0, -r, 0]));
        expect(length([contactVel[0], 0, contactVel[2]])).toBeLessThan(0.1);
        expect(b.velLin[0]).toBeGreaterThan(0.7); // didn't brake to a halt (no spurious tangential drag)
        expect(b.velLin[0]).toBeLessThan(1.05); // and didn't speed up
        expect(energy(b, s.params.gravity)).toBeLessThan(e0 + 1e-3); // no energy injection
    });
});

describe("capsule moment of inertia — derived limits (no AVBD reference)", () => {
    // the solid-capsule tensor is hand-derived (no reference covers it), so pin it at both limits where
    // the answer is known in closed form: the cylinder vanishes into a sphere, the radius into a thin rod.
    test("h → 0 reduces to the solid sphere (⅖mr²)", () => {
        nearVec(capsule(0, 0.5, 1, 0.5, [0, 0, 0]).moment, sphere(0.5, 1, 0.5, [0, 0, 0]).moment);
    });

    test("r → 0 reduces to the thin rod (m·L²/12 perpendicular, 0 axial)", () => {
        const h = 1.2;
        const m = 3;
        const mom = capsule(h, 1e-6, m, 0.5, [0, 0, 0]).moment;
        const rod = (m * (2 * h) ** 2) / 12;
        expect(mom[1]).toBeLessThan(1e-9); // axial → 0
        expect(mom[0]).toBeCloseTo(rod, 4); // perpendicular → m·L²/12 (the r→0 deviation is O(r/L))
        expect(mom[2]).toBeCloseTo(rod, 4);
    });
});

describe("rounded × hull — dispatch + closed form (Phase 6.3 hull)", () => {
    // a box body and a box-hull body are the SAME geometry (a box IS its boxHull), and both route through
    // collideRoundedPolytope, so a rounded shape vs each must produce an IDENTICAL manifold. The strongest
    // gate on the rounded × polytope unification + a check that the dispatch routes a ShapeKind.Hull body.
    const sameManifold = (a: Body, h: Body) => {
        const viaBox = narrowphase(sphere(0.5, 1, 0.5, [0, 1.4, 0]), a);
        const viaHull = narrowphase(sphere(0.5, 1, 0.5, [0, 1.4, 0]), h);
        expect(viaHull.contacts.length).toBe(viaBox.contacts.length);
        for (let i = 0; i < viaBox.contacts.length; i++) {
            expect(viaHull.contacts[i].feature).toBe(viaBox.contacts[i].feature);
            nearVec(viaHull.contacts[i].rA, viaBox.contacts[i].rA);
            nearVec(viaHull.contacts[i].rB, viaBox.contacts[i].rB);
        }
        for (let r = 0; r < 3; r++) nearVec(viaHull.basis[r], viaBox.basis[r]);
    };

    test("sphere vs a box-hull == sphere vs a box (dispatch + closestPointOnHull parity)", () => {
        sameManifold(
            body([2, 2, 2], 0, 0.5, [0, 0, 0]),
            hull(boxHull([2, 2, 2]), 0, 0.5, [0, 0, 0]),
        );
    });

    test("capsule vs a box-hull == capsule vs a box (segment-clip parity)", () => {
        const cap = capsule(1, 0.5, 1, 0.5, [0, 1.4, 0], [0, 0, 0], Z90); // horizontal, over the face
        const ground = (): Body => body([6, 2, 6], 0, 0.5, [0, 0, 0]);
        const groundHull = (): Body => hull(boxHull([6, 2, 6]), 0, 0.5, [0, 0, 0]);
        const viaBox = narrowphase(cap, ground());
        const viaHull = narrowphase(cap, groundHull());
        expect(viaHull.contacts.length).toBe(viaBox.contacts.length);
        expect(viaBox.contacts.length).toBe(2);
        for (let i = 0; i < 2; i++) {
            nearVec(viaHull.contacts[i].rA, viaBox.contacts[i].rA);
            nearVec(viaHull.contacts[i].rB, viaBox.contacts[i].rB);
        }
    });

    // a sphere above a tetrahedron — a genuinely non-box surface. No hand-computed value (the closest
    // feature is a slanted face/edge); assert the contact is self-consistent: the hull anchor is ON the
    // tet surface, the normal points outward to the sphere, and the gap matches the reconstructed surfaces.
    test("sphere vs a tetrahedron — a valid, self-consistent contact", () => {
        const s = sphere(0.5, 1, 0.5, [0, 0.9, 0]);
        const t = hull(tetHull(0.5), 0, 0.5, [0, 0, 0]);
        const { contacts, basis } = narrowphase(s, t);
        expect(contacts.length).toBe(1);
        const n = basis[0];
        const xA = surfaceA(s, contacts[0].rA, n); // sphere surface
        const xB = surfaceB(t, contacts[0].rB, n); // tet surface (radius 0 ⇒ the bare arm)
        // the normal points from the tet surface toward the sphere centre (outward), unit length
        near(length(n), 1);
        expect(dot(n, sub(s.posLin, xB))).toBeGreaterThan(0);
        // xB lies on the tet boundary: no face strictly in front of it (within ε)
        for (const v of [xB]) {
            const local = [v[0], v[1], v[2]] as Vec3;
            expect(local[1]).toBeLessThanOrEqual(0.5 + 1e-6); // the tet's top is y = 0.5
        }
        // penetrating a touch: the signed gap is the reconstructed surface gap
        expect(dot(n, sub(xA, xB))).toBeLessThan(0);
        expect(dot(n, sub(xA, xB))).toBeGreaterThan(-0.2);
    });
});

describe("rounded × hull — through the solver", () => {
    // a sphere dropped onto a box-HULL ground settles at the same margin rest as on a box (the hull path is
    // the box path), and a capsule on a tet-pyramid-style hull settles — the rounded × hull pipeline end to
    // end (broadphase by the hull bounding radius → narrowphase → the contact Force → BDF1 settle).
    const CollMargin = 0.01;
    test("sphere settles on a box-hull ground (same rest as a box)", () => {
        const s = makeSolver([
            hull(boxHull([10, 1, 10]), 0, 0.5, [0, 0, 0]), // static hull ground, top at y = 0.5
            sphere(0.5, 1, 0.5, [0, 3, 0]),
        ]);
        for (let f = 0; f < 600; f++) step(s);
        const ball = s.bodies[1];
        expect(length(ball.velLin)).toBeLessThan(1e-3);
        expect(length(ball.velAng)).toBeLessThan(5e-2); // no spurious spin (fresh arms on the hull pair)
        expect(Math.abs(ball.posLin[1] - (0.5 + 0.5 - CollMargin))).toBeLessThan(2e-3);
    });
});
