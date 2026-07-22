// Fast-tier behavioral guards for the public step path. The bit-exact contract lives in
// step.fixture.ts (gated on the C reference); these run in `bun test` and catch gross breakage
// without the reference — gravity integration, contact resolution, and the sleep/wake/split
// transitions (islands + sleeping). Sleep is left on (the default) where the test exercises it.

import { expect, test } from "bun:test";
import { NULL_INDEX } from "./array";
import { AwakeContact, type Contact, ContactFlags } from "./contact";
import { SetType } from "./core";
import {
    type Body,
    BodyType,
    createCompound,
    createGrid,
    createGridMesh,
    defaultSurfaceMaterial,
    makeBoxHull,
    World,
} from "./index";
import { ShapeType } from "./types";
import type { WorldState } from "./world";

const QUAT_ID = { v: { x: 0, y: 0, z: 0 }, s: 1 };

const DT = 1 / 60;

// Recompute a contact's collide-partition membership from first principles (independent of the cached
// `recycleStable` / `collideKind` the code maintains), so the invariant test cross-checks rather than
// restates the implementation. Mirrors collide's eligibility: enumerated iff setIndex is Awake; recycle
// iff dynamic-dynamic direct-convex with both bodies awake.
function expectedKind(state: WorldState, contact: Contact): number {
    if (contact.setIndex !== SetType.Awake) {
        return AwakeContact.None;
    }
    const typeA = state.shapes[contact.shapeIdA].type;
    const stable =
        (contact.flags & ContactFlags.contactStaticFlag) === 0 &&
        (contact.flags & ContactFlags.simMeshContact) === 0 &&
        (typeA === ShapeType.Sphere || typeA === ShapeType.Capsule || typeA === ShapeType.Hull);
    if (stable) {
        const bodyA = state.bodies[contact.edges[0].bodyId];
        const bodyB = state.bodies[contact.edges[1].bodyId];
        if (bodyA.setIndex === SetType.Awake && bodyB.setIndex === SetType.Awake) {
            return AwakeContact.Recycle;
        }
    }
    return AwakeContact.Other;
}

// The one-sentence invariant: after any event sequence, awakeRecycleContacts + awakeOtherContacts hold
// exactly the eligible/enumerated contacts, each in the list its state dictates, with consistent indices.
function assertPartition(state: WorldState): { recycle: number; other: number } {
    const expectRecycle = new Set<number>();
    const expectOther = new Set<number>();
    for (const contact of state.contacts) {
        if (contact.contactId === NULL_INDEX) {
            continue; // freed slot
        }
        const kind = expectedKind(state, contact);
        expect(contact.collideKind).toBe(kind);
        if (kind === AwakeContact.Recycle) {
            expectRecycle.add(contact.contactId);
        } else if (kind === AwakeContact.Other) {
            expectOther.add(contact.contactId);
        }

        // bodySimIndex must be current for every side the solver/recycle reads: a static side is NULL,
        // an awake side is that body's live localIndex. A sleeping side is unread (seeded, not tracked)
        // — skip it. A stale awake side here is exactly a missed localIndex-propagation event.
        const bodyA = state.bodies[contact.edges[0].bodyId];
        const bodyB = state.bodies[contact.edges[1].bodyId];
        if (bodyA.type === BodyType.Static) {
            expect(contact.bodySimIndexA).toBe(NULL_INDEX);
        } else if (bodyA.setIndex === SetType.Awake) {
            expect(contact.bodySimIndexA).toBe(bodyA.localIndex);
        }
        if (bodyB.type === BodyType.Static) {
            expect(contact.bodySimIndexB).toBe(NULL_INDEX);
        } else if (bodyB.setIndex === SetType.Awake) {
            expect(contact.bodySimIndexB).toBe(bodyB.localIndex);
        }
    }

    expect(state.awakeRecycleContacts.length).toBe(expectRecycle.size);
    expect(state.awakeOtherContacts.length).toBe(expectOther.size);
    expect(new Set(state.awakeRecycleContacts)).toEqual(expectRecycle);
    expect(new Set(state.awakeOtherContacts)).toEqual(expectOther);
    // Each cached slot index round-trips to its list position (what swap-remove relies on).
    state.awakeRecycleContacts.forEach((id, i) => {
        expect(state.contacts[id].collideKind).toBe(AwakeContact.Recycle);
        expect(state.contacts[id].collideIndex).toBe(i);
    });
    state.awakeOtherContacts.forEach((id, i) => {
        expect(state.contacts[id].collideKind).toBe(AwakeContact.Other);
        expect(state.contacts[id].collideIndex).toBe(i);
    });
    return { recycle: expectRecycle.size, other: expectOther.size };
}

function ground(world: World): void {
    const g = world.createBody({ position: { x: 0, y: -1, z: 0 } });
    g.createHull({}, makeBoxHull(20, 1, 20));
}

function box(world: World, x: number, y: number, vx = 0): Body {
    const b = world.createBody({
        type: BodyType.Dynamic,
        position: { x, y, z: 0 },
        linearVelocity: { x: vx, y: 0, z: 0 },
    });
    b.createHull({}, makeBoxHull(0.5, 0.5, 0.5));
    return b;
}

test("a free body falls under gravity", () => {
    const world = new World({ gravity: { x: 0, y: -10, z: 0 }, enableSleep: false });
    const body = world.createBody({ type: BodyType.Dynamic, position: { x: 0, y: 5, z: 0 } });
    body.createSphere({}, { center: { x: 0, y: 0, z: 0 }, radius: 0.5 });

    const y0 = body.getPosition().y;
    for (let i = 0; i < 40; ++i) {
        world.step(DT, 4);
    }

    // Fell downward and is moving down; ~2/3 s of free fall is well over a meter.
    expect(body.getPosition().y).toBeLessThan(y0 - 1);
    expect(body.getLinearVelocity().y).toBeLessThan(0);
    expect(body.isAwake()).toBe(true);
});

test("a sphere settles on the ground without tunnelling", () => {
    const world = new World({ gravity: { x: 0, y: -10, z: 0 }, enableSleep: false });

    // Ground box: center y = -1, half-height 1, so its top face is at y = 0.
    const ground = world.createBody({ position: { x: 0, y: -1, z: 0 } });
    ground.createHull({}, makeBoxHull(20, 1, 20));

    const ball = world.createBody({ type: BodyType.Dynamic, position: { x: 0, y: 2, z: 0 } });
    ball.createSphere({}, { center: { x: 0, y: 0, z: 0 }, radius: 0.5 });

    for (let i = 0; i < 180; ++i) {
        world.step(DT, 4);
    }

    // Rests with its center about one radius above the ground top (small soft-contact penetration).
    const y = ball.getPosition().y;
    expect(y).toBeGreaterThan(0.45);
    expect(y).toBeLessThan(0.55);
});

test("a settled stack falls asleep, then an impact wakes it", () => {
    const world = new World({ gravity: { x: 0, y: -10, z: 0 } });
    ground(world);
    const bottom = box(world, 0, 0.5);
    const top = box(world, 0, 1.5);

    // Stacked touching with no initial motion: both settle and fall asleep well within a second.
    for (let i = 0; i < 90; ++i) {
        world.step(DT, 4);
    }
    expect(bottom.isAwake()).toBe(false);
    expect(top.isAwake()).toBe(false);

    // Drop a box onto the sleeping stack; the new contact must wake it (b3LinkContact wake path).
    const drop = box(world, 0, 8);
    let woke = false;
    for (let i = 0; i < 90; ++i) {
        world.step(DT, 4);
        if (bottom.isAwake()) {
            woke = true;
        }
    }
    expect(woke).toBe(true);
    // Drop landed on top of the stack (three boxes → top-of-stack center near y = 2.5).
    expect(drop.getPosition().y).toBeGreaterThan(2.0);
});

test("collide partition tracks the eligible awake contacts through create, sleep, and wake", () => {
    const world = new World({ gravity: { x: 0, y: -10, z: 0 } });
    const state = world.state;
    ground(world); // static ground: box↔ground contacts are static → awakeOtherContacts
    const bottom = box(world, 0, 0.5);
    const top = box(world, 0, 1.5); // box↔box is dynamic-dynamic convex → awakeRecycleContacts

    // Settle and sleep. The invariant must hold after every step; along the way both lists go non-empty
    // (a box↔box recycle contact plus box↔ground static contacts) and then empty as the island sleeps.
    let sawRecycle = false;
    let sawOther = false;
    let sawEmpty = false;
    for (let i = 0; i < 90; ++i) {
        world.step(DT, 4);
        const { recycle, other } = assertPartition(state);
        sawRecycle ||= recycle > 0;
        sawOther ||= other > 0;
        sawEmpty ||= recycle === 0 && other === 0;
    }
    expect(bottom.isAwake()).toBe(false);
    expect(sawRecycle).toBe(true); // a dynamic-dynamic recycle contact was tracked
    expect(sawOther).toBe(true); // a static box↔ground contact was tracked
    expect(sawEmpty).toBe(true); // the lists emptied when the island slept

    // Drop a box onto the sleeping stack: the wake path (linkContact → wakeSolverSet) must repopulate
    // the lists, and the invariant must hold through the wake + re-settle.
    const drop = box(world, 0, 8);
    let woke = false;
    let sawRefill = false;
    for (let i = 0; i < 120; ++i) {
        world.step(DT, 4);
        const { recycle } = assertPartition(state);
        woke ||= bottom.isAwake();
        sawRefill ||= recycle > 0;
    }
    expect(woke).toBe(true);
    expect(sawRefill).toBe(true);

    // Flip a body's type through bodySetType: it destroys the body's contacts (dropping them from the
    // partition) and recreates them next step with a fresh recycleStable — dynamic→static flips the
    // box↔box pair's contactStaticFlag, moving it recycle→other; static→dynamic flips it back.
    bottom.setType(BodyType.Static);
    assertPartition(state); // contacts destroyed synchronously, none recreated yet
    let sawStaticOther = false;
    for (let i = 0; i < 40; ++i) {
        world.step(DT, 4);
        const { recycle } = assertPartition(state);
        // While `bottom` is static, top↔bottom is a static contact — no recycle member survives it
        // unless another dynamic pair forms, which this scene has none of.
        sawStaticOther ||= recycle === 0;
    }
    expect(sawStaticOther).toBe(true);

    bottom.setType(BodyType.Dynamic);
    assertPartition(state);
    let sawRecycleAgain = false;
    for (let i = 0; i < 60; ++i) {
        world.step(DT, 4);
        const { recycle } = assertPartition(state);
        sawRecycleAgain ||= recycle > 0; // top↔bottom is dynamic-dynamic again
    }
    expect(sawRecycleAgain).toBe(true);

    // Destroy a body that still has live contacts: destroyBodyContacts must pull each from the partition.
    expect(state.bodies[top.id.index1 - 1].contactCount).toBeGreaterThan(0);
    top.destroy();
    assertPartition(state);
    world.step(DT, 4);
    assertPartition(state);
    drop.destroy();
    assertPartition(state);
});

test("bodySimIndex follows a surviving awake body's localIndex swap on destroy", () => {
    // Sleep off, so every dynamic body stays awake and the swap-remove is deterministic. Three separated
    // columns rest on the ground (each a static box↔ground contact; no box↔box), awake localIndex 0/1/2.
    const world = new World({ gravity: { x: 0, y: -10, z: 0 }, enableSleep: false });
    const state = world.state;
    ground(world);
    const a = box(world, -3, 0.5);
    const b = box(world, 0, 0.5);
    const c = box(world, 3, 0.5);
    for (let i = 0; i < 60; ++i) {
        world.step(DT, 4);
        assertPartition(state);
    }
    expect(b.isAwake()).toBe(true);
    expect(c.isAwake()).toBe(true);

    // Destroy the middle body: destroyBody swap-removes its awake slot, migrating the last awake body
    // (c, localIndex 2) into slot 1 while c still holds a live ground contact. assertPartition then
    // asserts c's contact bodySimIndex followed the localIndex change — the moved-body hook. A missing
    // hook leaves it stale at 2 and this fails.
    b.destroy();
    assertPartition(state);
    for (let i = 0; i < 20; ++i) {
        world.step(DT, 4);
        assertPartition(state);
    }
    expect(a.isValid()).toBe(true);
    expect(c.isValid()).toBe(true);
});

test("continuous collision catches a fast body that would tunnel a thin floor", () => {
    // A tiny box fired straight down fast enough to clear a thin floor within a single discrete step.
    const drop = (enableContinuous: boolean): number => {
        const world = new World({
            gravity: { x: 0, y: -10, z: 0 },
            enableSleep: false,
            enableContinuous,
        });
        // Thin static floor, top face at y = 0.
        const floor = world.createBody({ position: { x: 0, y: -0.05, z: 0 } });
        floor.createHull({}, makeBoxHull(5, 0.05, 5));
        const b = world.createBody({
            type: BodyType.Dynamic,
            position: { x: 0, y: 3, z: 0 },
            linearVelocity: { x: 0, y: -120, z: 0 },
        });
        b.createHull({}, makeBoxHull(0.1, 0.1, 0.1));
        for (let i = 0; i < 20; ++i) {
            world.step(DT, 4);
        }
        return b.getPosition().y;
    };

    // Without CCD the box passes straight through the floor — its AABB never overlaps in any step.
    expect(drop(false)).toBeLessThan(-1);
    // With CCD the sweep stops it at the surface: it rests just above the floor top (y = 0).
    const caught = drop(true);
    expect(caught).toBeGreaterThan(0);
    expect(caught).toBeLessThan(0.5);
});

test("touching bodies that separate sleep as independent islands", () => {
    const world = new World({ gravity: { x: 0, y: -10, z: 0 } });
    ground(world);
    // Start exactly touching (one island via their mutual contact), pushed apart.
    const left = box(world, -0.5, 0.5, -1);
    const right = box(world, 0.5, 0.5, 1);

    for (let i = 0; i < 120; ++i) {
        world.step(DT, 4);
    }

    // They can only fall asleep once the shared island fractures (b3SplitIsland): a two-body
    // split-pending island refuses to sleep. Both asleep ⇒ the split path ran.
    expect(left.isAwake()).toBe(false);
    expect(right.isAwake()).toBe(false);
    // And they ended up apart, not stuck together.
    expect(right.getPosition().x - left.getPosition().x).toBeGreaterThan(1.0);
});

test("a box rests on a triangle-mesh floor without tunnelling", () => {
    const world = new World({ gravity: { x: 0, y: -10, z: 0 }, enableSleep: false });

    // A static grid-mesh floor in the xz-plane at y = 0.
    const floor = world.createBody({ position: { x: 0, y: 0, z: 0 } });
    floor.createMesh({}, createGridMesh(8, 8, 1.0, 0, true), { x: 1, y: 1, z: 1 });

    const b = world.createBody({ type: BodyType.Dynamic, position: { x: 0, y: 2, z: 0 } });
    b.createHull({}, makeBoxHull(1.0, 1.0, 1.0));

    for (let i = 0; i < 180; ++i) {
        world.step(DT, 4);
    }

    // Rests flat on the mesh: its half-height (1) above the floor top (y = 0), no tunnelling.
    const y = b.getPosition().y;
    expect(y).toBeGreaterThan(0.9);
    expect(y).toBeLessThan(1.1);
});

test("continuous collision catches a fast body swept at a triangle-mesh floor", () => {
    // A grid-mesh floor is a zero-thickness plane of triangles, so a fast body's AABB never overlaps
    // it in any discrete step — only the swept mesh time-of-impact path can catch it.
    const drop = (enableContinuous: boolean): number => {
        const world = new World({
            gravity: { x: 0, y: -10, z: 0 },
            enableSleep: false,
            enableContinuous,
        });
        const floor = world.createBody({ position: { x: 0, y: 0, z: 0 } });
        floor.createMesh({}, createGridMesh(8, 8, 1.0, 0, true), { x: 1, y: 1, z: 1 });
        const b = world.createBody({
            type: BodyType.Dynamic,
            position: { x: 0, y: 3, z: 0 },
            linearVelocity: { x: 0, y: -60, z: 0 },
        });
        b.createHull({}, makeBoxHull(0.25, 0.25, 0.25));
        for (let i = 0; i < 40; ++i) {
            world.step(DT, 4);
        }
        return b.getPosition().y;
    };

    // Without CCD the box passes straight through the mesh plane.
    expect(drop(false)).toBeLessThan(-1);
    // With CCD the swept per-triangle TOI catches it, and it settles on the surface (half-extent 0.25).
    const caught = drop(true);
    expect(caught).toBeGreaterThan(0);
    expect(caught).toBeLessThan(0.5);
});

test("a box rests on a height-field floor without tunnelling", () => {
    const world = new World({ gravity: { x: 0, y: -10, z: 0 }, enableSleep: false });

    // A flat 8x8 grid height field, offset to centre on the origin (surface at y ~ 0).
    const floor = world.createBody({ position: { x: -3.5, y: 0, z: -3.5 } });
    floor.createHeightField({}, createGrid(8, 8, { x: 1, y: 1, z: 1 }, false));

    const b = world.createBody({ type: BodyType.Dynamic, position: { x: 0, y: 2, z: 0 } });
    b.createHull({}, makeBoxHull(1.0, 1.0, 1.0));

    for (let i = 0; i < 180; ++i) {
        world.step(DT, 4);
    }

    // Rests flat on the field: its half-height (1) above the surface (y ~ 0), no tunnelling.
    const y = b.getPosition().y;
    expect(y).toBeGreaterThan(0.9);
    expect(y).toBeLessThan(1.1);
});

test("continuous collision catches a fast body swept at a height-field floor", () => {
    // A flat height field is a plane of triangles: a fast body's discrete AABB never overlaps it, so
    // only the swept height-field time-of-impact path (queryHeightField) can catch it.
    const drop = (enableContinuous: boolean): number => {
        const world = new World({
            gravity: { x: 0, y: -10, z: 0 },
            enableSleep: false,
            enableContinuous,
        });
        const floor = world.createBody({ position: { x: -3.5, y: 0, z: -3.5 } });
        floor.createHeightField({}, createGrid(8, 8, { x: 1, y: 1, z: 1 }, false));
        const b = world.createBody({
            type: BodyType.Dynamic,
            position: { x: 0, y: 3, z: 0 },
            linearVelocity: { x: 0, y: -60, z: 0 },
        });
        b.createHull({}, makeBoxHull(0.25, 0.25, 0.25));
        for (let i = 0; i < 40; ++i) {
            world.step(DT, 4);
        }
        return b.getPosition().y;
    };

    // Without CCD the box passes straight through the field plane.
    expect(drop(false)).toBeLessThan(-1);
    // With CCD the swept per-triangle TOI catches it, and it settles on the surface (half-extent 0.25).
    const caught = drop(true);
    expect(caught).toBeGreaterThan(0);
    expect(caught).toBeLessThan(0.5);
});

test("a box rests on a compound floor without tunnelling", () => {
    const world = new World({ gravity: { x: 0, y: -10, z: 0 }, enableSleep: false });

    // A static compound floor built from two offset box-hull children; their tops sit at y = 0.
    const mat = defaultSurfaceMaterial();
    const slab = makeBoxHull(1.5, 0.25, 1.5);
    const compound = createCompound({
        hulls: [
            { hull: slab, transform: { p: { x: 1.4, y: -0.25, z: 0 }, q: QUAT_ID }, material: mat },
            {
                hull: slab,
                transform: { p: { x: -1.4, y: -0.25, z: 0 }, q: QUAT_ID },
                material: mat,
            },
        ],
    });
    const floor = world.createBody({});
    floor.createCompound({}, compound!);

    const b = world.createBody({ type: BodyType.Dynamic, position: { x: 0, y: 2, z: 0 } });
    b.createHull({}, makeBoxHull(1.0, 1.0, 1.0));

    for (let i = 0; i < 180; ++i) {
        world.step(DT, 4);
    }

    // Rests flat on the compound: its half-height (1) above the child tops (y = 0), no tunnelling.
    const y = b.getPosition().y;
    expect(y).toBeGreaterThan(0.9);
    expect(y).toBeLessThan(1.1);
});

test("continuous collision catches a fast body swept at a compound floor", () => {
    // The two hull children are thin slabs (0.1 thick, tops at y = 0): a fast body clears their
    // discrete AABB across a step, so only the swept compound time-of-impact path (queryCompound +
    // per-child dispatch) can catch it. The box drops over the seam where both children overlap.
    const drop = (enableContinuous: boolean): number => {
        const world = new World({
            gravity: { x: 0, y: -10, z: 0 },
            enableSleep: false,
            enableContinuous,
        });
        const mat = defaultSurfaceMaterial();
        const slab = makeBoxHull(1.5, 0.05, 1.5);
        const compound = createCompound({
            hulls: [
                {
                    hull: slab,
                    transform: { p: { x: 1.4, y: -0.05, z: 0 }, q: QUAT_ID },
                    material: mat,
                },
                {
                    hull: slab,
                    transform: { p: { x: -1.4, y: -0.05, z: 0 }, q: QUAT_ID },
                    material: mat,
                },
            ],
        });
        const floor = world.createBody({});
        floor.createCompound({}, compound!);
        const b = world.createBody({
            type: BodyType.Dynamic,
            position: { x: 0, y: 3, z: 0 },
            // -40 clears the slabs' discrete AABB every step (so only the swept TOI can catch it), yet
            // is gentle enough that the caught body settles rather than rebounding off the seam. A much
            // faster impact (e.g. -100) rebounds and re-tunnels — a shared limit of the colored solver,
            // verified bit-exact against box3d, not a port bug.
            linearVelocity: { x: 0, y: -40, z: 0 },
        });
        b.createHull({}, makeBoxHull(0.25, 0.25, 0.25));
        for (let i = 0; i < 60; ++i) {
            world.step(DT, 4);
        }
        return b.getPosition().y;
    };

    // Without CCD the box passes straight through the compound slabs.
    expect(drop(false)).toBeLessThan(-1);
    // With CCD the swept per-child TOI catches it, and it settles on the surface (half-extent 0.25).
    const caught = drop(true);
    expect(caught).toBeGreaterThan(0);
    expect(caught).toBeLessThan(0.5);
});

test("a capsule child on a compound floor supports a box (non-f32 radius)", () => {
    // Regression: a capsule radius of 0.3 is not f32-exact. If the child geometry is stored as the raw
    // f64 value instead of rounded to f32, the contact solve diverges from the C reference. Behaviorally
    // the box must still come to rest on the capsule tops (y = radius + half-height = 1.3), not tunnel.
    const world = new World({ gravity: { x: 0, y: -10, z: 0 }, enableSleep: false });
    const mat = defaultSurfaceMaterial();
    const compound = createCompound({
        capsules: [
            {
                capsule: {
                    center1: { x: -2, y: 0, z: 0.7 },
                    center2: { x: 2, y: 0, z: 0.7 },
                    radius: 0.3,
                },
                material: mat,
            },
            {
                capsule: {
                    center1: { x: -2, y: 0, z: -0.7 },
                    center2: { x: 2, y: 0, z: -0.7 },
                    radius: 0.3,
                },
                material: mat,
            },
        ],
    });
    const floor = world.createBody({});
    floor.createCompound({}, compound!);

    const b = world.createBody({ type: BodyType.Dynamic, position: { x: 0, y: 2, z: 0 } });
    b.createHull({}, makeBoxHull(1.0, 1.0, 1.0));

    for (let i = 0; i < 180; ++i) {
        world.step(DT, 4);
    }

    const y = b.getPosition().y;
    expect(y).toBeGreaterThan(1.2);
    expect(y).toBeLessThan(1.4);
});
