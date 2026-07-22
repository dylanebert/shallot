// Stage 2 (events + read surface) tests. Contact begin/end/hit, body move (+ fellAsleep + CCD
// rewind), and joint events all read sim state and push to arrays the world-state hash never covers,
// so — like sensors and queries — the contract is behavioral, not bit-exact. These mirror the
// upstream executable specs (reference/box3d test_world.c: TestContactEvents, TestHitEvents,
// TestCompoundHitEvents, TestContinuousMoveEvent) and pin the move/joint/out-param behavior beside them.

import { describe, expect, test } from "bun:test";
import {
    BodyType,
    createCompound,
    defaultSurfaceMaterial,
    makeBoxHull,
    type SurfaceMaterial,
    World,
} from "./index";

const sphere = (radius: number) => ({ center: { x: 0, y: 0, z: 0 }, radius });
const material = (over: Partial<SurfaceMaterial>): SurfaceMaterial => ({
    ...defaultSurfaceMaterial(),
    ...over,
});
const IDENTITY = { v: { x: 0, y: 0, z: 0 }, s: 1 };

describe("contact events", () => {
    // Mirrors TestContactEvents: a restitutive sphere bounces on the ground, so we see both begin and
    // end touch events, the ids resolve to the two shapes, and the carried contact handle is valid.
    test("a bouncing sphere fires begin and end touch events with a valid contact", () => {
        const world = new World();
        const ground = world.createBody({
            type: BodyType.Static,
            position: { x: 0, y: -0.5, z: 0 },
        });
        const groundShape = ground.createHull({}, makeBoxHull(10, 0.5, 10));

        const body = world.createBody({ type: BodyType.Dynamic, position: { x: 0, y: 5, z: 0 } });
        const sphereShape = body.createSphere(
            { density: 1, enableContactEvents: true, baseMaterial: material({ restitution: 0.6 }) },
            sphere(0.5),
        );

        let beginCount = 0;
        let endCount = 0;
        let idsChecked = false;

        for (let i = 0; i < 120; ++i) {
            world.step(1 / 60);
            const ev = world.getContactEvents();

            if (ev.beginEvents.length > 0 && idsChecked === false) {
                const be = ev.beginEvents[0];
                const aSphere = be.shapeA.id.index1 === sphereShape.id.index1;
                const bSphere = be.shapeB.id.index1 === sphereShape.id.index1;
                const aGround = be.shapeA.id.index1 === groundShape.id.index1;
                const bGround = be.shapeB.id.index1 === groundShape.id.index1;
                expect((aSphere && bGround) || (aGround && bSphere)).toBe(true);
                expect(be.contact.isValid()).toBe(true);
                // getData resolves the same two shapes and carries the live manifold.
                const data = be.contact.getData();
                expect(data.shapeA.id.index1).toBe(be.shapeA.id.index1);
                expect(data.shapeB.id.index1).toBe(be.shapeB.id.index1);
                expect(data.manifolds.length).toBeGreaterThan(0);
                idsChecked = true;
            }

            beginCount += ev.beginEvents.length;
            endCount += ev.endEvents.length;
        }

        expect(idsChecked).toBe(true);
        expect(beginCount).toBeGreaterThanOrEqual(1);
        expect(endCount).toBeGreaterThanOrEqual(1);

        world.destroy();
    });

    test("no contact events when neither shape enables them", () => {
        const world = new World();
        const ground = world.createBody({
            type: BodyType.Static,
            position: { x: 0, y: -0.5, z: 0 },
        });
        ground.createHull({}, makeBoxHull(10, 0.5, 10));
        const body = world.createBody({ type: BodyType.Dynamic, position: { x: 0, y: 1, z: 0 } });
        body.createSphere({ density: 1 }, sphere(0.5));

        let total = 0;
        for (let i = 0; i < 60; ++i) {
            world.step(1 / 60);
            const ev = world.getContactEvents();
            total += ev.beginEvents.length + ev.endEvents.length + ev.hitEvents.length;
        }
        expect(total).toBe(0);

        world.destroy();
    });

    // getContactEvents maps world.contactBeginEvents/contactHitEvents into fresh wrapper objects
    // per call (api.ts); the internal arrays are truncated in place at the top of the next step
    // (step.ts: "the API accessors map these into fresh wrapped objects per call ... never expose
    // the raw arrays, so a caller can only ever hold copies and the reuse is unobservable"). This
    // guards that invariant against a future accessor that returns a view into the reused array.
    test("held begin/hit events from step N survive step N+1's internal truncate-reuse", () => {
        const world = new World({ hitEventThreshold: 1 });
        const ground = world.createBody({
            type: BodyType.Static,
            position: { x: 0, y: -0.5, z: 0 },
        });
        const groundShape = ground.createHull({}, makeBoxHull(10, 0.5, 10));

        const body = world.createBody({ type: BodyType.Dynamic, position: { x: 0, y: 5, z: 0 } });
        const sphereShape = body.createSphere(
            {
                density: 1,
                enableContactEvents: true,
                enableHitEvents: true,
                baseMaterial: material({ restitution: 0.6 }),
            },
            sphere(0.5),
        );

        let held: ReturnType<typeof world.getContactEvents> | null = null;
        for (let i = 0; i < 120 && held === null; ++i) {
            world.step(1 / 60);
            const ev = world.getContactEvents();
            if (ev.beginEvents.length > 0 && ev.hitEvents.length > 0) {
                held = ev;
            }
        }
        if (held === null) {
            throw new Error("expected at least one begin+hit event before the truncate-reuse step");
        }

        const beginLength = held.beginEvents.length;
        const hitLength = held.hitEvents.length;
        const begin = held.beginEvents[0];
        const hit = held.hitEvents[0];
        const beginShapeA = begin.shapeA.id.index1;
        const beginShapeB = begin.shapeB.id.index1;
        const beginContact = begin.contact.id.index1;
        const hitShapeA = hit.shapeA.id.index1;
        const hitShapeB = hit.shapeB.id.index1;
        const hitPoint = { ...hit.point };
        const hitNormal = { ...hit.normal };
        const hitSpeed = hit.approachSpeed;

        // world.contactBeginEvents/contactHitEvents are truncated (`.length = 0`) at the top of the
        // very next step (step.ts) — a raw-array accessor would corrupt `held` right here.
        world.step(1 / 60);

        expect(held.beginEvents.length).toBe(beginLength);
        expect(held.hitEvents.length).toBe(hitLength);
        expect(held.beginEvents[0]).toBe(begin);
        expect(held.hitEvents[0]).toBe(hit);
        expect(begin.shapeA.id.index1).toBe(beginShapeA);
        expect(begin.shapeB.id.index1).toBe(beginShapeB);
        expect(begin.contact.id.index1).toBe(beginContact);
        expect(hit.shapeA.id.index1).toBe(hitShapeA);
        expect(hit.shapeB.id.index1).toBe(hitShapeB);
        expect(hit.point).toEqual(hitPoint);
        expect(hit.normal).toEqual(hitNormal);
        expect(hit.approachSpeed).toBe(hitSpeed);
        expect(
            (beginShapeA === sphereShape.id.index1 && beginShapeB === groundShape.id.index1) ||
                (beginShapeA === groundShape.id.index1 && beginShapeB === sphereShape.id.index1),
        ).toBe(true);

        world.destroy();
    });
});

describe("hit events", () => {
    // Mirrors TestHitEvents: a sphere driven straight down at 30 m/s clears the 1 m/s threshold; the
    // captured hit carries a >threshold approach speed, a Y-aligned normal, and the sphere's material.
    test("a fast head-on impact reports a hit event with speed, normal, and material", () => {
        const world = new World({ hitEventThreshold: 1 });
        const ground = world.createBody({
            type: BodyType.Static,
            position: { x: 0, y: -0.5, z: 0 },
        });
        ground.createHull({}, makeBoxHull(10, 0.5, 10));

        const body = world.createBody({
            type: BodyType.Dynamic,
            gravityScale: 0,
            position: { x: 0, y: 2, z: 0 },
            linearVelocity: { x: 0, y: -30, z: 0 },
        });
        body.createSphere(
            { density: 1, enableHitEvents: true, baseMaterial: material({ userMaterialId: 7n }) },
            sphere(0.5),
        );

        let hitCount = 0;
        let speed = 0;
        let normal = { x: 0, y: 0, z: 0 };
        let point = { x: 0, y: 0, z: 0 };
        let matA = 0n;
        let matB = 0n;

        for (let i = 0; i < 30; ++i) {
            world.step(1 / 60);
            const ev = world.getContactEvents();
            if (ev.hitEvents.length > 0 && hitCount === 0) {
                const hit = ev.hitEvents[0];
                speed = hit.approachSpeed;
                normal = hit.normal;
                point = hit.point;
                matA = hit.userMaterialIdA;
                matB = hit.userMaterialIdB;
            }
            hitCount += ev.hitEvents.length;
        }

        expect(hitCount).toBeGreaterThanOrEqual(1);
        expect(speed).toBeGreaterThan(1);
        expect(Math.abs(normal.x)).toBeLessThan(0.01);
        expect(Math.abs(normal.z)).toBeLessThan(0.01);
        // The point (midCenter + mid-anchor) lands on the ground surface under the drop, near origin.
        expect(Math.abs(point.x)).toBeLessThan(0.5);
        expect(Math.abs(point.y)).toBeLessThan(0.5);
        expect(matA === 7n || matB === 7n).toBe(true);

        world.destroy();
    });

    test("a contact below the threshold reports no hit event", () => {
        // A high threshold no slow settle can clear — the same drop that fires under threshold 1.
        const world = new World({ hitEventThreshold: 100 });
        const ground = world.createBody({
            type: BodyType.Static,
            position: { x: 0, y: -0.5, z: 0 },
        });
        ground.createHull({}, makeBoxHull(10, 0.5, 10));
        const body = world.createBody({ type: BodyType.Dynamic, position: { x: 0, y: 1, z: 0 } });
        body.createSphere({ density: 1, enableHitEvents: true }, sphere(0.5));

        let hits = 0;
        for (let i = 0; i < 60; ++i) {
            world.step(1 / 60);
            hits += world.getContactEvents().hitEvents.length;
        }
        expect(hits).toBe(0);

        world.destroy();
    });

    // Mirrors TestCompoundHitEvents: the struck compound child's material must be attributed, not
    // materials[0]. A sphere is driven onto each of two children at opposite x with distinct ids.
    test("a compound hit attributes the struck child's material", () => {
        const HullA = 11n;
        const HullB = 22n;
        const Sphere = 99n;
        const CenterX = 3;

        for (const side of [0, 1]) {
            const expectedHull = side === 0 ? HullA : HullB;
            const spawnX = side === 0 ? -CenterX : CenterX;

            const world = new World({ hitEventThreshold: 1 });
            const compound = createCompound({
                hulls: [
                    {
                        hull: makeBoxHull(1, 1, 1),
                        transform: { p: { x: -CenterX, y: 0, z: 0 }, q: IDENTITY },
                        material: material({ userMaterialId: HullA }),
                    },
                    {
                        hull: makeBoxHull(1, 1, 1),
                        transform: { p: { x: CenterX, y: 0, z: 0 }, q: IDENTITY },
                        material: material({ userMaterialId: HullB }),
                    },
                ],
            });
            if (compound === null) {
                throw new Error("compound creation failed");
            }

            const holder = world.createBody({ type: BodyType.Static });
            holder.createCompound({}, compound);

            const body = world.createBody({
                type: BodyType.Dynamic,
                gravityScale: 0,
                position: { x: spawnX, y: 3, z: 0 },
                linearVelocity: { x: 0, y: -30, z: 0 },
            });
            body.createSphere(
                {
                    density: 1,
                    enableHitEvents: true,
                    baseMaterial: material({ userMaterialId: Sphere }),
                },
                sphere(0.5),
            );

            let hitCount = 0;
            let matA = 0n;
            let matB = 0n;
            for (let i = 0; i < 30; ++i) {
                world.step(1 / 60);
                const ev = world.getContactEvents();
                if (ev.hitEvents.length > 0 && hitCount === 0) {
                    matA = ev.hitEvents[0].userMaterialIdA;
                    matB = ev.hitEvents[0].userMaterialIdB;
                }
                hitCount += ev.hitEvents.length;
            }

            expect(hitCount).toBeGreaterThanOrEqual(1);
            expect(matA === Sphere || matB === Sphere).toBe(true);
            expect(matA === expectedHull || matB === expectedHull).toBe(true);

            world.destroy();
        }
    });
});

describe("body move events", () => {
    test("a falling body reports a move event matching its transform", () => {
        const world = new World();
        const body = world.createBody({ type: BodyType.Dynamic, position: { x: 0, y: 5, z: 0 } });
        body.createSphere({ density: 1 }, sphere(0.5));

        world.step(1 / 60);
        const events = world.getBodyEvents();
        expect(events.count).toBe(1);

        const xf = body.getTransform();
        const move = events.moveEvents[0];
        expect(move.body.id.index1).toBe(body.id.index1);
        expect(move.transform.p.x).toBe(xf.p.x);
        expect(move.transform.p.y).toBe(xf.p.y);
        expect(move.transform.p.z).toBe(xf.p.z);
        expect(move.fellAsleep).toBe(false);

        world.destroy();
    });

    // Mirrors TestContinuousMoveEvent: a fast body swept to its impact by CCD reports the rewound
    // pose, not the pre-CCD discrete advance — the move event and getTransform() must agree exactly.
    test("a CCD-rewound body reports the impact pose", () => {
        const world = new World({ enableContinuous: true });
        const wall = world.createBody({ type: BodyType.Static, position: { x: 0, y: 0, z: 0 } });
        wall.createHull({}, makeBoxHull(0.1, 5, 5));

        const ball = world.createBody({
            type: BodyType.Dynamic,
            gravityScale: 0,
            position: { x: 3, y: 0, z: 0 },
            linearVelocity: { x: -30, y: 0, z: 0 },
        });
        ball.createSphere({ density: 1 }, sphere(0.25));

        let haveMove = false;
        for (let step = 0; step < 30; ++step) {
            world.step(1 / 60);
            const xf = ball.getTransform();
            for (const move of world.getBodyEvents().moveEvents) {
                if (move.body.id.index1 !== ball.id.index1) {
                    continue;
                }
                haveMove = true;
                expect(move.transform.p.x).toBe(xf.p.x);
                expect(move.transform.p.y).toBe(xf.p.y);
                expect(move.transform.p.z).toBe(xf.p.z);
                expect(move.transform.q.s).toBe(xf.q.s);
            }
        }
        expect(haveMove).toBe(true);

        world.destroy();
    });

    test("a settling body reports fellAsleep on the step it sleeps", () => {
        const world = new World();
        const ground = world.createBody({
            type: BodyType.Static,
            position: { x: 0, y: -0.5, z: 0 },
        });
        ground.createHull({}, makeBoxHull(10, 0.5, 10));
        const body = world.createBody({ type: BodyType.Dynamic, position: { x: 0, y: 0.5, z: 0 } });
        body.createHull({ density: 1 }, makeBoxHull(0.5, 0.5, 0.5));

        let sawFellAsleep = false;
        for (let i = 0; i < 300 && sawFellAsleep === false; ++i) {
            world.step(1 / 60);
            for (const move of world.getBodyEvents().moveEvents) {
                if (move.body.id.index1 === body.id.index1 && move.fellAsleep) {
                    sawFellAsleep = true;
                }
            }
        }
        expect(sawFellAsleep).toBe(true);

        world.destroy();
    });
});

describe("joint events", () => {
    test("a zero-threshold joint reports every step it is awake; a default joint never does", () => {
        const world = new World();
        const anchor = world.createBody({ type: BodyType.Static, position: { x: 0, y: 5, z: 0 } });
        const hung = world.createBody({ type: BodyType.Dynamic, position: { x: 0, y: 4, z: 0 } });
        hung.createHull({ density: 1 }, makeBoxHull(0.5, 0.5, 0.5));

        // A zero threshold reports the joint on every awake step (b3SolveJointsTask).
        const flagged = world.createWeldJoint(anchor, hung, {
            forceThreshold: 0,
            userData: "load",
        });
        world.step(1 / 60);

        const events = world.getJointEvents();
        expect(events.length).toBe(1);
        expect(events[0].joint.id.index1).toBe(flagged.id.index1);
        expect(events[0].userData).toBe("load");

        world.destroy();
    });

    test("a default (no-threshold) joint reports nothing", () => {
        const world = new World();
        const anchor = world.createBody({ type: BodyType.Static, position: { x: 0, y: 5, z: 0 } });
        const hung = world.createBody({ type: BodyType.Dynamic, position: { x: 0, y: 4, z: 0 } });
        hung.createHull({ density: 1 }, makeBoxHull(0.5, 0.5, 0.5));
        world.createWeldJoint(anchor, hung);

        world.step(1 / 60);
        expect(world.getJointEvents().length).toBe(0);

        world.destroy();
    });
});

describe("out-param reads", () => {
    test("getPosition/getRotation/getTransform fill a provided target without allocating", () => {
        const world = new World();
        const body = world.createBody({ type: BodyType.Dynamic, position: { x: 1, y: 2, z: 3 } });
        body.createSphere({ density: 1 }, sphere(0.5));

        const pos = { x: 0, y: 0, z: 0 };
        expect(body.getPosition(pos)).toBe(pos);
        expect(pos).toEqual(body.getPosition());

        const rot = { v: { x: 0, y: 0, z: 0 }, s: 0 };
        expect(body.getRotation(rot)).toBe(rot);
        expect(rot).toEqual(body.getRotation());

        const xf = { p: { x: 0, y: 0, z: 0 }, q: { v: { x: 0, y: 0, z: 0 }, s: 0 } };
        expect(body.getTransform(xf)).toBe(xf);
        expect(xf).toEqual(body.getTransform());

        world.destroy();
    });
});

describe("hit event threshold accessor", () => {
    test("get/set round-trips through the world", () => {
        const world = new World({ hitEventThreshold: 5 });
        expect(world.getHitEventThreshold()).toBe(5);
        world.setHitEventThreshold(2.5);
        expect(world.getHitEventThreshold()).toBe(2.5);
        world.destroy();
    });
});
