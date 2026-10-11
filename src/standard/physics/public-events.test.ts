import { expect, test } from "bun:test";
import { Time, World } from "@dylanebert/shallot";
import {
    Body,
    BodyType,
    PhysicsPlugin,
    RevoluteJoint,
    Shape,
    ShapeKind,
} from "@dylanebert/shallot/physics";
import {
    castRay,
    castRayClosest,
    castShape,
    castShapeClosest,
    defaultDebugDraw,
    drawPhysics,
    overlapAABB,
    overlapShape,
    PhysicsEvents,
    PhysicsWorldDefinition,
    physicsWorld,
    StandardPhysicsPlugin,
    setTransform,
} from "@dylanebert/shallot/standard/physics";

async function physicsWorldForTest(): Promise<World> {
    const world = new World();
    for (const component of PhysicsPlugin.components!)
        world.registry.register(component, PhysicsPlugin.name);
    await StandardPhysicsPlugin.initialize!(world);
    await StandardPhysicsPlugin.warm!(world);
    const recovery = StandardPhysicsPlugin.recovery!;
    if (recovery === "stateless") throw new Error("physics must declare recovery");
    world.registerRecovery(StandardPhysicsPlugin.name, recovery(world));
    for (const system of StandardPhysicsPlugin.systems!)
        world.addSystem(system, StandardPhysicsPlugin.name);
    return world;
}

async function overlappingSensor() {
    const world = await physicsWorldForTest();
    world.resource(PhysicsWorldDefinition).gravity = { x: 0, y: 0, z: 0 };
    world.step(Time.FIXED_DT);
    const sensorBody = world.create();
    world.add(sensorBody, Body, { type: BodyType.Static });
    const sensorShape = world.create();
    world.add(sensorShape, Shape, {
        body: sensorBody,
        kind: ShapeKind.Sphere,
        isSensor: 1,
        enableSensorEvents: 1,
    });
    const visitorBody = world.create();
    world.add(visitorBody, Body, { type: BodyType.Dynamic });
    const visitorShape = world.create();
    world.add(visitorShape, Shape, {
        body: visitorBody,
        kind: ShapeKind.Sphere,
        enableSensorEvents: 1,
    });
    world.step(Time.FIXED_DT);
    return {
        world,
        sensorBody,
        sensorShape,
        visitorBody,
        visitorShape,
        events: world.resource(PhysicsEvents),
    };
}

test("a sensor Shape reports its entering body's entity", async () => {
    const world = await physicsWorldForTest();
    try {
        world.resource(PhysicsWorldDefinition).gravity = {
            x: 0,
            y: 0,
            z: 0,
        };
        world.step(Time.FIXED_DT);
        const sensorBody = world.create();
        world.add(sensorBody, Body, { type: BodyType.Static });
        const sensor = world.create();
        world.add(sensor, Shape, {
            body: sensorBody,
            kind: ShapeKind.Sphere,
            isSensor: 1,
            enableSensorEvents: 1,
        });
        const visitorBody = world.create();
        world.add(visitorBody, Body, { type: BodyType.Dynamic });
        const visitor = world.create();
        world.add(visitor, Shape, {
            body: visitorBody,
            kind: ShapeKind.Sphere,
            enableSensorEvents: 1,
        });
        const disabledSensorBody = world.create();
        world.add(disabledSensorBody, Body, {
            type: BodyType.Static,
            position: [10, 0, 0, 0],
        });
        const disabledSensor = world.create();
        world.add(disabledSensor, Shape, {
            body: disabledSensorBody,
            kind: ShapeKind.Sphere,
            isSensor: 1,
        });
        const disabledVisitorBody = world.create();
        world.add(disabledVisitorBody, Body, {
            type: BodyType.Dynamic,
            position: [10, 0, 0, 0],
        });
        const disabledVisitor = world.create();
        world.add(disabledVisitor, Shape, {
            body: disabledVisitorBody,
            kind: ShapeKind.Sphere,
            enableSensorEvents: 1,
        });

        world.step(Time.FIXED_DT);
        const events = world.resource(PhysicsEvents);
        expect(events.sensorBegin.count).toBe(1);
        const entering = events.sensorBegin.records[0];
        expect(entering).toEqual({
            sensorShape: world.ref(sensor),
            sensorBody: world.ref(sensorBody),
            visitorShape: world.ref(visitor),
            visitorBody: world.ref(visitorBody),
        });
        expect(world.resolve(events.sensorBegin.records[0].sensorShape)).toBe(sensor);
        expect(world.resolve(events.sensorBegin.records[0].visitorShape)).toBe(visitor);
        expect(events.sensorBegin.records[0].sensorShape).not.toBe(world.ref(disabledSensor));
        expect(events.sensorBegin.records[0].visitorShape).not.toBe(world.ref(disabledVisitor));

        setTransform(world, visitorBody, [4, 0, 0], [0, 0, 0, 1]);
        world.step(Time.FIXED_DT);
        expect(events.sensorEnd.count).toBe(1);
        const saved = world.snapshot();

        setTransform(world, visitorBody, [0, 0, 0], [0, 0, 0, 1]);
        world.step(Time.FIXED_DT);
        expect(events.sensorBegin.count).toBe(1);
        expect(events.sensorBegin.records[0].visitorBody).toBe(world.ref(visitorBody));

        world.restore(saved);
        setTransform(world, visitorBody, [0, 0, 0], [0, 0, 0, 1]);
        world.step(Time.FIXED_DT);
        expect(events.sensorBegin.count).toBe(1);
        expect(events.sensorBegin.records[0]).toBe(entering);
        expect(events.sensorBegin.records[0].visitorBody).toBe(world.ref(visitorBody));
    } finally {
        await StandardPhysicsPlugin.dispose!(world);
        world.dispose();
    }
});

test("replaying a sensor entry reproduces its generation-aware ECS identities", async () => {
    const { world, sensorBody, sensorShape, visitorBody, visitorShape, events } =
        await overlappingSensor();
    try {
        setTransform(world, visitorBody, [4, 0, 0], [0, 0, 0, 1]);
        world.step(Time.FIXED_DT);
        const saved = world.snapshot();
        setTransform(world, visitorBody, [0, 0, 0], [0, 0, 0, 1]);
        world.step(Time.FIXED_DT);
        const first = events.sensorBegin.records[0];
        const firstIdentity = [
            first.sensorShape,
            first.sensorBody,
            first.visitorShape,
            first.visitorBody,
        ];

        world.restore(saved);
        setTransform(world, visitorBody, [0, 0, 0], [0, 0, 0, 1]);
        world.step(Time.FIXED_DT);
        const replay = events.sensorBegin.records[0];
        expect(events.sensorBegin.count).toBe(1);
        expect([
            replay.sensorShape,
            replay.sensorBody,
            replay.visitorShape,
            replay.visitorBody,
        ]).toEqual(firstIdentity);
        expect([
            world.resolve(replay.sensorShape),
            world.resolve(replay.sensorBody),
            world.resolve(replay.visitorShape),
            world.resolve(replay.visitorBody),
        ]).toEqual([sensorShape, sensorBody, visitorShape, visitorBody]);
    } finally {
        await StandardPhysicsPlugin.dispose!(world);
        world.dispose();
    }
});

test("a retained sensor-begin record keeps its visitor EntityRef after that eid is reused", async () => {
    const { world, visitorShape, events } = await overlappingSensor();
    try {
        expect(events.sensorBegin.count).toBe(1);
        const oldVisitor = world.ref(visitorShape);
        world.destroy(visitorShape);
        const replacement = world.create();
        expect(replacement).toBe(visitorShape);
        expect(world.ref(replacement)).not.toBe(oldVisitor);
        expect(events.sensorBegin.records[0].visitorShape).toBe(oldVisitor);
        expect(world.resolve(events.sensorBegin.records[0].visitorShape)).toBe(0);
    } finally {
        await StandardPhysicsPlugin.dispose!(world);
        world.dispose();
    }
});

test("a sensor-end record keeps the visitor Body EntityRef when its eid is reused", async () => {
    const { world, visitorBody, events } = await overlappingSensor();
    try {
        const oldVisitorBody = world.ref(visitorBody);
        world.destroy(visitorBody);
        const replacement = world.create();
        expect(replacement).toBe(visitorBody);
        world.add(replacement, Body, { type: BodyType.Dynamic });
        world.step(Time.FIXED_DT);
        expect(events.sensorEnd.count).toBe(1);
        expect(events.sensorEnd.records[0].visitorBody).toBe(oldVisitorBody);
        expect(events.sensorEnd.records[0].visitorBody).not.toBe(world.ref(replacement));
        expect(world.resolve(events.sensorEnd.records[0].visitorBody)).toBe(0);
    } finally {
        await StandardPhysicsPlugin.dispose!(world);
        world.dispose();
    }
});

test("a sensor-end event retains a despawned Shape EntityRef", async () => {
    const { world, visitorShape, events } = await overlappingSensor();
    try {
        const oldVisitorShape = world.ref(visitorShape);
        world.destroy(visitorShape);
        world.step(Time.FIXED_DT);
        expect(events.sensorEnd.count).toBe(1);
        expect(events.sensorEnd.records[0].visitorShape).toBe(oldVisitorShape);
        expect(world.resolve(events.sensorEnd.records[0].visitorShape)).toBe(0);
    } finally {
        await StandardPhysicsPlugin.dispose!(world);
        world.dispose();
    }
});

test("recreating a sensor visitor Shape on its recycled eid preserves the old end event", async () => {
    const { world, visitorBody, visitorShape, events } = await overlappingSensor();
    try {
        const oldVisitorShape = world.ref(visitorShape);
        const visitorBodyRef = world.ref(visitorBody);
        world.destroy(visitorShape);
        const replacementShape = world.create();
        expect(replacementShape).toBe(visitorShape);
        world.add(replacementShape, Shape, {
            body: visitorBody,
            kind: ShapeKind.Sphere,
            enableSensorEvents: 1,
        });
        const replacementRef = world.ref(replacementShape);
        expect(replacementRef).not.toBe(oldVisitorShape);
        world.step(Time.FIXED_DT);
        expect(events.sensorEnd.count).toBe(1);
        expect(events.sensorEnd.records[0].visitorShape).toBe(oldVisitorShape);
        expect(events.sensorEnd.records[0].visitorBody).toBe(visitorBodyRef);
    } finally {
        await StandardPhysicsPlugin.dispose!(world);
        world.dispose();
    }
});

test("contact, hit and joint events retain ECS identity and Box3D payloads", async () => {
    const world = await physicsWorldForTest();
    try {
        world.resource(PhysicsWorldDefinition).gravity = { x: 0, y: -10, z: 0 };
        world.resource(PhysicsWorldDefinition).hitEventThreshold = 0;
        world.step(Time.FIXED_DT);

        const groundBody = world.create();
        world.add(groundBody, Body, { type: BodyType.Static, position: [0, -0.5, 0, 0] });
        const groundShape = world.create();
        world.add(groundShape, Shape, {
            body: groundBody,
            materialUserIdLow: 0x89abcdef,
            materialUserIdHigh: 0x01234567,
        });
        const fallingBody = world.create();
        world.add(fallingBody, Body, {
            type: BodyType.Dynamic,
            position: [0, 2, 0, 0],
            linearVelocity: [0, -8, 0, 0],
        });
        const fallingShape = world.create();
        world.add(fallingShape, Shape, {
            body: fallingBody,
            kind: ShapeKind.Sphere,
            sphere: [0, 0, 0, 0.5],
            density: 1,
            enableContactEvents: 1,
            enableHitEvents: 1,
            materialUserIdLow: 0x76543210,
            materialUserIdHigh: 0xfedcba98,
        });
        const quietGround = world.create();
        world.add(quietGround, Body, {
            type: BodyType.Static,
            position: [10, -0.5, 0, 0],
        });
        const quietGroundShape = world.create();
        world.add(quietGroundShape, Shape, { body: quietGround });
        const quietBody = world.create();
        world.add(quietBody, Body, {
            type: BodyType.Dynamic,
            position: [10, 2, 0, 0],
            linearVelocity: [0, -8, 0, 0],
        });
        const quietShape = world.create();
        world.add(quietShape, Shape, { body: quietBody, kind: ShapeKind.Sphere, density: 1 });

        const anchor = world.create();
        world.add(anchor, Body, { type: BodyType.Static, position: [20, 5, 0, 0] });
        const bob = world.create();
        world.add(bob, Body, {
            type: BodyType.Dynamic,
            position: [20, 5, 0, 0],
        });
        world.add(bob, Shape, { kind: ShapeKind.Sphere, density: 1 });
        const joint = world.create();
        world.add(joint, RevoluteJoint, {
            a: anchor,
            b: bob,
            forceThreshold: 0,
            torqueThreshold: 0,
        });

        const events = world.resource(PhysicsEvents);
        let sawContact = false;
        let sawHit = false;
        let sawJoint = false;
        for (let tick = 0; tick < 30 && !(sawContact && sawHit && sawJoint); tick++) {
            world.step(Time.FIXED_DT);
            if (events.contactBegin.count !== 0) {
                const event = events.contactBegin.records[0];
                expect([
                    [event.shapeA, event.bodyA],
                    [event.shapeB, event.bodyB],
                ]).toContainEqual([world.ref(groundShape), world.ref(groundBody)]);
                expect([
                    [event.shapeA, event.bodyA],
                    [event.shapeB, event.bodyB],
                ]).toContainEqual([world.ref(fallingShape), world.ref(fallingBody)]);
                expect(event.normalImpulse).toBeGreaterThan(0);
                for (let i = 0; i < events.contactBegin.count; i++) {
                    expect([
                        events.contactBegin.records[i].shapeA,
                        events.contactBegin.records[i].shapeB,
                    ]).not.toContain(world.ref(quietGroundShape));
                    expect([
                        events.contactBegin.records[i].shapeA,
                        events.contactBegin.records[i].shapeB,
                    ]).not.toContain(world.ref(quietShape));
                }
                sawContact = true;
            }
            if (events.contactHit.count !== 0) {
                const handleHits = physicsWorld(world)!.getContactEvents().hitEvents;
                expect(handleHits.length).toBe(events.contactHit.count);
                for (let i = 0; i < events.contactHit.count; i++) {
                    const event = events.contactHit.records[i];
                    const handleHit = handleHits[i];
                    expect(event.point).toEqual(handleHit.point);
                    expect(event.normal).toEqual(handleHit.normal);
                    expect(event.approachSpeed).toBe(handleHit.approachSpeed);
                    expect(event.userMaterialIdA).toBe(handleHit.userMaterialIdA);
                    expect(event.userMaterialIdB).toBe(handleHit.userMaterialIdB);
                }
                const event = events.contactHit.records[0];
                expect([
                    [event.shapeA, event.bodyA],
                    [event.shapeB, event.bodyB],
                ]).toContainEqual([world.ref(groundShape), world.ref(groundBody)]);
                expect([
                    [event.shapeA, event.bodyA],
                    [event.shapeB, event.bodyB],
                ]).toContainEqual([world.ref(fallingShape), world.ref(fallingBody)]);
                expect(event.approachSpeed).toBeGreaterThan(0);
                expect(Number.isFinite(event.point.x + event.normal.x)).toBe(true);
                expect([event.userMaterialIdA, event.userMaterialIdB]).toContain(
                    0xfedcba9876543210n,
                );
                for (let i = 0; i < events.contactHit.count; i++) {
                    expect([
                        events.contactHit.records[i].shapeA,
                        events.contactHit.records[i].shapeB,
                    ]).not.toContain(world.ref(quietGroundShape));
                    expect([
                        events.contactHit.records[i].shapeA,
                        events.contactHit.records[i].shapeB,
                    ]).not.toContain(world.ref(quietShape));
                }
                sawHit = true;
            }
            if (events.joints.count !== 0) {
                expect(events.joints.records[0].joint).toBe(world.ref(joint));
                sawJoint = true;
            }
        }
        expect([sawContact, sawHit, sawJoint]).toEqual([true, true, true]);

        setTransform(world, fallingBody, [0, 4, 0], [0, 0, 0, 1]);
        world.step(Time.FIXED_DT);
        expect(events.contactEnd.count).toBeGreaterThan(0);
        const end = events.contactEnd.records[0];
        expect([
            [end.shapeA, end.bodyA],
            [end.shapeB, end.bodyB],
        ]).toContainEqual([world.ref(fallingShape), world.ref(fallingBody)]);
    } finally {
        await StandardPhysicsPlugin.dispose!(world);
        world.dispose();
    }
});

test("ray queries skip a recycled Shape eid before the next physics sync", async () => {
    const world = await physicsWorldForTest();
    try {
        world.resource(PhysicsWorldDefinition).gravity = { x: 0, y: 0, z: 0 };
        world.step(Time.FIXED_DT);
        const body = world.create();
        world.add(body, Body, { type: BodyType.Static });
        const shape = world.create();
        world.add(shape, Shape, {
            body,
            kind: ShapeKind.Sphere,
            sphere: [0, 0, 0, 0.5],
        });
        world.step(Time.FIXED_DT);

        world.destroy(shape);
        const unrelated = world.create();
        expect(unrelated).toBe(shape);
        let callbacks = 0;
        castRay(world, { x: -3, y: 0, z: 0 }, { x: 6, y: 0, z: 0 }, () => {
            callbacks++;
            return 1;
        });
        const hit = {
            shape: 0,
            body: 0,
            point: { x: 0, y: 0, z: 0 },
            normal: { x: 0, y: 0, z: 0 },
            fraction: 0,
            userMaterialId: 0n,
            triangleIndex: 0,
            childIndex: 0,
        };
        expect(callbacks).toBe(0);
        expect(castRayClosest(world, { x: -3, y: 0, z: 0 }, { x: 6, y: 0, z: 0 }, hit)).toBe(false);
        const proxy = { points: [{ x: 0, y: 0, z: 0 }], count: 1, radius: 0.25 };
        let shapeCastCallbacks = 0;
        castShape(world, { x: -3, y: 0, z: 0 }, proxy, { x: 6, y: 0, z: 0 }, () => {
            shapeCastCallbacks++;
            return 1;
        });
        let overlapCallbacks = 0;
        overlapShape(world, { x: 0, y: 0, z: 0 }, proxy, () => {
            overlapCallbacks++;
            return true;
        });
        overlapAABB(
            world,
            {
                lowerBound: { x: -1, y: -1, z: -1 },
                upperBound: { x: 1, y: 1, z: 1 },
            },
            () => {
                overlapCallbacks++;
                return true;
            },
        );
        expect(shapeCastCallbacks).toBe(0);
        expect(overlapCallbacks).toBe(0);
        expect(world.has(unrelated, Shape)).toBe(false);
    } finally {
        await StandardPhysicsPlugin.dispose!(world);
        world.dispose();
    }
});

test("ray, shape-cast and overlap queries return the authored Shape and Body entities", async () => {
    const world = await physicsWorldForTest();
    try {
        world.resource(PhysicsWorldDefinition).gravity = { x: 0, y: 0, z: 0 };
        world.step(Time.FIXED_DT);
        const body = world.create();
        world.add(body, Body, { type: BodyType.Static });
        const shape = world.create();
        world.add(shape, Shape, {
            body,
            kind: ShapeKind.Sphere,
            sphere: [0, 0, 0, 0.5],
        });
        world.step(Time.FIXED_DT);

        const hit = {
            shape: 0,
            body: 0,
            point: { x: 0, y: 0, z: 0 },
            normal: { x: 0, y: 0, z: 0 },
            fraction: 0,
            userMaterialId: 0n,
            triangleIndex: 0,
            childIndex: 0,
        };
        expect(castRayClosest(world, { x: -3, y: 0, z: 0 }, { x: 6, y: 0, z: 0 }, hit)).toBe(true);
        expect([hit.shape, hit.body]).toEqual([shape, body]);
        expect(
            castRayClosest(world, { x: -3, y: 0, z: 0 }, { x: 6, y: 0, z: 0 }, hit, {
                categoryBits: 0n,
                maskBits: 0n,
            }),
        ).toBe(false);
        expect([hit.shape, hit.body]).toEqual([0, 0]);

        const proxy = { points: [{ x: 0, y: 0, z: 0 }], count: 1, radius: 0.25 };
        expect(
            castShapeClosest(world, { x: -3, y: 0, z: 0 }, proxy, { x: 6, y: 0, z: 0 }, hit),
        ).toBe(true);
        expect([hit.shape, hit.body]).toEqual([shape, body]);

        let shapeCastCount = 0;
        castShape(world, { x: -3, y: 0, z: 0 }, proxy, { x: 6, y: 0, z: 0 }, (candidate) => {
            expect([candidate.shape, candidate.body]).toEqual([shape, body]);
            shapeCastCount++;
            return candidate.fraction;
        });
        expect(shapeCastCount).toBe(1);

        const overlaps: number[][] = [];
        overlapShape(world, { x: 0, y: 0, z: 0 }, proxy, (shapeEid, bodyEid) => {
            overlaps.push([shapeEid, bodyEid]);
            return true;
        });
        expect(overlaps).toEqual([[shape, body]]);

        const aabbOverlaps: number[][] = [];
        overlapAABB(
            world,
            {
                lowerBound: { x: -0.75, y: -0.75, z: -0.75 },
                upperBound: { x: 0.75, y: 0.75, z: 0.75 },
            },
            (shapeEid, bodyEid) => {
                aabbOverlaps.push([shapeEid, bodyEid]);
                return true;
            },
        );
        expect(aabbOverlaps).toEqual([[shape, body]]);

        let castCount = 0;
        castRay(world, { x: -3, y: 0, z: 0 }, { x: 6, y: 0, z: 0 }, (candidate) => {
            expect([candidate.shape, candidate.body]).toEqual([shape, body]);
            castCount++;
            return candidate.fraction;
        });
        expect(castCount).toBe(1);

        const debug = defaultDebugDraw();
        debug.drawShapes = true;
        let draws = 0;
        debug.drawSolidSphere = () => draws++;
        drawPhysics(world, debug);
        expect(draws).toBe(1);
    } finally {
        await StandardPhysicsPlugin.dispose!(world);
        world.dispose();
    }
});
