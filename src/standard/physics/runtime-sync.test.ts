import { afterAll, beforeAll, describe, expect, spyOn, test } from "bun:test";
import { Time, World } from "@dylanebert/shallot";
import {
    Body,
    BodyMotionLock,
    BodyType,
    Compounds,
    HeightFields,
    Hulls,
    PhysicsMeshes,
    PhysicsPlugin,
    PrismaticJoint,
    RevoluteJoint,
    Shape,
    ShapeKind,
    ShapeMaterials,
    WheelJoint,
} from "@dylanebert/shallot/physics";
import {
    StandardPhysicsPlugin,
    setAwake,
    setLinearVelocity,
    setTransform,
} from "@dylanebert/shallot/standard/physics";
import { GlobalTransform, Transform, TransformPlugin } from "@dylanebert/shallot/transform";
import { expectWarning } from "../../../scripts/test-console";
import { makeShapeId } from "./api/config";
import { BodyField, bodyField } from "./kernel/bodyrecords";
import { RJ_MOTOR_SPEED } from "./kernel/columns";
import { readJointFloat, writeJointFloat } from "./kernel/jointcolumns";
import { JointField, jointField } from "./kernel/jointrecords";
import { kernel } from "./kernel/kernel";
import { ShapeField, ShapeFlags, shapeField, shapeFlag } from "./kernel/shaperecords";
import { physicsBodyIndex, physicsState } from "./runtime";
import {
    getShapeDensity,
    getShapeFilter,
    getShapeGeometryReference,
    getShapeGeometryScale,
    getShapeMaterials,
    getShapeType,
    isSensorShape,
    readShapeCapsule,
    readShapeSphere,
    setShapeFriction,
    shapeIsValid,
} from "./shapes/shape";
import { jointIds } from "./solver/joint.fixture";
import {
    BodyProperty,
    bodyGetLinearVelocity,
    bodyGetProperty,
    bodyGetTransform,
    bodyGetType,
    bodyIsAwake,
} from "./world/body";
import { worldCounters } from "./world/world";

async function createPhysicsWorld(): Promise<World> {
    const world = new World();
    for (const component of [...TransformPlugin.components!, ...PhysicsPlugin.components!])
        world.registry.register(component, PhysicsPlugin.name);
    await StandardPhysicsPlugin.initialize!(world);
    await StandardPhysicsPlugin.warm!(world);
    const recovery = StandardPhysicsPlugin.recovery!;
    if (recovery === "stateless") throw new Error("StandardPhysicsPlugin must recover");
    world.registerRecovery(StandardPhysicsPlugin.name, recovery(world));
    for (const system of StandardPhysicsPlugin.systems!)
        world.addSystem(system, StandardPhysicsPlugin.name);
    return world;
}

async function withPhysics(run: (world: World) => void | Promise<void>): Promise<void> {
    const world = await createPhysicsWorld();
    try {
        await run(world);
    } finally {
        await StandardPhysicsPlugin.dispose!(world);
        world.dispose();
    }
}

function stateFor(world: World) {
    const state = physicsState(world);
    if (!state) throw new Error("physics state is unavailable");
    return state;
}

function bodyIndex(world: World, eid: number): number {
    const index = physicsBodyIndex(world, eid);
    if (index < 0) throw new Error(`physics body for entity ${eid} is unavailable`);
    return index;
}

function shapeId(world: World, index = 0) {
    return makeShapeId(stateFor(world), index);
}

test("a Body type edit after spawn reaches the solver", async () => {
    await withPhysics((world) => {
        const eid = world.create();
        world.add(eid, Body, { type: BodyType.Dynamic });
        world.tick();

        world.storage(Body).type.set(eid, BodyType.Static);
        world.tick();

        expect(bodyGetType(stateFor(world), bodyIndex(world, eid))).toBe(BodyType.Static);
    });
});

test("a post-spawn awake write survives the spawn-time Body mark", async () => {
    await withPhysics((world) => {
        const eid = world.create();
        world.add(eid, Body, {
            type: BodyType.Dynamic,
            linearVelocity: [0, 0, 0, 0],
            angularVelocity: [0, 0, 0, 0],
            linearDamping: 0,
            angularDamping: 0,
            gravityScale: 0,
            sleepThreshold: 0.05,
            motionLocks: 0,
            enableSleep: 1,
            isAwake: 1,
            isBullet: 0,
            isEnabled: 1,
            allowFastRotation: 0,
            enableContactRecycling: 1,
        });
        world.add(eid, Shape);
        world.step(Time.FIXED_DT);

        const solver = bodyIndex(world, eid);
        expect(bodyField(stateFor(world), solver, BodyField.mass)).toBeGreaterThan(0);
        expect(bodyIsAwake(stateFor(world), solver)).toBe(true);
        setAwake(world, eid, false);
        expect(bodyIsAwake(stateFor(world), solver)).toBe(false);

        world.step(Time.FIXED_DT);
        expect(bodyIsAwake(stateFor(world), solver)).toBe(false);
    });
});

test("Shape material, filter and event writes reach Box3D setters only when values differ", async () => {
    await withPhysics((world) => {
        const eid = world.create();
        world.add(eid, Body, { type: BodyType.Dynamic });
        world.add(eid, Shape);
        world.tick();
        const shape = shapeId(world);
        const k = kernel(world);
        const setters = [
            spyOn(k, "shapeSetDensity"),
            spyOn(k, "shapeSetFriction"),
            spyOn(k, "shapeSetRestitution"),
            spyOn(k, "shapeSetSurfaceMaterial"),
        ] as const;
        try {
            const authored = world.storage(Shape);
            authored.density.set(eid, 1250);
            authored.friction.set(eid, 0.25);
            authored.restitution.set(eid, 0.4);
            authored.rollingResistance.set(eid, 0.2);
            authored.tangentVelocity.set(eid, 1, 2, 3, 0);
            authored.userMaterialIdLow.set(eid, 0x12345678);
            authored.userMaterialIdHigh.set(eid, 0x9abcdef0);
            authored.customColor.set(eid, 0xff00ff);
            authored.filterCategoryLow.set(eid, 0x1234);
            authored.filterMaskHigh.set(eid, 0x5678);
            authored.filterGroupIndex.set(eid, -4);
            authored.enableSensorEvents.set(eid, 1);
            authored.enableContactEvents.set(eid, 1);
            authored.enableHitEvents.set(eid, 1);
            authored.enablePreSolveEvents.set(eid, 1);
            world.tick();

            const shapeSlot = shape.index1 - 1;
            const state = stateFor(world);
            expect(getShapeDensity(state, shapeSlot)).toBe(1250);
            const material = getShapeMaterials(state, shapeSlot)[0]!;
            expect(material.friction).toBe(0.25);
            expect(material.restitution).toBeCloseTo(0.4);
            expect(material.rollingResistance).toBeCloseTo(0.2);
            expect(material.tangentVelocity).toEqual([1, 2, 3]);
            expect(material.userMaterialId).toBe(0x9abcdef012345678n);
            expect(material.customColor).toBe(0xff00ff);
            expect(getShapeFilter(state, shapeSlot)).toEqual({
                categoryBits: 0xffffffff00001234n,
                maskBits: 0x00005678ffffffffn,
                groupIndex: -4,
            });
            expect(shapeFlag(state, shapeSlot, ShapeFlags.enableSensorEvents)).toBe(true);
            expect(shapeFlag(state, shapeSlot, ShapeFlags.enableContactEvents)).toBe(true);
            expect(shapeFlag(state, shapeSlot, ShapeFlags.enableHitEvents)).toBe(true);
            expect(shapeFlag(state, shapeSlot, ShapeFlags.enablePreSolveEvents)).toBe(true);
            expect(setters.map((setter) => setter.mock.calls.length)).toEqual([1, 1, 1, 1]);

            authored.friction.set(eid, 0.25);
            world.tick();
            expect(setters[1]).toHaveBeenCalledTimes(1);
            expect(worldCounters(stateFor(world)).shapeCount).toBe(1);
        } finally {
            for (const setter of setters) setter.mockRestore();
        }
    });
});

test("Sphere and capsule Shape geometry writes use Box3D setters and getters", async () => {
    await withPhysics((world) => {
        const sphereBody = world.create();
        world.add(sphereBody, Body, { type: BodyType.Dynamic });
        world.add(sphereBody, Shape, { kind: ShapeKind.Sphere });
        const capsuleBody = world.create();
        world.add(capsuleBody, Body, { type: BodyType.Dynamic });
        world.add(capsuleBody, Shape, { kind: ShapeKind.Capsule });
        world.tick();

        const sphere = shapeId(world, 0);
        const capsule = shapeId(world, 1);
        const k = kernel(world);
        const setSphere = spyOn(k, "shapeSetSphere");
        const setCapsule = spyOn(k, "shapeSetCapsule");
        try {
            world.storage(Shape).sphereCenter.set(sphereBody, 0.25, -0.5, 0.75, 0);
            world.storage(Shape).radius.set(sphereBody, 0.8);
            world.storage(Shape).capsuleCenter1.set(capsuleBody, -1, 0, 0, 0);
            world.storage(Shape).capsuleCenter2.set(capsuleBody, 1, 0, 0, 0);
            world.storage(Shape).radius.set(capsuleBody, 0.4);
            world.tick();

            const state = stateFor(world);
            expect(shapeIsValid(state, sphere)).toBe(true);
            expect(getShapeType(state, sphere.index1 - 1)).toBe(ShapeKind.Sphere);
            expect(readShapeSphere(state, sphere.index1 - 1).center).toEqual({
                x: 0.25,
                y: -0.5,
                z: 0.75,
            });
            expect(readShapeSphere(state, sphere.index1 - 1).radius).toBeCloseTo(0.8);
            expect(shapeIsValid(state, capsule)).toBe(true);
            expect(getShapeType(state, capsule.index1 - 1)).toBe(ShapeKind.Capsule);
            expect(readShapeCapsule(state, capsule.index1 - 1).center1).toEqual({
                x: -1,
                y: 0,
                z: 0,
            });
            expect(readShapeCapsule(state, capsule.index1 - 1).center2).toEqual({
                x: 1,
                y: 0,
                z: 0,
            });
            expect(readShapeCapsule(state, capsule.index1 - 1).radius).toBeCloseTo(0.4);
            expect(setSphere).toHaveBeenCalledTimes(1);
            expect(setCapsule).toHaveBeenCalledTimes(1);
            expect(worldCounters(stateFor(world)).shapeCount).toBe(2);
        } finally {
            setSphere.mockRestore();
            setCapsule.mockRestore();
        }
    });
});

test("a Shape radius write alone reaches the sphere and capsule setters", async () => {
    await withPhysics((world) => {
        const sphereBody = world.create();
        world.add(sphereBody, Body, { type: BodyType.Dynamic });
        world.add(sphereBody, Shape, { kind: ShapeKind.Sphere });
        const capsuleBody = world.create();
        world.add(capsuleBody, Body, { type: BodyType.Dynamic });
        world.add(capsuleBody, Shape, { kind: ShapeKind.Capsule });
        world.tick();
        // Clearing the spawn marks leaves the radius mark as the next sync's only one.
        world.clearChanges();

        world.storage(Shape).radius.set(sphereBody, 0.8);
        world.storage(Shape).radius.set(capsuleBody, 0.4);
        world.tick();

        const state = stateFor(world);
        expect(readShapeSphere(state, shapeId(world, 0).index1 - 1).radius).toBeCloseTo(0.8);
        expect(readShapeCapsule(state, shapeId(world, 1).index1 - 1).radius).toBeCloseTo(0.4);
    });
});

test("a hull scale edit reaches b3Shape_SetHull without replacing the ECS-bound shape", async () => {
    await withPhysics((world) => {
        const eid = world.create();
        world.add(eid, Body, { type: BodyType.Dynamic });
        world.add(eid, Shape);
        world.tick();
        const shape = shapeId(world);
        const previousGeometry = getShapeGeometryReference(stateFor(world), shape.index1 - 1);
        const setHull = spyOn(kernel(world), "shapeSetHull");
        try {
            world.storage(Shape).scale.x.set(eid, 2);
            world.tick();

            expect(shapeIsValid(stateFor(world), shape)).toBe(true);
            expect(getShapeType(stateFor(world), shape.index1 - 1)).toBe(ShapeKind.Hull);
            expect(getShapeGeometryReference(stateFor(world), shape.index1 - 1)).not.toBe(
                previousGeometry,
            );
            expect(setHull).toHaveBeenCalledTimes(1);
            expect(worldCounters(stateFor(world)).shapeCount).toBe(1);
        } finally {
            setHull.mockRestore();
        }
    });
});

test("mesh geometry and per-triangle material edits use Box3D setters and getters", async () => {
    await withPhysics((world) => {
        const meshes = world.resource(PhysicsMeshes);
        const firstId = meshes.add({
            name: "shape-sync-first-mesh",
            vertices: new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0]),
            indices: new Uint32Array([0, 1, 2]),
        });
        const secondId = meshes.add({
            name: "shape-sync-second-mesh",
            vertices: new Float32Array([0, 0, 0, 2, 0, 0, 0, 1, 0]),
            indices: new Uint32Array([0, 1, 2]),
        });
        const material = world.resource(ShapeMaterials).add({
            name: "shape-sync-material-set",
            materials: [
                {
                    friction: 0.15,
                    restitution: 0.35,
                    rollingResistance: 0.1,
                    tangentVelocity: [1, 0, -1],
                    userMaterialId: 23n,
                    customColor: 0x336699,
                },
            ],
        });
        const eid = world.create();
        world.add(eid, Body);
        world.add(eid, Shape, { kind: ShapeKind.Mesh, geometry: firstId });
        world.tick();

        const shape = shapeId(world);
        const previousGeometry = getShapeGeometryReference(stateFor(world), shape.index1 - 1);
        const k = kernel(world);
        const setMesh = spyOn(k, "shapeSetMesh");
        const setMaterial = spyOn(k, "shapeSetMeshMaterial");
        try {
            world.storage(Shape).geometry.set(eid, secondId);
            world.storage(Shape).scale.set(eid, 2, 0.5, 1, 0);
            world.storage(Shape).materialSet.set(eid, material);
            world.tick();

            expect(shapeIsValid(stateFor(world), shape)).toBe(true);
            expect(getShapeType(stateFor(world), shape.index1 - 1)).toBe(ShapeKind.Mesh);
            expect(getShapeGeometryReference(stateFor(world), shape.index1 - 1)).not.toBe(
                previousGeometry,
            );
            const state = stateFor(world);
            const shapeSlot = shape.index1 - 1;
            expect(getShapeGeometryScale(state, shapeSlot)).toEqual({ x: 2, y: 0.5, z: 1 });
            expect(getShapeMaterials(state, shapeSlot)).toEqual([
                {
                    friction: Math.fround(0.15),
                    restitution: Math.fround(0.35),
                    rollingResistance: Math.fround(0.1),
                    tangentVelocity: [1, 0, -1],
                    userMaterialId: 23n,
                    customColor: 0x336699,
                },
            ]);
            expect(setMesh).toHaveBeenCalledTimes(1);
            expect(setMaterial).toHaveBeenCalledTimes(1);
            world.storage(Shape).materialSet.set(eid, material);
            world.tick();
            expect(setMaterial).toHaveBeenCalledTimes(1);
            expect(worldCounters(stateFor(world)).shapeCount).toBe(1);
        } finally {
            setMesh.mockRestore();
            setMaterial.mockRestore();
        }
    });
});

test("two material handles, including zero and a recycled generation, retain distinct materials through spawn, sync and restore", async () => {
    await withPhysics((world) => {
        if (typeof PhysicsPlugin.recovery !== "function")
            throw new Error("missing physics recovery");
        world.registerRecovery(PhysicsPlugin.name, PhysicsPlugin.recovery(world));
        const assets = world.resource(ShapeMaterials);
        const material = (friction: number, userMaterialId: bigint) => ({
            friction,
            userMaterialId,
            restitution: 0,
            rollingResistance: 0,
            tangentVelocity: [0, 0, 0] as [number, number, number],
            customColor: 0,
        });
        const first = assets.add({ name: "same", materials: [material(0.25, 11n)] });
        const removed = assets.add({ name: "same", materials: [] });
        assets.remove(removed);
        const second = assets.add({ name: "same", materials: [material(0.75, 22n)] });
        expect(first).toBe(0);
        expect(second).toBeGreaterThan(removed);
        const geometry = world.resource(PhysicsMeshes).add({
            name: "triangle",
            vertices: new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0]),
            indices: new Uint32Array([0, 1, 2]),
        });
        const entities = [first, second].map((materialSet) => {
            const eid = world.create();
            world.add(eid, Body);
            world.add(eid, Shape, { kind: ShapeKind.Mesh, geometry, materialSet });
            return eid;
        });
        const check = (ids: bigint[]) => {
            for (let i = 0; i < ids.length; i++) {
                const materials = getShapeMaterials(stateFor(world), shapeId(world, i).index1 - 1);
                expect(materials[0].userMaterialId).toBe(ids[i]);
                expect(materials[0].friction).toBe(ids[i] === 11n ? 0.25 : 0.75);
            }
        };
        world.tick();
        check([11n, 22n]);
        const saved = world.snapshot();
        world.storage(Shape).materialSet.set(entities[0], second);
        world.storage(Shape).materialSet.set(entities[1], first);
        world.tick();
        check([22n, 11n]);
        assets.clear();
        assets.add({ name: "same", materials: [material(0.5, 33n)] });
        world.restore(saved);
        expect(world.storage(Shape).materialSet.get(entities[0])).toBe(first);
        expect(world.storage(Shape).materialSet.get(entities[1])).toBe(second);
        expect(assets.has(removed)).toBe(false);
        world.tick();
        check([11n, 22n]);
        world.storage(Shape).materialSet.set(entities[0], -1);
        world.tick();
        expect(
            getShapeMaterials(stateFor(world), shapeId(world).index1 - 1)[0].userMaterialId,
        ).toBe(0n);
    });
});

test("changing Shape.body reattaches the collider to its new Body entity", async () => {
    await withPhysics((world) => {
        const firstBody = world.create();
        world.add(firstBody, Transform, { translation: [1, 0, 0] });
        world.add(firstBody, Body);
        const secondBody = world.create();
        world.add(secondBody, Transform, { translation: [2, 0, 0] });
        world.add(secondBody, Body);
        const shapeEntity = world.create();
        world.add(shapeEntity, Shape, { body: firstBody });
        world.tick();

        const previous = shapeId(world);
        const state = stateFor(world);
        const body = shapeField(state, previous.index1 - 1, ShapeField.bodyId);
        const pose = { p: { x: 0, y: 0, z: 0 }, q: { v: { x: 0, y: 0, z: 0 }, s: 1 } };
        bodyGetTransform(state, body, pose);
        expect(pose.p.x).toBe(1);
        world.storage(Shape).body.set(shapeEntity, secondBody);
        world.tick();

        expect(shapeIsValid(stateFor(world), previous)).toBe(false);
        const nextState = stateFor(world);
        const shape = shapeId(world);
        const nextBody = shapeField(nextState, shape.index1 - 1, ShapeField.bodyId);
        const nextPose = { p: { x: 0, y: 0, z: 0 }, q: { v: { x: 0, y: 0, z: 0 }, s: 1 } };
        bodyGetTransform(nextState, nextBody, nextPose);
        expect(nextPose.p.x).toBe(2);
        expect(worldCounters(stateFor(world)).shapeCount).toBe(1);
    });
});

test("an external Shape waits while its Body is removed and attaches when it returns", async () => {
    await withPhysics((world) => {
        const body = world.create();
        world.add(body, Body);
        const collider = world.create();
        world.add(collider, Shape, { body });
        world.tick();
        const previous = shapeId(world);
        expect(shapeIsValid(stateFor(world), previous)).toBe(true);

        world.remove(body, Body);
        world.tick();
        expect(shapeIsValid(stateFor(world), previous)).toBe(false);
        expect(worldCounters(stateFor(world)).shapeCount).toBe(0);

        world.add(body, Body);
        world.tick();
        expect(shapeIsValid(stateFor(world), shapeId(world))).toBe(true);
        expect(worldCounters(stateFor(world)).shapeCount).toBe(1);
    });
});

test("changing updateBodyMass does not recreate a Shape and later density writes honor it", async () => {
    await withPhysics((world) => {
        const eid = world.create();
        world.add(eid, Body, { type: BodyType.Dynamic });
        world.add(eid, Shape, { kind: ShapeKind.Sphere, density: 2 });
        world.tick();

        const shape = shapeId(world);
        const body = bodyIndex(world, eid);
        const mass = bodyField(stateFor(world), body, BodyField.mass);
        expect(mass).toBeGreaterThan(0);

        world.storage(Shape).updateBodyMass.set(eid, 0);
        world.tick();
        expect(shapeIsValid(stateFor(world), shape)).toBe(true);
        expect(bodyField(stateFor(world), body, BodyField.mass)).toBe(mass);

        world.storage(Shape).density.set(eid, 4);
        world.tick();
        expect(getShapeDensity(stateFor(world), shape.index1 - 1)).toBe(4);
        expect(bodyField(stateFor(world), body, BodyField.mass)).toBe(mass);
    });
});

test("recreating a Shape with updateBodyMass disabled preserves body mass", async () => {
    await withPhysics((world) => {
        const bodyEid = world.create();
        world.add(bodyEid, Body, { type: BodyType.Dynamic });
        world.add(bodyEid, Shape, {
            kind: ShapeKind.Sphere,
            density: 2,
            updateBodyMass: 0,
        });
        const sensorEid = world.create();
        world.add(sensorEid, Shape, {
            body: bodyEid,
            kind: ShapeKind.Sphere,
            density: 4,
            updateBodyMass: 0,
        });
        world.tick();

        const body = bodyIndex(world, bodyEid);
        const mass = bodyField(stateFor(world), body, BodyField.mass);
        const previous = shapeId(world, 1);
        world.storage(Shape).isSensor.set(sensorEid, 1);
        world.tick();

        expect(shapeIsValid(stateFor(world), previous)).toBe(false);
        expect(isSensorShape(stateFor(world), shapeId(world, 1).index1 - 1)).toBe(true);
        expect(bodyField(stateFor(world), body, BodyField.mass)).toBe(mass);
    });
});

test("a marked friction write compares against the live Box3D value", async () => {
    await withPhysics((world) => {
        const eid = world.create();
        world.add(eid, Body, { type: BodyType.Dynamic });
        world.add(eid, Shape, { kind: ShapeKind.Sphere });
        world.tick();

        const shape = shapeId(world);
        const authored = world.storage(Shape).friction.get(eid);
        setShapeFriction(stateFor(world), shape.index1 - 1, 0.125);
        world.storage(Shape).friction.set(eid, authored);
        world.tick();

        expect(getShapeMaterials(stateFor(world), shape.index1 - 1)[0]!.friction).toBe(authored);
    });
});

test("a sensor flag change recreates the Box3D shape at the next fixed sync", async () => {
    await withPhysics((world) => {
        const eid = world.create();
        world.add(eid, Body, { type: BodyType.Dynamic });
        world.add(eid, Shape);
        world.tick();
        const previous = shapeId(world);
        expect(isSensorShape(stateFor(world), previous.index1 - 1)).toBe(false);

        world.storage(Shape).isSensor.set(eid, 1);
        world.tick();

        expect(shapeIsValid(stateFor(world), previous)).toBe(false);
        const current = shapeId(world);
        expect(shapeIsValid(stateFor(world), current)).toBe(true);
        expect(isSensorShape(stateFor(world), current.index1 - 1)).toBe(true);
        expect(worldCounters(stateFor(world)).shapeCount).toBe(1);
    });
});

test("Body definition writes reach Box3D through its setters after spawn", async () => {
    await withPhysics((world) => {
        const eid = world.create();
        world.add(eid, Body, {
            type: BodyType.Dynamic,
            linearVelocity: [0, 0, 0, 0],
            angularVelocity: [0, 0, 0, 0],
            linearDamping: 0,
            angularDamping: 0,
            gravityScale: 1,
            sleepThreshold: 0.05,
            motionLocks: 0,
            enableSleep: 1,
            isAwake: 1,
            isBullet: 0,
            isEnabled: 1,
            allowFastRotation: 0,
            enableContactRecycling: 1,
        });
        world.add(eid, Shape);
        world.tick();
        const authored = world.storage(Body);
        const solver = bodyIndex(world, eid);

        authored.type.set(eid, BodyType.Kinematic);
        authored.linearDamping.set(eid, 0.25);
        authored.angularDamping.set(eid, 0.5);
        authored.gravityScale.set(eid, 0.75);
        authored.sleepThreshold.set(eid, 0.2);
        authored.motionLocks.set(eid, BodyMotionLock.linearX | BodyMotionLock.angularZ);
        authored.enableSleep.set(eid, 0);
        authored.isBullet.set(eid, 1);
        authored.allowFastRotation.set(eid, 1);
        authored.enableContactRecycling.set(eid, 0);
        world.tick();

        expect(bodyGetType(stateFor(world), solver)).toBe(BodyType.Kinematic);
        expect(bodyGetProperty(stateFor(world), solver, BodyProperty.linearDamping)).toBe(0.25);
        expect(bodyGetProperty(stateFor(world), solver, BodyProperty.angularDamping)).toBe(0.5);
        expect(bodyGetProperty(stateFor(world), solver, BodyProperty.gravityScale)).toBe(0.75);
        expect(bodyGetProperty(stateFor(world), solver, BodyProperty.sleepThreshold)).toBeCloseTo(
            0.2,
        );
        expect(bodyGetProperty(stateFor(world), solver, BodyProperty.motionLocks)).toBe(
            BodyMotionLock.linearX | BodyMotionLock.angularZ,
        );
        expect(bodyGetProperty(stateFor(world), solver, BodyProperty.enableSleep) !== 0).toBe(
            false,
        );
        expect(bodyGetProperty(stateFor(world), solver, BodyProperty.isBullet) !== 0).toBe(true);
        expect(bodyGetProperty(stateFor(world), solver, BodyProperty.allowFastRotation) !== 0).toBe(
            true,
        );
        expect(
            bodyGetProperty(stateFor(world), solver, BodyProperty.enableContactRecycling) !== 0,
        ).toBe(false);

        authored.type.set(eid, BodyType.Dynamic);
        world.tick();
        authored.enableSleep.set(eid, 1);
        world.tick();
        expect(bodyGetProperty(stateFor(world), solver, BodyProperty.enableSleep) !== 0).toBe(true);
        setAwake(world, eid, false);
        world.tick();
        expect(bodyIsAwake(stateFor(world), solver)).toBe(false);
        setAwake(world, eid, true);
        world.tick();
        expect(bodyIsAwake(stateFor(world), solver)).toBe(true);
        authored.isEnabled.set(eid, 0);
        world.tick();
        expect(bodyGetProperty(stateFor(world), solver, BodyProperty.isEnabled) !== 0).toBe(false);
        authored.isEnabled.set(eid, 1);
        world.tick();
        expect(bodyGetProperty(stateFor(world), solver, BodyProperty.isEnabled) !== 0).toBe(true);
    });
});

test("unchanged Body definition writes call no Box3D body setter", async () => {
    await withPhysics((world) => {
        const eid = world.create();
        world.add(eid, Body, {
            type: BodyType.Dynamic,
            linearVelocity: [0, 0, 0, 0],
            angularVelocity: [0, 0, 0, 0],
            linearDamping: 0,
            angularDamping: 0,
            gravityScale: 1,
            sleepThreshold: 0.05,
            motionLocks: 0,
            enableSleep: 1,
            isAwake: 1,
            isBullet: 0,
            isEnabled: 1,
            allowFastRotation: 0,
            enableContactRecycling: 1,
        });
        world.add(eid, Shape);
        world.tick();
        const k = kernel(world);
        const setters = [
            ["bodySetType", spyOn(k, "bodySetType")],
            ["bodySetProperty", spyOn(k, "bodySetProperty")],
            ["bodySetAwake", spyOn(k, "bodySetAwake")],
            ["bodyEnable", spyOn(k, "bodyEnable")],
            ["bodyDisable", spyOn(k, "bodyDisable")],
        ] as const;
        try {
            const authored = world.storage(Body);
            authored.type.set(eid, BodyType.Dynamic);
            authored.linearDamping.set(eid, 0);
            authored.angularDamping.set(eid, 0);
            authored.gravityScale.set(eid, 1);
            authored.sleepThreshold.set(eid, 0.05);
            authored.motionLocks.set(eid, 0);
            authored.enableSleep.set(eid, 1);
            authored.isAwake.set(eid, 1);
            authored.isBullet.set(eid, 0);
            authored.isEnabled.set(eid, 1);
            authored.allowFastRotation.set(eid, 0);
            authored.enableContactRecycling.set(eid, 1);
            world.tick();
            for (const [name, setter] of setters) {
                if (setter.mock.calls.length !== 0)
                    throw new Error(`unchanged Body field called ${name}`);
            }
        } finally {
            for (const [, setter] of setters) setter.mockRestore();
        }
    });
});

test("Body.isAwake is spawn-only", async () => {
    await withPhysics((world) => {
        const eid = world.create();
        world.add(eid, Body, {
            type: BodyType.Dynamic,
            linearVelocity: [0, 0, 0, 0],
            angularVelocity: [0, 0, 0, 0],
            linearDamping: 0,
            angularDamping: 0,
            gravityScale: 0,
            sleepThreshold: 0.05,
            motionLocks: 0,
            enableSleep: 1,
            isAwake: 1,
            isBullet: 0,
            isEnabled: 1,
            allowFastRotation: 0,
            enableContactRecycling: 1,
        });
        world.add(eid, Shape);
        world.step(Time.FIXED_DT);
        const solver = bodyIndex(world, eid);
        expect(bodyIsAwake(stateFor(world), solver)).toBe(true);

        const setter = spyOn(kernel(world), "bodySetAwake");
        try {
            world.storage(Body).isAwake.set(eid, 0);
            world.step(Time.FIXED_DT);
            expect(bodyIsAwake(stateFor(world), solver)).toBe(true);
            expect(setter).not.toHaveBeenCalled();
        } finally {
            setter.mockRestore();
        }
    });
});

test("a Body field edit survives frame mark clearing and snapshot restore", async () => {
    await withPhysics((world) => {
        const eid = world.create();
        world.add(eid, Body, {
            type: BodyType.Dynamic,
            linearVelocity: [0, 0, 0, 0],
            angularVelocity: [0, 0, 0, 0],
            gravityScale: 1,
            sleepThreshold: 0.05,
            enableSleep: 1,
            isAwake: 1,
            isEnabled: 1,
        });
        world.add(eid, Shape);
        world.tick();
        const authored = world.storage(Body).linearDamping;
        const solver = bodyIndex(world, eid);
        authored.set(eid, 0.2);
        const saved = world.snapshot();
        world.step(0);
        authored.set(eid, 0.3);
        world.tick();
        expect(bodyGetProperty(stateFor(world), solver, BodyProperty.linearDamping)).toBeCloseTo(
            0.3,
        );

        world.restore(saved);
        world.tick();
        expect(bodyGetProperty(stateFor(world), solver, BodyProperty.linearDamping)).toBeCloseTo(
            0.2,
        );
    });
});

test("a corrected dynamic body's published velocity agrees with its solver body after one tick", async () => {
    await withPhysics((world) => {
        const floor = world.create();
        world.add(floor, Transform, { translation: [0, -0.5, 0] });
        world.add(floor, Body);
        world.add(floor, Shape, { scale: [20, 1, 20, 0] });
        const eid = world.create();
        world.add(eid, Transform, { translation: [0, 0.5, 0] });
        world.add(eid, Body, { type: BodyType.Dynamic });
        world.tick();
        setLinearVelocity(world, eid, { x: 4, y: 0, z: 0 });
        const saved = world.snapshot();
        world.tick();
        world.restore(saved);

        const body = bodyIndex(world, eid);
        const position = bodyGetTransform(stateFor(world), body, {
            p: { x: 0, y: 0, z: 0 },
            q: { v: { x: 0, y: 0, z: 0 }, s: 1 },
        }).p;
        setTransform(world, eid, [position.x, position.y, position.z], [0, 0, 0, 1]);
        world.tick();

        const published = world.storage(GlobalTransform).linearVelocity;
        const solved = bodyGetLinearVelocity(stateFor(world), bodyIndex(world, eid), {
            x: 0,
            y: 0,
            z: 0,
        });
        expect(published.x.get(eid)).toBeCloseTo(solved.x, 5);
        expect(published.y.get(eid)).toBeCloseTo(solved.y, 5);
        expect(published.z.get(eid)).toBeCloseTo(solved.z, 5);
    });
});

function addRevoluteMotor(world: World): number {
    const anchor = world.create();
    const body = world.create();
    world.add(anchor, Body, { type: BodyType.Static });
    world.add(body, Body);
    const joint = world.create();
    world.add(joint, RevoluteJoint, {
        a: anchor,
        b: body,
        enableMotor: 1,
        localRotationA: [0, 0, 0, 1],
        localRotationB: [0, 0, 0, 1],
    });
    return joint;
}

test("a RevoluteJoint motor-speed edit reuses its solver joint", async () => {
    await withPhysics((world) => {
        const joint = addRevoluteMotor(world);
        world.tick();

        expect(
            world.storage(RevoluteJoint).localRotationA.read(joint, new Float32Array(4)),
        ).toEqual(new Float32Array([0, 0, 0, 1]));
        const state = stateFor(world);
        expect(worldCounters(state).jointCount).toBe(1);
        const solverId = jointIds(state).find((id) => state.jointUserData[id] === world.ref(joint));
        if (solverId === undefined) throw new Error("authored RevoluteJoint was not created");
        const generation = jointField(state, solverId!, JointField.generation);
        const create = spyOn(kernel(world), "jointCreateRevolute");
        world.storage(RevoluteJoint).motorSpeed.set(joint, 4);
        world.tick();

        expect(create).not.toHaveBeenCalled();
        expect(jointIds(state)).toContain(solverId);
        expect(jointField(state, solverId!, JointField.generation)).toBe(generation);
        expect(readJointFloat(state, solverId!, RJ_MOTOR_SPEED)).toBe(4);
    });
});

function addDynamicMotorPair(world: World): [number, number] {
    const a = world.create();
    const b = world.create();
    for (const [eid, x] of [
        [a, -1],
        [b, 1],
    ] as const) {
        world.add(eid, Transform, { translation: [x, 0, 0] });
        world.add(eid, Body, {
            type: BodyType.Dynamic,
            gravityScale: 0,
            linearDamping: 0,
            angularDamping: 0,
        });
        world.add(eid, Shape, {
            kind: ShapeKind.Sphere,
            sphereCenter: [0, 0, 0],
            radius: 0.25,
            density: 1,
            friction: 0,
        });
    }
    return [a, b];
}

function movedFrom(
    before: ReturnType<typeof bodyGetTransform>,
    after: ReturnType<typeof bodyGetTransform>,
) {
    return (
        Math.abs(after.p.x - before.p.x) +
            Math.abs(after.p.y - before.p.y) +
            Math.abs(after.p.z - before.p.z) +
            Math.abs(after.q.v.x - before.q.v.x) +
            Math.abs(after.q.v.y - before.q.v.y) +
            Math.abs(after.q.v.z - before.q.v.z) >
        1e-5
    );
}

async function expectSleepingMotorEditToWakeAndMove(
    world: World,
    a: number,
    b: number,
    edit: () => void,
): Promise<void> {
    world.tick();
    const state = stateFor(world);
    const bodyA = bodyIndex(world, a);
    const bodyB = bodyIndex(world, b);
    setAwake(world, a, false);
    setAwake(world, b, false);
    expect(bodyIsAwake(state, bodyA)).toBe(false);
    expect(bodyIsAwake(state, bodyB)).toBe(false);
    const beforeA = bodyGetTransform(state, bodyA, {
        p: { x: 0, y: 0, z: 0 },
        q: { v: { x: 0, y: 0, z: 0 }, s: 1 },
    });
    const beforeB = bodyGetTransform(state, bodyB, {
        p: { x: 0, y: 0, z: 0 },
        q: { v: { x: 0, y: 0, z: 0 }, s: 1 },
    });
    edit();
    world.tick();
    expect(bodyIsAwake(state, bodyA)).toBe(true);
    expect(bodyIsAwake(state, bodyB)).toBe(true);
    for (let i = 0; i < 12; i++) world.tick();
    const afterA = bodyGetTransform(state, bodyA, {
        p: { x: 0, y: 0, z: 0 },
        q: { v: { x: 0, y: 0, z: 0 }, s: 1 },
    });
    const afterB = bodyGetTransform(state, bodyB, {
        p: { x: 0, y: 0, z: 0 },
        q: { v: { x: 0, y: 0, z: 0 }, s: 1 },
    });
    expect(movedFrom(beforeA, afterA) || movedFrom(beforeB, afterB)).toBe(true);
}

test("RevoluteJoint motor-speed ECS edits wake sleeping endpoints and move them", async () => {
    await withPhysics((world) => {
        const [a, b] = addDynamicMotorPair(world);
        const joint = world.create();
        world.add(joint, RevoluteJoint, {
            a,
            b,
            localAnchorA: [1, 0, 0, 0],
            localAnchorB: [-1, 0, 0, 0],
            enableMotor: 1,
            maxMotorTorque: 1000,
        });
        return expectSleepingMotorEditToWakeAndMove(world, a, b, () => {
            world.storage(RevoluteJoint).motorSpeed.set(joint, 8);
        });
    });
});

test("PrismaticJoint motor-speed ECS edits wake sleeping endpoints and move them", async () => {
    await withPhysics((world) => {
        const [a, b] = addDynamicMotorPair(world);
        const joint = world.create();
        world.add(joint, PrismaticJoint, {
            a,
            b,
            localAnchorA: [1, 0, 0, 0],
            localAnchorB: [-1, 0, 0, 0],
            enableMotor: 1,
            maxMotorForce: 1000,
        });
        return expectSleepingMotorEditToWakeAndMove(world, a, b, () => {
            world.storage(PrismaticJoint).motorSpeed.set(joint, 3);
        });
    });
});

test("WheelJoint spin-speed ECS edits wake sleeping endpoints and move them", async () => {
    await withPhysics((world) => {
        const [a, b] = addDynamicMotorPair(world);
        const joint = world.create();
        world.add(joint, WheelJoint, {
            a,
            b,
            localAnchorA: [1, 0, 0, 0],
            localAnchorB: [-1, 0, 0, 0],
            enableSpinMotor: 1,
            maxSpinTorque: 1000,
        });
        return expectSleepingMotorEditToWakeAndMove(world, a, b, () => {
            world.storage(WheelJoint).spinSpeed.set(joint, 8);
        });
    });
});

test("WheelJoint steering-angle ECS edits wake sleeping endpoints and move them", async () => {
    await withPhysics((world) => {
        const [a, b] = addDynamicMotorPair(world);
        const joint = world.create();
        world.add(joint, WheelJoint, {
            a,
            b,
            localAnchorA: [1, 0, 0, 0],
            localAnchorB: [-1, 0, 0, 0],
            enableSteering: 1,
            steeringHertz: 10,
            steeringDampingRatio: 0.7,
            maxSteeringTorque: 1000,
        });
        return expectSleepingMotorEditToWakeAndMove(world, a, b, () => {
            world.storage(WheelJoint).targetSteeringAngle.set(joint, 0.6);
        });
    });
});

test("a joint field write survives a frame with no fixed tick", async () => {
    await withPhysics((world) => {
        const joint = addRevoluteMotor(world);
        world.tick();
        const state = stateFor(world);
        const solverId = jointIds(state).find(
            (id) => state.jointUserData[id] === world.ref(joint),
        )!;
        const generation = jointField(state, solverId, JointField.generation);
        const create = spyOn(kernel(world), "jointCreateRevolute");

        world.storage(RevoluteJoint).motorSpeed.set(joint, 6);
        const saved = world.snapshot();
        world.step(0);
        expect(readJointFloat(state, solverId, RJ_MOTOR_SPEED)).toBe(0);

        world.storage(RevoluteJoint).motorSpeed.set(joint, 8);
        world.tick();
        expect(readJointFloat(state, solverId, RJ_MOTOR_SPEED)).toBe(8);

        world.restore(saved);
        world.tick();

        expect(create).not.toHaveBeenCalled();
        expect(jointField(state, solverId, JointField.generation)).toBe(generation);
        expect(readJointFloat(state, solverId, RJ_MOTOR_SPEED)).toBe(6);
    });
});

test("a CPU-only draw edit survives frame mark clearing without a fixed tick", async () => {
    await withPhysics((world) => {
        const joint = addRevoluteMotor(world);
        world.storage(RevoluteJoint).motorSpeed.set(joint, 11);
        world.tick();
        const state = stateFor(world);
        const solverId = jointIds(state).find(
            (id) => state.jointUserData[id] === world.ref(joint),
        )!;
        expect(readJointFloat(state, solverId, RJ_MOTOR_SPEED)).toBe(11);

        world.addSystem(
            {
                name: "write-joint-speed-in-draw",
                group: "draw",
                update() {
                    world.storage(RevoluteJoint).motorSpeed.set(joint, 12);
                },
            },
            "JointDrawWrite",
        );
        world.step(0);
        world.tick();

        expect(readJointFloat(state, solverId, RJ_MOTOR_SPEED)).toBe(12);
    });
});

test("a CPU-only draw edit followed by a throw reaches the next joint sync", async () => {
    await withPhysics((world) => {
        const joint = addRevoluteMotor(world);
        const motorSpeed = world.storage(RevoluteJoint).motorSpeed;
        motorSpeed.set(joint, 12);
        world.tick();
        const state = stateFor(world);
        const solverId = jointIds(state).find(
            (id) => state.jointUserData[id] === world.ref(joint),
        )!;
        expect(readJointFloat(state, solverId, RJ_MOTOR_SPEED)).toBe(12);
        world.clearChanges();

        const brokenDraw = {
            name: "write-joint-speed-then-throw",
            group: "draw" as const,
            update() {
                motorSpeed.set(joint, 13);
                throw new Error("draw failed");
            },
        };
        world.addSystem(brokenDraw, "JointDrawFailure");
        expect(() => world.step(0)).toThrow(
            'System "JointDrawFailure/write-joint-speed-then-throw" threw: draw failed',
        );
        world.removeSystem(brokenDraw);

        world.tick();

        expect(readJointFloat(state, solverId, RJ_MOTOR_SPEED)).toBe(13);
    });
});

describe("dirty field synchronization", () => {
    let world: World | undefined;

    beforeAll(async () => {
        world = await createPhysicsWorld();
    });

    afterAll(async () => {
        if (!world) return;
        await StandardPhysicsPlugin.dispose!(world);
        world.dispose();
    });

    test("applies an ECS value that matches an earlier value", () => {
        const fixture = world;
        if (!fixture) throw new Error("physics fixture did not initialize");
        const joint = addRevoluteMotor(fixture);
        const authored = fixture.storage(RevoluteJoint).motorSpeed;
        authored.set(joint, 6);
        fixture.tick();
        const state = stateFor(fixture);
        const solverId = jointIds(state).find(
            (id) => state.jointUserData[id] === fixture.ref(joint),
        )!;
        writeJointFloat(state, solverId, RJ_MOTOR_SPEED, 8);
        expect(readJointFloat(state, solverId, RJ_MOTOR_SPEED)).toBe(8);

        const create = spyOn(kernel(fixture), "jointCreateRevolute");
        authored.set(joint, 6);
        fixture.tick();

        expect(create).not.toHaveBeenCalled();
        expect(readJointFloat(state, solverId, RJ_MOTOR_SPEED)).toBe(6);
    });
});

test("joint snapshots restore the matching live constraints when worlds have different joints", async () => {
    await withPhysics((world) => {
        const first = addRevoluteMotor(world);
        world.tick();
        const firstSnapshot = world.snapshot();

        world.remove(first, RevoluteJoint);
        const second = addRevoluteMotor(world);
        world.tick();
        expect(worldCounters(stateFor(world)).jointCount).toBe(1);
        expect(jointIds(stateFor(world)).map((id) => stateFor(world).jointUserData[id])).toEqual([
            world.ref(second),
        ]);
        const secondSnapshot = world.snapshot();

        world.restore(firstSnapshot);
        world.tick();
        expect(worldCounters(stateFor(world)).jointCount).toBe(1);
        expect(jointIds(stateFor(world)).map((id) => stateFor(world).jointUserData[id])).toEqual([
            world.ref(first),
        ]);

        world.restore(secondSnapshot);
        world.tick();
        expect(worldCounters(stateFor(world)).jointCount).toBe(1);
        expect(jointIds(stateFor(world)).map((id) => stateFor(world).jointUserData[id])).toEqual([
            world.ref(second),
        ]);
    });
});

test("joint endpoint changes recreate the solver joint", async () => {
    await withPhysics((world) => {
        const joint = addRevoluteMotor(world);
        world.tick();
        const state = stateFor(world);
        const originalId = jointIds(state).find(
            (id) => state.jointUserData[id] === world.ref(joint),
        )!;
        const generation = jointField(state, originalId, JointField.generation);
        const create = spyOn(kernel(state.ecsState), "jointCreateRevolute");
        const replacement = world.create();
        world.add(replacement, Body, { type: BodyType.Static });
        world.tick();
        world.storage(RevoluteJoint).a.set(joint, replacement);
        world.tick();

        expect(create).toHaveBeenCalledTimes(1);
        const newId = jointIds(state).find((id) => state.jointUserData[id] === world.ref(joint))!;
        expect(jointField(state, newId, JointField.generation)).not.toBe(generation);
        expect(jointIds(state).map((id) => state.jointUserData[id])).toEqual([world.ref(joint)]);
    });
});

test("a bound static Body applies a marked Transform pose on the next step", async () => {
    await withPhysics((world) => {
        const eid = world.create();
        world.add(eid, Body);
        world.tick();
        expect(world.has(eid, Transform)).toBe(true);
        world.storage(Transform).translation.set(eid, 4, 5, 6);
        world.tick();
        expect(
            bodyGetTransform(stateFor(world), bodyIndex(world, eid), {
                p: { x: 0, y: 0, z: 0 },
                q: { v: { x: 0, y: 0, z: 0 }, s: 1 },
            }).p,
        ).toEqual({ x: 4, y: 5, z: 6 });
        expect(world.storage(GlobalTransform).translation.x.get(eid)).toBe(4);
    });
});

test("physics sync walks Bodies added before a restored snapshot was bound", async () => {
    await withPhysics((world) => {
        const first = world.create();
        world.add(first, Body);
        world.tick();

        const second = world.create();
        world.add(second, Body);
        const saved = world.snapshot();
        expect(worldCounters(stateFor(world)).bodyCount).toBe(1);

        world.tick();
        expect(worldCounters(stateFor(world)).bodyCount).toBe(2);
        world.restore(saved);
        expect(worldCounters(stateFor(world)).bodyCount).toBe(1);

        world.tick();

        expect(world.has(second, Body)).toBe(true);
        expect(worldCounters(stateFor(world)).bodyCount).toBe(2);
        expect(physicsBodyIndex(world, second) >= 0).toBe(true);
    });
});

test("physics sync retries a missing Shape hull when the registry grows", async () => {
    expectWarning(/Shape 1 was not attached:.*missing hull 1/);
    await withPhysics((world) => {
        const hulls = world.resource(Hulls);
        const eid = world.create();
        world.add(eid, Body);
        world.add(eid, Shape, { kind: ShapeKind.Hull, geometry: 1, scale: [1, 1, 1, 0] });
        world.tick();
        expect(worldCounters(stateFor(world)).bodyCount).toBe(1);
        expect(worldCounters(stateFor(world)).shapeCount).toBe(0);

        const cube = structuredClone(hulls.get(0)!);
        expect(hulls.add({ ...cube, name: "runtime-sync-recovery-hull" })).toBe(1);
        world.tick();

        expect(worldCounters(stateFor(world)).bodyCount).toBe(1);
        expect(worldCounters(stateFor(world)).shapeCount).toBe(1);
        expect(physicsBodyIndex(world, eid) >= 0).toBe(true);
    });
});

for (const [kind, label, register] of [
    [
        ShapeKind.Mesh,
        "mesh",
        (world: World) =>
            world.resource(PhysicsMeshes).add({
                name: "late-mesh",
                vertices: new Float32Array([0, 0, 0, 1, 0, 0, 0, 0, 1]),
                indices: new Uint32Array([0, 2, 1]),
            }),
    ],
    [
        ShapeKind.HeightField,
        "height-field",
        (world: World) =>
            world.resource(HeightFields).add({
                name: "late-height-field",
                heights: new Float32Array(4),
                scale: [1, 1, 1],
                countX: 2,
                countZ: 2,
                globalMinimumHeight: 0,
                globalMaximumHeight: 0,
            }),
    ],
    [
        ShapeKind.Compound,
        "compound",
        (world: World) =>
            world.resource(Compounds).add({
                name: "late-compound",
                spheres: [{ center: [0, 0, 0], radius: 0.5 }],
            }),
    ],
] as const) {
    test(`physics sync retries a missing Shape ${label} when the registry gains it`, async () => {
        expectWarning(new RegExp(`Shape 1 was not attached:.*missing ${label} 0`));
        await withPhysics((world) => {
            const eid = world.create();
            world.add(eid, Body);
            world.add(eid, Shape, { kind, geometry: 0 });
            world.tick();
            expect(worldCounters(stateFor(world)).shapeCount).toBe(0);
            expect(register(world)).toBe(0);
            world.tick();
            expect(worldCounters(stateFor(world)).shapeCount).toBe(1);
        });
    });
}

test("physics sync refuses a hull whose points length is not a multiple of 3", async () => {
    expectWarning(
        /Shape 1 was not attached:.*hull "ragged" has 25 point coordinates, not a multiple of 3/,
    );
    await withPhysics((world) => {
        const hulls = world.resource(Hulls);
        const cube = hulls.get(0)!.points!;
        const points = new Float32Array(cube.length + 1);
        points.set(cube);
        const geometry = hulls.add({ name: "ragged", points });
        const eid = world.create();
        world.add(eid, Body);
        world.add(eid, Shape, { kind: ShapeKind.Hull, geometry, scale: [1, 1, 1] });
        world.tick();
        expect(worldCounters(stateFor(world)).bodyCount).toBe(1);
        expect(worldCounters(stateFor(world)).shapeCount).toBe(0);
    });
});

test("physics sync visits changed Bodies beyond the first eid word", async () => {
    await withPhysics((world) => {
        world.tick();
        for (let i = 0; i < 40; i++) world.create();
        const eid = world.create();
        expect(eid).toBeGreaterThanOrEqual(32);
        world.add(eid, Body);

        world.tick();

        expect(worldCounters(stateFor(world)).bodyCount).toBe(1);
        expect(physicsBodyIndex(world, eid) >= 0).toBe(true);
    });
});

test("physics sync forgets a Body destroyed before the next tick", async () => {
    await withPhysics((world) => {
        const eid = world.create();
        world.add(eid, Body);
        world.tick();
        expect(worldCounters(stateFor(world)).bodyCount).toBe(1);

        world.destroy(eid);
        world.tick();

        expect(world.exists(eid)).toBe(false);
        expect(worldCounters(stateFor(world)).bodyCount).toBe(0);
        expect(physicsBodyIndex(world, eid)).toBe(-1);
    });
});

test("physics sync restores a pending Shape hull when its registry snapshot predates sync", async () => {
    expectWarning(/Shape 1 was not attached:.*missing hull 1/);
    await withPhysics((world) => {
        const hulls = world.resource(Hulls);
        const eid = world.create();
        world.add(eid, Body);
        world.add(eid, Shape, { kind: ShapeKind.Hull, geometry: 1, scale: [1, 1, 1, 0] });
        world.tick();
        expect(worldCounters(stateFor(world)).bodyCount).toBe(1);
        expect(worldCounters(stateFor(world)).shapeCount).toBe(0);

        const cube = structuredClone(hulls.get(0)!);
        hulls.add({ ...cube, name: "runtime-sync-restored-hull" });
        const saved = world.snapshot();
        world.tick();
        expect(worldCounters(stateFor(world)).bodyCount).toBe(1);
        world.restore(saved);
        expect(worldCounters(stateFor(world)).bodyCount).toBe(1);
        expect(worldCounters(stateFor(world)).shapeCount).toBe(0);

        world.tick();

        expect(worldCounters(stateFor(world)).bodyCount).toBe(1);
        expect(worldCounters(stateFor(world)).shapeCount).toBe(1);
        expect(physicsBodyIndex(world, eid) >= 0).toBe(true);
    });
});
