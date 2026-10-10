import { expect, spyOn, test } from "bun:test";
import { Time, World } from "@dylanebert/shallot";
import {
    Body,
    BodyMotionLock,
    BodyType,
    Hulls,
    PhysicsMeshes,
    PhysicsPlugin,
    RevoluteJoint,
    Shape,
    ShapeKind,
    ShapeMaterials,
} from "@dylanebert/shallot/physics";
import {
    physicsWorld,
    StandardPhysicsPlugin,
    setAwake,
    setTransform,
} from "@dylanebert/shallot/standard/physics";
import { GlobalTransform, Transform } from "@dylanebert/shallot/transform";
import { Body as SolverBody } from "./api/body";
import { makeJointId, makeShapeId } from "./api/config";
import { RevoluteJoint as SolverRevoluteJoint } from "./api/joint";
import { Shape as SolverShape } from "./api/shape";
import { RJ_MOTOR_SPEED } from "./kernel/columns";
import { readJointFloat } from "./kernel/jointcolumns";
import { JointField, jointField } from "./kernel/jointrecords";
import { createMesh } from "./shapes/mesh";
import { jointIds } from "./solver/joint.fixture";

async function createPhysicsWorld(): Promise<World> {
    const world = new World();
    for (const component of PhysicsPlugin.components!)
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

function solverShape(world: World, index = 0): SolverShape {
    const state = physicsWorld(world)!.state;
    return new SolverShape(state, makeShapeId(state, index));
}

test("a Body type edit after spawn reaches the solver", async () => {
    await withPhysics((world) => {
        const eid = world.create();
        world.add(eid, Body, { type: BodyType.Dynamic });
        world.tick();

        world.storage(Body).type.set(eid, BodyType.Static);
        world.tick();

        expect(physicsWorld(world)!.getBody(eid)!.getType()).toBe(BodyType.Static);
    });
});

test("a post-spawn awake write survives the spawn-time Body mark", async () => {
    await withPhysics((world) => {
        const eid = world.create();
        world.add(eid, Body, {
            type: BodyType.Dynamic,
            position: [0, 0, 0, 0],
            rotation: [0, 0, 0, 1],
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

        const solver = physicsWorld(world)!.getBody(eid)!;
        expect(solver.getMass()).toBeGreaterThan(0);
        expect(solver.isAwake()).toBe(true);
        setAwake(world, eid, false);
        expect(solver.isAwake()).toBe(false);

        world.step(Time.FIXED_DT);
        expect(solver.isAwake()).toBe(false);
    });
});

test("Shape material, filter and event writes reach Box3D setters only when values differ", async () => {
    await withPhysics((world) => {
        const eid = world.create();
        world.add(eid, Body, { type: BodyType.Dynamic });
        world.add(eid, Shape);
        world.tick();
        const shape = solverShape(world);
        const setters = [
            spyOn(SolverShape.prototype, "setDensity"),
            spyOn(SolverShape.prototype, "setFriction"),
            spyOn(SolverShape.prototype, "setRestitution"),
            spyOn(SolverShape.prototype, "setSurfaceMaterial"),
            spyOn(SolverShape.prototype, "setFilter"),
        ] as const;
        try {
            const authored = world.storage(Shape);
            authored.density.set(eid, 1250);
            authored.friction.set(eid, 0.25);
            authored.restitution.set(eid, 0.4);
            authored.rollingResistance.set(eid, 0.2);
            authored.tangentVelocity.set(eid, 1, 2, 3, 0);
            authored.materialUserIdLow.set(eid, 0x12345678);
            authored.materialUserIdHigh.set(eid, 0x9abcdef0);
            authored.customColor.set(eid, 0xff00ff);
            authored.filterCategoryLow.set(eid, 0x1234);
            authored.filterMaskHigh.set(eid, 0x5678);
            authored.filterGroupIndex.set(eid, -4);
            authored.enableSensorEvents.set(eid, 1);
            authored.enableContactEvents.set(eid, 1);
            authored.enableHitEvents.set(eid, 1);
            authored.enablePreSolveEvents.set(eid, 1);
            world.tick();

            expect(shape.getDensity()).toBe(1250);
            const material = shape.getSurfaceMaterial();
            expect(material.friction).toBe(0.25);
            expect(material.restitution).toBeCloseTo(0.4);
            expect(material.rollingResistance).toBeCloseTo(0.2);
            expect(material.tangentVelocity).toEqual({ x: 1, y: 2, z: 3 });
            expect(material.userMaterialId).toBe(0x9abcdef012345678n);
            expect(material.customColor).toBe(0xff00ff);
            expect(shape.getFilter()).toEqual({
                categoryBits: 0xffffffff00001234n,
                maskBits: 0x00005678ffffffffn,
                groupIndex: -4,
            });
            expect(shape.areSensorEventsEnabled()).toBe(true);
            expect(shape.areContactEventsEnabled()).toBe(true);
            expect(shape.areHitEventsEnabled()).toBe(true);
            expect(shape.arePreSolveEventsEnabled()).toBe(true);
            expect(setters.map((setter) => setter.mock.calls.length)).toEqual([1, 1, 1, 1, 1]);

            authored.friction.set(eid, 0.25);
            world.tick();
            expect(setters[1]).toHaveBeenCalledTimes(1);
            expect(physicsWorld(world)!.getCounters().shapeCount).toBe(1);
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

        const sphere = solverShape(world, 0);
        const capsule = solverShape(world, 1);
        const setSphere = spyOn(SolverShape.prototype, "setSphere");
        const setCapsule = spyOn(SolverShape.prototype, "setCapsule");
        try {
            world.storage(Shape).sphere.set(sphereBody, 0.25, -0.5, 0.75, 0.8);
            world.storage(Shape).capsuleA.set(capsuleBody, -1, 0, 0, 0);
            world.storage(Shape).capsuleB.set(capsuleBody, 1, 0, 0, 0.4);
            world.tick();

            expect(sphere.isValid()).toBe(true);
            expect(sphere.getType()).toBe(ShapeKind.Sphere);
            expect(sphere.getSphere().center).toEqual({ x: 0.25, y: -0.5, z: 0.75 });
            expect(sphere.getSphere().radius).toBeCloseTo(0.8);
            expect(capsule.isValid()).toBe(true);
            expect(capsule.getType()).toBe(ShapeKind.Capsule);
            expect(capsule.getCapsule().center1).toEqual({ x: -1, y: 0, z: 0 });
            expect(capsule.getCapsule().center2).toEqual({ x: 1, y: 0, z: 0 });
            expect(capsule.getCapsule().radius).toBeCloseTo(0.4);
            expect(setSphere).toHaveBeenCalledTimes(1);
            expect(setCapsule).toHaveBeenCalledTimes(1);
            expect(physicsWorld(world)!.getCounters().shapeCount).toBe(2);
        } finally {
            setSphere.mockRestore();
            setCapsule.mockRestore();
        }
    });
});

test("a hull scale edit reaches b3Shape_SetHull without replacing the ECS-bound shape", async () => {
    await withPhysics((world) => {
        const eid = world.create();
        world.add(eid, Body, { type: BodyType.Dynamic });
        world.add(eid, Shape);
        world.tick();
        const shape = solverShape(world);
        const previousGeometry = shape.getGeometryReference();
        const setHull = spyOn(SolverShape.prototype, "setHull");
        try {
            world.storage(Shape).scale.x.set(eid, 1);
            world.tick();

            expect(shape.isValid()).toBe(true);
            expect(shape.getType()).toBe(ShapeKind.Hull);
            expect(shape.getGeometryReference()).not.toBe(previousGeometry);
            expect(setHull).toHaveBeenCalledTimes(1);
            expect(physicsWorld(world)!.getCounters().shapeCount).toBe(1);
        } finally {
            setHull.mockRestore();
        }
    });
});

test("mesh geometry and per-triangle material edits use Box3D setters and getters", async () => {
    await withPhysics((world) => {
        const firstMesh = createMesh({
            vertices: [
                { x: 0, y: 0, z: 0 },
                { x: 1, y: 0, z: 0 },
                { x: 0, y: 1, z: 0 },
            ],
            indices: [0, 1, 2],
        })!;
        const secondMesh = createMesh({
            vertices: [
                { x: 0, y: 0, z: 0 },
                { x: 2, y: 0, z: 0 },
                { x: 0, y: 1, z: 0 },
            ],
            indices: [0, 1, 2],
        })!;
        const meshes = world.resource(PhysicsMeshes);
        const firstId = meshes.register({ name: "shape-sync-first-mesh", data: firstMesh });
        const secondId = meshes.register({ name: "shape-sync-second-mesh", data: secondMesh });
        world.resource(ShapeMaterials).register({
            name: "shape-sync-material-set",
            materials: [
                {
                    friction: 0.15,
                    restitution: 0.35,
                    rollingResistance: 0.1,
                    tangentVelocity: { x: 1, y: 0, z: -1 },
                    userMaterialId: 23n,
                    customColor: 0x336699,
                },
            ],
        });
        const eid = world.create();
        world.add(eid, Body);
        world.add(eid, Shape, { kind: ShapeKind.Mesh, geometry: firstId });
        world.tick();

        const shape = solverShape(world);
        const previousGeometry = shape.getGeometryReference();
        const setMesh = spyOn(SolverShape.prototype, "setMesh");
        const setMaterial = spyOn(SolverShape.prototype, "setMeshMaterial");
        try {
            world.storage(Shape).geometry.set(eid, secondId);
            world.storage(Shape).scale.set(eid, 2, 0.5, 1, 0);
            world.storage(Shape).materialSet.set(eid, 1);
            world.tick();

            expect(shape.isValid()).toBe(true);
            expect(shape.getType()).toBe(ShapeKind.Mesh);
            expect(shape.getGeometryReference()).not.toBe(previousGeometry);
            expect(shape.getGeometryScale()).toEqual({ x: 2, y: 0.5, z: 1 });
            expect(shape.getMaterials()).toEqual([
                {
                    friction: Math.fround(0.15),
                    restitution: Math.fround(0.35),
                    rollingResistance: Math.fround(0.1),
                    tangentVelocity: { x: 1, y: 0, z: -1 },
                    userMaterialId: 23n,
                    customColor: 0x336699,
                },
            ]);
            expect(setMesh).toHaveBeenCalledTimes(1);
            expect(setMaterial).toHaveBeenCalledTimes(1);
            world.storage(Shape).materialSet.set(eid, 1);
            world.tick();
            expect(setMaterial).toHaveBeenCalledTimes(1);
            expect(physicsWorld(world)!.getCounters().shapeCount).toBe(1);
        } finally {
            setMesh.mockRestore();
            setMaterial.mockRestore();
        }
    });
});

test("changing Shape.body reattaches the collider to its new Body entity", async () => {
    await withPhysics((world) => {
        const firstBody = world.create();
        world.add(firstBody, Body, { position: [1, 0, 0, 0] });
        const secondBody = world.create();
        world.add(secondBody, Body, { position: [2, 0, 0, 0] });
        const shapeEntity = world.create();
        world.add(shapeEntity, Shape, { body: firstBody });
        world.tick();

        const previous = solverShape(world);
        expect(previous.getBody().getPosition().x).toBe(1);
        world.storage(Shape).body.set(shapeEntity, secondBody);
        world.tick();

        expect(previous.isValid()).toBe(false);
        expect(solverShape(world).getBody().getPosition().x).toBe(2);
        expect(physicsWorld(world)!.getCounters().shapeCount).toBe(1);
    });
});

test("an external Shape waits while its Body is removed and attaches when it returns", async () => {
    await withPhysics((world) => {
        const body = world.create();
        world.add(body, Body);
        const collider = world.create();
        world.add(collider, Shape, { body });
        world.tick();
        const previous = solverShape(world);
        expect(previous.isValid()).toBe(true);

        world.remove(body, Body);
        world.tick();
        expect(previous.isValid()).toBe(false);
        expect(physicsWorld(world)!.getCounters().shapeCount).toBe(0);

        world.add(body, Body);
        world.tick();
        expect(solverShape(world).isValid()).toBe(true);
        expect(physicsWorld(world)!.getCounters().shapeCount).toBe(1);
    });
});

test("changing updateBodyMass does not recreate a Shape and later density writes honor it", async () => {
    await withPhysics((world) => {
        const eid = world.create();
        world.add(eid, Body, { type: BodyType.Dynamic });
        world.add(eid, Shape, { kind: ShapeKind.Sphere, density: 2 });
        world.tick();

        const shape = solverShape(world);
        const body = physicsWorld(world)!.getBody(eid)!;
        const mass = body.getMass();
        expect(mass).toBeGreaterThan(0);

        world.storage(Shape).updateBodyMass.set(eid, 0);
        world.tick();
        expect(shape.isValid()).toBe(true);
        expect(body.getMass()).toBe(mass);

        world.storage(Shape).density.set(eid, 4);
        world.tick();
        expect(shape.getDensity()).toBe(4);
        expect(body.getMass()).toBe(mass);
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

        const body = physicsWorld(world)!.getBody(bodyEid)!;
        const mass = body.getMass();
        const previous = solverShape(world, 1);
        world.storage(Shape).isSensor.set(sensorEid, 1);
        world.tick();

        expect(previous.isValid()).toBe(false);
        expect(solverShape(world, 1).isSensor()).toBe(true);
        expect(body.getMass()).toBe(mass);
    });
});

test("a marked friction write compares against the live Box3D value", async () => {
    await withPhysics((world) => {
        const eid = world.create();
        world.add(eid, Body, { type: BodyType.Dynamic });
        world.add(eid, Shape, { kind: ShapeKind.Sphere });
        world.tick();

        const shape = solverShape(world);
        const authored = world.storage(Shape).friction.get(eid);
        shape.setFriction(0.125);
        world.storage(Shape).friction.set(eid, authored);
        world.tick();

        expect(shape.getFriction()).toBe(authored);
    });
});

test("changing the spawn-only sensor flag recreates its Box3D shape definition", async () => {
    await withPhysics((world) => {
        const eid = world.create();
        world.add(eid, Body, { type: BodyType.Dynamic });
        world.add(eid, Shape);
        world.tick();
        const previous = solverShape(world);
        expect(previous.isSensor()).toBe(false);

        world.storage(Shape).isSensor.set(eid, 1);
        world.tick();

        expect(previous.isValid()).toBe(false);
        const current = solverShape(world);
        expect(current.isValid()).toBe(true);
        expect(current.isSensor()).toBe(true);
        expect(physicsWorld(world)!.getCounters().shapeCount).toBe(1);
    });
});

test("Body definition writes reach Box3D through its setters after spawn", async () => {
    await withPhysics((world) => {
        const eid = world.create();
        world.add(eid, Body, {
            type: BodyType.Dynamic,
            position: [0, 0, 0, 0],
            rotation: [0, 0, 0, 1],
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
        const solver = physicsWorld(world)!.getBody(eid)!;

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

        expect(solver.getType()).toBe(BodyType.Kinematic);
        expect(solver.getLinearDamping()).toBe(0.25);
        expect(solver.getAngularDamping()).toBe(0.5);
        expect(solver.getGravityScale()).toBe(0.75);
        expect(solver.getSleepThreshold()).toBeCloseTo(0.2);
        expect(solver.getMotionLocks()).toBe(BodyMotionLock.linearX | BodyMotionLock.angularZ);
        expect(solver.isSleepEnabled()).toBe(false);
        expect(solver.isBullet()).toBe(true);
        expect(solver.isFastRotationAllowed()).toBe(true);
        expect(solver.isContactRecyclingEnabled()).toBe(false);

        authored.type.set(eid, BodyType.Dynamic);
        world.tick();
        authored.enableSleep.set(eid, 1);
        world.tick();
        expect(solver.isSleepEnabled()).toBe(true);
        setAwake(world, eid, false);
        world.tick();
        expect(solver.isAwake()).toBe(false);
        setAwake(world, eid, true);
        world.tick();
        expect(solver.isAwake()).toBe(true);
        authored.isEnabled.set(eid, 0);
        world.tick();
        expect(solver.isEnabled()).toBe(false);
        authored.isEnabled.set(eid, 1);
        world.tick();
        expect(solver.isEnabled()).toBe(true);
    });
});

test("unchanged Body definition writes call no Box3D body setter", async () => {
    await withPhysics((world) => {
        const eid = world.create();
        world.add(eid, Body, {
            type: BodyType.Dynamic,
            position: [0, 0, 0, 0],
            rotation: [0, 0, 0, 1],
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
        const setters = [
            ["setType", spyOn(SolverBody.prototype, "setType")],
            ["setLinearDamping", spyOn(SolverBody.prototype, "setLinearDamping")],
            ["setAngularDamping", spyOn(SolverBody.prototype, "setAngularDamping")],
            ["setGravityScale", spyOn(SolverBody.prototype, "setGravityScale")],
            ["setSleepThreshold", spyOn(SolverBody.prototype, "setSleepThreshold")],
            ["setMotionLocks", spyOn(SolverBody.prototype, "setMotionLocks")],
            ["enableSleep", spyOn(SolverBody.prototype, "enableSleep")],
            ["setAwake", spyOn(SolverBody.prototype, "setAwake")],
            ["setBullet", spyOn(SolverBody.prototype, "setBullet")],
            ["allowFastRotation", spyOn(SolverBody.prototype, "allowFastRotation")],
            ["enableContactRecycling", spyOn(SolverBody.prototype, "enableContactRecycling")],
            ["enable", spyOn(SolverBody.prototype, "enable")],
            ["disable", spyOn(SolverBody.prototype, "disable")],
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
            position: [0, 0, 0, 0],
            rotation: [0, 0, 0, 1],
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
        const solver = physicsWorld(world)!.getBody(eid)!;
        expect(solver.isAwake()).toBe(true);

        const setter = spyOn(SolverBody.prototype, "setAwake");
        try {
            world.storage(Body).isAwake.set(eid, 0);
            world.step(Time.FIXED_DT);
            expect(solver.isAwake()).toBe(true);
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
            position: [0, 0, 0, 0],
            rotation: [0, 0, 0, 1],
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
        const solver = physicsWorld(world)!.getBody(eid)!;
        authored.set(eid, 0.2);
        const saved = world.snapshot();
        world.step(0);
        authored.set(eid, 0.3);
        world.tick();
        expect(solver.getLinearDamping()).toBeCloseTo(0.3);

        world.restore(saved);
        world.tick();
        expect(solver.getLinearDamping()).toBeCloseTo(0.2);
    });
});

test("a corrected dynamic body's published velocity agrees with its solver body after one tick", async () => {
    await withPhysics((world) => {
        const floor = world.create();
        world.add(floor, Body, { position: [0, -0.5, 0, 0] });
        world.add(floor, Shape, { scale: [10, 0.5, 10, 0] });
        const eid = world.create();
        world.add(eid, Body, { type: BodyType.Dynamic, position: [0, 0.5, 0, 0] });
        world.tick();
        physicsWorld(world)!.getBody(eid)!.setLinearVelocity({ x: 4, y: 0, z: 0 });
        const saved = world.snapshot();
        world.tick();
        world.restore(saved);

        const body = physicsWorld(world)!.getBody(eid)!;
        const position = body.getPosition();
        setTransform(world, eid, [position.x, position.y, position.z], [0, 0, 0, 1]);
        world.tick();

        const published = world.storage(GlobalTransform).linearVelocity;
        const solved = physicsWorld(world)!.getBody(eid)!.getLinearVelocity();
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
        const solver = physicsWorld(world)!;
        expect(solver.getCounters().jointCount).toBe(1);
        const solverId = jointIds(solver.state).find(
            (id) => solver.state.jointUserData[id] === joint,
        );
        if (solverId === undefined) throw new Error("authored RevoluteJoint was not created");
        const generation = jointField(solver.state, solverId!, JointField.generation);
        const create = spyOn(solver, "createRevoluteJoint");
        world.storage(RevoluteJoint).motorSpeed.set(joint, 4);
        world.tick();

        expect(create).not.toHaveBeenCalled();
        expect(jointIds(solver.state)).toContain(solverId);
        expect(jointField(solver.state, solverId!, JointField.generation)).toBe(generation);
        expect(readJointFloat(solver.state, solverId!, RJ_MOTOR_SPEED)).toBe(4);
    });
});

test("a joint field write survives a frame with no fixed tick", async () => {
    await withPhysics((world) => {
        const joint = addRevoluteMotor(world);
        world.tick();
        const solver = physicsWorld(world)!;
        const solverId = jointIds(solver.state).find(
            (id) => solver.state.jointUserData[id] === joint,
        )!;
        const generation = jointField(solver.state, solverId, JointField.generation);
        const create = spyOn(solver, "createRevoluteJoint");

        world.storage(RevoluteJoint).motorSpeed.set(joint, 6);
        const saved = world.snapshot();
        world.step(0);
        expect(readJointFloat(solver.state, solverId, RJ_MOTOR_SPEED)).toBe(0);

        world.storage(RevoluteJoint).motorSpeed.set(joint, 8);
        world.tick();
        expect(readJointFloat(solver.state, solverId, RJ_MOTOR_SPEED)).toBe(8);

        world.restore(saved);
        world.tick();

        expect(create).not.toHaveBeenCalled();
        expect(jointField(solver.state, solverId, JointField.generation)).toBe(generation);
        expect(readJointFloat(solver.state, solverId, RJ_MOTOR_SPEED)).toBe(6);
    });
});

test("a CPU-only draw edit survives frame mark clearing without a fixed tick", async () => {
    await withPhysics((world) => {
        const joint = addRevoluteMotor(world);
        world.storage(RevoluteJoint).motorSpeed.set(joint, 11);
        world.tick();
        const solver = physicsWorld(world)!;
        const solverId = jointIds(solver.state).find(
            (id) => solver.state.jointUserData[id] === joint,
        )!;
        expect(readJointFloat(solver.state, solverId, RJ_MOTOR_SPEED)).toBe(11);

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

        expect(readJointFloat(solver.state, solverId, RJ_MOTOR_SPEED)).toBe(12);
    });
});

test("a CPU-only draw edit followed by a throw reaches the next joint sync", async () => {
    await withPhysics((world) => {
        const joint = addRevoluteMotor(world);
        const motorSpeed = world.storage(RevoluteJoint).motorSpeed;
        motorSpeed.set(joint, 12);
        world.tick();
        const solver = physicsWorld(world)!;
        const solverId = jointIds(solver.state).find(
            (id) => solver.state.jointUserData[id] === joint,
        )!;
        expect(readJointFloat(solver.state, solverId, RJ_MOTOR_SPEED)).toBe(12);
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

        expect(readJointFloat(solver.state, solverId, RJ_MOTOR_SPEED)).toBe(13);
    });
});

test("a dirty field is applied when its ECS value matches an earlier value", async () => {
    await withPhysics((world) => {
        const joint = addRevoluteMotor(world);
        const authored = world.storage(RevoluteJoint).motorSpeed;
        authored.set(joint, 6);
        world.tick();
        const solver = physicsWorld(world)!;
        const solverId = jointIds(solver.state).find(
            (id) => solver.state.jointUserData[id] === joint,
        )!;
        const handle = new SolverRevoluteJoint(solver.state, makeJointId(solver.state, solverId));
        handle.setMotorSpeed(8);
        expect(readJointFloat(solver.state, solverId, RJ_MOTOR_SPEED)).toBe(8);

        const create = spyOn(solver, "createRevoluteJoint");
        authored.set(joint, 6);
        world.tick();

        expect(create).not.toHaveBeenCalled();
        expect(readJointFloat(solver.state, solverId, RJ_MOTOR_SPEED)).toBe(6);
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
        expect(physicsWorld(world)!.getCounters().jointCount).toBe(1);
        expect(
            jointIds(physicsWorld(world)!.state).map(
                (id) => physicsWorld(world)!.state.jointUserData[id],
            ),
        ).toEqual([second]);
        const secondSnapshot = world.snapshot();

        world.restore(firstSnapshot);
        world.tick();
        expect(physicsWorld(world)!.getCounters().jointCount).toBe(1);
        expect(
            jointIds(physicsWorld(world)!.state).map(
                (id) => physicsWorld(world)!.state.jointUserData[id],
            ),
        ).toEqual([first]);

        world.restore(secondSnapshot);
        world.tick();
        expect(physicsWorld(world)!.getCounters().jointCount).toBe(1);
        expect(
            jointIds(physicsWorld(world)!.state).map(
                (id) => physicsWorld(world)!.state.jointUserData[id],
            ),
        ).toEqual([second]);
    });
});

test("joint endpoint changes recreate the solver joint", async () => {
    await withPhysics((world) => {
        const joint = addRevoluteMotor(world);
        world.tick();
        const solver = physicsWorld(world)!;
        const originalId = jointIds(solver.state).find(
            (id) => solver.state.jointUserData[id] === joint,
        )!;
        const generation = jointField(solver.state, originalId, JointField.generation);
        const create = spyOn(solver, "createRevoluteJoint");
        const replacement = world.create();
        world.add(replacement, Body, { type: BodyType.Static });
        world.tick();
        world.storage(RevoluteJoint).a.set(joint, replacement);
        world.tick();

        expect(create).toHaveBeenCalledTimes(1);
        const newId = jointIds(solver.state).find(
            (id) => solver.state.jointUserData[id] === joint,
        )!;
        expect(jointField(solver.state, newId, JointField.generation)).not.toBe(generation);
        expect(jointIds(solver.state).map((id) => solver.state.jointUserData[id])).toEqual([joint]);
    });
});

test("physics sync warns when Transform is added to a bound Body", async () => {
    await withPhysics((world) => {
        const warn = spyOn(console, "warn").mockImplementation(() => {});
        try {
            const eid = world.create();
            world.add(eid, Body);
            world.tick();
            expect(physicsWorld(world)!.getBody(eid)?.isValid()).toBe(true);

            world.add(eid, Transform);
            world.tick();

            expect(warn).toHaveBeenCalledWith(
                expect.stringContaining(`entity ${eid} carries both Body and Transform`),
            );
        } finally {
            warn.mockRestore();
        }
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
        expect(physicsWorld(world)!.getCounters().bodyCount).toBe(1);

        world.tick();
        expect(physicsWorld(world)!.getCounters().bodyCount).toBe(2);
        world.restore(saved);
        expect(physicsWorld(world)!.getCounters().bodyCount).toBe(1);

        world.tick();

        expect(world.has(second, Body)).toBe(true);
        expect(physicsWorld(world)!.getCounters().bodyCount).toBe(2);
        expect(physicsWorld(world)!.getBody(second)?.isValid()).toBe(true);
    });
});

test("physics sync retries a missing Shape hull when the registry grows", async () => {
    await withPhysics((world) => {
        const hulls = world.resource(Hulls);
        const eid = world.create();
        world.add(eid, Body);
        world.add(eid, Shape, { kind: ShapeKind.Hull, geometry: 1, scale: [1, 1, 1, 0] });
        world.tick();
        expect(physicsWorld(world)!.getCounters().bodyCount).toBe(1);
        expect(physicsWorld(world)!.getCounters().shapeCount).toBe(0);

        const cube = structuredClone(hulls.get(hulls.name(0)!)!);
        expect(hulls.register({ ...cube, name: "runtime-sync-recovery-hull" })).toBe(1);
        world.tick();

        expect(physicsWorld(world)!.getCounters().bodyCount).toBe(1);
        expect(physicsWorld(world)!.getCounters().shapeCount).toBe(1);
        expect(physicsWorld(world)!.getBody(eid)?.isValid()).toBe(true);
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

        expect(physicsWorld(world)!.getCounters().bodyCount).toBe(1);
        expect(physicsWorld(world)!.getBody(eid)?.isValid()).toBe(true);
    });
});

test("physics sync forgets a Body destroyed before the next tick", async () => {
    await withPhysics((world) => {
        const eid = world.create();
        world.add(eid, Body);
        world.tick();
        expect(physicsWorld(world)!.getCounters().bodyCount).toBe(1);

        world.destroy(eid);
        world.tick();

        expect(world.exists(eid)).toBe(false);
        expect(physicsWorld(world)!.getCounters().bodyCount).toBe(0);
        expect(physicsWorld(world)!.getBody(eid)).toBeNull();
    });
});

test("physics sync restores a pending Shape hull when its registry snapshot predates sync", async () => {
    await withPhysics((world) => {
        const hulls = world.resource(Hulls);
        const eid = world.create();
        world.add(eid, Body);
        world.add(eid, Shape, { kind: ShapeKind.Hull, geometry: 1, scale: [1, 1, 1, 0] });
        world.tick();
        expect(physicsWorld(world)!.getCounters().bodyCount).toBe(1);
        expect(physicsWorld(world)!.getCounters().shapeCount).toBe(0);

        const cube = structuredClone(hulls.get(hulls.name(0)!)!);
        hulls.register({ ...cube, name: "runtime-sync-restored-hull" });
        const saved = world.snapshot();
        world.tick();
        expect(physicsWorld(world)!.getCounters().bodyCount).toBe(1);
        world.restore(saved);
        expect(physicsWorld(world)!.getCounters().bodyCount).toBe(1);
        expect(physicsWorld(world)!.getCounters().shapeCount).toBe(0);

        world.tick();

        expect(physicsWorld(world)!.getCounters().bodyCount).toBe(1);
        expect(physicsWorld(world)!.getCounters().shapeCount).toBe(1);
        expect(physicsWorld(world)!.getBody(eid)?.isValid()).toBe(true);
    });
});
