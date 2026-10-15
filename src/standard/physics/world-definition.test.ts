import { expect, test } from "bun:test";
import { Body, BodyType, Shape, ShapeKind } from "../../core/physics";
import { GlobalTransform, Transform } from "../../core/transform";
import { createApp, type EntityRef, Time } from "../../engine";
import { hashPhysics, PhysicsWorldDefinition, StandardPhysicsPlugin } from ".";
import { physicsState } from "./runtime";

test("a world gravity write after warm changes acceleration on the next tick", async () => {
    const app = await createApp({ defaults: false, plugins: [StandardPhysicsPlugin] });
    try {
        const { world } = app;
        const eid = world.create();
        world.add(eid, Transform, { translation: [0, 10, 0] });
        world.add(eid, Body, { type: BodyType.Dynamic });
        world.add(eid, Shape);

        world.tick();
        const velocity = world.storage(GlobalTransform).linearVelocity.y;
        const before = velocity.get(eid);
        const definition = world.resource(PhysicsWorldDefinition);
        definition.gravity = { x: 0, y: -2, z: 0 };
        world.tick();

        expect(velocity.get(eid) - before).toBeCloseTo(-2 * Time.FIXED_DT, 6);
    } finally {
        app.dispose();
    }
});

test("world settings apply at the next fixed sync", async () => {
    const app = await createApp({ defaults: false, plugins: [StandardPhysicsPlugin] });
    try {
        const { world } = app;
        const definition = world.resource(PhysicsWorldDefinition);
        definition.gravity = { x: 0, y: -3, z: 0 };
        definition.restitutionThreshold = 0.25;
        definition.hitEventThreshold = 2;
        definition.contactHertz = 20;
        definition.contactDampingRatio = 3;
        definition.contactSpeed = 1.5;
        definition.maximumLinearSpeed = 50;
        definition.contactRecycleDistance = 0.1;
        definition.enableSleep = false;
        definition.enableContinuous = false;
        definition.enableWarmStarting = false;
        definition.enableSpeculative = false;
        definition.subStepCount = 2;
        world.tick();

        const state = physicsState(world)!;
        expect(state.gravity.y).toBe(-3);
        expect(state.restitutionThreshold).toBe(0.25);
        expect(state.hitEventThreshold).toBe(2);
        expect(state.contactHertz).toBe(20);
        expect(state.contactDampingRatio).toBe(3);
        expect(state.contactSpeed).toBe(1.5);
        expect(state.maxLinearSpeed).toBe(50);
        expect(state.contactRecycleDistance).toBeCloseTo(0.1, 7);
        expect(state.enableSleep).toBe(false);
        expect(state.enableContinuous).toBe(false);
        expect(state.enableWarmStarting).toBe(false);
        expect(state.enableSpeculative).toBe(false);
        expect(hashPhysics(world)).toBeTypeOf("bigint");
    } finally {
        app.dispose();
    }
});

test("world callbacks receive the owning World and its ECS Shape EntityRefs", async () => {
    const app = await createApp({ defaults: false, plugins: [StandardPhysicsPlugin] });
    try {
        const { world } = app;
        const definition = world.resource(PhysicsWorldDefinition);
        const calls = { filter: 0, preSolve: 0, friction: 0, restitution: 0 };
        const anchor = world.create();
        world.add(anchor, Body, { type: BodyType.Static });
        world.add(anchor, Shape, {
            kind: ShapeKind.Sphere,
            sphereCenter: [0, 0, 0],
            radius: 1,
            enableCustomFiltering: 1,
            enablePreSolveEvents: 1,
        });
        const moving = world.create();
        world.add(moving, Transform, { translation: [0.5, 0, 0] });
        world.add(moving, Body, { type: BodyType.Dynamic });
        world.add(moving, Shape, {
            kind: ShapeKind.Sphere,
            sphereCenter: [0, 0, 0],
            radius: 1,
            enablePreSolveEvents: 1,
        });
        const checkPair = (shapeA: EntityRef, shapeB: EntityRef) =>
            expect([world.resolve(shapeA), world.resolve(shapeB)].sort((a, b) => a - b)).toEqual(
                [anchor, moving].sort((a, b) => a - b),
            );
        definition.customFilterCallback = (owner, shapeA, shapeB) => {
            expect(owner).toBe(world);
            checkPair(shapeA, shapeB);
            calls.filter++;
            return true;
        };
        definition.preSolveCallback = (owner, shapeA, shapeB, point, normal) => {
            expect(owner).toBe(world);
            checkPair(shapeA, shapeB);
            expect(Number.isFinite(point.x + normal.x)).toBe(true);
            calls.preSolve++;
            return true;
        };
        definition.frictionCallback = (owner) => {
            expect(owner).toBe(world);
            calls.friction++;
            return 0.25;
        };
        definition.restitutionCallback = (owner) => {
            expect(owner).toBe(world);
            calls.restitution++;
            return 0.125;
        };

        world.tick();
        world.tick();
        expect(calls.filter).toBeGreaterThan(0);
        expect(calls.preSolve).toBeGreaterThan(0);
        expect(calls.friction).toBeGreaterThan(0);
        expect(calls.restitution).toBeGreaterThan(0);

        const image = world.snapshot();
        const preSolve = definition.preSolveCallback;
        definition.preSolveCallback = null;
        world.tick();
        const callsAfterClear = calls.preSolve;
        world.restore(image);
        expect<unknown>(definition.preSolveCallback).toBe(preSolve);
        world.tick();
        expect(calls.preSolve).toBeGreaterThan(callsAfterClear);
    } finally {
        app.dispose();
    }
});

test("snapshot restore restores the world definition with the solver", async () => {
    const app = await createApp({ defaults: false, plugins: [StandardPhysicsPlugin] });
    try {
        const { world } = app;
        const eid = world.create();
        world.add(eid, Transform, { translation: [0, 10, 0] });
        world.add(eid, Body, { type: BodyType.Dynamic });
        world.add(eid, Shape);
        const definition = world.resource(PhysicsWorldDefinition);
        definition.gravity = { x: 0, y: -2, z: 0 };
        world.tick();
        const image = world.snapshot();

        definition.gravity = { x: 0, y: -8, z: 0 };
        world.tick();
        world.restore(image);

        expect(definition.gravity.y).toBe(-2);
        const velocity = world.storage(GlobalTransform).linearVelocity.y;
        const before = velocity.get(eid);
        world.tick();
        expect(velocity.get(eid) - before).toBeCloseTo(-2 * Time.FIXED_DT, 6);
    } finally {
        app.dispose();
    }
});
