import { expect, test } from "bun:test";
import { Body, BodyType, Shape } from "../../core/physics";
import { GlobalTransform } from "../../core/transform";
import { createApp, Time } from "../../engine";
import { hashPhysics, PhysicsWorldDefinition, physicsWorld, StandardPhysicsPlugin } from ".";

test("a world gravity write after warm changes acceleration on the next tick", async () => {
    const app = await createApp({ defaults: false, plugins: [StandardPhysicsPlugin] });
    try {
        const { world } = app;
        const eid = world.create();
        world.add(eid, Body, { type: BodyType.Dynamic, position: [0, 10, 0, 0] });
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

        const state = physicsWorld(world)!.state;
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

test("world callbacks receive the owning World", async () => {
    const app = await createApp({ defaults: false, plugins: [StandardPhysicsPlugin] });
    try {
        const { world } = app;
        const definition = world.resource(PhysicsWorldDefinition);
        const calls = { filter: 0, preSolve: 0, friction: 0, restitution: 0 };
        definition.customFilterCallback = (owner) => {
            expect(owner).toBe(world);
            calls.filter++;
            return true;
        };
        definition.preSolveCallback = (owner) => {
            expect(owner).toBe(world);
            calls.preSolve++;
            return true;
        };
        definition.frictionCallback = (owner, a, _idA, b) => {
            expect(owner).toBe(world);
            calls.friction++;
            return Math.sqrt(a * b);
        };
        definition.restitutionCallback = (owner, a, _idA, b) => {
            expect(owner).toBe(world);
            calls.restitution++;
            return Math.max(a, b);
        };

        const solver = physicsWorld(world)!;
        const shape = {
            enableCustomFiltering: true,
            enablePreSolveEvents: true,
            enableContactEvents: true,
        };
        const geometry = { center: { x: 0, y: 0, z: 0 }, radius: 1 };
        solver.createBody().createSphere(shape, geometry);
        solver
            .createBody({ type: BodyType.Dynamic, position: { x: 0.5, y: 0, z: 0 } })
            .createSphere(shape, geometry);
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
        world.add(eid, Body, { type: BodyType.Dynamic, position: [0, 10, 0, 0] });
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
