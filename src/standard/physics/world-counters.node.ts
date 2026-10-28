import { expect, setDefaultTimeout, test } from "bun:test";
import {
    Body,
    BodyType,
    createApp,
    Shape,
    ShapeKind,
    SphericalJoint,
    Transform,
} from "@dylanebert/shallot";
import { TIMEOUT } from "@dylanebert/shallot/testing";
import { getWorldCounters, PhysicsWorldDefinition, StandardPhysicsPlugin } from ".";
import { counterPile } from "./fixtures/world-counters";
import { kernel } from "./kernel/kernel";

setDefaultTimeout(TIMEOUT.node);
test("world counter records and arrays are borrowed independently; reads do not allocate kernel bytes", async () => {
    const a = await counterPile(true);
    const b = await counterPile(false);
    try {
        const first = getWorldCounters(a.world)!;
        const colors = first.colorCounts;
        const manifolds = first.manifoldCounts;
        const other = getWorldCounters(b.world)!;
        expect(other).not.toBe(first);
        expect(other.colorCounts).not.toBe(colors);
        expect(other.manifoldCounts).not.toBe(manifolds);
        const bytes = first.byteCount;
        expect(bytes).toBeGreaterThan(0);
        for (let i = 0; i < 1000; i++) {
            expect(getWorldCounters(a.world)).toBe(first);
            expect(first.byteCount).toBe(bytes);
        }
        a.world.resource(PhysicsWorldDefinition).enableAutomaticStep = true;
        a.world.tick();
        expect(getWorldCounters(a.world)).toBe(first);
        expect(first.colorCounts).toBe(colors);
        expect(first.manifoldCounts).toBe(manifolds);
        expect(first.taskCount).toBeGreaterThan(0);
        expect(first.taskCount).toBeLessThanOrEqual(256);
        expect(first.stackUsed).toBeGreaterThan(0);
        expect(first.arenaCapacity).toBeGreaterThanOrEqual(0);
        expect(first.satCallCount).toBeGreaterThan(0);
        const peak = first.stackUsed;
        const arena = first.arenaCapacity;
        const measured = first.byteCount;
        for (let i = 0; i < 100; i++) getWorldCounters(a.world);
        expect(first.byteCount).toBe(measured);
        a.world.tick();
        getWorldCounters(a.world);
        expect(first.stackUsed).toBeGreaterThanOrEqual(peak);
        expect(first.arenaCapacity).toBeGreaterThanOrEqual(arena);
        expect(other.bodyCount).toBe(4);
        expect(other.taskCount).toBe(0);
    } finally {
        a.dispose();
        b.dispose();
    }
});

test("world counter views refresh after linear memory grows without replacing the borrowed record", async () => {
    for (const workers of [0, 1]) {
        const app = await counterPile(true, workers);
        try {
            const first = getWorldCounters(app.world)!;
            const colors = first.colorCounts;
            const bytes = first.byteCount;
            kernel(app.world).memory.grow(1);
            expect(getWorldCounters(app.world)).toBe(first);
            expect(first.byteCount).toBe(bytes);
            app.world.resource(PhysicsWorldDefinition).enableAutomaticStep = true;
            app.world.tick();
            expect(getWorldCounters(app.world)).toBe(first);
            expect(first.colorCounts).toBe(colors);
            expect(first.colorCounts[23]).toBeGreaterThan(0);
            expect(first.contactCount).toBeGreaterThan(0);
        } finally {
            app.dispose();
        }
    }
});

test("world counter reads track synchronized and restored entities, refuse solver callbacks, and retain host diagnostics", async () => {
    let callbackRead: unknown;
    const app = await createApp({
        defaults: false,
        plugins: [StandardPhysicsPlugin],
        setup(world) {
            const definition = world.resource(PhysicsWorldDefinition);
            definition.workerCount = 0;
            definition.preSolveCallback = (world) => {
                callbackRead = getWorldCounters(world);
                return true;
            };
        },
    });
    try {
        const world = app.world;
        const empty = world.snapshot();
        const a = world.create();
        world.add(a, Body, { type: BodyType.Dynamic });
        world.add(a, Shape, { kind: ShapeKind.Sphere, radius: 1, enablePreSolveEvents: 1 });
        const b = world.create();
        world.add(b, Transform, { translation: [1, 0, 0] });
        world.add(b, Body, { type: BodyType.Dynamic });
        world.add(b, Shape, { kind: ShapeKind.Sphere, radius: 1 });
        expect(getWorldCounters(world)!.bodyCount).toBe(0);
        world.tick();
        expect(callbackRead).toBeNull();
        expect(getWorldCounters(world)).toMatchObject({
            bodyCount: 2,
            shapeCount: 2,
            jointCount: 0,
        });
        const joint = world.create();
        world.add(joint, SphericalJoint, { a, b });
        world.tick();
        const counters = getWorldCounters(world)!;
        expect(counters.jointCount).toBe(1);
        expect(counters.colorCounts.reduce((sum, count) => sum + count, 0)).toBeGreaterThan(0);
        world.destroy(joint);
        world.tick();
        expect(getWorldCounters(world)!.jointCount).toBe(0);
        const tasks = getWorldCounters(world)!.taskCount;
        world.restore(empty);
        expect(getWorldCounters(world)).toMatchObject({
            bodyCount: 0,
            shapeCount: 0,
            contactCount: 0,
            jointCount: 0,
            taskCount: tasks,
        });
    } finally {
        app.dispose();
    }
});
