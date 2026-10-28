import { expect, setDefaultTimeout, test } from "bun:test";
import { TIMEOUT } from "@dylanebert/shallot/testing";
import { Body, BodyType, Shape, ShapeKind } from "../../core/physics";
import { GlobalTransform, Transform } from "../../core/transform";
import { createApp, Time, type World } from "../../engine";
import {
    hashPhysics,
    PhysicsEvents,
    PhysicsWorldDefinition,
    StandardPhysicsPlugin,
    StepPhysicsSystem,
    stepPhysics,
} from ".";

setDefaultTimeout(TIMEOUT.node);

function populate(world: World): number {
    const sensor = world.create();
    world.add(sensor, Body, { type: BodyType.Static });
    world.add(sensor, Shape, {
        kind: ShapeKind.Sphere,
        radius: 2,
        isSensor: 1,
        enableSensorEvents: 1,
    });
    const body = world.create();
    world.add(body, Transform, { translation: [0, 1, 0] });
    world.add(body, Body, { type: BodyType.Dynamic });
    world.add(body, Shape, { kind: ShapeKind.Sphere, radius: 0.5, enableSensorEvents: 1 });
    return body;
}

function events(world: World) {
    return Object.fromEntries(
        Object.entries(world.resource(PhysicsEvents)).map(([name, value]) => [
            name,
            structuredClone(value.records.slice(0, value.count)),
        ]),
    );
}

async function manual(run: (world: World) => void) {
    const app = await createApp({
        defaults: false,
        plugins: [
            StandardPhysicsPlugin,
            {
                name: "ManualPhysics",
                recovery: () => ({ snapshot: () => null, restore: () => {} }),
                systems: [
                    {
                        name: "manual-step",
                        group: "fixed",
                        before: [StepPhysicsSystem],
                        update: run,
                    },
                ],
            },
        ],
    });
    app.world.resource(PhysicsWorldDefinition).enableAutomaticStep = false;
    return app;
}

test("one public step equals automatic hash, events, poses and velocities", async () => {
    const a = await createApp({ defaults: false, plugins: [StandardPhysicsPlugin] });
    const b = await manual((world) => stepPhysics(world, Time.FIXED_DT, 4));
    try {
        const body = populate(a.world);
        populate(b.world);
        for (let i = 0; i < 20; i++) {
            a.world.tick();
            b.world.tick();
            expect(hashPhysics(b.world)).toBe(hashPhysics(a.world));
            expect(events(b.world)).toEqual(events(a.world));
            for (const field of ["translation", "rotation", "linearVelocity"] as const)
                expect(Array.from(b.world.storage(GlobalTransform)[field].column)).toEqual(
                    Array.from(a.world.storage(GlobalTransform)[field].column),
                );
            expect(b.world.storage(Transform).translation.y.get(body)).toBe(
                a.world.storage(Transform).translation.y.get(body),
            );
        }
    } finally {
        a.dispose();
        b.dispose();
    }
});

test("two half steps replace events and recover for replay", async () => {
    let firstEvents = 0;
    const app = await manual((world) => {
        stepPhysics(world, Time.FIXED_DT / 2, 4);
        firstEvents = world.resource(PhysicsEvents).sensorBegin.count;
        stepPhysics(world, Time.FIXED_DT / 2, 4);
    });
    try {
        populate(app.world);
        app.world.tick();
        expect(firstEvents).toBe(1);
        expect(app.world.resource(PhysicsEvents).sensorBegin.count).toBe(0);
        const snapshot = app.world.snapshot();
        const advance = () => {
            for (let i = 0; i < 10; i++) app.world.tick();
            return {
                hash: hashPhysics(app.world),
                events: events(app.world),
                pose: Array.from(app.world.storage(GlobalTransform).translation.column),
            };
        };
        const expected = advance();
        app.world.resource(PhysicsWorldDefinition).enableAutomaticStep = true;
        app.world.restore(snapshot);
        expect(app.world.resource(PhysicsWorldDefinition).enableAutomaticStep).toBe(false);
        expect(advance()).toEqual(expected);
    } finally {
        app.dispose();
    }
});

test("each call synchronizes bodies and changed authoring within the same fixed system", async () => {
    const app = await manual((world) => {
        world.resource(PhysicsWorldDefinition).gravity = { x: 0, y: 0, z: 0 };
        const body = populate(world);
        stepPhysics(world, Time.FIXED_DT / 2, 4);
        expect(world.resource(PhysicsEvents).sensorBegin.count).toBe(1);
        expect(world.storage(GlobalTransform).translation.y.get(body)).toBe(1);
        world.storage(Transform).translation.y.set(body, 10);
        stepPhysics(world, Time.FIXED_DT / 2, 4);
        expect(world.resource(PhysicsEvents).sensorBegin.count).toBe(0);
        expect(world.resource(PhysicsEvents).sensorEnd.count).toBe(1);
        expect(world.storage(GlobalTransform).translation.y.get(body)).toBe(10);
    });
    try {
        app.world.tick();
    } finally {
        app.dispose();
    }
});

test("disabled automatic solve still syncs authoring without advancing bodies", async () => {
    const app = await manual(() => {});
    try {
        const body = populate(app.world);
        app.world.tick();
        const hash = hashPhysics(app.world);
        app.world.tick();
        expect(hashPhysics(app.world)).toBe(hash);
        expect(app.world.storage(GlobalTransform).translation.y.get(body)).toBe(1);
        expect(app.world.resource(PhysicsEvents).sensorBegin.count).toBe(0);
        app.world.storage(Transform).translation.y.set(body, 10);
        app.world.tick();
        expect(app.world.storage(GlobalTransform).translation.y.get(body)).toBe(10);
        expect(hashPhysics(app.world)).not.toBe(hash);
    } finally {
        app.dispose();
    }
});

test("stepPhysics refuses outside the fixed group", async () => {
    const app = await createApp({
        defaults: false,
        plugins: [
            StandardPhysicsPlugin,
            {
                name: "WrongGroup",
                systems: [
                    {
                        group: "simulation",
                        update(world) {
                            expect(() => stepPhysics(world, Time.FIXED_DT, 4)).toThrow(
                                /stepPhysics.*fixed group/,
                            );
                        },
                    },
                ],
            },
        ],
    });
    try {
        expect(() => stepPhysics(app.world, Time.FIXED_DT, 4)).toThrow(/stepPhysics.*fixed group/);
        app.world.step(0);
    } finally {
        app.dispose();
    }
});
