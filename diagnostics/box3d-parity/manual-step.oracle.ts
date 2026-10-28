import { expect, test } from "bun:test";
import { Body, BodyType, Shape, ShapeKind } from "../../src/core/physics";
import { Transform } from "../../src/core/transform";
import { createApp } from "../../src/engine";
import { hashPhysics, PhysicsEvents, PhysicsWorldDefinition, StandardPhysicsPlugin, stepPhysics, StepPhysicsSystem } from "../../src/standard/physics";
import { nativeSseOutput } from "./native-evidence";
import { assertPublicOracleKernel } from "./oracle-kernel";

await assertPublicOracleKernel();
const native = nativeSseOutput("manual-step.c", "").trim().split("\n").map(line => {
    const [hash, begin, end] = line.split(" ");
    return { hash: BigInt(hash), begin: Number(begin), end: Number(end) };
});

test("two public half steps match native b3World_Step hashes and per-call sensor events", async () => {
    const observed: typeof native = [];
    const app = await createApp({ defaults: false, plugins: [StandardPhysicsPlugin, {
        name: "ManualStep",
        systems: [{ group: "fixed", before: [StepPhysicsSystem], update(world) {
            for (let i = 0; i < 2; i++) {
                stepPhysics(world, 1 / 120, 4);
                const events = world.resource(PhysicsEvents);
                observed.push({ hash: hashPhysics(world), begin: events.sensorBegin.count, end: events.sensorEnd.count });
            }
        } }],
    }] });
    try {
        const world = app.world;
        world.resource(PhysicsWorldDefinition).enableAutomaticStep = false;
        const sensor = world.create();
        world.add(sensor, Body, { type: BodyType.Static });
        world.add(sensor, Shape, { kind: ShapeKind.Sphere, radius: 2, isSensor: 1, enableSensorEvents: 1 });
        const body = world.create();
        world.add(body, Transform, { translation: [0, 1, 0] });
        world.add(body, Body, { type: BodyType.Dynamic });
        world.add(body, Shape, { kind: ShapeKind.Sphere, radius: 0.5, enableSensorEvents: 1 });
        world.tick();
        expect(observed).toEqual(native);
        expect(native.map(step => step.begin)).toEqual([1, 0]);
    } finally { app.dispose(); }
});
