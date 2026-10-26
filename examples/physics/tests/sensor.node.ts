import { expect, setDefaultTimeout, test } from "bun:test";
import { createApp, hashPhysics, Shape, StandardPhysicsPlugin } from "@dylanebert/shallot";
import { TIMEOUT } from "@dylanebert/shallot/testing";
import { BenchmarkSamplePlugin } from "../src/benchmarks";
import { shapeMaterial, shapePresentation } from "../src/presentation";
import { createScene } from "../src/scenes";
import { SensorSample } from "../src/sensor";

setDefaultTimeout(TIMEOUT.node);

test("Sensor draws its roles and reacts on overlap, with recoverable spawn and event state", async () => {
    const app = await createApp({
        defaults: false,
        plugins: [StandardPhysicsPlugin, BenchmarkSamplePlugin],
        setup(world) {
            createScene(world, "benchmarkSensor");
        },
    });
    const world = app.world;
    try {
        const sample = world.resource(SensorSample);
        expect(sample.sensors.size).toBe(1681);
        expect([...sample.sensors.values()].filter((s) => s.active)).toHaveLength(81);
        expect(
            [...world.query([Shape])].filter((eid) =>
                world.storage(Shape).enableCustomFiltering.get(eid),
            ),
        ).toHaveLength(40);
        for (const [ref, sensor] of sample.sensors) {
            const eid = world.resolve(ref);
            expect(shapePresentation(world, "benchmarkSensor", eid).role).toBe(
                sensor.active ? "ground" : "sensor",
            );
            if (sensor.row === 20)
                expect(Array.from(shapeMaterial(world, "sensor", eid).baseColor)).toEqual(
                    [1, 0.22, 0.035, 1].map(Math.fround),
                );
        }
        for (let i = 0; i < 100; i++) world.tick();
        expect(sample.bodyRefs).toHaveLength(1681 + 120);
        expect(sample.reactions.size).toBeGreaterThan(0);
        expect(sample.maxBeginCount).toBeGreaterThan(0);
        for (const ref of sample.reactions)
            expect(
                Array.from(shapeMaterial(world, "subject", world.resolve(ref)).baseColor),
            ).toEqual([1, 0.22, 0.035, 1].map(Math.fround));
        const snapshot = world.snapshot();
        for (let i = 0; i < 64; i++) world.tick();
        const hash = hashPhysics(world);
        const state = structuredClone(sample);
        world.restore(snapshot);
        for (let i = 0; i < 64; i++) world.tick();
        expect(hashPhysics(world)).toBe(hash);
        expect(world.resource(SensorSample)).toEqual(state);
        expect(sample.maxEndCount).toBeGreaterThan(0);
    } finally {
        app.dispose();
    }
});
