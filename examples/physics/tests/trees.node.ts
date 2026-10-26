import { expect, setDefaultTimeout, test } from "bun:test";
import {
    Body,
    BodyType,
    createApp,
    hashPhysics,
    Shape,
    StandardPhysicsPlugin,
    shapeMesh,
} from "@dylanebert/shallot";
import { TIMEOUT } from "@dylanebert/shallot/testing";
import { BenchmarkSamplePlugin } from "../src/benchmarks";
import { sceneSwitcher } from "../src/sample";
import { changeControl, controlValues, enabledControls } from "../src/scenes";

setDefaultTimeout(TIMEOUT.node);

test("Falling Trees grid controls rebuild a fresh world, retain the body list and recover simulation", async () => {
    const app = await createApp({
        defaults: false,
        plugins: [StandardPhysicsPlugin, BenchmarkSamplePlugin],
    });
    try {
        const bodies = sceneSwitcher(app.world)("fallingTrees");
        const values = controlValues("fallingTrees");
        const control = enabledControls("fallingTrees", values)["Grid size"]!;
        expect(control.options).toEqual([
            { label: "100cm", value: 100 },
            { label: "50cm", value: 50 },
            { label: "25cm", value: 25 },
        ]);
        const firstSteps: bigint[] = [];
        for (const size of [100, 50, 25, 100]) {
            changeControl(app.world, "fallingTrees", bodies, values, "Grid size", size);
            expect(values["Grid size"]).toBe(size);
            expect(bodies).toHaveLength(50);
            expect([...app.world.query([Shape])]).toHaveLength(1101);
            const ground = [...app.world.query([Body, Shape])][0]!;
            const mesh = shapeMesh(app.world, app.world.get(ground, Shape)!);
            expect(mesh.indices.length).toBe(6 * 150 * 200 * (100 / size) ** 2);
            for (const body of bodies)
                expect(app.world.storage(Body).type.get(body)).toBe(BodyType.Dynamic);
            app.world.tick();
            firstSteps.push(hashPhysics(app.world));
            const saved = app.world.snapshot();
            app.world.tick();
            const expected = hashPhysics(app.world);
            app.world.restore(saved);
            app.world.tick();
            expect(hashPhysics(app.world)).toBe(expected);
        }
        expect(firstSteps[0]).toBe(firstSteps[3]);

        expect(() =>
            changeControl(app.world, "fallingTrees", bodies, values, "Grid size", 75),
        ).toThrow("Grid size must be 100, 50 or 25 cm");
    } finally {
        app.dispose();
    }
});
