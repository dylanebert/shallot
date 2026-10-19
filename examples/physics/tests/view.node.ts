import { expect, setDefaultTimeout, test } from "bun:test";
import { Body, BodyType, createApp, Shape, StandardPhysicsPlugin } from "@dylanebert/shallot";
import { TIMEOUT } from "@dylanebert/shallot/testing";
import { EventSample } from "../src/events";
import { scenePresentation } from "../src/presentation";
import { categories, controlValues, createScene, visibleControls } from "../src/scenes";

setDefaultTimeout(TIMEOUT.node);

for (const { id, name } of categories.flatMap((category) => category.scenes)) {
    test(`${name} builds every shape mesh, role material and controls panel without a GPU`, async () => {
        const app = await createApp({
            defaults: false,
            plugins: [StandardPhysicsPlugin],
            setup(world) {
                createScene(world, id);
            },
        });
        try {
            const world = app.world;
            const presentation = scenePresentation(world, id);
            expect(presentation.shapes.map((shape) => shape.eid).sort()).toEqual(
                [...world.query([Shape])].sort(),
            );
            expect(presentation.shapes.length).toBeGreaterThan(0);
            for (const { eid, body, role, mesh } of presentation.shapes) {
                expect(mesh.vertices.length).toBeGreaterThan(0);
                const expected = world.get(eid, Shape)!.isSensor
                    ? "sensor"
                    : world.get(body, Body)!.type === BodyType.Static
                      ? "ground"
                      : ["revolute", "prismatic", "wheel"].includes(id) ||
                          body === world.resource(EventSample).player
                        ? "motorized"
                        : "subject";
                expect(role).toBe(expected);
                expect(presentation.materials[role]).toBeDefined();
            }
            expect(Array.from(presentation.materials.sensor.baseColor)).toEqual(
                [0.04, 0.43, 0.53, 1].map(Math.fround),
            );
            if (id === "sensor") {
                const ground = presentation.ground!;
                const bounds = [0, 1, 2].map((axis) => {
                    const values = Array.from(ground.mesh.vertices).filter(
                        (_, index) => index % 8 === axis,
                    );
                    return [
                        Math.min(...values) + ground.translation[axis]!,
                        Math.max(...values) + ground.translation[axis]!,
                    ];
                });
                expect(bounds[0]).toEqual([-10, 10]);
                expect(bounds[2]).toEqual([-10, 10]);
                expect(bounds[1]![0]).toBeCloseTo(-0.2);
                expect(bounds[1]![1]).toBeCloseTo(0);
                expect(ground.mesh.indices.length).toBeGreaterThan(0);
            } else expect(presentation.ground).toBeUndefined();
            expect(presentation.controls).toEqual(visibleControls(id, controlValues(id)));
        } finally {
            app.dispose();
        }
    });
}
