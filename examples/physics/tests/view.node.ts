import { expect, setDefaultTimeout, test } from "bun:test";
import { Body, BodyType, createApp, Shape, StandardPhysicsPlugin } from "@dylanebert/shallot";
import { TIMEOUT } from "@dylanebert/shallot/testing";
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
            for (const { body, role, mesh } of presentation.shapes) {
                expect(mesh.vertices.length).toBeGreaterThan(0);
                const expected =
                    world.get(body, Body)!.type === BodyType.Static
                        ? "ground"
                        : ["revolute", "prismatic", "wheel"].includes(id)
                          ? "motorized"
                          : "subject";
                expect(role).toBe(expected);
                expect(presentation.materials[role]).toBeDefined();
            }
            expect(presentation.controls).toEqual(visibleControls(id, controlValues(id)));
        } finally {
            app.dispose();
        }
    });
}
