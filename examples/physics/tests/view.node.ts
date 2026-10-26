import { expect, setDefaultTimeout, test } from "bun:test";
import { Body, BodyType, createApp, Shape, StandardPhysicsPlugin } from "@dylanebert/shallot";
import {
    applyLinearImpulseToCenter,
    isAwake,
    setAwake,
} from "@dylanebert/shallot/standard/physics";
import { TIMEOUT } from "@dylanebert/shallot/testing";
import { EventSample, EventSamplePlugin } from "../src/events";
import { scenePresentation, shapeMaterial } from "../src/presentation";
import { categories, controlValues, createScene, enabledControls, scenes } from "../src/scenes";

setDefaultTimeout(TIMEOUT.node);

test("sleep changes a body's colour without changing its role", async () => {
    const app = await createApp({ defaults: false, plugins: [StandardPhysicsPlugin] });
    try {
        const [body] = createScene(app.world, "boxStack");
        app.world.tick();
        expect(isAwake(app.world, body!)).toBe(true);
        const before = Array.from(shapeMaterial(app.world, "subject", body!).baseColor);
        for (let i = 0; i < 1800 && isAwake(app.world, body!); i++) app.world.tick();
        expect(isAwake(app.world, body!)).toBe(false);
        expect(Array.from(shapeMaterial(app.world, "subject", body!).baseColor)).not.toEqual(
            before,
        );
        // Compare chromaticity: brightness changes with state and lighting, affordance does not.
        const chromaticity = (color: ArrayLike<number>) => {
            const sum = color[0]! + color[1]! + color[2]!;
            return [color[0]! / sum, color[1]! / sum, color[2]! / sum];
        };
        const distance = (a: number[], b: number[]) =>
            Math.hypot(...a.map((value, i) => value - b[i]!));
        const ground = chromaticity(shapeMaterial(app.world, "ground", body!).baseColor);
        for (const role of ["subject", "motorized", "sensor", "reaction"] as const) {
            if (role === "reaction")
                app.world.resource(EventSample).reactions.add(app.world.ref(body!));
            const materialRole = role === "reaction" ? "subject" : role;
            setAwake(app.world, body!, true);
            const awake = shapeMaterial(app.world, materialRole, body!);
            setAwake(app.world, body!, false);
            const asleep = shapeMaterial(app.world, materialRole, body!);
            const tint = chromaticity(asleep.baseColor);
            expect(distance(tint, chromaticity(awake.baseColor)), role).toBeLessThan(
                distance(tint, ground),
            );
            expect(asleep.perceptualRoughness).toBeGreaterThan(awake.perceptualRoughness);
            expect(asleep.baseColor[0]! + asleep.baseColor[1]! + asleep.baseColor[2]!).toBeLessThan(
                awake.baseColor[0]! + awake.baseColor[1]! + awake.baseColor[2]!,
            );
        }
    } finally {
        app.dispose();
    }
});

test("a broken joint recolours its body", async () => {
    const app = await createApp({ defaults: false, plugins: [EventSamplePlugin] });
    try {
        const bodies = createScene(app.world, "joint");
        const before = bodies.map((body) =>
            Array.from(shapeMaterial(app.world, "subject", body).baseColor),
        );
        app.world.tick();
        applyLinearImpulseToCenter(app.world, bodies[0]!, { x: 0, y: -1000, z: 0 });
        for (let i = 0; i < 120; i++) app.world.tick();
        expect(
            bodies.some(
                (body, i) =>
                    JSON.stringify(
                        Array.from(shapeMaterial(app.world, "subject", body).baseColor),
                    ) !== JSON.stringify(before[i]),
            ),
        ).toBe(true);
    } finally {
        app.dispose();
    }
});

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
            const extent = {
                boxStack: 40,
                bridge: 25,
                contact: 80,
                hit: 60,
                joint: 20,
                sensor: 10,
                motor: 10,
                revolute: 10,
                prismatic: 10,
                wheel: 10,
                filter: 10,
                parallel: 20,
                largePyramid: 400,
                widePyramid: 100,
                manyPyramids: 77,
                fallingBoxes: 100,
                jointGrid: 100,
                rain: 75,
                junkyard: 120,
            }[id];
            const grid = presentation.grid;
            const spacing = grid.lines[2]!.a[2] - grid.lines[0]!.a[2];
            expect(spacing).toBe(scenes[id].gridSpacing);
            // At the 720 CSS-pixel review camera, adjacent cells at the ground focus
            // reserve at least 4 pixels in both screen directions (at most 1 line / 4 px).
            const camera = scenes[id].camera;
            const pitch = (camera.pitch * Math.PI) / 180;
            const depth = camera.distance + camera.target[1]! * Math.sin(pitch);
            const focal = 720 / (2 * Math.tan(Math.PI / 6));
            const across = (focal * spacing) / depth;
            const along =
                (focal * spacing * (camera.distance * Math.sin(pitch) + camera.target[1]!)) /
                (depth * depth - ((Math.cos(pitch) * spacing) / 2) ** 2);
            expect(
                Math.min(across, along),
                `${name} grid cell pixels at the ground focus`,
            ).toBeGreaterThanOrEqual(4);
            expect(grid).toBeDefined();
            expect(grid.width).toBe(1);
            expect(grid.opacity).toBeLessThan(0.45);
            for (let offset = -extent; offset <= extent; offset += spacing) {
                const color = offset === 0 ? 0x797369 : 0x49453f;
                expect(grid.lines).toContainEqual({
                    a: [-extent, 0.005, offset],
                    b: [extent, 0.005, offset],
                    color,
                });
                expect(grid.lines).toContainEqual({
                    a: [offset, 0.005, -extent],
                    b: [offset, 0.005, extent],
                    color,
                });
            }
            expect(grid.lines).toHaveLength(((2 * extent) / spacing + 1) * 2);
            expect(presentation.controls).toEqual(enabledControls(id, controlValues(id)));
        } finally {
            app.dispose();
        }
    });
}
