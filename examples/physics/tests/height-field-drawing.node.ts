import { expect, setDefaultTimeout, test } from "bun:test";
import { Camera, createApp, GlobalTransform, resizeViewport } from "@dylanebert/shallot";
import { type View, Views, worldToViewport } from "@dylanebert/shallot/rendering";
import { TIMEOUT } from "@dylanebert/shallot/testing";
import { drawHeightFieldView, heightFieldDrawing } from "../src/height-field-presentation";
import { MeshSamplePlugin } from "../src/mesh";
import { HeightFieldState } from "../src/mesh-height-field";
import { type Controls, controlValues, createScene, scenes } from "../src/scenes";

setDefaultTimeout(TIMEOUT.node);
test("Height Field draws its native-sized ray endpoints and hit, sphere outline and no axes in scene ink", async () => {
    const app = await createApp({ defaults: false, plugins: [MeshSamplePlugin] });
    const world = app.world;
    try {
        createScene(world, "meshHeightField");
        const camera = world.create();
        world.add(camera, Camera, { fov: 60, near: 0.01, mode: 0 });
        world.add(camera, GlobalTransform, {
            translation: [0, 1, 30],
            rotation: [0, 0, 0, 1],
            scale: [1, 1, 1],
        });
        world.resource(Views).set(camera, { viewportIndex: 0, camera: world.ref(camera) } as View);
        const values = controlValues("meshHeightField");
        for (const radius of [0.2, 0]) {
            values.radius = radius;
            (scenes.meshHeightField.controls as Controls).radius!.set(world, [], radius, values);
            world.tick();
            expect(world.resource(HeightFieldState).hit).toBe(true);
            const records = heightFieldDrawing(world);
            expect(records.filter((r) => r[0] === 0)).toHaveLength(3);
            expect(records.filter((r) => r[0] === 1)).toHaveLength(2);
            expect(records.filter((r) => r[0] === 4)).toHaveLength(radius === 0 ? 0 : 1);
            for (const [width, height, dpr] of [
                [1280, 720, 1],
                [390, 844, 2],
            ]) {
                resizeViewport(world, 0, width!, height!, dpr!);
                const calls: {
                    a: [number, number, number];
                    b: [number, number, number];
                    color: number;
                    options: { width: number; depthBias: number };
                }[] = [];
                drawHeightFieldView(world, camera, (a, b, color, options) =>
                    calls.push({ a, b, color, options }),
                );
                expect(
                    calls.every((call) => call.color === 0xeee9df && call.options.depthBias === -1),
                ).toBe(true);
                const dots = calls.filter((call) => call.options.width !== 1);
                expect(dots).toHaveLength(3);
                for (const [i, mark] of dots.entries()) {
                    const a = worldToViewport(world, camera, mark.a)!,
                        b = worldToViewport(world, camera, mark.b)!;
                    expect(Math.hypot(a[0] - b[0], a[1] - b[1])).toBeCloseTo(
                        radius === 0 ? (i === 2 ? 10 : 6) : i === 2 ? 6 : 2,
                        5,
                    );
                }
                const lines = calls.filter((call) => call.options.width === 1);
                for (const record of records.filter((r) => r[0] === 1))
                    expect(lines).toContainEqual({
                        a: record.slice(1, 4) as [number, number, number],
                        b: record.slice(4, 7) as [number, number, number],
                        color: 0xeee9df,
                        options: { width: 1, depthBias: -1 },
                    });
                if (radius === 0) expect(lines).toHaveLength(2);
                else {
                    const center = records.find((r) => r[0] === 4)!.slice(1, 4);
                    const outline = lines.filter(
                        (call) =>
                            !records.some(
                                (r) => r[0] === 1 && r.slice(1, 4).every((v, i) => v === call.a[i]),
                            ),
                    );
                    expect(outline.length).toBeGreaterThan(20);
                    for (const edge of outline)
                        for (const point of [edge.a, edge.b])
                            expect(Math.hypot(...point.map((v, i) => v - center[i]!))).toBeCloseTo(
                                radius,
                                5,
                            );
                }
            }
        }
    } finally {
        app.dispose();
    }
});
