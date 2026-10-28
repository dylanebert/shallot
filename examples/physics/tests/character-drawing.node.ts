import { expect, setDefaultTimeout, test } from "bun:test";
import { Camera, createApp, GlobalTransform, resizeViewport, Shape } from "@dylanebert/shallot";
import { type View, Views, worldToViewport } from "@dylanebert/shallot/rendering";
import { TIMEOUT } from "@dylanebert/shallot/testing";
import { CharacterSample, CharacterSamplePlugin } from "../src/character";
import { characterDrawing, drawCharacterView } from "../src/character-presentation";
import { createScene, scenes } from "../src/scenes";

setDefaultTimeout(TIMEOUT.node);
for (const scene of ["capsulePlane", "moverOverlap"] as const)
    test(`${scene} draws capsule outlines, native-sized contact marks and only its settled overlays`, async () => {
        const app = await createApp({ defaults: false, plugins: [CharacterSamplePlugin] });
        const world = app.world;
        try {
            createScene(world, scene);
            const s = world.resource(CharacterSample);
            if (scene === "moverOverlap") Object.assign(s.position, { x: -3, y: 1, z: 0 });
            world.tick();
            expect(s.planes.count).toBeGreaterThan(0);
            if (scene === "moverOverlap") {
                expect(scenes.moverOverlap.controls.Planes.read(world)).toBe(s.planes.count);
                expect(scenes.moverOverlap.controls.Degenerate.read(world)).toBe(s.zeroNormalCount);
            }
            const camera = world.create();
            world.add(camera, Camera, { fov: 60, near: 0.01, mode: 0 });
            world.add(camera, GlobalTransform, {
                translation: [0, 1, 15],
                rotation: [0, 0, 0, 1],
                scale: [1, 1, 1],
            });
            world
                .resource(Views)
                .set(camera, { viewportIndex: 0, camera: world.ref(camera) } as View);
            expect([...world.query([Shape])]).toHaveLength(scene === "capsulePlane" ? 1 : 3);
            const drawing = characterDrawing(world);
            expect(drawing.records.filter((r) => r[0] === 3)).toHaveLength(
                scene === "capsulePlane" ? 1 : 2,
            );
            expect(drawing.colors[0]).toBe(0xeee9df);
            if (scene === "moverOverlap") expect(drawing.colors.at(-1)).toBe(0x38afc0);
            else
                expect(drawing.records.slice(-3)).toEqual([
                    [1, 0, 0, 0, 2, 0, 0],
                    [1, 0, 0, 0, 0, 2, 0],
                    [1, 0, 0, 0, 0, 0, 2],
                ]);
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
                drawCharacterView(world, camera, (a, b, color, options) =>
                    calls.push({ a, b, color, options }),
                );
                expect(calls.length).toBeGreaterThan(60);
                const marks = calls.filter((call) => call.options.width !== 1);
                expect(marks).toHaveLength(s.planes.count);
                for (const mark of marks) {
                    const a = worldToViewport(world, camera, mark.a)!,
                        b = worldToViewport(world, camera, mark.b)!;
                    expect(Math.hypot(a[0] - b[0], a[1] - b[1])).toBeCloseTo(
                        scene === "capsulePlane" ? 5 : 6,
                        5,
                    );
                    expect(mark.color).toBe(0xeee9df);
                    expect(mark.options.depthBias).toBe(-1);
                }
            }
            if (scene === "moverOverlap") {
                Object.assign(s.planes.records[0]!.plane.plane.normal, { x: 0, y: 0, z: 0 });
                const invalid = characterDrawing(world);
                expect(invalid.colors[1]).toBe(0xff8135);
                expect(invalid.colors[2]).toBe(0xff8135);
            }
        } finally {
            app.dispose();
        }
    });
