import { expect, setDefaultTimeout, test } from "bun:test";
import { Camera, createApp, Transform } from "@dylanebert/shallot";
import { Orbit } from "@dylanebert/shallot/extras";
import { TIMEOUT } from "@dylanebert/shallot/testing";
import {
    CollisionDebug,
    CollisionDebugPlugin,
    debugDrag,
    debugNames,
} from "../src/collision-debug";
import { debugDrawing } from "../src/collision-debug-presentation";
import { changeControl, controlValues, createScene, scenes } from "../src/scenes";
import { setSceneCamera } from "../src/view";

setDefaultTimeout(TIMEOUT.node);
const point = (v: { x: number; y: number; z: number }): [number, number, number] => [v.x, v.y, v.z];
for (const scene of debugNames)
    test(`${scene}: query geometry, witnesses and vertex identity are drawn in scene ink`, async () => {
        const app = await createApp({
            defaults: false,
            plugins: [CollisionDebugPlugin],
            setup(w) {
                createScene(w, scene);
            },
        });
        try {
            app.world.tick();
            const s = app.world.resource(CollisionDebug),
                drawing = debugDrawing(app.world);
            expect(drawing.segments.length).toBeGreaterThan(0);
            expect(
                drawing.segments.every((v) =>
                    [0xeee9df, 0x38a4b5, 0xff8135, 0x797369].includes(v.color),
                ),
            ).toBe(true);
            if (scene === "shapeCastDebug") {
                expect(s.cast!.hit).toBe(true);
                expect(drawing.segments.some((v) => v.color === 0xff8135)).toBe(true);
                expect(drawing.segments.some((v) => v.color === 0x797369)).toBe(true);
                expect(drawing.labels).toHaveLength(0);
                expect(scenes[scene].controls).toBeUndefined();
            } else if (scene === "timeOfImpact") {
                expect(drawing.labels.map((v) => v.label)).toEqual(["0", "1", "2"]);
                expect(drawing.labels[0]!.point).toEqual([-4, 0, -4]);
                expect(drawing.marks).toContainEqual({
                    point: point(s.toi!.point),
                    color: 0xeee9df,
                });
                expect(scenes[scene].controls!["shape A"]!.options!.map((v) => v.label)).toEqual([
                    "box",
                    "capsule",
                    "triangle",
                ]);
            } else {
                expect(drawing.marks).toContainEqual({
                    point: point(s.distance!.pointA),
                    color: 0x38a4b5,
                });
                expect(drawing.marks).toContainEqual({
                    point: point(s.distance!.pointB),
                    color: 0xeee9df,
                });
                if (scene === "distanceDebug") expect(drawing.labels).toHaveLength(8);
                else {
                    const values = controlValues(scene);
                    changeControl(app.world, scene, [], values, "show indices", 1);
                    expect(debugDrawing(app.world).labels.map((v) => v.label)).toEqual([
                        "0",
                        "1",
                        "2",
                        "0",
                        "1",
                        "2",
                        "3",
                        "4",
                        "5",
                        "6",
                        "7",
                    ]);
                    changeControl(app.world, scene, [], values, "draw simplex", 1);
                    s.index = 1;
                }
                if (scene === "shapeDistance") {
                    s.transform.p.y = Math.fround(0.1);
                    app.world.tick();
                    s.index = s.distance!.simplexCount - 1;
                    expect(s.distance!.simplexes[s.index]!.count).toBe(4);
                }
                const simplex = s.distance!.simplexes[s.index]!;
                const selected = debugDrawing(app.world);
                if (scene === "distanceDebug") {
                    expect(simplex.count).toBe(1);
                    expect(selected.marks[2]!.point).toEqual(point(simplex.vertices[0]!.wA));
                    expect(selected.marks[3]!.point).toEqual(point(simplex.vertices[0]!.wB));
                    expect(
                        scenes[scene].controls!["Current distance, m"]!.read!(app.world),
                    ).toBeGreaterThan(0);
                }
                for (let i = 0; i < simplex.count; i++) {
                    const color = [0xeee9df, 0x38a4b5, 0xff8135, 0xffb62e][i]!;
                    expect(selected.marks).toContainEqual({
                        point: point(simplex.vertices[i]!.wA),
                        color,
                    });
                    expect(selected.marks).toContainEqual({
                        point: point(simplex.vertices[i]!.wB),
                        color,
                    });
                }
            }
        } finally {
            app.dispose();
        }
    });
test("Shape Distance drags only its query shape and stops on release", async () => {
    const app = await createApp({
        defaults: false,
        plugins: [CollisionDebugPlugin],
        setup(w) {
            createScene(w, "shapeDistance");
        },
    });
    try {
        app.world.tick();
        const direction = { x: 0, y: 0, z: -20 };
        expect(debugDrag(app.world, { x: 10, y: 1, z: 10 }, direction, true, true)).toBe(false);
        expect(debugDrag(app.world, { x: 0, y: 1, z: 10 }, direction, true, true)).toBe(true);
        expect(debugDrag(app.world, { x: 1, y: 2, z: 10 }, direction, false, true)).toBe(true);
        expect(app.world.resource(CollisionDebug).transform.p).toEqual({ x: 1, y: 2, z: 0 });
        expect(debugDrag(app.world, { x: 1, y: 2, z: 10 }, direction, false, false)).toBe(false);
        debugDrag(app.world, { x: 2, y: 3, z: 10 }, direction, false, false);
        expect(app.world.resource(CollisionDebug).transform.p).toEqual({ x: 1, y: 2, z: 0 });
    } finally {
        app.dispose();
    }
});
test("Shape Cast Debug's full query remains inside its camera's clipping and orbit limits", async () => {
    const app = await createApp({ defaults: false, plugins: [CollisionDebugPlugin] });
    try {
        const w = app.world,
            camera = w.create();
        w.add(camera, Camera);
        w.add(camera, Orbit);
        w.add(camera, Transform);
        setSceneCamera(w, "shapeCastDebug", camera);
        expect(w.storage(Camera).far.get(camera)).toBeGreaterThan(18000);
        expect(w.storage(Orbit).maxDistance.get(camera)).toBeGreaterThanOrEqual(18000);
        setSceneCamera(w, "shapeDistance", camera);
        expect(w.storage(Camera).far.get(camera)).toBe(1000);
        expect(w.storage(Orbit).maxDistance.get(camera)).toBe(900);
    } finally {
        app.dispose();
    }
});
