import { expect, setDefaultTimeout, test } from "bun:test";
import {
    Camera,
    createApp,
    GlobalTransform,
    InputPlugin,
    pointerMove,
    pressButton,
    pressKey,
    releaseButton,
    resizeViewport,
    touchPoint,
} from "@dylanebert/shallot";
import { Orbit, OrbitMode } from "@dylanebert/shallot/extras";
import { type View, Views } from "@dylanebert/shallot/rendering";
import { TIMEOUT } from "@dylanebert/shallot/testing";
import { Grab, GrabPlugin } from "../src/grab";
import {
    ManifoldPlugin,
    ManifoldSample,
    manifoldNames,
    manifoldTransform,
    vector,
} from "../src/manifold";
import { createScene } from "../src/scenes";

setDefaultTimeout(TIMEOUT.node);
for (const scene of manifoldNames)
    for (const touch of [false, true])
        test(`${scene} ${touch ? "touch" : "pointer"}: only B drags, orbit locks, and Shift never rotates`, async () => {
            const app = await createApp({
                    defaults: false,
                    plugins: [ManifoldPlugin, InputPlugin],
                }),
                world = app.world;
            try {
                createScene(world, scene);
                world.tick();
                const s = world.resource(ManifoldSample),
                    center = manifoldTransform(
                        s.b,
                        scene.endsWith("Sphere") ? s.sphere.center : vector(),
                    ),
                    camera = world.create();
                world.add(camera, Camera, { fov: 60, near: 0.01, far: 100, mode: 0 });
                world.add(camera, GlobalTransform, {
                    translation: [center.x, center.y, center.z + 10],
                    rotation: [0, 0, 0, 1],
                    scale: [1, 1, 1],
                });
                world.add(camera, Orbit);
                world
                    .resource(Views)
                    .set(camera, { viewportIndex: 0, camera: world.ref(camera) } as View);
                resizeViewport(world, 0, 1000, 1000, 1);
                for (const system of GrabPlugin.systems!) world.addSystem(system);
                const down = (x: number, y: number) => {
                    if (touch) touchPoint(world, 1, x, y);
                    else {
                        pointerMove(world, x, y);
                        pressButton(world, "left");
                    }
                };
                const move = (x: number, y: number) => {
                    if (touch) touchPoint(world, 1, x, y);
                    else pointerMove(world, x, y);
                };
                const up = (x: number, y: number) => {
                    if (touch) touchPoint(world, 1, x, y, false);
                    else releaseButton(world, "left");
                };
                const initial = structuredClone(s.b),
                    a = structuredClone(s.a);
                down(0, 0);
                world.step(0);
                move(500, 500);
                world.step(0);
                expect(s.tracking).toBe(false);
                expect(s.b).toEqual(initial);
                up(500, 500);
                world.step(0);
                pressKey(world, "ShiftLeft");
                down(500, 500);
                world.step(0);
                expect(s.tracking).toBe(true);
                expect(world.storage(Orbit).mode.get(camera)).toBe(OrbitMode.Locked);
                move(550, 500);
                world.step(1 / 60);
                expect(s.b.p.x).toBeGreaterThan(initial.p.x);
                expect(s.b.q).toEqual(initial.q);
                expect(s.a).toEqual(a);
                expect(world.resource(Grab).mouse).toBe(0);
                expect(world.resource(Grab).joint).toBe(0);
                const draggedX = s.b.p.x;
                move(600, 500);
                up(600, 500);
                world.step(0);
                expect(s.b.p.x).toBeGreaterThan(draggedX);
                expect(s.tracking).toBe(false);
                expect(world.storage(Orbit).mode.get(camera)).toBe(OrbitMode.Free);
                const released = structuredClone(s.b);
                move(650, 500);
                world.step(0);
                expect(s.b).toEqual(released);
            } finally {
                app.dispose();
            }
        });
