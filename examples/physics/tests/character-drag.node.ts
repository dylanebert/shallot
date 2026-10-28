import { expect, setDefaultTimeout, test } from "bun:test";
import {
    Camera,
    createApp,
    GlobalTransform,
    InputPlugin,
    pointerMove,
    pressButton,
    releaseButton,
    resizeViewport,
    touchPoint,
} from "@dylanebert/shallot";
import { Orbit, OrbitMode } from "@dylanebert/shallot/extras";
import { type View, Views } from "@dylanebert/shallot/rendering";
import { TIMEOUT } from "@dylanebert/shallot/testing";
import { CharacterSample, CharacterSamplePlugin } from "../src/character";
import { Grab, GrabPlugin } from "../src/grab";
import { createScene } from "../src/scenes";

setDefaultTimeout(TIMEOUT.node);

test("constructing another scene clears the character query and drag state", async () => {
    const app = await createApp({ defaults: false, plugins: [CharacterSamplePlugin] });
    try {
        createScene(app.world, "capsulePlane");
        app.world.tick();
        app.world.resource(CharacterSample).tracking = true;
        createScene(app.world, "boxStack");
        expect(app.world.resource(CharacterSample)).toEqual(CharacterSample.create());
    } finally {
        app.dispose();
    }
});
for (const scene of ["capsulePlane", "moverOverlap"] as const)
    for (const touch of [false, true])
        test(`${scene} ${touch ? "touch" : "pointer"} drags only its query capsule, locks orbit and creates no grab`, async () => {
            const app = await createApp({
                defaults: false,
                plugins: [CharacterSamplePlugin, InputPlugin],
            });
            const world = app.world;
            try {
                createScene(world, scene);
                world.tick();
                const s = world.resource(CharacterSample);
                const camera = world.create();
                world.add(camera, Camera, { fov: 60, near: 0.01, far: 100, mode: 0 });
                world.add(camera, GlobalTransform, {
                    translation: [s.position.x, s.position.y, s.position.z + 10],
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
                const initial = { ...s.position };
                down(0, 0);
                world.step(0);
                move(500, 500);
                world.step(0);
                expect(s.tracking).toBe(false);
                expect(s.position).toEqual(initial);
                up(500, 500);
                world.step(0);
                down(500, 500);
                world.step(0);
                expect(s.tracking).toBe(true);
                expect(world.storage(Orbit).mode.get(camera)).toBe(OrbitMode.Locked);
                move(550, 500);
                world.step(1 / 60);
                expect(s.position.x).toBeGreaterThan(initial.x);
                expect(world.resource(Grab).mouse).toBe(0);
                expect(world.resource(Grab).joint).toBe(0);
                const draggedX = s.position.x;
                move(600, 500);
                up(600, 500);
                world.step(0);
                expect(s.position.x).toBeGreaterThan(draggedX);
                expect(s.tracking).toBe(false);
                expect(world.storage(Orbit).mode.get(camera)).toBe(OrbitMode.Free);
                const released = { ...s.position };
                move(650, 500);
                world.step(0);
                expect(s.position).toEqual(released);
            } finally {
                app.dispose();
            }
        });
