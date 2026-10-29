import { expect, setDefaultTimeout, test } from "bun:test";
import {
    Camera,
    cancelTouch,
    createApp,
    focusCanvas,
    pointerLockChanged,
    pointerMove,
    pointerWheel,
    pressButton,
    pressKey,
    releaseKey,
    resizeViewport,
    setInputEnabled,
    Time,
    Transform,
    touchPoint,
} from "@dylanebert/shallot";
import { Orbit, OrbitMode, OrbitPlugin } from "@dylanebert/shallot/extras";
import { TIMEOUT } from "@dylanebert/shallot/testing";
import { Grab, GrabPlugin } from "../src/grab";
import { readPlayerControls } from "../src/player-controls";

setDefaultTimeout(TIMEOUT.node);

test("player input orbits on locked pointer look, moves relative to yaw, and keeps a short Jump until a tick", async () => {
    const keys = { W: 0, A: 0, S: 0, D: 0, Space: 0, Shift: 0 },
        forward = { x: 0, y: 0, z: -1 },
        right = { x: 1, y: 0, z: 0 };
    const app = await createApp({
        defaults: false,
        plugins: [
            OrbitPlugin,
            {
                name: "Controls",
                systems: [
                    {
                        group: "setup",
                        update(w) {
                            readPlayerControls(w, keys, forward, right);
                        },
                    },
                ],
            },
        ],
    });
    try {
        const w = app.world,
            camera = w.create();
        w.add(camera, Camera);
        w.add(camera, Transform);
        w.add(camera, Orbit, { yaw: 0, pitch: 0, mode: OrbitMode.Manual, smoothness: 1 });
        resizeViewport(w, 0, 390, 844, 1);
        focusCanvas(w);
        pointerLockChanged(w, true);
        pointerMove(w, 0, 0, 1080, 0);
        pressButton(w, "right");
        pointerWheel(w, 100);
        pressKey(w, "ArrowRight");
        pressKey(w, "KeyW");
        pressKey(w, "ShiftRight");
        pressKey(w, "Space");
        releaseKey(w, "Space");
        w.step(Time.FIXED_DT / 2);
        expect(w.get(camera, Orbit)!.yaw).toBeCloseTo(-1.5, 5);
        expect(w.get(camera, Orbit)!.distance).toBe(10);
        expect(w.get(camera, Orbit)!.pan).toEqual([0, 0, 0]);
        expect(forward.x).toBeCloseTo(Math.sin(1.5), 5);
        expect(forward.y).toBe(0);
        expect(keys.W).toBe(1);
        expect(keys.Shift).toBe(1);
        expect(keys.Space).toBe(1);
        w.step(Time.FIXED_DT / 2);
        expect(keys.Space).toBe(1);
        w.step(0);
        expect(keys.Space).toBe(0);
        const pan = w.get(camera, Orbit)!.pan;
        releaseKey(w, "KeyW");
        touchPoint(w, 1, 80, 600);
        touchPoint(w, 1, 80, 530);
        touchPoint(w, 2, 300, 500);
        touchPoint(w, 2, 330, 520);
        w.step(0);
        expect(keys.W).toBe(1);
        expect(w.get(camera, Orbit)!.yaw).toBeCloseTo(-1.5 - (30 * 0.35 * Math.PI) / 180, 5);
        expect(w.get(camera, Orbit)!.pan).toEqual(pan);
        cancelTouch(w, 1);
        cancelTouch(w, 2);
        w.step(0);
        expect(keys.W).toBe(0);
        setInputEnabled(w, false);
        pressKey(w, "KeyW");
        w.step(0);
        expect(keys.W).toBe(0);
    } finally {
        app.dispose();
    }
});

test("a player-owned camera leaves pointer presses and touch to the player instead of the scene grab handler", async () => {
    const app = await createApp({ defaults: false, plugins: [OrbitPlugin] });
    try {
        const world = app.world;
        for (const system of GrabPlugin.systems!) world.addSystem(system);
        const camera = world.create();
        world.add(camera, Camera);
        world.add(camera, Transform);
        world.add(camera, Orbit, { mode: OrbitMode.Manual });
        pressButton(world, "left");
        world.step(Time.FIXED_DT);
        const grab = world.resource(Grab);
        expect([grab.pending, grab.held, grab.body, grab.joint]).toEqual([false, false, 0, 0]);
        expect(world.get(camera, Orbit)!.mode).toBe(OrbitMode.Manual);
        touchPoint(world, 1, 80, 600);
        touchPoint(world, 2, 300, 450);
        world.step(Time.FIXED_DT);
        expect([grab.pending, grab.held, grab.body, grab.joint]).toEqual([false, false, 0, 0]);
        expect(world.get(camera, Orbit)!.mode).toBe(OrbitMode.Manual);
    } finally {
        app.dispose();
    }
});
