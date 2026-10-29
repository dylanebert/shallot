import { expect, test } from "bun:test";
import {
    Camera,
    createApp,
    focusCanvas,
    pointerMove,
    pointerWheel,
    pressButton,
    pressKey,
    resizeViewport,
    Time,
    Transform,
    touchPoint,
} from "@dylanebert/shallot";
import { Orbit, OrbitMode, OrbitPlugin } from "@dylanebert/shallot/extras";

test("Manual follows authored Orbit fields while refusing mouse, keyboard and multi-touch gestures", async () => {
    const app = await createApp({ defaults: false, plugins: [OrbitPlugin] });
    try {
        const w = app.world,
            camera = w.create();
        w.add(camera, Camera);
        w.add(camera, Transform);
        w.add(camera, Orbit, {
            mode: OrbitMode.Manual,
            smoothness: 1,
            yaw: 0,
            pitch: 0,
            distance: 6,
            pan: [2, 0, 0],
        });
        resizeViewport(w, 0, 390, 844, 1);
        focusCanvas(w);
        pressButton(w, "right");
        pressKey(w, "KeyW");
        pressKey(w, "ArrowRight");
        pointerMove(w, { x: 100, y: 100, deltaX: 40, deltaY: 20 });
        pointerWheel(w, 100);
        w.step(Time.FIXED_DT);
        expect(w.get(camera, Transform)!.translation).toEqual([2, 0, 6]);
        touchPoint(w, 1, 80, 600);
        touchPoint(w, 2, 300, 450);
        touchPoint(w, 1, 80, 530);
        touchPoint(w, 2, 330, 450);
        w.step(Time.FIXED_DT);
        expect(w.get(camera, Transform)!.translation).toEqual([2, 0, 6]);
        const orbit = w.storage(Orbit);
        orbit.yaw.set(camera, Math.PI / 2);
        orbit.distance.set(camera, 8);
        w.step(Time.FIXED_DT);
        const pose = w.get(camera, Transform)!;
        expect(pose.translation[0]).toBeCloseTo(10, 5);
        expect(pose.translation[2]).toBeCloseTo(0, 5);
        expect(w.get(camera, Orbit)!.pan).toEqual([2, 0, 0]);
    } finally {
        app.dispose();
    }
});
