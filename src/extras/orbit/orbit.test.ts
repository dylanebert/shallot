import { expect, test } from "bun:test";
import {
    Camera,
    CameraMode,
    cancelTouch,
    createApp,
    Devices,
    focusCanvas,
    Orbit,
    OrbitMode,
    OrbitPlugin,
    pointerMove,
    pointerWheel,
    pressButton,
    pressKey,
    releaseButton,
    releaseKey,
    resizeViewport,
    Time,
    Transform,
    touchPoint,
} from "@dylanebert/shallot";

async function subject(mode: number, button?: "left" | "right", locked = false) {
    const app = await createApp({
        defaults: false,
        plugins: [OrbitPlugin, { name: "CameraData", components: [Camera] }],
    });
    const w = app.world;
    const camera = w.create();
    w.add(camera, Transform);
    w.add(camera, Camera, { mode });
    w.add(camera, Orbit, { mode: locked ? OrbitMode.Locked : OrbitMode.Free });
    resizeViewport(w, 0, 320, 200, 1);
    focusCanvas(w, 0);
    w.step(0);
    if (button) pressButton(w, button);
    return { app, w, camera };
}

function pose(w: Awaited<ReturnType<typeof createApp>>["world"], camera: number) {
    const t = w.storage(Transform);
    return [
        t.translation.x.get(camera),
        t.translation.y.get(camera),
        t.translation.z.get(camera),
        t.rotation.x.get(camera),
        t.rotation.y.get(camera),
        t.rotation.z.get(camera),
        t.rotation.w.get(camera),
    ];
}

test("one finger rotates yaw and pitch regardless of mouse orbit binding and stops after cancellation", async () => {
    const { app, w, camera } = await subject(CameraMode.Perspective);
    try {
        const orbit = w.storage(Orbit);
        orbit.orbitButton.set(camera, 2);
        const yaw = orbit.yaw.get(camera);
        const pitch = orbit.pitch.get(camera);
        touchPoint(w, 1, 100, 100);
        w.step(Time.FIXED_DT);
        touchPoint(w, 1, 120, 88);
        w.step(Time.FIXED_DT);
        expect(orbit.yaw.get(camera)).toBeCloseTo(yaw - 20 * orbit.sensitivity.get(camera), 6);
        expect(orbit.pitch.get(camera)).toBeCloseTo(pitch - 12 * orbit.sensitivity.get(camera), 6);
        cancelTouch(w, 1);
        const releasedYaw = orbit.yaw.get(camera);
        const releasedPitch = orbit.pitch.get(camera);
        w.step(Time.FIXED_DT);
        w.step(Time.FIXED_DT);
        expect(orbit.yaw.get(camera)).toBe(releasedYaw);
        expect(orbit.pitch.get(camera)).toBe(releasedPitch);
    } finally {
        await app.dispose();
    }
});

for (const mode of [CameraMode.Perspective, CameraMode.Orthographic]) {
    test(`a pinch zooms ${mode === CameraMode.Perspective ? "perspective distance" : "orthographic size"} without rotating`, async () => {
        const { app, w, camera } = await subject(mode);
        try {
            const orbit = w.storage(Orbit);
            const yaw = orbit.yaw.get(camera);
            const pitch = orbit.pitch.get(camera);
            touchPoint(w, 1, 100, 100);
            touchPoint(w, 2, 200, 100);
            w.step(Time.FIXED_DT);
            touchPoint(w, 1, 90, 100);
            touchPoint(w, 2, 210, 100);
            w.step(Time.FIXED_DT);
            if (mode === CameraMode.Perspective) {
                expect(orbit.distance.get(camera)).toBeCloseTo(9.6, 6);
                expect(orbit.size.get(camera)).toBe(5);
            } else {
                expect(orbit.size.get(camera)).toBeCloseTo(4.8, 6);
                expect(w.storage(Camera).size.get(camera)).toBeLessThan(5);
                expect(orbit.distance.get(camera)).toBe(10);
            }
            expect(orbit.yaw.get(camera)).toBe(yaw);
            expect(orbit.pitch.get(camera)).toBe(pitch);
        } finally {
            await app.dispose();
        }
    });
}

test("two-finger drag pans by the centroid without rotating or zooming, and cancellation returns to one-finger rotation", async () => {
    const { app, w, camera } = await subject(CameraMode.Perspective);
    try {
        resizeViewport(w, 0, 320, 200, 1);
        focusCanvas(w, 0);
        const orbit = w.storage(Orbit);
        const yaw = orbit.yaw.get(camera);
        const pitch = orbit.pitch.get(camera);
        touchPoint(w, 1, 100, 100);
        touchPoint(w, 2, 200, 100);
        w.step(Time.FIXED_DT);
        touchPoint(w, 1, 120, 110);
        touchPoint(w, 2, 220, 110);
        w.step(Time.FIXED_DT);
        const scale = (2 * 10 * Math.tan(Math.PI / 6)) / 200;
        expect(orbit.pan.x.get(camera)).toBeCloseTo(
            scale * (-10 * Math.sin(pitch) * Math.sin(yaw) - 20 * Math.cos(yaw)),
            6,
        );
        expect(orbit.pan.y.get(camera)).toBeCloseTo(scale * 10 * Math.cos(pitch), 6);
        expect(orbit.pan.z.get(camera)).toBeCloseTo(
            scale * (-10 * Math.sin(pitch) * Math.cos(yaw) + 20 * Math.sin(yaw)),
            6,
        );
        expect(orbit.yaw.get(camera)).toBe(yaw);
        expect(orbit.pitch.get(camera)).toBe(pitch);
        expect(orbit.distance.get(camera)).toBe(10);
        const pan = [orbit.pan.x.get(camera), orbit.pan.y.get(camera), orbit.pan.z.get(camera)];
        cancelTouch(w, 1);
        touchPoint(w, 2, 230, 110);
        w.step(Time.FIXED_DT);
        expect(orbit.yaw.get(camera)).toBeCloseTo(yaw - 10 * orbit.sensitivity.get(camera), 6);
        expect([orbit.pan.x.get(camera), orbit.pan.y.get(camera), orbit.pan.z.get(camera)]).toEqual(
            pan,
        );
    } finally {
        await app.dispose();
    }
});

for (const flyButton of [0, 1, 2]) {
    test(`touch never engages fly with flyButton=${flyButton}, even with its mouse button and movement keys held`, async () => {
        const { app, w, camera } = await subject(CameraMode.Perspective);
        try {
            const orbit = w.storage(Orbit);
            orbit.flyButton.set(camera, flyButton);
            orbit.orbitButton.set(camera, (flyButton + 1) % 3);
            orbit.panButton.set(camera, (flyButton + 1) % 3);
            touchPoint(w, 1, 100, 100);
            pressButton(w, (["left", "middle", "right"] as const)[flyButton]!);
            pressKey(w, "KeyW");
            pressKey(w, "KeyE");
            pressKey(w, "ShiftLeft");
            const initial = pose(w, camera);
            w.step(Time.FIXED_DT);
            expect(pose(w, camera)).toEqual(initial);
            expect(orbit.flySpeed.get(camera)).toBe(5);
        } finally {
            await app.dispose();
        }
    });
}

test("orthographic fly-button drag uses orbit sensitivity and smoothing, ignores movement keys, and scroll zooms size without changing fly speed", async () => {
    const orbit = await subject(CameraMode.Orthographic, "left");
    const fly = await subject(CameraMode.Orthographic, "right");
    try {
        const initial = pose(fly.w, fly.camera);
        for (const code of ["KeyW", "KeyA", "KeyQ", "ShiftLeft"]) pressKey(fly.w, code);
        fly.w.step(Time.FIXED_DT);
        expect(pose(fly.w, fly.camera)).toEqual(initial);
        for (const s of [orbit, fly]) {
            pointerMove(s.w, { x: 100, y: 80, deltaX: 30, deltaY: -12 });
            pointerWheel(s.w, 100);
            s.w.step(Time.FIXED_DT);
        }
        expect(pose(fly.w, fly.camera)).toEqual(pose(orbit.w, orbit.camera));
        expect(pose(fly.w, fly.camera)).not.toEqual(initial);
        expect(fly.w.storage(Camera).size.get(fly.camera)).toBeGreaterThan(5);
        expect(fly.w.storage(Camera).size.get(fly.camera)).toBe(
            orbit.w.storage(Camera).size.get(orbit.camera),
        );
        expect(fly.w.storage(Orbit).flySpeed.get(fly.camera)).toBe(5);
    } finally {
        await orbit.app.dispose();
        await fly.app.dispose();
    }
});

test("Locked orthographic fly-button drag leaves look unchanged while scroll still zooms", async () => {
    const { app, w, camera } = await subject(CameraMode.Orthographic, "right", true);
    try {
        const initial = pose(w, camera);
        pointerMove(w, { x: 100, y: 80, deltaX: 30, deltaY: -12 });
        pointerWheel(w, 100);
        pressKey(w, "KeyW");
        w.step(Time.FIXED_DT);
        expect(pose(w, camera)).toEqual(initial);
        expect(w.storage(Camera).size.get(camera)).toBeGreaterThan(5);
    } finally {
        await app.dispose();
    }
});

test("perspective fly follows the recorded mouse, WASD/QE, Shift and wheel path", async () => {
    const { app, w, camera } = await subject(CameraMode.Perspective, "right");
    try {
        const frames = [
            { dx: 23, dy: -9, scroll: 100, keys: ["KeyW", "KeyD", "KeyE", "ShiftLeft"] },
            { dx: -14, dy: 7, scroll: -200, keys: ["KeyS", "KeyA", "KeyQ"] },
            { dx: 5, dy: 3, scroll: 40, keys: ["KeyW", "KeyE", "ShiftRight"] },
        ];
        for (const frame of frames) {
            for (const code of frame.keys) pressKey(w, code);
            pointerMove(w, { x: 100, y: 80, deltaX: frame.dx, deltaY: frame.dy });
            pointerWheel(w, frame.scroll);
            w.step(Time.FIXED_DT);
            for (const code of frame.keys) releaseKey(w, code);
        }
        expect(pose(w, camera)).toEqual([
            4.630765438079834, 3.630070686340332, 7.838225364685059, -0.16749423742294312,
            0.2366694211959839, 0.04145892336964607, 0.9561455845832825,
        ]);
    } finally {
        await app.dispose();
    }
});

test("the public Orbit consumer consumes held, released and neutral pointer facts to produce a sensitivity-scaled camera pose without a canvas, browser producer or renderer", async () => {
    const app = await createApp({ defaults: false, plugins: [OrbitPlugin] });
    try {
        const world = app.world;
        const _devices = world.resource(Devices);
        const camera = world.create();
        world.add(camera, Transform);
        world.add(camera, Orbit);
        world.storage(Orbit).sensitivity.set(camera, 0.01);
        resizeViewport(world, 0, 320, 180, 2);
        focusCanvas(world, 0);

        world.step(0); // initializes OrbitSmooth and produces the initial pose
        const initialYaw = world.storage(Orbit).yaw.get(camera);
        const initialX = world.storage(Transform).translation.x.get(camera);
        const initialZ = world.storage(Transform).translation.z.get(camera);

        pressButton(world, "left");
        pointerMove(world, {
            x: 160,
            y: 90,
            deltaX: 20,
            deltaY: -6,
            canvasIndex: 0,
        });
        world.step(Time.FIXED_DT);
        if (Math.abs(world.storage(Orbit).yaw.get(camera) - (initialYaw - 20 * 0.01)) > 0.000001)
            throw new Error("Orbit did not consume the held drag at its sensitivity");
        if (
            world.storage(Transform).translation.x.get(camera) === initialX &&
            world.storage(Transform).translation.z.get(camera) === initialZ
        )
            throw new Error("Orbit did not produce a camera pose from the supplied drag");
        if (!_devices.pointer.left) throw new Error("Orbit lost the held button fact");

        releaseButton(world, "left");
        world.step(Time.FIXED_DT);
        const releasedYaw = world.storage(Orbit).yaw.get(camera);
        if (_devices.pointer.left) throw new Error("Orbit retained a released button");
        world.step(Time.FIXED_DT); // neutral: no stale drag delta may be replayed
        if (world.storage(Orbit).yaw.get(camera) !== releasedYaw)
            throw new Error("Orbit replayed released drag input on a neutral step");
        if (_devices.keys.released.size !== 0)
            throw new Error("unrelated keyboard release state leaked into Orbit");
    } finally {
        await app.dispose();
    }
});
