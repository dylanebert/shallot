import { expect, test } from "bun:test";
import {
    Camera,
    CameraMode,
    cancelTouch,
    createApp,
    focusCanvas,
    Orbit,
    OrbitMode,
    OrbitPlugin,
    pointerMove,
    pressButton,
    pressKey,
    releaseButton,
    releaseKey,
    resizeViewport,
    setInputEnabled,
    Time,
    Transform,
    touchPoint,
} from "@dylanebert/shallot";

async function subject(mode = CameraMode.Perspective as number) {
    const app = await createApp({
        defaults: false,
        plugins: [OrbitPlugin, { name: "CameraData", components: [Camera] }],
    });
    const w = app.world;
    const eid = w.create();
    w.add(eid, Transform);
    w.add(eid, Camera, { mode });
    w.add(eid, Orbit, {
        rotateSpeed: 1,
        orbitButton: 0,
        panButton: 2,
        flyButton: 255,
        zoomButton: 1,
        swapPanOrbit: 1,
    });
    resizeViewport(w, 0, 320, 200, 1);
    focusCanvas(w, 0);
    w.step(0);
    return { app, w, eid, o: w.storage(Orbit) };
}

for (const mode of [CameraMode.Perspective, CameraMode.Orthographic]) {
    test(`zoom drag shares zoom scale and clamps in projection ${mode}`, async () => {
        const { app, w, eid, o } = await subject(mode);
        try {
            const yaw = o.yaw.get(eid);
            const pitch = o.pitch.get(eid);
            pressButton(w, 1);
            pointerMove(w, 10, 20, 10, 20);
            w.step(Time.FIXED_DT);
            expect(o.yaw.get(eid)).toBe(yaw);
            expect(o.pitch.get(eid)).toBe(pitch);
            expect(o.pan.x.get(eid)).toBe(0);
            expect(o.pan.y.get(eid)).toBe(0);
            expect(o.pan.z.get(eid)).toBe(0);
            expect(o.distance.get(eid)).toBeCloseTo(mode === CameraMode.Perspective ? 10.4 : 10, 5);
            expect(o.size.get(eid)).toBeCloseTo(mode === CameraMode.Orthographic ? 5.2 : 5, 5);
            pointerMove(w, 10, 1e6, 0, 1e6);
            w.step(Time.FIXED_DT);
            expect((mode === CameraMode.Perspective ? o.distance : o.size).get(eid)).toBe(900);
            releaseButton(w, 1);
            pointerMove(w, 20, 30, 10, 10);
            w.step(Time.FIXED_DT);
            expect((mode === CameraMode.Perspective ? o.distance : o.size).get(eid)).toBe(900);
        } finally {
            await app.dispose();
        }
    });
}

for (const modifier of ["shiftKey", "ctrlKey", "metaKey"] as const) {
    for (const button of [0, 2] as const) {
        test(`${modifier} swaps button ${button} at press and remains latched`, async () => {
            const { app, w, eid, o } = await subject();
            try {
                const yaw = o.yaw.get(eid);
                pressButton(w, button, {
                    shiftKey: false,
                    ctrlKey: false,
                    metaKey: false,
                    altKey: false,
                    [modifier]: true,
                });
                pointerMove(w, 20, 0, 20, 0);
                w.step(Time.FIXED_DT);
                if (button === 0) {
                    expect(o.yaw.get(eid)).toBe(yaw);
                    expect(o.pan.x.get(eid)).not.toBe(0);
                } else expect(o.yaw.get(eid)).toBeCloseTo(yaw - 0.1, 6);
                pressKey(w, "ShiftLeft");
                releaseKey(w, "ShiftLeft");
                pressKey(w, "ControlLeft");
                pointerMove(w, 40, 0, 20, 0);
                w.step(Time.FIXED_DT);
                expect(o.yaw.get(eid)).toBeCloseTo(button === 0 ? yaw : yaw - 0.2, 6);
                releaseButton(w, button);
                const stopped = o.yaw.get(eid);
                const pan = o.pan.x.get(eid);
                pointerMove(w, 60, 0, 20, 0);
                w.step(Time.FIXED_DT);
                expect(o.yaw.get(eid)).toBe(stopped);
                expect(o.pan.x.get(eid)).toBe(pan);
                pressButton(w, button);
                pointerMove(w, 80, 0, 20, 0);
                w.step(Time.FIXED_DT);
                if (button === 0) expect(o.yaw.get(eid)).toBeCloseTo(stopped - 0.1, 6);
                else expect(o.yaw.get(eid)).toBe(stopped);
            } finally {
                await app.dispose();
            }
        });
    }
}

test("plain left rotates and right pans without fly, including Locked pan", async () => {
    const { app, w, eid, o } = await subject();
    try {
        const yaw = o.yaw.get(eid);
        pressButton(w, 0);
        pointerMove(w, 20, 0, 20, 0);
        w.step(Time.FIXED_DT);
        expect(o.yaw.get(eid)).toBeCloseTo(yaw - 0.1, 6);
        releaseButton(w, 0);
        for (const mode of [OrbitMode.Free, OrbitMode.Locked]) {
            o.mode.set(eid, mode);
            const pan = o.pan.x.get(eid);
            const angle = o.yaw.get(eid);
            pressButton(w, 2);
            pressKey(w, "KeyW");
            pointerMove(w, 40, 20, 20, 20);
            w.step(Time.FIXED_DT);
            expect(o.pan.x.get(eid)).not.toBe(pan);
            expect(o.yaw.get(eid)).toBe(angle);
            expect(o.distance.get(eid)).toBe(10);
            releaseButton(w, 2);
            releaseKey(w, "KeyW");
        }
    } finally {
        await app.dispose();
    }
});

test("touch ignores a swapped mouse press and cancellation cannot replay it", async () => {
    const { app, w, eid, o } = await subject();
    try {
        pressButton(w, 0, { shiftKey: true, ctrlKey: false, metaKey: false, altKey: false });
        w.step(Time.FIXED_DT);
        releaseButton(w, 0);
        touchPoint(w, 1, 100, 100);
        w.step(Time.FIXED_DT);
        const yaw = o.yaw.get(eid);
        touchPoint(w, 1, 120, 100);
        w.step(Time.FIXED_DT);
        expect(o.yaw.get(eid)).toBeCloseTo(yaw - 0.1, 6);
        expect(o.pan.x.get(eid)).toBe(0);
        cancelTouch(w, 1);
        pointerMove(w, 140, 100, 20, 0);
        w.step(Time.FIXED_DT);
        expect(o.yaw.get(eid)).toBeCloseTo(yaw - 0.1, 6);
    } finally {
        await app.dispose();
    }
});

test("press modifiers are per world and remain device facts across recycled camera rows", async () => {
    const first = await subject();
    const second = await subject();
    try {
        pressButton(first.w, 0, { shiftKey: true, ctrlKey: false, metaKey: false, altKey: false });
        first.w.step(Time.FIXED_DT);
        pressButton(second.w, 0);
        pointerMove(second.w, 20, 0, 20, 0);
        second.w.step(Time.FIXED_DT);
        expect(second.o.pan.x.get(second.eid)).toBe(0);
        expect(second.o.yaw.get(second.eid)).toBeCloseTo(Math.PI / 6 - 0.1, 6);
        first.w.destroy(first.eid);
        first.w.step(Time.FIXED_DT);
        const recycled = first.w.create();
        expect(recycled).toBe(first.eid);
        first.w.add(recycled, Transform);
        first.w.add(recycled, Orbit, { swapPanOrbit: 1 });
        first.w.step(0);
        pointerMove(first.w, 20, 0, 20, 0);
        first.w.step(Time.FIXED_DT);
        expect(first.o.pan.x.get(recycled)).not.toBe(0);
        expect(first.o.yaw.get(recycled)).toBeCloseTo(Math.PI / 6, 6);
    } finally {
        await first.app.dispose();
        await second.app.dispose();
    }
});

test("disabled buttons do not alias right; Locked retains zoom and pan; Manual ignores devices", async () => {
    const { app, w, eid, o } = await subject();
    try {
        o.orbitButton.set(eid, 255);
        o.panButton.set(eid, 255);
        o.zoomButton.set(eid, 255);
        const yaw = o.yaw.get(eid);
        pressButton(w, 2);
        pointerMove(w, 20, 20, 20, 20);
        w.step(Time.FIXED_DT);
        expect(o.yaw.get(eid)).toBe(yaw);
        expect(o.pan.x.get(eid)).toBe(0);
        expect(o.distance.get(eid)).toBe(10);
        releaseButton(w, 2);
        o.zoomButton.set(eid, 1);
        o.mode.set(eid, OrbitMode.Locked);
        pressButton(w, 1);
        pointerMove(w, 20, 40, 0, 20);
        w.step(Time.FIXED_DT);
        expect(o.distance.get(eid)).toBeCloseTo(10.4, 5);
        o.mode.set(eid, OrbitMode.Manual);
        pointerMove(w, 20, 60, 0, 20);
        w.step(Time.FIXED_DT);
        expect(o.distance.get(eid)).toBeCloseTo(10.4, 5);
        setInputEnabled(w, false);
        setInputEnabled(w, true);
        o.mode.set(eid, OrbitMode.Free);
        pointerMove(w, 20, 80, 0, 20);
        w.step(Time.FIXED_DT);
        expect(o.distance.get(eid)).toBeCloseTo(10.4, 5);
    } finally {
        await app.dispose();
    }
});
