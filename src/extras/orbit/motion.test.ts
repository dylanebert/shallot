import { expect, test } from "bun:test";
import {
    blurCanvas,
    cancelTouch,
    createApp,
    Devices,
    focusCanvas,
    Orbit,
    OrbitPlugin,
    pointerLockChanged,
    pointerMove,
    pointerWheel,
    pressButton,
    releaseButton,
    requirePointerLock,
    resizeViewport,
    setInputEnabled,
    Transform,
    touchPoint,
} from "@dylanebert/shallot";

test("stationary touch after released mouse motion does not replay mouse displacement", async () => {
    const { app, w, eid, o } = await subject();
    try {
        pressButton(w, 0);
        pointerMove(w, 30, 12, 30, 12);
        releaseButton(w, 0);
        touchPoint(w, 1, 30, 12);
        w.step(1 / 60);
        expect(o.yaw.get(eid)).toBeCloseTo(-0.15, 6);
        expect(o.pitch.get(eid)).toBeCloseTo(0.06, 6);
        touchPoint(w, 1, 60, 24);
        w.step(1 / 60);
        expect(o.yaw.get(eid)).toBeCloseTo(-0.3, 6);
        expect(o.pitch.get(eid)).toBeCloseTo(0.12, 6);
    } finally {
        await app.dispose();
    }
});

test("released pan moves the actual fly pose and survives fly-exit target reanchoring", async () => {
    const { app, w, eid, o } = await subject();
    try {
        o.panButton.set(eid, 1);
        o.zoomButton.set(eid, 255);
        o.flyButton.set(eid, 2);
        const transform = w.storage(Transform);
        const before = [
            transform.translation.x.get(eid),
            transform.translation.y.get(eid),
            transform.translation.z.get(eid),
        ];
        pressButton(w, 1);
        pointerMove(w, 30, 12, 30, 12);
        releaseButton(w, 1);
        pressButton(w, 2);
        w.step(1 / 60);
        const pan = [o.pan.x.get(eid), o.pan.y.get(eid), o.pan.z.get(eid)];
        expect(pan[0]).not.toBe(0);
        expect(pan[1]).not.toBe(0);
        const flyPose = [
            transform.translation.x.get(eid),
            transform.translation.y.get(eid),
            transform.translation.z.get(eid),
        ];
        for (let i = 0; i < 3; i++) expect(flyPose[i]! - before[i]!).toBeCloseTo(pan[i]!, 6);
        releaseButton(w, 2);
        w.step(1 / 60);
        const target = [o.pan.x.get(eid), o.pan.y.get(eid), o.pan.z.get(eid)];
        const exitPose = [
            transform.translation.x.get(eid),
            transform.translation.y.get(eid),
            transform.translation.z.get(eid),
        ];
        for (let i = 0; i < 3; i++) {
            expect(target[i]).toBeCloseTo(pan[i]!, 6);
            expect(exitPose[i]).toBeCloseTo(flyPose[i]!, 6);
        }
    } finally {
        await app.dispose();
    }
});

for (const cancel of [false, true]) {
    for (const split of [false, true]) {
        test(`moving touch retains event-time totals: cancel=${cancel}, split=${split}`, async () => {
            const { app, w, eid, o } = await subject();
            try {
                touchPoint(w, 1, 0, 0);
                touchPoint(w, 2, 100, 0);
                touchPoint(w, 1, 10, 6);
                touchPoint(w, 2, 130, 6);
                // Centroid (20,6), distance 100 -> 120, no single-contact motion.
                if (split) w.step(1 / 60);
                if (cancel) {
                    cancelTouch(w, 1);
                    cancelTouch(w, 2);
                } else {
                    touchPoint(w, 1, 10, 6, false);
                    touchPoint(w, 2, 130, 6, false);
                }
                w.step(1 / 60);
                expect(o.yaw.get(eid)).toBe(0);
                expect(o.pitch.get(eid)).toBe(0);
                const scale = (20 * Math.tan(Math.PI / 6)) / 670;
                expect(o.pan.x.get(eid)).toBeCloseTo(-20 * scale, 6);
                expect(o.pan.y.get(eid)).toBeCloseTo(6 * scale, 6);
                expect(o.distance.get(eid)).toBeCloseTo(9.6, 5);
                w.step(1 / 60);
                expect(o.distance.get(eid)).toBeCloseTo(9.6, 5);
            } finally {
                await app.dispose();
            }
        });
    }
}

test("single then two contacts retain separate motion after both release", async () => {
    const { app, w, eid, o } = await subject();
    try {
        touchPoint(w, 1, 0, 0);
        touchPoint(w, 1, 30, 12);
        touchPoint(w, 2, 130, 12);
        touchPoint(w, 2, 150, 12);
        touchPoint(w, 1, 30, 12, false);
        touchPoint(w, 2, 150, 12, false);
        w.step(1 / 60);
        expect(o.yaw.get(eid)).toBeCloseTo(-0.15, 6);
        expect(o.pitch.get(eid)).toBeCloseTo(0.06, 6);
        expect(o.pan.x.get(eid)).toBeCloseTo((-10 * 20 * Math.tan(Math.PI / 6)) / 670, 6);
        expect(o.distance.get(eid)).toBeCloseTo(9.6, 5);
        w.step(1 / 60);
        expect(o.yaw.get(eid)).toBeCloseTo(-0.15, 6);
    } finally {
        await app.dispose();
    }
});

test("wheel provenance filters only event-time drag wheel when opted in", async () => {
    for (const ignore of [0, 1]) {
        const { app, w, eid, o } = await subject();
        try {
            o.ignoreDragWheel.set(eid, ignore);
            pointerWheel(w, 20);
            pressButton(w, 0);
            pointerWheel(w, 120);
            releaseButton(w, 0);
            pointerWheel(w, -10);
            expect(w.resource(Devices).pointer.scroll).toBe(130);
            w.step(1 / 60);
            expect(o.distance.get(eid)).toBeCloseTo(ignore ? 10.2 : 12.6, 5);
        } finally {
            await app.dispose();
        }
    }
});

test("wheel filtering sees touch contacts even after all contacts cancel before the frame", async () => {
    const { app, w, eid, o } = await subject();
    try {
        o.ignoreDragWheel.set(eid, 1);
        pointerWheel(w, 20);
        touchPoint(w, 1, 0, 0);
        touchPoint(w, 2, 100, 0);
        pointerWheel(w, 120);
        cancelTouch(w, 1);
        cancelTouch(w, 2);
        pointerWheel(w, -10);
        w.step(1 / 60);
        expect(o.distance.get(eid)).toBeCloseTo(10.2, 5);
    } finally {
        await app.dispose();
    }
});

async function subject() {
    const app = await createApp({ defaults: false, plugins: [OrbitPlugin] });
    const w = app.world;
    const eid = w.create();
    w.add(eid, Transform);
    w.add(eid, Orbit, {
        yaw: 0,
        pitch: 0,
        sensitivity: 0.005,
        orbitButton: 0,
        panButton: 2,
        zoomButton: 1,
        flyButton: 255,
        swapPanOrbit: 1,
    });
    resizeViewport(w, 0, 880, 670, 1);
    focusCanvas(w);
    w.step(0);
    return { app, w, eid, o: w.storage(Orbit) };
}

test("sequential gestures and chords route event-time motion, excluding idle movement", async () => {
    const { app, w, eid, o } = await subject();
    try {
        pointerMove(w, 100, 100, 100, 100);
        pressButton(w, 0, { shiftKey: true, ctrlKey: false, metaKey: false, altKey: false });
        pointerMove(w, 130, 112, 30, 12);
        releaseButton(w, 0);
        pressButton(w, 0);
        pointerMove(w, 160, 124, 30, 12);
        pressButton(w, 1);
        pointerMove(w, 190, 136, 30, 12);
        releaseButton(w, 1);
        releaseButton(w, 0);
        pointerMove(w, 290, 236, 100, 100);
        w.step(1 / 60);
        expect(o.yaw.get(eid)).toBeCloseTo(-0.15, 6);
        expect(o.pitch.get(eid)).toBeCloseTo(0.06, 6);
        expect(o.pan.x.get(eid)).not.toBe(0);
        expect(o.distance.get(eid)).toBeCloseTo(10.24, 5);
        const pose = [o.yaw.get(eid), o.pan.x.get(eid), o.distance.get(eid)];
        w.step(1 / 60);
        expect([o.yaw.get(eid), o.pan.x.get(eid), o.distance.get(eid)]).toEqual(pose);
    } finally {
        await app.dispose();
    }
});

test("released zoom drag survives a later held fly button without driving fly movement from released levels", async () => {
    const { app, w, eid, o } = await subject();
    try {
        o.flyButton.set(eid, 2);
        o.panButton.set(eid, 255);
        pressButton(w, 1);
        pointerMove(w, 30, 12, 30, 12);
        releaseButton(w, 1);
        pressButton(w, 2);
        const speed = o.flySpeed.get(eid);
        w.step(1 / 60);
        expect(o.distance.get(eid)).toBeCloseTo(10.24, 5);
        expect(o.flySpeed.get(eid)).toBe(speed);
        releaseButton(w, 2);
        const yaw = o.yaw.get(eid);
        pressButton(w, 2);
        pointerMove(w, 60, 24, 30, 12);
        releaseButton(w, 2);
        w.step(1 / 60);
        expect(o.yaw.get(eid)).not.toBe(yaw);
    } finally {
        await app.dispose();
    }
});

test("pooled motion preserves aggregates, excludes touch, clears at boundaries and stays world-local", async () => {
    const first = await subject();
    const second = await subject();
    try {
        const { w } = first;
        const p = w.resource(Devices).pointer;
        pressButton(w, 0);
        pointerMove(w, 30, 12, 30, 12);
        releaseButton(w, 0);
        expect(p.motion).toHaveLength(1);
        expect(p.deltaX).toBe(30);
        expect(p.motion[0]).toMatchObject({ buttons: 1, modifiers: 0, deltaX: 30, deltaY: 12 });
        const pooled = p.motion[0];
        expect(second.w.resource(Devices).pointer.motion).toHaveLength(0);
        w.step(1 / 60);
        expect(p.motion).toHaveLength(0);
        pressButton(w, 0);
        pointerMove(w, 60, 24, 30, 12);
        expect(p.motion[0]).toBe(pooled);
        blurCanvas(w);
        expect(p.motion).toHaveLength(0);
        pressButton(w, 0);
        pointerMove(w, 90, 36, 30, 12);
        setInputEnabled(w, false);
        setInputEnabled(w, true);
        expect(p.motion).toHaveLength(0);
        requirePointerLock(w, true);
        pressButton(w, 0);
        pointerMove(w, 120, 48, 30, 12);
        expect(p.motion).toHaveLength(0);
        pointerLockChanged(w, true);
        pressButton(w, 0);
        pointerMove(w, 150, 60, 30, 12);
        expect(p.motion).toHaveLength(1);
        pointerLockChanged(w, false);
        expect(p.motion).toHaveLength(0);
        requirePointerLock(w, false);
        touchPoint(w, 1, 0, 0);
        touchPoint(w, 1, 30, 12);
        touchPoint(w, 1, 30, 12, false);
        expect(p.motion).toHaveLength(0);
        const touch = w.resource(Devices).touch;
        expect([touch.pointerDeltaX, touch.pointerDeltaY]).toEqual([30, 12]);
        pointerMove(w, 180, 72, 30, 12);
        expect(p.motion).toHaveLength(0);
        expect([touch.pointerDeltaX, touch.pointerDeltaY]).toEqual([30, 12]);
        w.step(1 / 60);
        expect([touch.pointerDeltaX, touch.pointerDeltaY]).toEqual([0, 0]);
        touchPoint(w, 2, 0, 0);
        touchPoint(w, 2, 30, 12);
        blurCanvas(w);
        expect([touch.pointerDeltaX, touch.pointerDeltaY]).toEqual([0, 0]);
    } finally {
        await first.app.dispose();
        await second.app.dispose();
    }
});

for (const button of [0, 1] as const) {
    for (const split of [false, true]) {
        test(`button ${button} preserves 30x12 moves, split=${split}, release before frame`, async () => {
            async function run(releaseFirst: boolean) {
                const app = await createApp({ defaults: false, plugins: [OrbitPlugin] });
                const w = app.world;
                const eid = w.create();
                w.add(eid, Transform);
                w.add(eid, Orbit);
                resizeViewport(w, 0, 880, 670, 1);
                focusCanvas(w);
                w.step(0);
                pressButton(w, button);
                pointerMove(w, 30, 12, 30, 12);
                if (split) w.step(1 / 60);
                pointerMove(w, 60, 24, 30, 12);
                if (releaseFirst) releaseButton(w, button);
                w.step(1 / 60);
                if (!releaseFirst) releaseButton(w, button);
                const o = w.storage(Orbit);
                const result = [
                    o.yaw.get(eid),
                    o.pitch.get(eid),
                    o.pan.x.get(eid),
                    o.pan.y.get(eid),
                ];
                w.step(1 / 60);
                expect([
                    o.yaw.get(eid),
                    o.pitch.get(eid),
                    o.pan.x.get(eid),
                    o.pan.y.get(eid),
                ]).toEqual(result);
                await app.dispose();
                return result;
            }
            const held = await run(false);
            const released = await run(true);
            for (let i = 0; i < held.length; i++) expect(released[i]).toBeCloseTo(held[i]!, 6);
        });
    }
}
