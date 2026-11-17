import { expect, test } from "bun:test";
import {
    Camera,
    CameraMode,
    cancelTouch,
    createApp,
    Devices,
    focusCanvas,
    Orbit,
    OrbitPlugin,
    pointerMove,
    pointerWheel,
    pressButton,
    releaseButton,
    resizeViewport,
    setInputEnabled,
    Transform,
    touchPoint,
} from "@dylanebert/shallot";

async function subject(options = {}) {
    const app = await createApp({ defaults: false, plugins: [OrbitPlugin] });
    const w = app.world;
    const eid = w.create();
    w.add(eid, Transform);
    w.add(eid, Orbit, { yaw: 0, pitch: 0, rotateSpeed: 3.35, ...options });
    resizeViewport(w, 0, 880, 670, 1);
    focusCanvas(w);
    w.step(0);
    return { app, w, eid, o: w.storage(Orbit), t: w.storage(Transform) };
}

for (const ortho of [false, true])
    for (const minimum of [false, true]) {
        test(`proportional pinch clamps each delivered move before reversal: ortho=${ortho}, minimum=${minimum}`, async () => {
            for (const split of [false, true]) {
                const { app, w, eid, o, t } = await subject({
                    distance: 10,
                    size: 10,
                    minDistance: 5,
                    minSize: 5,
                    maxDistance: 15,
                    maxSize: 15,
                    proportionalPinch: 1,
                    zoomSmoothness: 1,
                });
                try {
                    w.add(eid, Camera, {
                        mode: ortho ? CameraMode.Orthographic : CameraMode.Perspective,
                    });
                    w.step(1 / 60);
                    touchPoint(w, 1, 0, 0);
                    touchPoint(w, 2, 100, 0);
                    touchPoint(w, 2, minimum ? 300 : 100 / 3, 0);
                    if (split) w.step(1 / 60);
                    touchPoint(w, 2, 100, 0);
                    w.step(1 / 60);
                    // Spread: max(5,10/3)*3=15. Contract: min(15,10*3)/3=5.
                    const expected = minimum ? 15 : 5;
                    expect((ortho ? o.size : o.distance).get(eid)).toBeCloseTo(expected, 5);
                    expect(
                        ortho ? w.storage(Camera).size.get(eid) : t.translation.z.get(eid),
                    ).toBeCloseTo(expected, 5);
                } finally {
                    await app.dispose();
                }
            }
        });
    }

for (const cancel of [false, true])
    for (const split of [false, true]) {
        test(`proportional pinch composes spans and survives lift: cancel=${cancel}, split=${split}`, async () => {
            for (const span of [100, 200]) {
                const { app, w, eid, o, t } = await subject({
                    proportionalPinch: 1,
                    zoomSmoothness: 1,
                });
                try {
                    touchPoint(w, 1, 0, 0);
                    touchPoint(w, 2, span, 0);
                    touchPoint(w, 2, span * 1.1, 0);
                    if (split) w.step(1 / 60);
                    touchPoint(w, 2, span * 1.2, 0);
                    if (cancel) {
                        cancelTouch(w, 1);
                        cancelTouch(w, 2);
                    } else {
                        touchPoint(w, 1, 0, 0, false);
                        touchPoint(w, 2, span * 1.2, 0, false);
                    }
                    w.step(1 / 60);
                    expect(o.distance.get(eid)).toBeCloseTo(10 / 1.2, 5);
                    expect(t.translation.z.get(eid)).toBeCloseTo(10 / 1.2, 5);
                    for (const frames of [1, 5, 234]) {
                        for (let i = 0; i < frames; i++) w.step(1 / 60);
                        expect(o.distance.get(eid)).toBeCloseTo(10 / 1.2, 5);
                        expect(t.translation.z.get(eid)).toBeCloseTo(10 / 1.2, 5);
                    }
                } finally {
                    await app.dispose();
                }
            }
        });

        test(`pan input decays into actual target and pose: cancel=${cancel}, split=${split}`, async () => {
            const { app, w, eid, o, t } = await subject({ panSmoothness: 0.05 });
            try {
                touchPoint(w, 1, 0, 0);
                touchPoint(w, 2, 100, 0);
                touchPoint(w, 1, 20, 6);
                touchPoint(w, 2, 120, 6);
                let n = 0;
                if (split) {
                    w.step(1 / 60);
                    n++;
                }
                if (cancel) {
                    cancelTouch(w, 1);
                    cancelTouch(w, 2);
                } else {
                    touchPoint(w, 1, 20, 6, false);
                    touchPoint(w, 2, 120, 6, false);
                }
                const totalX = (-20 * 20 * Math.tan(Math.PI / 6)) / 670;
                const totalY = (6 * 20 * Math.tan(Math.PI / 6)) / 670;
                for (const frames of [1, 1, 5, 233]) {
                    for (let i = 0; i < frames; i++) {
                        w.step(1 / 60);
                        n++;
                    }
                    const fraction = 1 - 0.95 ** n;
                    expect(o.pan.x.get(eid)).toBeCloseTo(totalX * fraction, 5);
                    expect(o.pan.y.get(eid)).toBeCloseTo(totalY * fraction, 5);
                    expect(t.translation.x.get(eid)).toBeCloseTo(totalX * fraction, 5);
                    expect(t.translation.y.get(eid)).toBeCloseTo(totalY * fraction, 5);
                }
            } finally {
                await app.dispose();
            }
        });
    }

for (const cancel of [false, true])
    for (const resume of [0, 1]) {
        test(`touch continuation is independent of pinch policy: cancel=${cancel}, resume=${resume}`, async () => {
            for (const proportionalPinch of [0, 1]) {
                const { app, w, eid, o } = await subject({
                    touchResume: resume,
                    proportionalPinch,
                });
                try {
                    touchPoint(w, 1, 0, 0);
                    touchPoint(w, 2, 100, 0);
                    if (cancel) cancelTouch(w, 2);
                    else touchPoint(w, 2, 100, 0, false);
                    touchPoint(w, 1, 30, 12);
                    w.step(1 / 60);
                    expect(o.yaw.get(eid)).toBeCloseTo(resume ? -0.15 : 0, 6);
                    touchPoint(w, 1, 60, 24);
                    w.step(1 / 60);
                    expect(o.yaw.get(eid)).toBeCloseTo(resume ? -0.3 : 0, 6);
                    touchPoint(w, 3, 160, 24);
                    touchPoint(w, 3, 180, 24);
                    w.step(1 / 60);
                    expect(o.distance.get(eid)).toBeLessThan(10);
                } finally {
                    await app.dispose();
                }
            }
        });
    }

for (const ortho of [false, true])
    for (const source of ["wheel", "pinch", "drag"]) {
        test(`configured input zoom snaps with no released tail: ortho=${ortho}, source=${source}`, async () => {
            const { app, w, eid, o, t } = await subject({
                zoomSmoothness: 1,
                smoothness: 0.05,
                zoomButton: 1,
                panButton: 2,
                proportionalPinch: 1,
            });
            try {
                w.add(eid, Camera, {
                    mode: ortho ? CameraMode.Orthographic : CameraMode.Perspective,
                });
                w.step(1 / 60);
                if (source === "wheel") pointerWheel(w, 20);
                if (source === "drag") {
                    pressButton(w, 1);
                    pointerMove(w, 0, 20, 0, 20);
                    releaseButton(w, 1);
                }
                if (source === "pinch") {
                    touchPoint(w, 1, 0, 0);
                    touchPoint(w, 2, 100, 0);
                    touchPoint(w, 2, 120, 0);
                    cancelTouch(w, 1);
                    cancelTouch(w, 2);
                }
                const expected = source === "pinch" ? (ortho ? 5 : 10) / 1.2 : ortho ? 5.2 : 10.4;
                for (const frames of [1, 1, 6, 240]) {
                    for (let i = 0; i < frames; i++) w.step(1 / 60);
                    expect((ortho ? o.size : o.distance).get(eid)).toBeCloseTo(expected, 5);
                    expect(
                        ortho ? w.storage(Camera).size.get(eid) : t.translation.z.get(eid),
                    ).toBeCloseTo(expected, 5);
                }
                // An authored target change is not an input zoom response.
                if (ortho) o.size.set(eid, expected + 2);
                else o.distance.set(eid, expected + 2);
                w.step(1 / 60);
                expect(
                    ortho ? w.storage(Camera).size.get(eid) : t.translation.z.get(eid),
                ).toBeCloseTo(expected + 0.1, 5);
            } finally {
                await app.dispose();
            }
        });
    }

for (const ortho of [false, true]) {
    test(`default zoom inherits global easing; inactive authored projection is untouched: ortho=${ortho}`, async () => {
        const { app, w, eid, o, t } = await subject({ smoothness: 0.05, proportionalPinch: 1 });
        try {
            w.add(eid, Camera, { mode: ortho ? CameraMode.Orthographic : CameraMode.Perspective });
            // Out-of-range authored values are not silently clamped without zoom input.
            if (ortho) o.distance.set(eid, 1000);
            else o.size.set(eid, 1000);
            w.step(1 / 60);
            expect(ortho ? o.distance.get(eid) : o.size.get(eid)).toBe(1000);
            pointerWheel(w, 20);
            for (let n = 1; n <= 7; n++) {
                w.step(1 / 60);
                const expected = ortho ? 5 + 0.2 * (1 - 0.95 ** n) : 10 + 0.4 * (1 - 0.95 ** n);
                expect(
                    ortho ? w.storage(Camera).size.get(eid) : t.translation.z.get(eid),
                ).toBeCloseTo(expected, 5);
            }
            expect(ortho ? o.distance.get(eid) : o.size.get(eid)).toBe(1000);
        } finally {
            await app.dispose();
        }
    });
}

for (const ortho of [false, true]) {
    test(`zoom override separates simultaneous authored and input changes: ortho=${ortho}`, async () => {
        const { app, w, eid, o, t } = await subject({ smoothness: 0.05, zoomSmoothness: 1 });
        try {
            w.add(eid, Camera, { mode: ortho ? CameraMode.Orthographic : CameraMode.Perspective });
            w.step(1 / 60);
            const base = ortho ? 5 : 10;
            (ortho ? o.size : o.distance).set(eid, base + 2);
            pointerWheel(w, 20);
            const input = (base + 2) * 0.04;
            for (let n = 1; n <= 7; n++) {
                w.step(1 / 60);
                expect(
                    ortho ? w.storage(Camera).size.get(eid) : t.translation.z.get(eid),
                ).toBeCloseTo(base + input + 2 * (1 - 0.95 ** n), 5);
            }
        } finally {
            await app.dispose();
        }
    });
}

test("ratio baselines are finite and pair-local; pooled facts expire", async () => {
    const { app, w, eid, o } = await subject({ proportionalPinch: 1 });
    try {
        touchPoint(w, 1, 0, 0);
        touchPoint(w, 2, 0, 0);
        touchPoint(w, 2, 100, 0); // coincident baseline is neutral
        w.step(1 / 60);
        expect(o.distance.get(eid)).toBe(10);
        const events = w.resource(Devices).touch.events;
        expect(events).toHaveLength(0);
        cancelTouch(w, 2);
        touchPoint(w, 3, 200, 0);
        touchPoint(w, 3, 240, 0);
        const fact = events[0];
        w.step(1 / 60);
        expect(o.distance.get(eid)).toBeCloseTo(10 / 1.2, 5);
        cancelTouch(w, 3);
        expect(events[0]).toBe(fact);
        w.step(1 / 60);
        expect(o.distance.get(eid)).toBeCloseTo(10 / 1.2, 5);
    } finally {
        await app.dispose();
    }
});

test("authored pan, suspension and camera replacement discard old pan impulse", async () => {
    const { app, w, eid, o, t } = await subject({ panSmoothness: 0.05 });
    try {
        const deliver = () => {
            touchPoint(w, 1, 0, 0);
            touchPoint(w, 2, 100, 0);
            touchPoint(w, 1, 20, 0);
            touchPoint(w, 2, 120, 0);
            cancelTouch(w, 1);
            cancelTouch(w, 2);
            w.step(1 / 60);
        };
        deliver();
        o.pan.x.set(eid, 3);
        w.step(1 / 60);
        expect(o.pan.x.get(eid)).toBe(3);
        expect(t.translation.x.get(eid)).toBe(3);
        deliver();
        const pivot = w.create();
        w.add(pivot, Transform, { translation: [10, 0, 0] });
        o.target.set(eid, pivot);
        const beforeTarget = o.pan.x.get(eid);
        w.step(1 / 60);
        expect(o.pan.x.get(eid)).toBe(beforeTarget);
        expect(t.translation.x.get(eid)).toBeCloseTo(10 + beforeTarget, 5);
        t.translation.x.set(pivot, 20);
        w.step(1 / 60);
        expect(o.pan.x.get(eid)).toBe(beforeTarget);
        expect(t.translation.x.get(eid)).toBeCloseTo(20 + beforeTarget, 5);
        o.target.set(eid, 0);
        deliver();
        setInputEnabled(w, false);
        w.step(1 / 60);
        const stopped = o.pan.x.get(eid);
        setInputEnabled(w, true);
        w.step(1 / 60);
        expect(o.pan.x.get(eid)).toBe(stopped);
        deliver();
        w.remove(eid, Orbit);
        w.step(1 / 60);
        w.add(eid, Orbit, { yaw: 0, pitch: 0, pan: [7, 0, 0], panSmoothness: 0.05 });
        w.step(1 / 60);
        expect(o.pan.x.get(eid)).toBe(7);
        expect(t.translation.x.get(eid)).toBe(7);
        const snapshot = w.snapshot();
        deliver();
        w.restore(snapshot);
        w.step(1 / 60);
        expect(o.pan.x.get(eid)).toBe(7);
        expect(t.translation.x.get(eid)).toBe(7);
        deliver();
        w.destroy(eid);
        const replacement = w.create();
        w.add(replacement, Transform);
        w.add(replacement, Orbit, { yaw: 0, pitch: 0, pan: [9, 0, 0], panSmoothness: 0.05 });
        w.step(1 / 60);
        expect(o.pan.x.get(replacement)).toBe(9);
        expect(t.translation.x.get(replacement)).toBe(9);
    } finally {
        await app.dispose();
    }
});
