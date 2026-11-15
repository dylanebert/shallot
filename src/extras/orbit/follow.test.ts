import { expect, test } from "bun:test";
import {
    Camera,
    createApp,
    GlobalTransform,
    RenderTransformHistory,
    resetInterpolation,
    Time,
    Transform,
    VisualCorrection,
} from "@dylanebert/shallot";
import { Orbit, OrbitPlugin } from "@dylanebert/shallot/extras";
import { GlobalTransformHistory } from "../../core/rendering/global-transform";
import { TransformRuntime } from "../../core/transform";

test("retargeting Orbit across twenty entities retires each old target's history", async () => {
    const app = await createApp({ defaults: false, plugins: [OrbitPlugin] });
    try {
        const world = app.world;
        const targets = Array.from({ length: 20 }, () => {
            const eid = world.create();
            world.add(eid, Transform);
            return eid;
        });
        const camera = world.create();
        world.add(camera, Transform);
        world.add(camera, Camera);
        world.add(camera, Orbit, { yaw: 0, pitch: 0, distance: 6 });
        for (const target of targets) {
            world.storage(Orbit).target.set(camera, target);
            world.step(0);
        }
        const runtime = world.resource(GlobalTransformHistory);
        const history = runtime.cpuHistory;
        expect(targets.filter((eid) => history.has(eid))).toEqual([targets[19]!]);
        expect(runtime.cpuReaders.length).toBe(1);
        expect(runtime.cpuRequests.size).toBe(1);
        world.remove(camera, Orbit);
        expect(targets.some((eid) => history.has(eid))).toBe(false);
        expect(runtime.cpuReaders.length).toBe(0);
        expect(runtime.cpuRequests.size).toBe(0);
    } finally {
        app.dispose();
    }
});

test("Orbit cameras share target samples but not request or explicit-enrollment lifetimes", async () => {
    const app = await createApp({ defaults: false, plugins: [OrbitPlugin] });
    try {
        const world = app.world;
        const target = world.create();
        world.add(target, Transform);
        const cameras = Array.from({ length: 2 }, () => {
            const camera = world.create();
            world.add(camera, Transform);
            world.add(camera, Camera);
            world.add(camera, Orbit, { target });
            return camera;
        });
        world.step(0);
        const runtime = world.resource(GlobalTransformHistory);
        expect(runtime.cpuRequests.get(target)!.count).toBe(2);
        expect(world.has(target, RenderTransformHistory)).toBe(false);
        world.destroy(cameras[0]!);
        expect(runtime.cpuRequests.get(target)!.count).toBe(1);
        world.add(target, RenderTransformHistory);
        world.storage(Orbit).target.set(cameras[1]!, 0);
        world.step(0);
        expect(runtime.cpuRequests.has(target)).toBe(false);
        expect(runtime.cpuHistory.has(target)).toBe(true);
        world.remove(target, RenderTransformHistory);
        expect(runtime.cpuHistory.has(target)).toBe(false);
    } finally {
        app.dispose();
    }
});

test("Orbit restores request ownership without retaining the abandoned target", async () => {
    const app = await createApp({ defaults: false, plugins: [OrbitPlugin] });
    try {
        const world = app.world;
        const a = world.create(),
            b = world.create(),
            camera = world.create();
        world.add(a, Transform);
        world.add(b, Transform);
        world.add(camera, Transform);
        world.add(camera, Camera);
        world.add(camera, Orbit, { target: a });
        world.tick();
        world.step(0);
        const snapshot = world.snapshot();
        world.storage(Orbit).target.set(camera, b);
        world.step(0);
        const runtime = world.resource(GlobalTransformHistory);
        expect(runtime.cpuHistory.has(a)).toBe(false);
        expect(runtime.cpuHistory.has(b)).toBe(true);
        world.restore(snapshot);
        expect(runtime.cpuHistory.has(b)).toBe(false);
        world.tick();
        world.step(0);
        expect(runtime.cpuHistory.has(a)).toBe(true);
        expect(runtime.cpuHistory.has(b)).toBe(false);
    } finally {
        app.dispose();
    }
});

test("follow uses the target's corrected rendered placement", async () => {
    const app = await createApp({ defaults: false, plugins: [OrbitPlugin] });
    try {
        const world = app.world;
        const target = world.create();
        world.add(target, Transform, { translation: [3, 4, 5] });
        world.add(target, VisualCorrection, { translation: [10, 0, 0] });
        const camera = world.create();
        world.add(camera, Transform);
        world.add(camera, Camera);
        world.add(camera, Orbit, { target, yaw: 0, pitch: 0, distance: 6 });
        world.step(0);
        expect(world.get(camera, GlobalTransform)!.translation).toEqual([13, 4, 11]);
    } finally {
        app.dispose();
    }
});

test("the first frame follows an authored target before any fixed tick", async () => {
    const app = await createApp({ defaults: false, plugins: [OrbitPlugin] });
    try {
        const world = app.world;
        const target = world.create();
        world.add(target, Transform, { translation: [3, 4, 5] });
        const camera = world.create();
        world.add(camera, Transform);
        world.add(camera, Camera);
        world.add(camera, Orbit, { target, yaw: 0, pitch: 0, distance: 6 });
        world.step(0);
        expect(world.time.fixedTick).toBe(0);
        expect(world.get(camera, GlobalTransform)!.translation).toEqual([3, 4, 11]);
    } finally {
        app.dispose();
    }
});

test("a between-tick teleport snaps every follow coordinate and discards the old history", async () => {
    let target = 0;
    const app = await createApp({
        defaults: false,
        plugins: [
            OrbitPlugin,
            {
                name: "Move",
                systems: [
                    {
                        group: "fixed",
                        update(world) {
                            world.storage(Transform).translation.set(target, 1, 1, 0);
                        },
                    },
                ],
            },
        ],
    });
    try {
        const world = app.world;
        target = world.create();
        world.add(target, Transform);
        const camera = world.create();
        world.add(camera, Transform);
        world.add(camera, Camera);
        world.add(camera, Orbit, { target, yaw: 0, pitch: 0, distance: 6 });
        world.tick();
        expect(world.get(target, GlobalTransform)!.translation).toEqual([1, 1, 0]);
        world.storage(Transform).translation.set(target, 20, 1, 0);
        world.storage(GlobalTransform).translation.set(target, 20, 1, 0);
        resetInterpolation(world, target);
        world.step(Time.FIXED_DT / 2);
        expect(world.time.fixedTick).toBe(1);
        expect(world.time.fixedAlpha).toBe(0.5);
        expect(world.get(camera, GlobalTransform)!.translation).toEqual([20, 1, 6]);
        // A presentation consumer may clear the notification after this frame.
        world.resource(TransformRuntime).discontinuities.fill(0);
        world.step(0);
        expect(world.get(camera, GlobalTransform)!.translation).toEqual([20, 1, 6]);
    } finally {
        app.dispose();
    }
});

test("a fixed target advances its follow camera 0.05 every 120 Hz frame, not alternating 0 and 0.10", async () => {
    let target = 0;
    const app = await createApp({
        defaults: false,
        plugins: [
            OrbitPlugin,
            {
                name: "Move",
                systems: [
                    {
                        group: "fixed",
                        update(w) {
                            const p = w.storage(Transform).translation;
                            p.x.set(target, p.x.get(target) + 0.1);
                        },
                    },
                ],
            },
        ],
    });
    try {
        const w = app.world;
        target = w.create();
        w.add(target, Transform);
        const camera = w.create();
        w.add(camera, Transform);
        w.add(camera, Camera);
        w.add(camera, Orbit, { target, yaw: 0, pitch: 0, distance: 6 });
        w.step(0);
        const positions: number[] = [];
        for (let i = 0; i < 12; i++) {
            w.step(Time.FIXED_DT / 2);
            positions.push(w.get(camera, GlobalTransform)!.translation[0]);
        }
        for (let i = 2; i < positions.length; i++)
            expect(positions[i]! - positions[i - 1]!).toBeCloseTo(0.05, 6);
        w.storage(Transform).translation.x.set(target, 20);
        resetInterpolation(w, target);
        w.step(Time.FIXED_DT);
        expect(w.get(camera, Transform)!.translation[0]).toBeCloseTo(20.1, 5);
    } finally {
        app.dispose();
    }
});
