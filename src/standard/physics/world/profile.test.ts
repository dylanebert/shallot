import { expect, spyOn, test } from "bun:test";
import { BodyType, makeBoxHull, PhysicsWorld } from "../api";
import { splitAllocationSubject } from "../joint-allocation.fixture";
import { kernel } from "../kernel/kernel";

test("the kernel times every solve substage through its platform clock", () => {
    const world = new PhysicsWorld({ gravity: { x: 0, y: -10, z: 0 } });
    const body = world.createBody({ type: BodyType.Dynamic, position: { x: 0, y: 5, z: 0 } });
    body.createHull({}, makeBoxHull(0.5, 0.5, 0.5));
    let clock = 0;
    const now = spyOn(performance, "now").mockImplementation(() => ++clock);
    try {
        world.step(1 / 60, 4);
        const profile = world.getProfile();
        for (const name of [
            "prepareConstraints",
            "warmStart",
            "solveImpulses",
            "relaxImpulses",
            "applyRestitution",
            "storeImpulses",
            "transforms",
        ] as const)
            expect(profile[name]).toBeGreaterThan(0);
        expect(profile.step).toBeGreaterThan(profile.solve);
        expect(profile.solve).toBeGreaterThan(profile.constraints);
        expect(profile.constraints).toBeGreaterThan(profile.prepareConstraints);
    } finally {
        now.mockRestore();
        world.destroy();
    }
});

test("split timers measure only a candidate task and disabled sleep stays untimed", () => {
    const disabled = new PhysicsWorld({ enableSleep: false, gravity: { x: 0, y: 0, z: 0 } });
    const enabled = new PhysicsWorld({ enableSleep: true, gravity: { x: 0, y: 0, z: 0 } });
    disabled.createBody({ type: BodyType.Dynamic, enableSleep: false });
    enabled.createBody({ type: BodyType.Dynamic, enableSleep: false });
    let clock = 0;
    const now = spyOn(performance, "now").mockImplementation(() => ++clock);
    try {
        disabled.step(Math.fround(1 / 60), 4);
        enabled.step(Math.fround(1 / 60), 4);
        const a = disabled.getProfile(),
            b = enabled.getProfile();
        expect(a.splitIslands).toBe(0);
        expect(b.splitIslands).toBe(0);
        expect(a.sleepIslands).toBe(0);
        expect(b.sleepIslands).toBe(1);
        expect(a.transforms).toBe(b.transforms);
        const split = splitAllocationSubject(disabled);
        split();
        expect(disabled.getProfile().splitIslands).toBe(1);
        expect(disabled.getProfile().sleepIslands).toBe(0);
    } finally {
        now.mockRestore();
        disabled.destroy();
        enabled.destroy();
    }
});

test("sleep gathers the deferred split candidate before the next step's split task", () => {
    const world = new PhysicsWorld({ gravity: { x: 0, y: 0, z: 0 }, enableSleep: true });
    const a = world.createBody({ type: BodyType.Dynamic });
    const b = world.createBody({ type: BodyType.Dynamic, position: { x: 1, y: 0, z: 0 } });
    world.createDistanceJoint(a, b);
    world.createDistanceJoint(a, b).destroy();
    const k = kernel(world.state.ecsState);
    let clock = 0;
    const now = spyOn(performance, "now").mockImplementation(() => ++clock);
    try {
        for (let i = 0; i < 40 && k.islandSplitCandidate() === -1; ++i)
            world.step(Math.fround(1 / 60), 4);
        expect(k.islandSplitCandidate()).toBeGreaterThanOrEqual(0);
        expect(world.getProfile().splitIslands).toBe(0);
        expect(world.getProfile().sleepIslands).toBe(1);
        world.step(Math.fround(1 / 60), 4);
        expect(k.islandSplitCandidate()).toBe(-1);
        expect(world.getProfile().splitIslands).toBe(1);
        expect(world.getProfile().sleepIslands).toBe(1);
    } finally {
        now.mockRestore();
        world.destroy();
    }
});

test("a bare falling box records step and solve timings without setup", () => {
    const world = new PhysicsWorld({ gravity: { x: 0, y: -10, z: 0 } });
    try {
        const body = world.createBody({ type: BodyType.Dynamic, position: { x: 0, y: 5, z: 0 } });
        body.createHull({}, makeBoxHull(0.5, 0.5, 0.5));
        world.step(1 / 60, 4);
        const profile = world.getProfile();
        expect(profile.step).toBeGreaterThan(0);
        expect(profile.solve).toBeGreaterThan(0);
        expect(profile.step).toBeGreaterThanOrEqual(profile.solve);
        profile.step = -1;
        expect(world.getProfile().step).toBeGreaterThan(0);
    } finally {
        world.destroy();
    }
});
