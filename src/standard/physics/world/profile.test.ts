import { expect, spyOn, test } from "bun:test";
import { BodyType, makeBoxHull, PhysicsWorld } from "../api";

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
