import { expect, test } from "bun:test";
import { BodyType, makeBoxHull, PhysicsWorld } from "../api";

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
