import { expect, test } from "bun:test";
import { BodyType, PhysicsWorld } from "../api";

test("setting velocity leaves static and zero-length sleeping bodies asleep, and wakes a nonzero dynamic body", () => {
    const world = new PhysicsWorld({ gravity: { x: 0, y: 0, z: 0 } });
    try {
        const fixed = world.createBody({ type: BodyType.Static });
        fixed.setLinearVelocity({ x: 0.5, y: 0, z: 0 });
        expect(fixed.isAwake()).toBe(false);
        expect(fixed.getLinearVelocity()).toEqual({ x: 0, y: 0, z: 0 });
        const sleeping = world.createBody({ type: BodyType.Dynamic, isAwake: false });
        for (const x of [0, -0, 1e-30]) {
            sleeping.setLinearVelocity({ x, y: 0, z: 0 });
            expect(sleeping.isAwake()).toBe(false);
        }
        sleeping.setLinearVelocity({ x: 0.5, y: 0, z: 0 });
        expect(sleeping.isAwake()).toBe(true);
        expect(sleeping.getLinearVelocity()).toEqual({ x: 0.5, y: 0, z: 0 });
        expect(sleeping.getType()).toBe(BodyType.Dynamic);
    } finally {
        world.destroy();
    }
});
