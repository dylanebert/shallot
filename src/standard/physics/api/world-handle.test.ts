import { expect, test } from "bun:test";
import { BodyType } from "../common/types";
import { PhysicsWorld } from "./world";

test("a world from Joint.getWorld polls every event kind its creating world reports", () => {
    const world = new PhysicsWorld({ gravity: { x: 0, y: -10, z: 0 } });
    try {
        const anchor = world.createBody({ position: { x: 0, y: 5, z: 0 } });
        const arm = world.createBody({ type: BodyType.Dynamic, position: { x: 1, y: 5, z: 0 } });
        arm.createSphere({}, { center: { x: 0, y: 0, z: 0 }, radius: 0.25 });
        const joint = world.createRevoluteJoint(anchor, arm, { forceThreshold: 0 });
        world.step(1 / 60);
        const handle = joint.getWorld();
        expect(handle.getBodyEvents().count).toBe(world.getBodyEvents().count);
        expect(handle.getBodyEvents().count).toBeGreaterThan(0);
        expect(handle.getJointEvents().length).toBe(world.getJointEvents().length);
        expect(handle.getJointEvents().length).toBeGreaterThan(0);
        expect(handle.getContactEvents().beginEvents.length).toBe(
            world.getContactEvents().beginEvents.length,
        );
        expect(handle.getSensorEvents().beginEvents.length).toBe(
            world.getSensorEvents().beginEvents.length,
        );
    } finally {
        world.destroy();
    }
});
