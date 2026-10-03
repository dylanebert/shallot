import { expect, test } from "bun:test";
import { World } from "../../../engine";
import { BodyType } from "../common/types";
import { kernel } from "../kernel/kernel";
import { PhysicsWorld } from "./world";

test("World.restore refuses a shared-kernel snapshot that could rewind a sibling World", () => {
    const target = new PhysicsWorld({ gravity: { x: 0, y: 0, z: 0 } });
    const sibling = new PhysicsWorld({ gravity: { x: 0, y: 0, z: 0 } });
    try {
        target.createBody({ type: BodyType.Kinematic, linearVelocity: { x: 1, y: 0, z: 0 } });
        const siblingBody = sibling.createBody({
            type: BodyType.Kinematic,
            linearVelocity: { x: 1, y: 0, z: 0 },
        });
        const saved = target.snapshot();
        siblingBody.setLinearVelocity({ x: 8, y: 0, z: 0 });

        let restoreError: unknown;
        try {
            target.restore(saved);
        } catch (error) {
            restoreError = error;
        }
        expect(siblingBody.getLinearVelocity()).toEqual({ x: 8, y: 0, z: 0 });
        expect(String(restoreError)).toContain("other live Worlds share its kernel");
    } finally {
        sibling.destroy();
        target.destroy();
    }
});

test("World.restore refuses a destroyed target even when a sibling is the only live World", () => {
    const target = new PhysicsWorld({ gravity: { x: 0, y: 0, z: 0 } });
    const sibling = new PhysicsWorld({ gravity: { x: 0, y: 0, z: 0 } });
    let targetDestroyed = false;
    try {
        target.createBody({ type: BodyType.Kinematic, linearVelocity: { x: 1, y: 0, z: 0 } });
        const siblingBody = sibling.createBody({
            type: BodyType.Kinematic,
            linearVelocity: { x: 1, y: 0, z: 0 },
        });
        const saved = target.snapshot();
        siblingBody.setLinearVelocity({ x: 8, y: 0, z: 0 });
        target.destroy();
        targetDestroyed = true;

        let restoreError: unknown;
        try {
            target.restore(saved);
        } catch (error) {
            restoreError = error;
        }
        expect(siblingBody.getLinearVelocity()).toEqual({ x: 8, y: 0, z: 0 });
        expect(String(restoreError)).toContain("target World is not live");
    } finally {
        sibling.destroy();
        if (!targetDestroyed) target.destroy();
    }
});

test("restoring a snapshot rebinds tree and pair views before another proxy mutation", () => {
    const world = new PhysicsWorld({ gravity: { x: 0, y: 0, z: 0 } }, new World());
    try {
        const body = world.createBody({ type: BodyType.Dynamic });
        body.createSphere({}, { center: { x: 0, y: 0, z: 0 }, radius: 1 });
        const saved = world.snapshot();
        world.restore(saved);
        const broad = world.state.broadPhase;
        expect(broad.store.trees).toBe(broad.trees);
        expect(broad.store.set).toBe(broad.pairSet);
        expect(broad.trees[BodyType.Dynamic].ni.buffer).toBe(
            kernel(world.state.ecsState).memory.buffer,
        );
        const second = world.createBody({ type: BodyType.Dynamic, position: { x: 4, y: 0, z: 0 } });
        second.createSphere({}, { center: { x: 0, y: 0, z: 0 }, radius: 1 });
        expect(broad.trees[BodyType.Dynamic].proxyCount).toBe(2);
        world.step(Math.fround(1 / 60), 4);
    } finally {
        world.destroy();
    }
});

test.todo("physics-hardening: restoring one World leaves a sibling World on the same kernel unchanged", () => {
    const target = new PhysicsWorld({ gravity: { x: 0, y: 0, z: 0 } });
    const sibling = new PhysicsWorld({ gravity: { x: 0, y: 0, z: 0 } });
    try {
        target.createBody({ type: BodyType.Kinematic, linearVelocity: { x: 1, y: 0, z: 0 } });
        const siblingBody = sibling.createBody({
            type: BodyType.Kinematic,
            linearVelocity: { x: 1, y: 0, z: 0 },
        });
        const saved = target.snapshot();
        siblingBody.setLinearVelocity({ x: 8, y: 0, z: 0 });

        expect(() => target.restore(saved)).not.toThrow();
        expect(siblingBody.getLinearVelocity()).toEqual({ x: 8, y: 0, z: 0 });
    } finally {
        sibling.destroy();
        target.destroy();
    }
});
