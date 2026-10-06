import { expect, test } from "bun:test";
import { BodyType, PhysicsWorld } from "../api";
import { SetType } from "../common/constants";
import { makeBoxHull } from "../shapes/hull";
import { BODY_RECORD_STRIDE, BodyField, bodyField, setBodyField } from "./bodyrecords";
import { kernel } from "./kernel";

function words(world: PhysicsWorld): number[] {
    const k = kernel(world.state.ecsState);
    const length = k.bodyLength(world.state.worldId) * BODY_RECORD_STRIDE;
    return Array.from(world.state.bodyStore.recordU.subarray(0, length));
}

test("body records and their LIFO pool restore every word, including free-slot generation, topology and mass", () => {
    const world = new PhysicsWorld({ gravity: { x: 0, y: 0, z: 0 } });
    const sibling = new PhysicsWorld();
    try {
        const hull = makeBoxHull(0.5, 0.5, 0.5);
        const a = world.createBody({ type: BodyType.Dynamic, sleepThreshold: 0.125 });
        const b = world.createBody({ type: BodyType.Dynamic, position: { x: 0.9, y: 0, z: 0 } });
        a.createHull({ density: 2 }, hull);
        b.createHull({ density: 3 }, hull);
        world.createDistanceJoint(a, b, { length: 0.9, collideConnected: true });
        world.step(1 / 60);
        const c = world.createBody();
        const free = c.id.index1 - 1;
        const generation = c.id.generation;
        c.destroy();
        setBodyField(world.state, a.id.index1 - 1, BodyField.sleepTime, 0.125);
        setBodyField(world.state, a.id.index1 - 1, BodyField.sleepVelocity, 0.25);
        const firstMass = a.getMassData();
        const secondMass = b.getMassData();
        expect(firstMass.mass).toBe(2);
        expect(secondMass.mass).toBe(3);
        expect(firstMass.inertia.cx.x).toBeCloseTo(1 / 3, 6);
        const before = words(world);
        const saved = world.snapshot();
        sibling.createBody({ type: BodyType.Dynamic });
        const other = words(sibling);
        a.setAwake(false);
        expect(bodyField(world.state, a.id.index1 - 1, BodyField.setIndex)).toBeGreaterThanOrEqual(
            SetType.FirstSleeping,
        );
        a.setAwake(true);
        b.setType(BodyType.Static);
        world.createBody({ type: BodyType.Dynamic });
        world.restore(saved);
        expect(words(world)).toEqual(before);
        expect(words(sibling)).toEqual(other);
        const reused = world.createBody();
        expect(reused.id.index1 - 1).toBe(free);
        expect(reused.id.generation).toBe((generation + 1) & 0xffff);
        expect(a.getMassData()).toEqual(firstMass);
        expect(b.getMassData()).toEqual(secondMass);
    } finally {
        sibling.destroy();
        world.destroy();
    }
});
