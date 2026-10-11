import { expect, test } from "bun:test";
import { BodyType, makeBoxHull, PhysicsWorld } from "../api";
import { ContactField, contactCount, setContactField } from "../collision/contact";
import { updateBroadPhasePairs } from "../collision/pairs";

// Low-level generation-width ABI parity test; the consumer-visible impulse sequence is in ecs-event-claims.test.ts.

test("contact begin, hit and end events retain a full u32 generation and the solved begin impulse", () => {
    const world = new PhysicsWorld({ gravity: { x: 0, y: -10, z: 0 }, enableContinuous: false });
    try {
        world
            .createBody({ position: { x: 0, y: -0.5, z: 0 } })
            .createHull({}, makeBoxHull(10, 0.5, 10));
        const body = world.createBody({
            type: BodyType.Dynamic,
            position: { x: 0, y: 0.5, z: 0 },
            linearVelocity: { x: 0, y: -10, z: 0 },
        });
        body.createHull(
            { enableContactEvents: true, enableHitEvents: true },
            makeBoxHull(0.5, 0.5, 0.5),
        );
        updateBroadPhasePairs(world.state);
        expect(contactCount(world.state)).toBe(1);
        const generation = 65536;
        setContactField(world.state, 0, ContactField.generation, generation);
        world.step(1 / 60, 4);
        const events = world.getContactEvents();
        expect(events.beginEvents).toHaveLength(1);
        const begin = events.beginEvents[0];
        expect(begin.contact.id.generation).toBe(generation);
        expect(begin.contact.isValid()).toBe(true);
        expect(begin.normalImpulse).toBeGreaterThan(0);
        expect(events.hitEvents).toHaveLength(1);
        expect(events.hitEvents[0].contact.id.generation).toBe(generation);
        expect(events.hitEvents[0].contact.isValid()).toBe(true);
        const impulse = begin.normalImpulse;
        body.destroy();
        expect(world.getContactEvents().beginEvents[0].normalImpulse).toBe(impulse);
        world.step(1 / 60, 4);
        const end = world.getContactEvents().endEvents;
        expect(end).toHaveLength(1);
        expect(end[0].contact.id.generation).toBe(generation);
        expect(end[0].contact.isValid()).toBe(false);
    } finally {
        world.destroy();
    }
});
