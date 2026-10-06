import { expect, test } from "bun:test";
import { BodyType, makeBoxHull, PhysicsWorld } from "../api";

function output(): number[] {
    const world = new PhysicsWorld({
        gravity: { x: 0, y: -10, z: 0 },
        enableContinuous: false,
    });
    try {
        world
            .createBody({ position: { x: 0, y: -0.5, z: 0 } })
            .createHull({}, makeBoxHull(10, 0.5, 10));
        const body = world.createBody({ type: BodyType.Dynamic, position: { x: 0, y: 3, z: 0 } });
        body.createHull({ enableContactEvents: true }, makeBoxHull(0.5, 0.5, 0.5));
        for (let step = 0; step < 120; ++step) {
            world.step(1 / 60, 4);
            const event = world.getContactEvents().beginEvents[0];
            if (!event) continue;
            const before = event.normalImpulse;
            body.destroy();
            const after = world.getContactEvents().beginEvents[0];
            return [step, before, after.normalImpulse, Number(after.contact.isValid())];
        }
        throw new Error("scene produced no begin event");
    } finally {
        world.destroy();
    }
}

test("the begin-step impulse remains available after the contact is destroyed before event delivery", async () => {
    expect(output()).toEqual(
        await Bun.file(new URL("contact-event-output.gold.json", import.meta.url)).json(),
    );
});
