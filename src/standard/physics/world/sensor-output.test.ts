import { expect, test } from "bun:test";
import { PhysicsWorld } from "../api/world";
import { BodyType } from "../common/types";

function output(): unknown[] {
    const world = new PhysicsWorld({ gravity: { x: 0, y: 0, z: 0 } });
    const records: unknown[] = [];
    try {
        const sensor = world
            .createBody({ type: BodyType.Static })
            .createSphere(
                { isSensor: true, enableSensorEvents: true },
                { center: { x: 0, y: 0, z: 0 }, radius: 2 },
            );
        const moving = world.createBody({
            type: BodyType.Dynamic,
            position: { x: -4, y: 0, z: 0 },
            linearVelocity: { x: 2, y: 0, z: 0 },
        });
        moving.createSphere(
            { enableSensorEvents: true, density: 1 },
            { center: { x: 0, y: 0, z: 0 }, radius: 0.5 },
        );
        const sleeping = world.createBody({
            type: BodyType.Dynamic,
            position: { x: 0, y: 1, z: 0 },
            isAwake: false,
        });
        const destroyed = sleeping.createSphere(
            { enableSensorEvents: true, density: 1 },
            { center: { x: 0, y: 0, z: 0 }, radius: 0.25 },
        );
        const bullet = world.createBody({
            type: BodyType.Dynamic,
            position: { x: -10, y: -1, z: 0 },
            linearVelocity: { x: 1200, y: 0, z: 0 },
            isBullet: true,
        });
        bullet.createSphere(
            { enableSensorEvents: true, density: 1 },
            { center: { x: 0, y: 0, z: 0 }, radius: 0.1 },
        );
        for (let step = 0; step < 240; ++step) {
            if (step === 20) destroyed.destroy();
            if (step === 200) sensor.destroy();
            world.step(1 / 60, 4);
            const events = world.getSensorEvents();
            for (const [kind, list] of [
                ["begin", events.beginEvents],
                ["end", events.endEvents],
            ] as const) {
                for (const e of list)
                    records.push([
                        step,
                        kind,
                        e.sensor.id.index1,
                        e.sensor.id.generation,
                        e.visitor.id.index1,
                        e.visitor.id.generation,
                    ]);
            }
        }
    } finally {
        world.destroy();
    }
    return records;
}
test("sensor begin and end output preserves moving, sleeping, destroyed and continuous visitor order", async () => {
    const expected = await Bun.file(new URL("sensor-output.gold.json", import.meta.url)).json();
    expect(output()).toEqual(expected);
});

export { output };
