import { Body, createApp, StandardPhysicsPlugin } from "@dylanebert/shallot";
import { physicsWorld } from "@dylanebert/shallot/standard/physics";
import { awakeContactCount } from "../collision/contact";
import { BodyType } from "../common/types";
import { kernel } from "../kernel/kernel";
import { makeBoxHull } from "../shapes/hull";

export let controlSink: { frame: number } | undefined;
export const control = () => {
    controlSink = { frame: 0 };
};

// Boxes resting on a sliding kinematic platform stay awake inside one static sensor: every step
// refreshes the sensor's overlaps, and none begins or ends. A game polls every event kind each step
// whether or not any arrived.
export default async function create(input: string) {
    const count = Number(input);
    const app = await createApp({ defaults: false, plugins: [StandardPhysicsPlugin] });
    const world = app.world;
    const physics = physicsWorld(world)!;
    const platform = world.create();
    world.add(platform, Body, {
        type: BodyType.Kinematic,
        position: [0, -0.5, 0, 0],
        halfExtents: [100, 0.5, 100, 0],
    });
    const hull = makeBoxHull(0.5, 0.5, 0.5);
    const side = Math.ceil(Math.sqrt(count));
    for (let i = 0; i < count; i++) {
        const box = physics.createBody({
            type: BodyType.Dynamic,
            position: {
                x: (i % side) * 3 - side * 1.5,
                y: 0.5,
                z: Math.floor(i / side) * 3 - side * 1.5,
            },
        });
        box.createHull({ density: 1, enableSensorEvents: true }, hull);
    }
    const sensorBody = physics.createBody({
        type: BodyType.Static,
        position: { x: 0, y: 0.5, z: 0 },
    });
    const sensor = sensorBody.createHull(
        { isSensor: true, enableSensorEvents: true },
        makeBoxHull(side * 1.5 + 2, 1, side * 1.5 + 2),
    );
    world.step(1 / 60);
    if (sensor.getSensorOverlaps().length !== count)
        throw new Error("allocation subject's sensor does not hold every box");
    const velocity = { x: 0, y: 0, z: 0 };
    let tick = 0;
    return {
        step: () => {
            velocity.x = tick++ % 240 < 120 ? 0.5 : -0.5;
            physics.getBody(platform)!.setLinearVelocity(velocity);
            world.step(1 / 60);
            const state = physics.state;
            if (
                kernel(state.ecsState).eventCount(state.worldId, 0) !== 0 ||
                kernel(state.ecsState).eventCount(state.worldId, 7) !== 0
            )
                throw new Error("allocation subject's sensor overlaps changed in steady play");
            const sensorEvents = physics.getSensorEvents();
            const contactEvents = physics.getContactEvents();
            if (
                sensorEvents.beginEvents.length +
                    sensorEvents.endEvents.length +
                    contactEvents.beginEvents.length +
                    contactEvents.endEvents.length +
                    contactEvents.hitEvents.length +
                    physics.getJointEvents().length !==
                0
            )
                throw new Error("allocation subject polled events in steady play");
            if (awakeContactCount(state) !== count)
                throw new Error("allocation subject lost its awake box contacts");
        },
        dispose: () => app.dispose(),
    };
}
