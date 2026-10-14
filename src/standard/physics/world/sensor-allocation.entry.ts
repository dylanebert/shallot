import { Body, createApp, Shape, StandardPhysicsPlugin } from "@dylanebert/shallot";
import { awakeContactCount } from "../collision/contact";
import { BodyType } from "../common/types";
import { EventKind } from "../kernel/eventbuffers";
import { kernel } from "../kernel/kernel";
import { ShapeField, shapeField } from "../kernel/shaperecords";
import { physicsBodyIndex, physicsState } from "../runtime";
import { makeBoxHull } from "../shapes/build-hull";
import { createHullShape } from "../shapes/shape";
import { createTestBody, testShapeDef } from "../testing";
import { bodySetLinearVelocity } from "./body";

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
    const state = physicsState(world)!;
    const platform = world.create();
    world.add(platform, Body, {
        type: BodyType.Kinematic,
        position: [0, -0.5, 0, 0],
    });
    world.add(platform, Shape, { scale: [100, 0.5, 100, 0] });
    const hull = makeBoxHull(0.5, 0.5, 0.5);
    const side = Math.ceil(Math.sqrt(count));
    for (let i = 0; i < count; i++) {
        const box = createTestBody(state, {
            type: BodyType.Dynamic,
            position: {
                x: (i % side) * 3 - side * 1.5,
                y: 0.5,
                z: Math.floor(i / side) * 3 - side * 1.5,
            },
        });
        createHullShape(state, box, testShapeDef({ density: 1, enableSensorEvents: true }), hull);
    }
    const sensorBody = createTestBody(state, {
        type: BodyType.Static,
        position: { x: 0, y: 0.5, z: 0 },
    });
    const sensor = createHullShape(
        state,
        sensorBody,
        testShapeDef({ isSensor: true, enableSensorEvents: true }),
        makeBoxHull(side * 1.5 + 2, 1, side * 1.5 + 2),
    )!;
    world.step(1 / 60);
    const sensorIndex = shapeField(state, sensor, ShapeField.sensorIndex);
    if (kernel(state.ecsState).sensorVisitorCount(state.worldId, sensorIndex) !== count)
        throw new Error("allocation subject's sensor does not hold every box");
    const platformBody = physicsBodyIndex(world, platform);
    if (platformBody < 0) throw new Error("allocation subject platform has no solver body");
    const velocity = { x: 0, y: 0, z: 0 };
    const k = kernel(state.ecsState);
    let tick = 0;
    return {
        step: () => {
            velocity.x = tick++ % 240 < 120 ? 0.5 : -0.5;
            bodySetLinearVelocity(state, platformBody, velocity);
            world.step(1 / 60);
            if (
                k.eventCount(state.worldId, EventKind.SensorBegin) !== 0 ||
                k.eventCount(state.worldId, EventKind.SensorEnd) !== 0 ||
                k.eventCount(state.worldId, EventKind.ContactBegin) !== 0 ||
                k.eventCount(state.worldId, EventKind.ContactEnd) !== 0 ||
                k.eventCount(state.worldId, EventKind.ContactHit) !== 0 ||
                k.eventCount(state.worldId, EventKind.Joint) !== 0 ||
                k.eventCount(state.worldId, EventKind.PendingSensorEnd) !== 0
            )
                throw new Error("allocation subject's sensor overlaps changed in steady play");
            if (awakeContactCount(state) !== count)
                throw new Error("allocation subject lost its awake box contacts");
        },
        dispose: () => app.dispose(),
    };
}
