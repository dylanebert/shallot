import { Body, createApp, Shape, StandardPhysicsPlugin } from "@dylanebert/shallot";
import { awakeContactCount } from "../collision/contact";
import { SetType } from "../common/constants";
import { BodyType, defaultSurfaceMaterial } from "../common/types";
import { BodyField, bodyField } from "../kernel/bodyrecords";
import { physicsBodyIndex, physicsState } from "../runtime";
import { makeBoxHull } from "../shapes/build-hull";
import { createHullShape } from "../shapes/shape";
import { createTestBody, testShapeDef } from "../testing";
import { bodySetLinearVelocity } from "../world/body";

export let controlSink: { frame: number } | undefined;
export const control = () => {
    controlSink = { frame: 0 };
};

// Separate boxes resting on a sliding kinematic platform: every contact persists and stays awake, and its
// relative transform changes, so each one reaches the narrowphase every step instead of recycling. Each
// box has its own user material id, so contacts mix distinct persistent materials.
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
    world.add(platform, Shape, { scale: [200, 1, 200, 0] });
    const hull = makeBoxHull(0.5, 0.5, 0.5);
    const side = Math.ceil(Math.sqrt(count));
    const boxes: number[] = [];
    for (let i = 0; i < count; i++) {
        const box = createTestBody(state, {
            type: BodyType.Dynamic,
            position: {
                x: (i % side) * 3 - side * 1.5,
                y: 0.5,
                z: Math.floor(i / side) * 3 - side * 1.5,
            },
        });
        createHullShape(
            state,
            box,
            testShapeDef({
                density: 1,
                baseMaterial: { ...defaultSurfaceMaterial(), userMaterialId: BigInt(i + 1) },
            }),
            hull,
        );
        boxes.push(box);
    }
    world.step(1 / 60);
    const platformBody = physicsBodyIndex(world, platform);
    if (platformBody < 0) throw new Error("allocation subject platform has no solver body");
    const velocity = { x: 0, y: 0, z: 0 };
    let tick = 0;
    return {
        step: () => {
            velocity.x = tick++ % 240 < 120 ? 0.5 : -0.5;
            bodySetLinearVelocity(state, platformBody, velocity);
            world.step(1 / 60);
            if (awakeContactCount(state) !== count)
                throw new Error("allocation subject lost its awake box contacts");
            for (let i = 0; i < count; i++) {
                const body = boxes[i];
                if (
                    bodyField(state, body, BodyField.setIndex) !== SetType.Awake ||
                    bodyField(state, body, BodyField.contactCount) !== 1
                )
                    throw new Error("allocation subject lost a box's one awake platform contact");
            }
        },
        dispose: () => app.dispose(),
    };
}
