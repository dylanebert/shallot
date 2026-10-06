import { Body, createApp, StandardPhysicsPlugin } from "@dylanebert/shallot";
import { physicsWorld } from "@dylanebert/shallot/standard/physics";
import { SetType } from "../common/constants";
import { BodyType, defaultSurfaceMaterial } from "../common/types";
import { BodyField, bodyField } from "../kernel/bodyrecords";
import { makeBoxHull } from "../shapes/hull";

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
    const physics = physicsWorld(world)!;
    const platform = world.create();
    world.add(platform, Body, {
        type: BodyType.Kinematic,
        position: [0, -0.5, 0, 0],
        halfExtents: [100, 0.5, 100, 0],
    });
    const hull = makeBoxHull(0.5, 0.5, 0.5);
    const side = Math.ceil(Math.sqrt(count));
    const boxes: number[] = [];
    for (let i = 0; i < count; i++) {
        const box = physics.createBody({
            type: BodyType.Dynamic,
            position: {
                x: (i % side) * 3 - side * 1.5,
                y: 0.5,
                z: Math.floor(i / side) * 3 - side * 1.5,
            },
        });
        box.createHull(
            {
                density: 1,
                baseMaterial: { ...defaultSurfaceMaterial(), userMaterialId: BigInt(i + 1) },
            },
            hull,
        );
        boxes.push(box.id.index1 - 1);
    }
    world.step(1 / 60);
    const velocity = { x: 0, y: 0, z: 0 };
    let tick = 0;
    return {
        step: () => {
            velocity.x = tick++ % 240 < 120 ? 0.5 : -0.5;
            physics.getBody(platform)!.setLinearVelocity(velocity);
            world.step(1 / 60);
            const state = physics.state;
            if (state.awakeContacts.length !== count)
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
        wait: () => world.gpu.device.queue.onSubmittedWorkDone(),
        dispose: () => app.dispose(),
    };
}
