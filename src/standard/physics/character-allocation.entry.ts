import { Body, createApp, ShapeKind } from "@dylanebert/shallot";
import {
    Character,
    CharacterPlugin,
    PhysicsWorldDefinition,
    physicsWorld,
} from "@dylanebert/shallot/standard/physics";
import { characterScratch } from "./character";
import { BodyType } from "./common/types";

export let controlSink: { frame: number } | undefined;
export const control = () => {
    controlSink = { frame: 0 };
};

export default async function create(input: string) {
    const app = await createApp({ defaults: false, plugins: [CharacterPlugin] });
    const world = app.world;
    const pushing = input === "push";
    const physics = physicsWorld(world)!;
    if (pushing) {
        world.resource(PhysicsWorldDefinition).gravity = { x: 0, y: 0, z: 0 };
        physics
            .createBody({ type: BodyType.Dynamic, position: { x: 0.8, y: 2, z: 0 } })
            .createSphere({ density: 1 }, { center: { x: 0, y: 0, z: 0 }, radius: 0.5 });
    }
    const platform = world.create();
    world.add(platform, Body, {
        type: BodyType.Kinematic,
        position: [0, -0.5, 0, 0],
        halfExtents: [1000, 0.5, 1000, 0],
    });
    const eid = world.create();
    world.add(eid, Body, {
        type: BodyType.Kinematic,
        shape: ShapeKind.Capsule,
        position: [0, pushing ? 2 : 1.3, 0, 0],
        halfExtents: [0, 0.5, 0, 0.3],
    });
    world.add(eid, Character);
    world.step(1 / 60);
    if (!pushing) physics.getBody(platform)!.setLinearVelocity({ x: 2, y: 0, z: 0 });
    const velocity = world.storage(Character).velocity;
    let tick = 0;
    return {
        step: () => {
            if (pushing) {
                velocity.set(eid, 3, 0, 0, 0);
                // Isolate SolveMove from the rigid solver's contact pipeline; the full fixed-step claim is the other mode.
                CharacterPlugin.systems![0].update!(world);
                if (world.resource(characterScratch).impulseCount !== 1)
                    throw new Error("allocation subject lost its push plane");
            } else {
                velocity.set(eid, tick++ % 120 < 60 ? 3 : -3, -12, 0, 0);
                world.step(1 / 60);
                if (tick > 600 && world.resource(characterScratch).count === 0)
                    throw new Error("allocation subject lost its floor planes");
            }
        },
        wait: async () => {
            await world.gpuIfAvailable?.device.queue.onSubmittedWorkDone();
        },
        dispose: () => app.dispose(),
    };
}
