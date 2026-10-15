import { Body, createApp, Shape, StandardPhysicsPlugin } from "@dylanebert/shallot";
import { BodyType } from "../common/types";
import { PhysicsEvents } from "../events";
import { setLinearVelocity } from "../runtime";
import { PhysicsWorldDefinition } from "../world-definition";

export let controlSink: { frame: number } | undefined;
export const control = () => {
    controlSink = { frame: 0 };
};

// Separate boxes driven down into a static ground every step: each contact reports a hit, and every
// shape carries its own nonzero user material id in both u32 halves, so each step publishes hit
// records with distinct 64-bit ids on both sides.
export default async function create(input: string) {
    const count = Number(input);
    const app = await createApp({ defaults: false, plugins: [StandardPhysicsPlugin] });
    const world = app.world;
    world.resource(PhysicsWorldDefinition).enableSleep = false;
    const ground = world.create();
    world.add(ground, Body, { type: BodyType.Static, position: [0, -0.5, 0] });
    world.add(ground, Shape, {
        scale: [200, 1, 200],
        enableHitEvents: 1,
        userMaterialIdLow: 0x89abcdef,
        userMaterialIdHigh: 0x01234567,
    });
    const side = Math.ceil(Math.sqrt(count));
    const boxes: number[] = [];
    for (let i = 0; i < count; i++) {
        const box = world.create();
        world.add(box, Body, {
            type: BodyType.Dynamic,
            position: [(i % side) * 3 - side * 1.5, 0.5, Math.floor(i / side) * 3 - side * 1.5],
        });
        world.add(box, Shape, {
            enableHitEvents: 1,
            userMaterialIdLow: i + 1,
            userMaterialIdHigh: 0xfedcba98,
        });
        boxes.push(box);
    }
    const events = world.resource(PhysicsEvents);
    const velocity = { x: 0, y: -4, z: 0 };
    const drive = () => {
        for (const box of boxes) setLinearVelocity(world, box, velocity);
        world.step(1 / 60);
    };
    // The first step creates the contacts; hits begin on the next.
    drive();
    return {
        step: () => {
            drive();
            const hits = events.contactHit;
            if (hits.count !== count)
                throw new Error("allocation subject did not report one hit per box");
            for (let i = 0; i < hits.count; i++) {
                const hit = hits.records[i];
                if (hit.userMaterialIdA === 0n || hit.userMaterialIdB === 0n)
                    throw new Error("allocation subject reported a hit without user material ids");
            }
        },
        dispose: () => app.dispose(),
    };
}
