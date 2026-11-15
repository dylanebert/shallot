import { Body, BodyType, Shape } from "../../core/physics";
import { Transform } from "../../core/transform";
import { createApp } from "../../engine";
import { overlapAABB, StandardPhysicsPlugin } from ".";

export let controlSink: object;
export function control() {
    controlSink = {};
}

export default async function create(input: string) {
    const app = await createApp({ defaults: false, plugins: [StandardPhysicsPlugin] });
    const world = app.world;
    const floor = world.create();
    world.add(floor, Transform, { translation: [0, -0.5, 0], scale: [30, 1, 30] });
    world.add(floor, Body, { type: BodyType.Static });
    world.add(floor, Shape);
    const body = world.create();
    world.add(body, Transform, { translation: [0, 0.8, 0] });
    world.add(body, Body, { type: BodyType.Dynamic, angularVelocity: [1, 2, 3], enableSleep: 0 });
    world.add(body, Shape);
    world.tick();
    overlapAABB(
        world,
        { lowerBound: { x: -10, y: -10, z: -10 }, upperBound: { x: 10, y: 10, z: 10 } },
        () => true,
    );
    const image = world.snapshot();
    return {
        step() {
            if (input === "restore") world.restore(image);
            world.tick();
        },
        allocationFunctions: [world.restore],
        dispose: () => app.dispose(),
    };
}
