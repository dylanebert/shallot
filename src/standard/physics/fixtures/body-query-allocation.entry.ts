import { Body, BodyType, createApp, Shape, ShapeKind } from "@dylanebert/shallot";
import {
    castBodyRay,
    castBodyShape,
    collideBodyMover,
    emptyQueryHit,
    overlapBodyShape,
    PhysicsWorldDefinition,
    StandardPhysicsPlugin,
} from "@dylanebert/shallot/standard/physics";

export let controlSink: object | undefined;
export const control = () => {
    controlSink = { query: 0 };
};
export let leaked: object | undefined;

export default async function create(input: string) {
    const app = await createApp({ defaults: false, plugins: [StandardPhysicsPlugin] });
    const world = app.world;
    world.resource(PhysicsWorldDefinition).enableAutomaticStep = false;
    const body = world.create();
    world.add(body, Body, { type: BodyType.Kinematic });
    for (let i = 0; i < 3; i++) {
        const shape = world.create();
        world.add(shape, Shape, {
            body,
            kind: ShapeKind.Sphere,
            sphereCenter: [-0.5 + 0.5 * i, 0, 0],
            radius: 1,
            userMaterialIdLow: 7,
        });
    }
    world.tick();
    const origin = { x: -4, y: 0, z: 0 };
    const translation = { x: 8, y: 0, z: 0 };
    const zero = { x: 0, y: 0, z: 0 };
    const placement = { p: zero, q: { v: zero, s: 1 } };
    const miss = { p: { x: 100, y: 0, z: 0 }, q: placement.q };
    const proxy = { points: [zero], count: 1, radius: 0.25 };
    const mover = {
        center1: { x: 0.8, y: -0.2, z: 0 },
        center2: { x: 0.8, y: 0.2, z: 0 },
        radius: 0.5,
    };
    const hit = emptyQueryHit();
    const filter = { categoryBits: 1n, maskBits: 1n };
    return {
        step() {
            if (
                !castBodyRay(world, body, origin, translation, placement, hit) ||
                hit.userMaterialId !== 7n
            )
                throw new Error("ray missed");
            if (castBodyRay(world, body, origin, translation, miss, hit))
                throw new Error("ray hit");
            if (!castBodyShape(world, body, origin, proxy, translation, placement, hit))
                throw new Error("shape missed");
            if (castBodyShape(world, body, origin, proxy, translation, miss, hit))
                throw new Error("shape hit");
            if (!overlapBodyShape(world, body, zero, proxy, placement))
                throw new Error("overlap missed");
            if (overlapBodyShape(world, body, zero, proxy, miss)) throw new Error("overlap hit");
            if (collideBodyMover(world, body, zero, mover, placement, 4).count !== 3)
                throw new Error("mover lost planes");
            if (collideBodyMover(world, body, zero, mover, placement, 1).count !== 1)
                throw new Error("mover exceeded capacity");
            if (collideBodyMover(world, body, zero, mover, miss).count !== 0)
                throw new Error("mover hit");
            if (!castBodyRay(world, body, origin, translation, placement, hit, filter))
                throw new Error("filtered ray missed");
            if (!castBodyShape(world, body, origin, proxy, translation, placement, hit, filter))
                throw new Error("filtered shape missed");
            if (!overlapBodyShape(world, body, zero, proxy, placement, filter))
                throw new Error("filtered overlap missed");
            if (collideBodyMover(world, body, zero, mover, placement, 4, filter).count !== 3)
                throw new Error("filtered mover lost planes");
            if (input === "leak") leaked = { ...hit };
        },
        dispose() {
            app.dispose();
            leaked = undefined;
        },
    };
}
