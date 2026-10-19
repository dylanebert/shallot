import { expect, test } from "bun:test";
import { Body, BodyType, Shape, ShapeKind } from "../../core/physics";
import { GlobalTransform } from "../../core/transform";
import { createApp, type World } from "../../engine";
import {
    computeBodyAABB,
    getLocalPoint,
    getMassData,
    isAwake,
    StandardPhysicsPlugin,
    setLinearVelocity,
    setTransform,
} from "./runtime";

function spawnBall(world: World): number {
    const eid = world.create();
    world.add(eid, Body, { type: BodyType.Dynamic });
    world.add(eid, Shape, { kind: ShapeKind.Sphere, radius: 0.5 });
    return eid;
}

function reads(world: World, eid: number) {
    return [
        isAwake(world, eid),
        getMassData(world, eid),
        computeBodyAABB(world, eid),
        getLocalPoint(world, eid, { x: 0, y: 0, z: 0 }),
    ];
}

test("a body destroyed or recycled since the last fixed sync reads null and publishes no predecessor state", async () => {
    const app = await createApp({ defaults: false, plugins: [StandardPhysicsPlugin] });
    try {
        const world = app.world;
        const falling = spawnBall(world);
        for (let i = 0; i < 30; i++) world.tick();
        expect(reads(world, falling).every((value) => value !== null)).toBe(true);
        world.destroy(falling);
        expect(reads(world, falling)).toEqual([null, null, null, null]);

        const recycled = spawnBall(world);
        expect(recycled).toBe(falling);
        expect(reads(world, recycled)).toEqual([null, null, null, null]);
        const velocity = () => Array.from(world.get(recycled, GlobalTransform)!.linearVelocity);
        setTransform(world, recycled, [0, 10, 0], [0, 0, 0, 1]);
        expect(velocity()).toEqual([0, 0, 0]);
        setLinearVelocity(world, recycled, { x: 0, y: 3, z: 0 });
        expect(velocity()).toEqual([0, 0, 0]);

        world.tick();
        expect(isAwake(world, recycled)).toBe(true);
        expect(world.get(recycled, GlobalTransform)!.linearVelocity[1]).toBeLessThan(0);
    } finally {
        app.dispose();
    }
});
