import { expect, test } from "bun:test";
import { Time, World } from "@dylanebert/shallot";
import { Body, BodyType, PhysicsPlugin, Shape, ShapeKind } from "@dylanebert/shallot/physics";
import type { QueryFilter } from "../common/types";
import { castRay, castShape, overlapAABB, overlapShape, StandardPhysicsPlugin } from "../runtime";

const origin = { x: 0, y: 0, z: 0 };
const translation = { x: 10, y: 0, z: 0 };
const proxy = { points: [origin], count: 1, radius: 0.1 };
const filter: QueryFilter = { categoryBits: 0xffffffffffffffffn, maskBits: 0xffffffffffffffffn };

async function queryWorld() {
    const world = new World();
    for (const component of PhysicsPlugin.components!)
        world.registry.register(component, PhysicsPlugin.name);
    await StandardPhysicsPlugin.initialize!(world);
    await StandardPhysicsPlugin.warm!(world);
    for (const system of StandardPhysicsPlugin.systems!)
        world.addSystem(system, StandardPhysicsPlugin.name);
    for (const [type, x] of [
        [BodyType.Static, 2],
        [BodyType.Static, 4],
        [BodyType.Kinematic, 6],
        [BodyType.Dynamic, 8],
    ] as const) {
        const body = world.create();
        world.add(body, Body, { type, position: [x, 0, 0, 0] });
        world.add(body, Shape, { kind: ShapeKind.Sphere, sphereCenter: [0, 0, 0], radius: 0.5 });
    }
    world.step(Time.FIXED_DT);
    return world;
}

async function dispose(world: World) {
    await StandardPhysicsPlugin.dispose!(world);
    world.dispose();
}

for (const kind of ["ray", "shape"] as const) {
    test(`${kind} callback fraction clips farther hits in this tree and later body-type trees`, async () => {
        const world = await queryWorld();
        try {
            const fractions: number[] = [];
            const callback = (hit: { fraction: number }) => {
                fractions.push(hit.fraction);
                return hit.fraction;
            };
            if (kind === "ray") castRay(world, origin, translation, callback, filter);
            else castShape(world, origin, proxy, translation, callback, filter);
            expect(fractions).toHaveLength(1);
            expect(fractions[0]).toBeGreaterThan(0);
            expect(fractions[0]).toBeLessThan(0.2);
        } finally {
            await dispose(world);
        }
    });
    test(`${kind} callback zero terminates this tree and all later body-type trees`, async () => {
        const world = await queryWorld();
        try {
            let calls = 0;
            const callback = () => {
                ++calls;
                return 0;
            };
            if (kind === "ray") castRay(world, origin, translation, callback, filter);
            else castShape(world, origin, proxy, translation, callback, filter);
            expect(calls).toBe(1);
        } finally {
            await dispose(world);
        }
    });
    test(`${kind} callback minus one filters a hit without clipping later hits`, async () => {
        const world = await queryWorld();
        try {
            const fractions: number[] = [];
            const callback = (hit: { fraction: number }) => {
                fractions.push(hit.fraction);
                return -1;
            };
            if (kind === "ray") castRay(world, origin, translation, callback, filter);
            else castShape(world, origin, proxy, translation, callback, filter);
            expect(fractions).toHaveLength(4);
            expect(fractions[3]).toBeGreaterThan(0.7);
        } finally {
            await dispose(world);
        }
    });
}

for (const kind of ["aabb", "overlap"] as const) {
    test(`${kind} false stops only the current body-type tree and still queries the next tree`, async () => {
        const world = await queryWorld();
        try {
            let calls = 0;
            const callback = () => {
                ++calls;
                return false;
            };
            if (kind === "aabb")
                overlapAABB(
                    world,
                    { lowerBound: { x: -2, y: -2, z: -2 }, upperBound: { x: 10, y: 2, z: 2 } },
                    callback,
                    filter,
                );
            else
                overlapShape(
                    world,
                    origin,
                    { points: [origin], count: 1, radius: 10 },
                    callback,
                    filter,
                );
            expect(calls).toBe(3);
        } finally {
            await dispose(world);
        }
    });
}
