import { expect, test } from "bun:test";
import { ContactFlags } from "../collision/contact";
import { BodyType, defaultBodyDef, defaultShapeDef, defaultWorldDef } from "../common/types";
import { makeBoxHull } from "../shapes/hull";
import { createHullShape, createSphereShape } from "../shapes/shape";
import { step } from "../solver/step";
import { createBody } from "../world/body";
import { createWorld, destroyWorld, getWorld, type WorldState } from "../world/world";
import { init } from "./kernel";

await init(undefined, { threads: 0 });

function scene(height: number): WorldState {
    const world = getWorld(
        createWorld(undefined, { ...defaultWorldDef(), gravity: { x: 0, y: 0, z: 0 } }),
    ) as WorldState;
    const ground = world.bodies[createBody(world, defaultBodyDef())];
    createHullShape(world, ground, defaultShapeDef(), makeBoxHull(2, height, 2));
    // Recycling would retain the old touching manifold without reading the evicted hull pool.
    const ball =
        world.bodies[
            createBody(world, {
                ...defaultBodyDef(),
                type: BodyType.Dynamic,
                enableContactRecycling: false,
                position: { x: 0, y: 1.5, z: 0 },
            })
        ];
    createSphereShape(world, ball, defaultShapeDef(), {
        center: { x: 0, y: 0, z: 0 },
        radius: 0.5,
    });
    return world;
}
const touching = (world: WorldState): number =>
    world.contacts.filter((c) => (c.flags & ContactFlags.contactTouchingFlag) !== 0).length;

test("alternating standalone worlds collide against their own hull geometry after another world takes residency", () => {
    const a = scene(1);
    let b: WorldState | undefined;
    try {
        step(a, 1 / 60, 4);
        expect(touching(a)).toBe(1);
        b = scene(0.25);
        step(b, 1 / 60, 4);
        expect(touching(b)).toBe(0);
        step(a, 1 / 60, 4);
        expect(touching(a)).toBe(1);
        step(b, 1 / 60, 4);
        expect(touching(b)).toBe(0);
    } finally {
        destroyWorld(a);
        if (b) destroyWorld(b);
    }
});
