import { expect, test } from "bun:test";
import type { ShapeProxy } from "../collision/distance";
import type { EntityId } from "../common/ids";
import type { AABB, Pos, Transform, WorldTransform } from "../common/math";
import { createTestBody, createTestWorld, destroyTestWorld } from "../testing";
import { makeBodyId } from "./body";
import {
    bodyCastRay,
    bodyCastShape,
    bodyCollideMover,
    bodyGetClosestPoint,
    bodyOverlapShape,
    worldCastMover,
    worldCastRay,
    worldCastRayClosest,
    worldCastShape,
    worldCollideMover,
    worldOverlapAABB,
    worldOverlapShape,
} from "./query";

const zero: Pos = { x: 0, y: 0, z: 0 };
const transform: Transform = { p: zero, q: { v: zero, s: 1 } };
const worldTransform: WorldTransform = transform;
const proxy: ShapeProxy = { points: [zero], count: 1, radius: 0 };
const mover = { center1: zero, center2: zero, radius: 1 };
const blockedOrigin = {
    get x(): number {
        throw new Error("query prepared while unavailable");
    },
    y: 0,
    z: 0,
};
const blockedTarget = {
    get x(): number {
        throw new Error("query prepared while unavailable");
    },
    y: 0,
    z: 0,
};
const callback = () => {
    throw new Error("query callback ran while unavailable");
};

function expectUnavailableQueries(world: ReturnType<typeof createTestWorld>, id: EntityId) {
    const aabb: AABB = { lowerBound: blockedOrigin, upperBound: blockedOrigin };
    expect(worldCastRayClosest(world, blockedOrigin, zero).hit).toBe(false);
    expect(worldCastRay(world, blockedOrigin, zero, callback)).toEqual({
        nodeVisits: 0,
        leafVisits: 0,
    });
    expect(worldOverlapAABB(world, aabb, callback)).toEqual({ nodeVisits: 0, leafVisits: 0 });
    expect(worldOverlapShape(world, blockedOrigin, proxy, callback)).toEqual({
        nodeVisits: 0,
        leafVisits: 0,
    });
    expect(worldCastShape(world, blockedOrigin, proxy, zero, callback)).toEqual({
        nodeVisits: 0,
        leafVisits: 0,
    });
    expect(worldCollideMover(world, blockedOrigin, mover, callback)).toBeUndefined();
    expect(worldCastMover(world, blockedOrigin, mover, zero)).toBe(1);
    expect(bodyCastRay(world, id, blockedOrigin, zero, transform).hit).toBe(false);
    expect(bodyCastShape(world, id, blockedOrigin, proxy, zero, transform).hit).toBe(false);
    expect(bodyOverlapShape(world, id, blockedOrigin, proxy, transform)).toBe(false);
    expect(bodyGetClosestPoint(world, id, blockedTarget)).toEqual({
        point: zero,
        distance: 0,
    });
    expect(bodyCollideMover(world, id, blockedOrigin, mover, worldTransform)).toEqual([]);
}

test("locked and destroyed direct queries return empty results before preparing inputs", () => {
    const world = createTestWorld();
    const body = createTestBody(world);
    const id = makeBodyId(world, body);
    world.locked = true;
    try {
        expectUnavailableQueries(world, id);
        world.locked = false;
        destroyTestWorld(world);
        expectUnavailableQueries(world, id);
    } finally {
        if (world.inUse) {
            world.locked = false;
            destroyTestWorld(world);
        }
    }
});
