import { expect, test } from "bun:test";
import type { ShapeProxy } from "../collision/distance";
import type { Pos, Transform, WorldTransform } from "../common/math";
import { createSphereShape } from "../shapes/shape";
import { createTestBody, createTestWorld, destroyTestWorld, testShapeDef } from "../testing";
import { bodyIsValid, destroyBody, makeBodyId } from "./body";
import {
    bodyCastRay,
    bodyCastShape,
    bodyCollideMover,
    bodyGetClosestPoint,
    bodyOverlapShape,
} from "./query";

const zero: Pos = { x: 0, y: 0, z: 0 };
const identity: Transform = { p: zero, q: { v: zero, s: 1 } };
const identityWorld: WorldTransform = identity;
const proxy: ShapeProxy = { points: [zero], count: 1, radius: 0.1 };
const rayOrigin = { x: 0, y: 0, z: -2 };
const rayTranslation = { x: 0, y: 0, z: 4 };
const mover = {
    center1: { x: 0, y: -0.25, z: 0 },
    center2: { x: 0, y: 0.25, z: 0 },
    radius: 0.4,
};

test("body queries reject a retained generation after its slot is reused", () => {
    const world = createTestWorld({ gravity: zero, enableSleep: false });
    try {
        const originalBody = createTestBody(world);
        const retained = makeBodyId(world, originalBody);
        destroyBody(world, originalBody);

        const replacementBody = createTestBody(world);
        expect(replacementBody).toBe(originalBody);
        createSphereShape(world, replacementBody, testShapeDef(), { center: zero, radius: 1 });
        const replacement = makeBodyId(world, replacementBody);
        expect(bodyIsValid(world, retained)).toBe(false);
        expect(bodyIsValid(world, replacement)).toBe(true);

        expect(bodyCastRay(world, replacement, rayOrigin, rayTranslation, identity)).toMatchObject({
            hit: true,
        });
        expect(
            bodyCastShape(world, replacement, rayOrigin, proxy, rayTranslation, identity),
        ).toMatchObject({ hit: true });
        expect(bodyOverlapShape(world, replacement, zero, proxy, identity)).toBe(true);
        expect(
            bodyGetClosestPoint(world, replacement, { x: 0, y: 0, z: 2 }).distance,
        ).toBeGreaterThan(0);
        expect(bodyCollideMover(world, replacement, zero, mover, identityWorld)).not.toHaveLength(
            0,
        );

        expect(bodyCastRay(world, retained, rayOrigin, rayTranslation, identity)).toMatchObject({
            hit: false,
        });
        expect(
            bodyCastShape(world, retained, rayOrigin, proxy, rayTranslation, identity),
        ).toMatchObject({ hit: false });
        expect(bodyOverlapShape(world, retained, zero, proxy, identity)).toBe(false);
        expect(bodyGetClosestPoint(world, retained, { x: 0, y: 0, z: 2 })).toEqual({
            point: zero,
            distance: 0,
        });
        expect(bodyCollideMover(world, retained, zero, mover, identityWorld)).toEqual([]);
    } finally {
        destroyTestWorld(world);
    }
});
