import { expect, test } from "bun:test";
import type { ShapeProxy } from "../collision/distance";
import type { Pos, Transform, WorldTransform } from "../common/records";
import { BodyType } from "../common/types";
import { kernel } from "../kernel/kernel";
import { createSphereShape } from "../shapes/shape";
import {
    createTestBody,
    createTestWorld,
    destroyTestWorld,
    stepTestWorld,
    testShapeDef,
} from "../testing";
import { bodyIsValid, destroyBody, makeBodyId } from "./body";
import {
    bodyCastRay,
    bodyCastShape,
    bodyCollideMover,
    bodyGetClosestPoint,
    bodyOverlapShape,
    worldCastRayClosest,
    worldOverlapAABB,
} from "./query";
import type { WorldState } from "./world";

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

for (const type of [BodyType.Static, BodyType.Dynamic]) {
    test(`two worlds in one kernel query their own trees before a step and after the other's edits (body type ${type})`, () => {
        const ray = { x: 10, y: 0, z: 0 };
        const sphere = (world: WorldState, x: number) =>
            createSphereShape(
                world,
                createTestBody(world, { type, position: { x, y: 0, z: 0 } }),
                testShapeDef(),
                { center: zero, radius: 0.5 },
            );
        const overlaps = (world: WorldState) => {
            let count = 0;
            worldOverlapAABB(
                world,
                { lowerBound: { x: -20, y: -1, z: -1 }, upperBound: { x: 20, y: 1, z: 1 } },
                () => ++count > 0,
            );
            return count;
        };
        const a = createTestWorld({ gravity: zero });
        const b = createTestWorld({ gravity: zero });
        try {
            sphere(a, 2);
            expect(worldCastRayClosest(a, zero, ray).fraction).toBeCloseTo(0.15, 6);
            sphere(b, 5);
            expect(worldCastRayClosest(a, zero, ray).fraction).toBeCloseTo(0.15, 6);
            expect(worldCastRayClosest(b, zero, ray).fraction).toBeCloseTo(0.45, 6);
            expect([overlaps(a), overlaps(b)]).toEqual([1, 1]);
            sphere(a, 1);
            kernel(undefined).memory.grow(4);
            expect(worldCastRayClosest(b, zero, ray).fraction).toBeCloseTo(0.45, 6);
            expect(worldCastRayClosest(a, zero, ray).fraction).toBeCloseTo(0.05, 6);
            for (let i = 0; i < 40; i++) sphere(b, 8 + i * 0.01);
            expect([overlaps(a), overlaps(b)]).toEqual([2, 41]);
            stepTestWorld(a, 1 / 60);
            stepTestWorld(b, 1 / 60);
            expect(worldCastRayClosest(a, zero, ray).fraction).toBeCloseTo(0.05, 6);
            expect(worldCastRayClosest(b, zero, ray).fraction).toBeCloseTo(0.45, 6);
            expect([overlaps(a), overlaps(b)]).toEqual([2, 41]);
        } finally {
            destroyTestWorld(b);
            destroyTestWorld(a);
        }
    });
}
