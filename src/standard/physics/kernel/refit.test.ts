import { expect, test } from "bun:test";
import { SPECULATIVE_DISTANCE } from "../common/constants";
import { aabb } from "../common/math";
import { BodyType } from "../common/types";
import { createMesh } from "../shapes/mesh";
import { computeFatShapeAABB, createMeshShape } from "../shapes/shape";
import {
    createTestBody,
    createTestWorld,
    destroyTestWorld,
    stepTestWorld,
    testShapeDef,
} from "../testing";
import { bodyGetTransform } from "../world/body";
import { readFatAabb, readShapeAabb } from "./shapecolumns";

test("kernel finalization commits a moving non-convex mesh's tight and fat bounds", () => {
    const world = createTestWorld({
        gravity: { x: 0, y: 0, z: 0 },
        enableSleep: false,
        enableContinuous: false,
    });
    try {
        const mesh = createMesh({
            vertices: [
                { x: -2, y: -1, z: -2 },
                { x: 2, y: 0, z: -2 },
                { x: 2, y: 1, z: 2 },
            ],
            indices: [0, 2, 1],
            identifyEdges: true,
        });
        if (!mesh) throw new Error("mesh construction failed");
        const body = createTestBody(world, {
            type: BodyType.Kinematic,
            linearVelocity: { x: 12, y: 0, z: 0 },
            angularVelocity: { x: 0, y: 1, z: 0 },
        });
        const shape = createMeshShape(world, body, testShapeDef(), mesh, { x: -2, y: 1, z: 0.5 })!;
        const initialFat = readFatAabb(world, shape, {
            lowerBound: { x: 0, y: 0, z: 0 },
            upperBound: { x: 0, y: 0, z: 0 },
        });
        for (let i = 0; i < 4; ++i) {
            stepTestWorld(world, 1 / 60, 4);
            const expected = computeFatShapeAABB(
                world,
                shape,
                bodyGetTransform(world, body, {
                    p: { x: 0, y: 0, z: 0 },
                    q: { v: { x: 0, y: 0, z: 0 }, s: 1 },
                }),
                SPECULATIVE_DISTANCE,
            );
            expect(
                readShapeAabb(world, shape, {
                    lowerBound: { x: 0, y: 0, z: 0 },
                    upperBound: { x: 0, y: 0, z: 0 },
                }),
            ).toEqual(expected);
            const fat = readFatAabb(world, shape, {
                lowerBound: { x: 0, y: 0, z: 0 },
                upperBound: { x: 0, y: 0, z: 0 },
            });
            expect(aabb.contains(fat, expected)).toBe(true);
            expect(fat).not.toEqual(initialFat);
        }
    } finally {
        destroyTestWorld(world);
    }
});
