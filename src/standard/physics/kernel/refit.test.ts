import { expect, test } from "bun:test";
import { BodyType, createMesh, PhysicsWorld } from "../api";
import { SPECULATIVE_DISTANCE } from "../common/constants";
import { aabb } from "../common/math";
import { computeFatShapeAABB } from "../shapes/shape";
import { readFatAabb } from "./shapecolumns";

test("kernel finalization commits a moving non-convex mesh's tight and fat bounds", () => {
    const world = new PhysicsWorld({
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
        const body = world.createBody({
            type: BodyType.Kinematic,
            linearVelocity: { x: 12, y: 0, z: 0 },
            angularVelocity: { x: 0, y: 1, z: 0 },
        });
        const shape = body.createMesh({}, mesh, { x: -2, y: 1, z: 0.5 });
        const id = shape.id.index1 - 1;
        const initialFat = readFatAabb(world.state, id, {
            lowerBound: { x: 0, y: 0, z: 0 },
            upperBound: { x: 0, y: 0, z: 0 },
        });
        for (let i = 0; i < 4; ++i) {
            world.step(1 / 60, 4);
            const expected = computeFatShapeAABB(
                world.state.shapes[id],
                body.getTransform(),
                SPECULATIVE_DISTANCE,
            );
            expect(shape.getAABB()).toEqual(expected);
            const fat = readFatAabb(world.state, id, {
                lowerBound: { x: 0, y: 0, z: 0 },
                upperBound: { x: 0, y: 0, z: 0 },
            });
            expect(aabb.contains(fat, expected)).toBe(true);
            expect(fat).not.toEqual(initialFat);
        }
    } finally {
        world.destroy();
    }
});
