import { expect, test } from "bun:test";
import { World } from "../../../engine";
import { physicsWorld, StandardPhysicsPlugin } from "../index";
import { shapeBodyId } from "./filtercolumns";
import { queryColumns } from "./querycolumns";

test("callback-free mover planes publish the first eight callback records and prepare clears body exclusion", async () => {
    const world = new World();
    await StandardPhysicsPlugin.initialize!(world);
    await StandardPhysicsPlugin.warm!(world);
    try {
        const physics = physicsWorld(world)!;
        const origin = { x: 0, y: 0, z: 0 };
        const capsule = {
            center1: { x: 0, y: -0.5, z: 0 },
            center2: { x: 0, y: 0.5, z: 0 },
            radius: 0.3,
        };
        for (let i = 0; i < 12; ++i)
            physics
                .createBody({ position: { x: 0.2, y: 0, z: 0 } })
                .createSphere({}, { center: origin, radius: 0.4 });
        const expected: { shape: number; values: number[] }[] = [];
        physics.collideMover(origin, capsule, (shape, planes) => {
            for (const result of planes)
                expected.push({
                    shape: shape.id.index1 - 1,
                    values: [
                        result.plane.normal.x,
                        result.plane.normal.y,
                        result.plane.normal.z,
                        result.plane.offset,
                        result.point.x,
                        result.point.y,
                        result.point.z,
                    ],
                });
            return true;
        });
        expect(expected).toHaveLength(12);
        const q = queryColumns(physics.state);
        const k = q.prepare(origin);
        q.mover(capsule.center1, capsule.center2, capsule.radius);
        const excluded = shapeBodyId(physics.state, expected[0].shape);
        q.headerU[19] = excluded + 1;
        k.worldQuery(physics.state.worldId, 5, 0);
        expect(q.resultU[0]).toBe(8);
        for (let i = 0; i < 8; ++i)
            expect(shapeBodyId(physics.state, q.resultU[16 + 8 * i])).not.toBe(excluded);
        q.prepare(origin);
        expect(q.headerU[19]).toBe(0);
        k.worldQuery(physics.state.worldId, 5, 0);
        expect(q.resultU[0]).toBe(8);
        for (let i = 0; i < 8; ++i) {
            const n = 16 + 8 * i;
            expect(q.resultU[n]).toBe(expected[i].shape);
            expect(Array.from(q.resultF.subarray(n + 1, n + 8))).toEqual(expected[i].values);
        }
    } finally {
        await StandardPhysicsPlugin.dispose!(world);
        world.dispose();
    }
});
