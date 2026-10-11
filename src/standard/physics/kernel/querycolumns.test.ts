import { expect, test } from "bun:test";
import { World } from "../../../engine";
import { StandardPhysicsPlugin } from "../index";
import { physicsState } from "../runtime";
import { createSphereShape } from "../shapes/shape";
import { createTestBody, testShapeDef } from "../testing";
import { shapeBodyId } from "./filtercolumns";
import { kernel } from "./kernel";
import { queryColumns } from "./querycolumns";

test("callback-free mover planes publish the first eight records and prepare clears body exclusion", async () => {
    const world = new World();
    await StandardPhysicsPlugin.initialize!(world);
    await StandardPhysicsPlugin.warm!(world);
    try {
        const state = physicsState(world)!;
        const origin = { x: 0, y: 0, z: 0 };
        const capsule = {
            center1: { x: 0, y: -0.5, z: 0 },
            center2: { x: 0, y: 0.5, z: 0 },
            radius: 0.3,
        };
        for (let i = 0; i < 12; ++i) {
            const body = createTestBody(state, { position: { x: 0.2, y: 0, z: 0 } });
            createSphereShape(state, body, testShapeDef(), { center: origin, radius: 0.4 });
        }
        const q = queryColumns(state);
        const k = kernel(state.ecsState);
        q.prepare(origin);
        q.mover(capsule.center1, capsule.center2, capsule.radius);
        k.worldQuery(state.worldId, 5, 0);
        expect(q.resultU[0]).toBe(8);
        const expected = Array.from({ length: 8 }, (_, i) => {
            const n = 16 + 8 * i;
            return {
                shape: q.resultU[n]!,
                values: Array.from(q.resultF.subarray(n + 1, n + 8)),
            };
        });
        const excluded = shapeBodyId(state, expected[0]!.shape);
        q.headerU[19] = excluded + 1;
        k.worldQuery(state.worldId, 5, 0);
        expect(q.resultU[0]).toBe(8);
        for (let i = 0; i < 8; ++i)
            expect(shapeBodyId(state, q.resultU[16 + 8 * i]!)).not.toBe(excluded);
        q.prepare(origin);
        expect(q.headerU[19]).toBe(0);
        k.worldQuery(state.worldId, 5, 0);
        expect(q.resultU[0]).toBe(8);
        for (let i = 0; i < 8; ++i) {
            const n = 16 + 8 * i;
            expect(q.resultU[n]).toBe(expected[i]!.shape);
            expect(Array.from(q.resultF.subarray(n + 1, n + 8))).toEqual(expected[i]!.values);
        }
    } finally {
        await StandardPhysicsPlugin.dispose!(world);
        world.dispose();
    }
});
