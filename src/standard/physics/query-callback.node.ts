import { expect, setDefaultTimeout, test } from "bun:test";
import { TIMEOUT } from "@dylanebert/shallot/testing";
import { BodyType } from "./common/types";
import { queryColumns } from "./kernel/querycolumns";
import { createSphereShape } from "./shapes/shape";
import { createTestBody, createTestWorld, destroyTestWorld, testShapeDef } from "./testing";
import { worldCastRay } from "./world/query";

setDefaultTimeout(TIMEOUT.node);

test("a cast callback throw stops at the first leaf, propagates, and leaves the next query usable", () => {
    const world = createTestWorld();
    try {
        for (let i = 0; i < 200; i++) {
            const body = createTestBody(world, {
                type: BodyType.Static,
                position: { x: 2 + i * 2, y: 0, z: 0 },
            });
            createSphereShape(world, body, testShapeDef(), {
                center: { x: 0, y: 0, z: 0 },
                radius: 0.5,
            });
        }
        const origin = { x: 0, y: 0, z: 0 };
        const translation = { x: 500, y: 0, z: 0 };
        // Leave the cast return word at 1 (continue), not 0 (terminate).
        const all = worldCastRay(world, origin, translation, () => 1);
        expect(all.leafVisits).toBe(200);

        const error = new Error("cast callback failed");
        let calls = 0;
        expect(() =>
            worldCastRay(world, origin, translation, () => {
                calls++;
                throw error;
            }),
        ).toThrow(error);
        expect(calls).toBe(1);
        // A throwing query cannot return its stats; read the kernel's completed result.
        expect(queryColumns(world).resultU[2]).toBe(1);

        calls = 0;
        const next = worldCastRay(world, origin, translation, () => {
            calls++;
            return 1;
        });
        expect(calls).toBe(200);
        expect(next).toEqual(all);
    } finally {
        destroyTestWorld(world);
    }
});
