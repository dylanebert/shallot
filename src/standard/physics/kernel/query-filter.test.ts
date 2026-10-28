import { expect, test } from "bun:test";
import { createTestWorld, destroyTestWorld } from "../testing";
import { queryColumns } from "./querycolumns";

test("query filter words survive default queries and refresh when custom bits change", () => {
    const world = createTestWorld();
    try {
        const q = queryColumns(world);
        const origin = { x: 0, y: 0, z: 0 };
        const a = { categoryBits: 0x123456789abcdef0n, maskBits: 0xfedcba9876543210n };
        const b = { categoryBits: 0x0000000100000002n, maskBits: 0x0000000300000004n };
        for (const filter of [a, undefined, a, b, undefined, b, a]) {
            q.prepare(origin, filter);
            const category = filter?.categoryBits ?? 0xffffffffffffffffn;
            const mask = filter?.maskBits ?? 0xffffffffffffffffn;
            expect(Array.from(q.headerU.subarray(6, 10))).toEqual([
                Number((category >> 32n) & 0xffffffffn),
                Number(category & 0xffffffffn),
                Number((mask >> 32n) & 0xffffffffn),
                Number(mask & 0xffffffffn),
            ]);
        }
    } finally {
        destroyTestWorld(world);
    }
});
