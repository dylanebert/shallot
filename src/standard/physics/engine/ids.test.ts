import { describe, expect, test } from "bun:test";
import { allocId, createIdPool, freeId, idCapacity, idCount, loadId, storeId } from "./ids";

// Ports test_id.c: the store/load roundtrip. Body/shape/joint ids share one packing, so one
// pair covers all three.
describe("id packing", () => {
    test("store∘load is identity", () => {
        const x = 0x0123456789abcdefn;
        expect(storeId(loadId(x))).toBe(x);
    });

    test("fields decode from the u64 layout", () => {
        const id = loadId(0x0123456789abcdefn);
        expect(id.index1).toBe(0x01234567);
        expect(id.world0).toBe(0x89ab);
        expect(id.generation).toBe(0xcdef);
    });

    test("a high-bit index1 sign-extends and still roundtrips", () => {
        const id = { index1: -1, world0: 0x1234, generation: 0x5678 };
        expect(loadId(storeId(id))).toEqual(id);
    });
});

describe("id pool", () => {
    test("hands out dense ids, then reuses freed ones LIFO", () => {
        const pool = createIdPool();
        expect(allocId(pool)).toBe(0);
        expect(allocId(pool)).toBe(1);
        expect(allocId(pool)).toBe(2);
        expect(idCount(pool)).toBe(3);
        expect(idCapacity(pool)).toBe(3);

        freeId(pool, 0);
        freeId(pool, 1);
        expect(idCount(pool)).toBe(1);
        // LIFO: last freed comes back first.
        expect(allocId(pool)).toBe(1);
        expect(allocId(pool)).toBe(0);
        // Range exhausted again, extend.
        expect(allocId(pool)).toBe(3);
        expect(idCapacity(pool)).toBe(4);
    });
});
