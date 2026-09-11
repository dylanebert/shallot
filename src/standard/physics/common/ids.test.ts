import { expect } from "bun:test";
import { check } from "../../../harness/check";
import { allocId, createIdPool, freeId, idCapacity, idCount, loadId, storeId } from "./ids";

// Ports test_id.c: the store/load roundtrip. Body/shape/joint ids share one packing, so one
// pair covers all three.

check(
    "storeId after loadId is the identity on a packed id",
    {
        claim: "the entity id packing loses or reorders bits on a full round trip through loadId and storeId",
    },
    () => {
        const x = 0x0123456789abcdefn;
        expect(storeId(loadId(x))).toBe(x);
    },
);

check(
    "id fields decode from the u64 layout",
    {
        claim: "loadId reads index1, world0 or generation from the wrong bit field of the packed u64",
    },
    () => {
        const id = loadId(0x0123456789abcdefn);
        expect(id.index1).toBe(0x01234567);
        expect(id.world0).toBe(0x89ab);
        expect(id.generation).toBe(0xcdef);
    },
);

check(
    "a high-bit index1 sign-extends and still round-trips",
    {
        claim: "an index1 with its top bit set fails to sign-extend on load, so a negative index round-trips as a large positive one",
    },
    () => {
        const id = { index1: -1, world0: 0x1234, generation: 0x5678 };
        expect(loadId(storeId(id))).toEqual(id);
    },
);

check(
    "the id pool hands out dense ids and reuses freed ones last in first out",
    {
        claim: "the id pool leaves holes in its dense range, miscounts live ids against capacity, or recycles freed ids in the wrong order",
    },
    () => {
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
    },
);
