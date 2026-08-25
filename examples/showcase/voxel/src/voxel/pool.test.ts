import { describe, expect, test } from "bun:test";
import { RegionAllocator } from "./pool";

describe("RegionAllocator", () => {
    test("starts as one free region spanning the whole capacity", () => {
        const pool = new RegionAllocator(100);
        expect(pool.regions()).toEqual([{ start: 0, size: 100 }]);
    });

    test("alloc hands back a contiguous region and shrinks the free list from the front", () => {
        const pool = new RegionAllocator(100);
        const a = pool.alloc(30);
        expect(a).toEqual({ start: 0, size: 30 });
        expect(pool.regions()).toEqual([{ start: 30, size: 70 }]);
    });

    test("an exact-size alloc consumes the whole free region, not just its size", () => {
        const pool = new RegionAllocator(10);
        const a = pool.alloc(10);
        expect(a).toEqual({ start: 0, size: 10 });
        expect(pool.regions()).toEqual([]);
    });

    test("first-fit: a later alloc lands in the first free region large enough, not the largest", () => {
        const pool = new RegionAllocator(100);
        const a = pool.alloc(10)!; // [0,10)
        const b = pool.alloc(20)!; // [10,30)
        pool.free(a.start); // free list: [0,10), [30,70)
        const c = pool.alloc(5); // fits the smaller hole first
        expect(c).toEqual({ start: 0, size: 5 });
        expect(pool.regions()).toEqual([
            { start: 5, size: 5 },
            { start: 30, size: 70 },
        ]);
        expect(b).toEqual({ start: 10, size: 20 });
    });

    test("alloc returns null (exhaustion signal) when no free region is large enough", () => {
        const pool = new RegionAllocator(10);
        pool.alloc(6);
        expect(pool.alloc(5)).toBeNull();
        expect(pool.alloc(4)).toEqual({ start: 6, size: 4 });
    });

    test("free coalesces with both neighbours, restoring one contiguous region", () => {
        const pool = new RegionAllocator(30);
        const a = pool.alloc(10)!; // [0,10)
        const b = pool.alloc(10)!; // [10,20)
        const c = pool.alloc(10)!; // [20,30)
        pool.free(a.start);
        pool.free(c.start);
        expect(pool.regions()).toEqual([
            { start: 0, size: 10 },
            { start: 20, size: 10 },
        ]);
        pool.free(b.start); // bridges the two, coalescing all three back into one
        expect(pool.regions()).toEqual([{ start: 0, size: 30 }]);
    });

    test("free of an unknown start throws rather than silently corrupting the free list", () => {
        const pool = new RegionAllocator(10);
        expect(() => pool.free(4)).toThrow();
    });

    test("freeing the same region twice throws on the second call", () => {
        const pool = new RegionAllocator(10);
        const a = pool.alloc(5)!;
        pool.free(a.start);
        expect(() => pool.free(a.start)).toThrow();
    });

    test("alloc of a non-positive size throws", () => {
        const pool = new RegionAllocator(10);
        expect(() => pool.alloc(0)).toThrow();
        expect(() => pool.alloc(-1)).toThrow();
    });

    test("repeated churn (alloc/free cycles) never grows the free list beyond one region on full release", () => {
        const pool = new RegionAllocator(64);
        for (let i = 0; i < 50; i++) {
            const regions = [pool.alloc(8), pool.alloc(16), pool.alloc(4)];
            for (const r of regions) pool.free(r!.start);
        }
        expect(pool.regions()).toEqual([{ start: 0, size: 64 }]);
    });

    test("a zero-capacity pool starts empty and every alloc signals exhaustion", () => {
        const pool = new RegionAllocator(0);
        expect(pool.regions()).toEqual([]);
        expect(pool.alloc(1)).toBeNull();
    });
});
