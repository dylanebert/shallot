import { describe, expect, test } from "bun:test";
import { buildTable, type ChunkPool, createChunkPool, fullRemesh, planRemesh } from "./chunkpool";
import {
    CHUNK_CELLS,
    chunkNeighbors,
    chunkSlot,
    faces,
    facesInChunk,
    SLOT_COUNT,
    set,
} from "./grid";
import { checker, single, sphere } from "./patterns";

const TOTAL_CELLS = SLOT_COUNT * CHUNK_CELLS;

function sumLiveRegions(pool: ChunkPool): number {
    let n = 0;
    for (const region of pool.region) if (region) n += region.size;
    return n;
}

describe("createChunkPool", () => {
    test("starts with every slot empty and a zero tail", () => {
        const pool = createChunkPool(100);
        expect(pool.tail).toBe(0);
        expect(pool.region.every((r) => r === null)).toBe(true);
        expect(pool.region.length).toBe(SLOT_COUNT);
    });
});

describe("fullRemesh", () => {
    test("allocates every non-empty chunk exactly its facesInChunk count, packed with no gaps", () => {
        const data = new Float32Array(TOTAL_CELLS);
        single(data, 40, 50, 60);
        const pool = fullRemesh(1024, data);

        const slot = chunkSlot(40, 50, 60);
        expect(pool.region[slot]).toEqual({ start: 0, size: 6 });
        expect(pool.tail).toBe(6);
        expect(sumLiveRegions(pool)).toBe(faces(data));

        // every other slot is empty — this fixture touches exactly one chunk.
        for (let s = 0; s < SLOT_COUNT; s++) {
            if (s === slot) continue;
            expect(pool.region[s]).toBeNull();
        }
    });

    test("packs multiple non-empty chunks back-to-back in ascending slot order", () => {
        const data = new Float32Array(TOTAL_CELLS);
        checker(data, 0, 0, 0, 16, 16, 16); // slot 0 — 6 · 16³/2 = 12,288 faces
        single(data, 200, 200, 200); // a later slot
        const pool = fullRemesh(20_000, data);

        const first = chunkSlot(0, 0, 0);
        const second = chunkSlot(200, 200, 200);
        expect(first).toBeLessThan(second);
        const a = pool.region[first]!;
        const b = pool.region[second]!;
        expect(a.start).toBe(0);
        expect(a.size).toBe(facesInChunk(data, first));
        // packed: the second region starts exactly where the first ends, even though many empty slots
        // sit between them in slot order.
        expect(b.start).toBe(a.size);
        expect(b.size).toBe(facesInChunk(data, second));
        expect(pool.tail).toBe(a.size + b.size);
    });

    test("a grid exceeding capacity silently caps — over-budget chunks stay unallocated", () => {
        const data = new Float32Array(TOTAL_CELLS);
        single(data, 40, 50, 60); // 6 faces
        const pool = fullRemesh(3, data); // smaller than the one chunk's 6 faces
        const slot = chunkSlot(40, 50, 60);
        expect(pool.region[slot]).toBeNull();
        expect(pool.tail).toBe(0);
    });
});

describe("planRemesh", () => {
    test("allocates a newly-touched empty slot and leaves everything else alone", () => {
        const data = new Float32Array(TOTAL_CELLS);
        single(data, 40, 50, 60);
        const pool = createChunkPool(1024);
        const slot = chunkSlot(40, 50, 60);

        const plan = planRemesh(pool, data, [slot]);
        expect(plan.exhausted).toBe(false);
        expect(plan.freed).toEqual([]);
        expect(pool.region[slot]).toEqual({ start: 0, size: 6 });
        expect(pool.tail).toBe(6);
    });

    test("a chunk whose face count is unchanged keeps its region (no free, no realloc)", () => {
        const data = new Float32Array(TOTAL_CELLS);
        single(data, 40, 50, 60);
        const pool = createChunkPool(1024);
        const slot = chunkSlot(40, 50, 60);
        planRemesh(pool, data, [slot]);
        const before = pool.region[slot];

        const plan = planRemesh(pool, data, [slot]); // re-plan the same unchanged chunk
        expect(plan.freed).toEqual([]);
        expect(pool.region[slot]).toBe(before);
    });

    test("a grown chunk frees its old region and grants a new, larger one", () => {
        const data = new Float32Array(TOTAL_CELLS);
        single(data, 40, 50, 60); // 6 faces
        const pool = createChunkPool(4096);
        const slot = chunkSlot(40, 50, 60);
        planRemesh(pool, data, [slot]);
        const oldRegion = pool.region[slot]!;

        // grows the same chunk's occupancy well past 6 faces (6 · 8³/2 = 1,536), still under capacity
        checker(data, 32, 32, 32, 8, 8, 8);
        const plan = planRemesh(pool, data, [slot]);
        expect(plan.freed).toEqual([oldRegion]);
        expect(plan.exhausted).toBe(false);
        const grown = pool.region[slot]!;
        expect(grown.size).toBe(facesInChunk(data, slot));
        expect(grown.size).toBeGreaterThan(oldRegion.size);
    });

    test("a chunk carved to empty frees its region and stays unallocated", () => {
        const data = new Float32Array(TOTAL_CELLS);
        single(data, 40, 50, 60);
        const pool = createChunkPool(1024);
        const slot = chunkSlot(40, 50, 60);
        planRemesh(pool, data, [slot]);
        const region = pool.region[slot]!;

        data.fill(0); // carve everything away
        const plan = planRemesh(pool, data, [slot]);
        expect(plan.freed).toEqual([region]);
        expect(pool.region[slot]).toBeNull();
    });

    test("exhaustion signals without granting the resized chunk a region", () => {
        const data = new Float32Array(TOTAL_CELLS);
        single(data, 40, 50, 60); // 6 faces, capacity 6 leaves no room to grow
        const pool = createChunkPool(6);
        const slot = chunkSlot(40, 50, 60);
        planRemesh(pool, data, [slot]);

        checker(data, 32, 32, 32, 8, 8, 8); // grows past the fixed 6-face capacity
        const plan = planRemesh(pool, data, [slot]);
        expect(plan.exhausted).toBe(true);
    });

    test("differential — Σ live region sizes always equals faces() after any sequence of plans", () => {
        const data = new Float32Array(TOTAL_CELLS);
        sphere(data, 128, 128, 128, 40);
        const pool = fullRemesh(1 << 20, data);
        expect(sumLiveRegions(pool)).toBe(faces(data));

        // carve a hole straddling a chunk seam (mirrors grid.test.ts's mutation fixture) and re-plan the
        // touched+halo set the same way `mesher.ts`'s `commitEdit` would (`chunkNeighbors` is its halo
        // primitive).
        const touched = new Set<number>();
        for (let z = 125; z <= 131; z++) {
            for (let y = 125; y <= 131; y++) {
                for (let x = 125; x <= 131; x++) {
                    touched.add(chunkSlot(x, y, z));
                    set(data, x, y, z, 0);
                }
            }
        }
        const affected = new Set<number>();
        for (const slot of touched) {
            affected.add(slot);
            for (const neighbor of chunkNeighbors(slot)) affected.add(neighbor);
        }
        const plan = planRemesh(pool, data, [...affected]);
        expect(plan.exhausted).toBe(false);
        expect(sumLiveRegions(pool)).toBe(faces(data));
    });
});

describe("buildTable", () => {
    test("encodes (start, capacity) per slot, zero for an unallocated chunk", () => {
        const data = new Float32Array(TOTAL_CELLS);
        single(data, 40, 50, 60);
        const pool = fullRemesh(1024, data);
        const slot = chunkSlot(40, 50, 60);

        const table = buildTable(pool);
        expect(table.length).toBe(SLOT_COUNT * 2);
        expect(table[slot * 2]).toBe(0);
        expect(table[slot * 2 + 1]).toBe(6);

        const other = (slot + 1) % SLOT_COUNT;
        expect(table[other * 2]).toBe(0);
        expect(table[other * 2 + 1]).toBe(0);
    });
});
