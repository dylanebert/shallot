import { describe, expect, test } from "bun:test";
import { documentDirtyTiles, type Polyline, type StrokeDocument } from "./document";
import { generateNetwork, ROAD_HALF_WIDTH, ROAD_MAX_LENGTH, ROAD_MIN_LENGTH } from "./network";
import { allocate, drain, invalidate, release } from "./queue";
import { ATLAS_LAYERS, THROTTLE, TILE_COUNT } from "./tiles";

describe("drain — the per-frame throttle", () => {
    test("never pops more than the throttle, even with a larger backlog", () => {
        const pending = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
        const set = new Set(pending);
        const popped = drain(pending, set, THROTTLE);
        expect(popped.length).toBe(THROTTLE);
        expect(pending.length).toBe(10 - THROTTLE);
        // every popped id left both the array and the de-dup set
        for (const id of popped) {
            expect(pending).not.toContain(id);
            expect(set.has(id)).toBe(false);
        }
    });

    test("pops FIFO — oldest marks redraw first", () => {
        const pending = [10, 20, 30];
        const set = new Set(pending);
        expect(drain(pending, set, 2)).toEqual([10, 20]);
        expect(pending).toEqual([30]);
    });

    test("a backlog under the throttle drains fully in one call, leaving nothing pending", () => {
        const pending = [1, 2, 3];
        const set = new Set(pending);
        const popped = drain(pending, set, THROTTLE);
        expect(popped).toEqual([1, 2, 3]);
        expect(pending.length).toBe(0);
        expect(set.size).toBe(0);
    });

    test("a burst larger than ATLAS_LAYERS still drains at exactly THROTTLE per call — the throttle bounds redraws, not capacity", () => {
        const pending = Array.from({ length: ATLAS_LAYERS + 20 }, (_, i) => i);
        const set = new Set(pending);
        let calls = 0;
        while (pending.length > 0) {
            expect(drain(pending, set, THROTTLE).length).toBeLessThanOrEqual(THROTTLE);
            calls++;
            if (calls > pending.length + ATLAS_LAYERS + 20)
                throw new Error("drain never converged");
        }
        expect(calls).toBe(Math.ceil((ATLAS_LAYERS + 20) / THROTTLE));
    });
});

describe("allocate — free-list layer assignment", () => {
    test("pops layers off the free list, writing the CPU indirection mirror in place", () => {
        const cpu = new Int32Array(8).fill(-1);
        const free = [2, 1, 0];
        expect(allocate(cpu, 3, free, 8)).toBe(0);
        expect(cpu[3]).toBe(0);
        expect(free).toEqual([2, 1]);
        expect(allocate(cpu, 5, free, 8)).toBe(1);
        expect(cpu[5]).toBe(1);
        expect(free).toEqual([2]);
    });

    test("returns the existing layer for an already-resident tile, free list unchanged", () => {
        const cpu = new Int32Array(8).fill(-1);
        const free = [1, 0];
        const first = allocate(cpu, 2, free, 8);
        const again = allocate(cpu, 2, free, 8);
        expect(again).toBe(first);
        expect(free).toEqual([1]);
    });

    test("throws rather than silently overwriting once the free list is empty", () => {
        const cpu = new Int32Array(4).fill(-1);
        const free = [1, 0];
        allocate(cpu, 0, free, 2);
        allocate(cpu, 1, free, 2);
        expect(free.length).toBe(0);
        expect(() => allocate(cpu, 2, free, 2)).toThrow(/capacity exceeded/);
    });
});

describe("release — returning layers to the free list", () => {
    test("clears indirection to -1, pushes layers back, and removes from pending", () => {
        const cpu = new Int32Array(8).fill(-1);
        const free = [3, 2, 1, 0];
        allocate(cpu, 3, free, 8);
        allocate(cpu, 5, free, 8);
        expect(cpu[3]).toBe(0);
        expect(cpu[5]).toBe(1);
        const pending = [3, 5, 7];
        const pendingSet = new Set(pending);

        release(cpu, [3], free, pending, pendingSet);

        expect(cpu[3]).toBe(-1);
        expect(free.length).toBe(3);
        expect(pending).toEqual([5, 7]);
        expect(pendingSet.has(3)).toBe(false);
        // the freed layer is immediately reusable
        expect(allocate(cpu, 3, free, 8)).toBe(0);
    });

    test("skips ids that are not resident (indirection already -1)", () => {
        const cpu = new Int32Array(8).fill(-1);
        const free = [1, 0];
        release(cpu, [2, 3], free, [], new Set());
        expect(free.length).toBe(2); // unchanged
        expect(cpu[2]).toBe(-1);
        expect(cpu[3]).toBe(-1);
    });
});

describe("invalidate — the atlas's document-swap reset", () => {
    test("releases every resident tile, refills the free list to capacity, and drops anything still queued", () => {
        const cpu = new Int32Array(8).fill(-1);
        const free: number[] = [];
        for (let i = 7; i >= 0; i--) free.push(i); // full capacity 8
        allocate(cpu, 3, free, 8);
        allocate(cpu, 5, free, 8);
        expect(cpu[3]).toBe(0);
        expect(cpu[5]).toBe(1);
        const pending = [6, 7];
        const pendingSet = new Set(pending);

        invalidate(cpu, free, 8, pending, pendingSet);

        expect(Array.from(cpu)).toEqual(new Array(8).fill(-1));
        expect(pending).toEqual([]);
        expect(pendingSet.size).toBe(0);
        expect(free.length).toBe(8);
        // the freed layers are immediately reusable
        expect(allocate(cpu, 3, free, 8)).toBeLessThanOrEqual(7);
    });

    // The regression this closes: `terrain.ts`'s `regenerate` used to call `markDirty` on the swapped-in
    // document without first releasing the outgoing document's resident layers, so repeated F9 presses
    // accumulated layers across reseeds until the fixed-size atlas ran out. `regenerate` now calls
    // `overlayAtlas.invalidate()` before `markDirty` — this drives that fixed shape (reset, then allocate)
    // against hundreds of real reseeds and asserts it never breaches ATLAS_LAYERS, since invalidation
    // means only the *current* document's own footprint is ever resident at once.
    //
    // No arm demonstrates the *unfixed* path overflowing any more, and none can: since stage 1 the road is
    // a fixed chord that does not move with the seed, so every reseed re-marks the same tiles and reseeding
    // stopped being a capacity input at all. The accumulation this guards is now reachable only from edits
    // (stage 4's drag), which is where stage 2's release path and its own red-first fixture live.
    test("real reseeds through the fixed invalidate-before-mark order never breach ATLAS_LAYERS", () => {
        const cpu = new Int32Array(TILE_COUNT).fill(-1);
        const free: number[] = [];
        const pending: number[] = [];
        const pendingSet = new Set<number>();
        let maxResident = 0;

        for (let seed = 0; seed <= 500; seed++) {
            invalidate(cpu, free, ATLAS_LAYERS, pending, pendingSet); // atlas.ts's swap invalidation
            const ids = documentDirtyTiles(generateNetwork());
            for (const id of ids) {
                allocate(cpu, id, free, ATLAS_LAYERS); // never throws post-fix
            }
            maxResident = Math.max(maxResident, ids.length);
        }

        expect(maxResident).toBeLessThanOrEqual(ATLAS_LAYERS);
        expect(maxResident).toBeGreaterThan(0); // sanity: reseeds actually allocated something
    });
});

// ─── property-based arm: tile release over random edit sequences ───────────────────────────────────
//
// RED-FIRST WITNESS (run against the pre-change free-running counter allocator, before the free list
// landed): with 65 edits each touching a fresh tile, the old counter threw at the 65th edit (0-indexed
// edit 64) because ATLAS_LAYERS is 64 and the counter never releases. The throw was:
//   "overlay atlas: capacity exceeded (64 layers) allocating tile 64"
// with nextLayer=64 at the point of failure. The free list replaces the counter: `release` pushes
// layers back between edits, so the same 65-edit sequence (and any sequence where each edit's footprint
// fits within ATLAS_LAYERS) never exhausts the pool. This arm is written fresh — stage 1 deleted the
// old cumulative-*reseed* overflow arm, since a fixed road's footprint no longer varies with the seed,
// so reseeds stopped being a capacity input.
//
// The property: over random document sequences, after every edit (retile + drain) the resident set
// equals documentDirtyTiles(current), released ids read -1, allocate never throws, and
// resident + free always sums to ATLAS_LAYERS.

const WORLD_HALF = 512;
const TILE_SIZE = 64;

/** a deterministic PRNG (mulberry32) so the property arm is reproducible across runs. */
function mulberry32(seed: number): () => number {
    let a = seed;
    return () => {
        a |= 0;
        a = (a + 0x6d2b79f5) | 0;
        let t = Math.imul(a ^ (a >>> 15), 1 | a);
        t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
        return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
}

/** a random road document whose chord stays within world bounds, between ROAD_MIN_LENGTH and
 *  ROAD_MAX_LENGTH — the same constraints the drag (stage 4) enforces, so every document's footprint
 *  fits within ATLAS_LAYERS (a 220 m road on a 64 m grid touches at most ~30 tiles). */
function randomDoc(rng: () => number): StrokeDocument {
    const margin = ROAD_HALF_WIDTH + 1;
    const lo = -WORLD_HALF + margin;
    const hi = WORLD_HALF - margin;
    const ax = lo + rng() * (hi - lo);
    const az = lo + rng() * (hi - lo);
    // pick a second endpoint within [ROAD_MIN_LENGTH, ROAD_MAX_LENGTH] of the first, within bounds
    let bx: number;
    let bz: number;
    for (;;) {
        const angle = rng() * Math.PI * 2;
        const len = ROAD_MIN_LENGTH + rng() * (ROAD_MAX_LENGTH - ROAD_MIN_LENGTH);
        bx = ax + Math.cos(angle) * len;
        bz = az + Math.sin(angle) * len;
        if (bx < lo || bx > hi || bz < lo || bz > hi) continue;
        break;
    }
    const poly: Polyline = {
        points: [
            [ax, az],
            [bx, bz],
        ],
        halfWidth: ROAD_HALF_WIDTH,
    };
    return { polylines: [poly] };
}

/** the set of resident tile ids (cpu[id] >= 0). */
function residentSet(cpu: Int32Array): Set<number> {
    const out = new Set<number>();
    for (let id = 0; id < cpu.length; id++) {
        if (cpu[id] >= 0) out.add(id);
    }
    return out;
}

describe("property: tile release over random edit sequences", () => {
    test("after every edit: resident = documentDirtyTiles(current), released read -1, allocate never throws, resident + free = ATLAS_LAYERS", () => {
        const cpu = new Int32Array(TILE_COUNT).fill(-1);
        const free: number[] = [];
        const pending: number[] = [];
        const pendingSet = new Set<number>();
        invalidate(cpu, free, ATLAS_LAYERS, pending, pendingSet);

        const rng = mulberry32(42);
        let current = generateNetwork();

        // bootstrap: drain the boot document so the invariant holds before the first edit
        for (const id of documentDirtyTiles(current)) {
            if (!pendingSet.has(id)) {
                pendingSet.add(id);
                pending.push(id);
            }
        }
        while (pending.length > 0) {
            const ids = drain(pending, pendingSet, THROTTLE);
            for (const id of ids) allocate(cpu, id, free, ATLAS_LAYERS);
        }

        // 200 random edits — far more than the 65 that broke the old counter
        for (let edit = 0; edit < 200; edit++) {
            const newDoc = randomDoc(rng);

            // retile: mark union dirty, release difference
            const oldTiles = new Set(documentDirtyTiles(current));
            const newTiles = new Set(documentDirtyTiles(newDoc));
            for (const id of newTiles) {
                if (!pendingSet.has(id)) {
                    pendingSet.add(id);
                    pending.push(id);
                }
            }
            for (const id of oldTiles) {
                if (!pendingSet.has(id)) {
                    pendingSet.add(id);
                    pending.push(id);
                }
            }
            const toRelease: number[] = [];
            for (const id of oldTiles) {
                if (!newTiles.has(id)) toRelease.push(id);
            }
            release(cpu, toRelease, free, pending, pendingSet);

            // drain: allocate a layer for every pending tile
            while (pending.length > 0) {
                const ids = drain(pending, pendingSet, THROTTLE);
                for (const id of ids) allocate(cpu, id, free, ATLAS_LAYERS);
            }

            current = newDoc;

            // ── invariants after every edit ──

            const resident = residentSet(cpu);
            const expected = new Set(documentDirtyTiles(current));

            // resident set equals documentDirtyTiles(current)
            expect(resident.size).toBe(expected.size);
            for (const id of resident) {
                expect(expected.has(id)).toBe(true);
            }
            for (const id of expected) {
                expect(resident.has(id)).toBe(true);
            }

            // released ids read -1: every tile NOT in the current document's dirty set has cpu[id] === -1
            for (let id = 0; id < TILE_COUNT; id++) {
                if (!expected.has(id)) {
                    expect(cpu[id]).toBe(-1);
                }
            }

            // resident + free always sums to ATLAS_LAYERS
            expect(resident.size + free.length).toBe(ATLAS_LAYERS);
        }
    });

    // The specific red-first input: ≥65 edits each touching a fresh tile. With the free list's release
    // between edits, this never throws — the old counter threw at the 65th (see the docblock above).
    test("65 edits each touching a fresh tile never throws with the free list", () => {
        const cpu = new Int32Array(TILE_COUNT).fill(-1);
        const free: number[] = [];
        const pending: number[] = [];
        const pendingSet = new Set<number>();
        invalidate(cpu, free, ATLAS_LAYERS, pending, pendingSet);

        let current = generateNetwork();
        // bootstrap
        for (const id of documentDirtyTiles(current)) {
            pendingSet.add(id);
            pending.push(id);
        }
        while (pending.length > 0) {
            const ids = drain(pending, pendingSet, THROTTLE);
            for (const id of ids) allocate(cpu, id, free, ATLAS_LAYERS);
        }

        for (let i = 0; i < 65; i++) {
            // place a road entirely inside one tile so each edit touches a fresh tile
            const tileX = i % 16;
            const tileZ = Math.floor(i / 16);
            const cx = tileX * TILE_SIZE - WORLD_HALF + 32;
            const cz = tileZ * TILE_SIZE - WORLD_HALF + 32;
            const newDoc: StrokeDocument = {
                polylines: [
                    {
                        points: [
                            [cx - 10, cz],
                            [cx + 10, cz],
                        ],
                        halfWidth: ROAD_HALF_WIDTH,
                    },
                ],
            };

            const oldTiles = new Set(documentDirtyTiles(current));
            const newTiles = new Set(documentDirtyTiles(newDoc));
            for (const id of newTiles) {
                if (!pendingSet.has(id)) {
                    pendingSet.add(id);
                    pending.push(id);
                }
            }
            for (const id of oldTiles) {
                if (!pendingSet.has(id)) {
                    pendingSet.add(id);
                    pending.push(id);
                }
            }
            const toRelease: number[] = [];
            for (const id of oldTiles) {
                if (!newTiles.has(id)) toRelease.push(id);
            }
            release(cpu, toRelease, free, pending, pendingSet);

            while (pending.length > 0) {
                const ids = drain(pending, pendingSet, THROTTLE);
                for (const id of ids) allocate(cpu, id, free, ATLAS_LAYERS);
            }

            current = newDoc;
        }

        // if we got here, allocate never threw across all 65 fresh-tile edits
        expect(residentSet(cpu).size + free.length).toBe(ATLAS_LAYERS);
    });
});
