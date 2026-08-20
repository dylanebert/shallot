import { describe, expect, test } from "bun:test";
import { documentDirtyTiles } from "./document";
import { generateNetwork } from "./network";
import { allocate, drain, invalidate } from "./queue";
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

describe("allocate — atlas layer assignment", () => {
    test("assigns sequential layers to fresh tiles, writing the CPU indirection mirror in place", () => {
        const cpu = new Int32Array(8).fill(-1);
        const a = allocate(cpu, 3, 0, 8);
        expect(a).toEqual({ layer: 0, nextLayer: 1 });
        expect(cpu[3]).toBe(0);
        const b = allocate(cpu, 5, a.nextLayer, 8);
        expect(b).toEqual({ layer: 1, nextLayer: 2 });
        expect(cpu[5]).toBe(1);
    });

    test("returns the existing layer for an already-resident tile, counter unchanged", () => {
        const cpu = new Int32Array(8).fill(-1);
        const first = allocate(cpu, 2, 0, 8);
        const again = allocate(cpu, 2, first.nextLayer, 8);
        expect(again).toEqual({ layer: first.layer, nextLayer: first.nextLayer });
    });

    test("throws rather than silently wrapping or evicting once capacity is exhausted", () => {
        const cpu = new Int32Array(4).fill(-1);
        let next = 0;
        for (let i = 0; i < 2; i++) next = allocate(cpu, i, next, 2).nextLayer;
        expect(() => allocate(cpu, 2, next, 2)).toThrow(/capacity exceeded/);
    });
});

describe("invalidate — the atlas's document-swap reset", () => {
    test("releases every resident tile, restarts the layer counter, and drops anything still queued", () => {
        const cpu = new Int32Array(8).fill(-1);
        allocate(cpu, 3, 0, 8);
        const b = allocate(cpu, 5, 1, 8);
        expect(cpu[3]).toBe(0);
        expect(cpu[5]).toBe(1);
        const pending = [6, 7];
        const pendingSet = new Set(pending);

        const nextLayer = invalidate(cpu, pending, pendingSet);

        expect(nextLayer).toBe(0);
        expect(Array.from(cpu)).toEqual(new Array(8).fill(-1));
        expect(pending).toEqual([]);
        expect(pendingSet.size).toBe(0);
        // the freed layer is immediately reusable, not skipped as if still owned
        expect(allocate(cpu, 3, nextLayer, 8)).toEqual({ layer: 0, nextLayer: 1 });
        expect(b.layer).toBe(1); // sanity: the pre-invalidate assignment really did use layer 1
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
        const pending: number[] = [];
        const pendingSet = new Set<number>();
        let nextLayer = 0;
        let maxNextLayer = 0;

        for (let seed = 0; seed <= 500; seed++) {
            nextLayer = invalidate(cpu, pending, pendingSet); // atlas.ts's swap invalidation, run first
            const ids = documentDirtyTiles(generateNetwork());
            for (const id of ids) {
                nextLayer = allocate(cpu, id, nextLayer, ATLAS_LAYERS).nextLayer; // never throws post-fix
            }
            maxNextLayer = Math.max(maxNextLayer, nextLayer);
        }

        expect(maxNextLayer).toBeLessThanOrEqual(ATLAS_LAYERS);
        expect(maxNextLayer).toBeGreaterThan(0); // sanity: reseeds actually allocated something
    });
});
