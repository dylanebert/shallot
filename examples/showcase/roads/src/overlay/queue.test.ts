import { describe, expect, test } from "bun:test";
import { documentDirtyTiles } from "./document";
import { generateNetwork } from "./network";
import { allocate, drain } from "./queue";
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

    // `tiles.ts`'s own comment sizes ATLAS_LAYERS off "well past stage 6's 'a handful of roads + one
    // carpark' network... on the order of ten tiles" — a *single* generated network never overflows by
    // design. `overlay/atlas.ts`'s ATLAS_LAYERS are also never evicted (tiles.ts's own architecture), so
    // the real overflow path is cumulative: repeated reseeds (`terrain.ts`'s `regenerate`, driven live by
    // `boot.ts`'s F9) each allocate a fresh handful of layers on top of whatever's already resident, until
    // the fixed 64-layer atlas runs out. This drives that path with real generator output, not a synthetic
    // fixture (stage 4's own residue note) — real `documentDirtyTiles` output from real `generateNetwork`
    // seeds, fed through the real `allocate`.
    test("real cumulative reseeds — not a single oversized network — overflow the real 64-layer atlas", () => {
        const cpu = new Int32Array(TILE_COUNT).fill(-1);
        let nextLayer = 0;
        let seed = 1;
        let overflowed: { seed: number; tileId: number } | null = null;
        const SeedBudget = 50; // generous: the spike above found overflow by seed 4

        while (overflowed === null && seed <= SeedBudget) {
            const ids = documentDirtyTiles(generateNetwork(seed));
            for (const id of ids) {
                if (cpu[id] >= 0) continue; // already resident from an earlier reseed — no new layer spent
                try {
                    nextLayer = allocate(cpu, id, nextLayer, ATLAS_LAYERS).nextLayer;
                } catch (cause) {
                    overflowed = { seed, tileId: id };
                    expect(String(cause)).toMatch(/capacity exceeded/);
                    break;
                }
            }
            seed++;
        }

        if (overflowed === null) {
            throw new Error(
                `generateNetwork's cumulative tile footprint never exceeded ATLAS_LAYERS (${ATLAS_LAYERS}) ` +
                    `across ${SeedBudget} reseeds (${nextLayer} layers allocated) — the capacity-exceeded ` +
                    "throw is real code but this run found no real input that reaches it; widen SeedBudget " +
                    "or revisit the network's own sizing rather than asserting a fixture.",
            );
        }
        expect(overflowed.seed).toBeLessThanOrEqual(SeedBudget);
        expect(nextLayer).toBe(ATLAS_LAYERS); // every real layer got spent before the real throw fired
    });
});
