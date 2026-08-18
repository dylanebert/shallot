import { describe, expect, test } from "bun:test";
import { allocate, drain } from "./queue";
import { ATLAS_LAYERS, THROTTLE } from "./tiles";

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
