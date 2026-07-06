import { describe, expect, test } from "bun:test";
import { coalesce, median } from "./coalesce";

describe("median", () => {
    test("an empty window is 0", () => {
        expect(median([], [])).toBe(0);
    });

    test("returns the middle value, ignoring transient short double-fire intervals", () => {
        // a 60Hz cadence with two double-fire shorts mixed in — the median stays at the cadence
        expect(median([16.7, 16.7, 4, 16.7, 16.7, 4, 16.7], [])).toBe(16.7);
    });

    test("adapts to a new sustained cadence once it dominates the window", () => {
        expect(median([4, 4, 4, 16.7, 16.7, 16.7, 16.7], [])).toBe(16.7);
    });

    test("reuses the scratch buffer rather than allocating", () => {
        const scratch: number[] = [];
        median([3, 1, 2], scratch);
        const same = scratch;
        median([9, 8, 7, 6], scratch);
        expect(scratch).toBe(same); // identity preserved — no realloc
        expect(scratch).toEqual([6, 7, 8, 9]);
    });
});

describe("coalesce (rAF double-fire)", () => {
    test("an unset cadence never coalesces — the first frames always render", () => {
        expect(coalesce(1000, 0, 0)).toBe(false);
    });

    test("a normal frame at the cadence renders", () => {
        expect(coalesce(1016.7, 1000, 16.7)).toBe(false); // 16.7 ≥ 8.35
    });

    test("a double-fire under half the cadence is coalesced", () => {
        expect(coalesce(1004, 1000, 16.7)).toBe(true); // 4 < 8.35
    });

    test("a frame just over half the cadence still renders", () => {
        expect(coalesce(1009, 1000, 16.7)).toBe(false); // 9 ≥ 8.35
    });

    test("the threshold scales with the cadence — a 240Hz desktop renders its 4ms frames", () => {
        expect(coalesce(1004.17, 1000, 1000 / 240)).toBe(false); // a whole 240Hz frame
        expect(coalesce(1001, 1000, 1000 / 240)).toBe(true); // a sub-half-interval double-fire
    });
});
