import { describe, expect, test } from "bun:test";
import type { Vec3 } from "./gizmo";
import { nextPivot, Pivot, pivotAnchor } from "./pivot";

describe("nextPivot — cycles the pivot mode (wraps)", () => {
    test("Median → Active → Median", () => {
        expect(nextPivot(Pivot.Median)).toBe(Pivot.Active);
        expect(nextPivot(Pivot.Active)).toBe(Pivot.Median);
    });
});

describe("pivotAnchor — the gizmo origin per pivot mode", () => {
    const a: Vec3 = [0, 0, 0];
    const b: Vec3 = [2, 0, 0];
    const c: Vec3 = [4, 3, 0];

    test("Median is the centroid of the selection", () => {
        expect(pivotAnchor(Pivot.Median, [a, b, c], 2)).toEqual([2, 1, 0]);
    });

    test("Active is the active (indexed) entity's own origin", () => {
        expect(pivotAnchor(Pivot.Active, [a, b, c], 2)).toEqual(c);
        expect(pivotAnchor(Pivot.Active, [a, b, c], 0)).toEqual(a);
    });

    test("a lone selection: both modes are that entity", () => {
        expect(pivotAnchor(Pivot.Median, [b], 0)).toEqual(b);
        expect(pivotAnchor(Pivot.Active, [b], 0)).toEqual(b);
    });

    test("an empty selection has no anchor", () => {
        expect(pivotAnchor(Pivot.Median, [], 0)).toBeNull();
        expect(pivotAnchor(Pivot.Active, [], -1)).toBeNull();
    });
});
