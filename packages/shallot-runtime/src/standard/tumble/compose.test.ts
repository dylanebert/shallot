import { describe, expect, test } from "bun:test";
import { ShapeKind } from "../physics";
import { nlerpShortest, renderScale } from "./compose";

// pure render-interpolation math (the CPU twin of AVBD's COMPOSE_PASS_WGSL) — no GPU, no tumble World.

describe("nlerpShortest", () => {
    test("returns curr exactly at t=1 and prev exactly at t=0", () => {
        const prev: [number, number, number, number] = [0, 0, 0, 1];
        const curr: [number, number, number, number] = [0, Math.SQRT1_2, 0, Math.SQRT1_2];
        expect(nlerpShortest(prev, curr, 0)).toEqual(prev);
        const at1 = nlerpShortest(prev, curr, 1);
        for (let i = 0; i < 4; i++) expect(at1[i]).toBeCloseTo(curr[i], 5);
    });

    test("blends the shortest arc — flips prev into curr's hemisphere before mixing", () => {
        // prev and -prev represent the same rotation; blending toward curr must agree regardless of sign.
        const curr: [number, number, number, number] = [0, Math.SQRT1_2, 0, Math.SQRT1_2];
        const a = nlerpShortest([0, 0, 0, 1], curr, 0.5);
        const b = nlerpShortest([0, 0, 0, -1], curr, 0.5);
        for (let i = 0; i < 4; i++) expect(a[i]).toBeCloseTo(b[i], 5);
    });

    test("result is always unit-length", () => {
        const q = nlerpShortest([0, 0, 0, 1], [1, 0, 0, 0], 0.3);
        const len = Math.sqrt(q[0] ** 2 + q[1] ** 2 + q[2] ** 2 + q[3] ** 2);
        expect(len).toBeCloseTo(1, 6);
    });
});

describe("renderScale", () => {
    test("box/hull scale to 2·halfExtents", () => {
        expect(renderScale(ShapeKind.Box, [0.5, 1, 1.5], 0)).toEqual([1, 2, 3]);
        expect(renderScale(ShapeKind.Hull, [0.5, 1, 1.5], 0)).toEqual([1, 2, 3]);
    });

    test("sphere scales uniformly to 2·radius", () => {
        expect(renderScale(ShapeKind.Sphere, [0, 0, 0], 0.5)).toEqual([1, 1, 1]);
    });

    test("capsule scales to (2r, halfHeight + r, 2r) — the caps distort under a non-proportional ratio", () => {
        expect(renderScale(ShapeKind.Capsule, [0, 1, 0], 0.3)).toEqual([0.6, 1.3, 0.6]);
    });
});
