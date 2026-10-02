import { expect, test } from "bun:test";
import { nlerpShortest } from "./compose";

// pure render-interpolation math — no GPU, no physics World.

test("render interpolation misses its endpoints, so a body at rest would show a pose that is neither the previous nor the current physics transform", () => {
    const prev: [number, number, number, number] = [0, 0, 0, 1];
    const curr: [number, number, number, number] = [0, Math.SQRT1_2, 0, Math.SQRT1_2];
    expect(nlerpShortest(prev, curr, 0)).toEqual(prev);
    const at1 = nlerpShortest(prev, curr, 1);
    for (let i = 0; i < 4; i++) expect(at1[i]).toBeCloseTo(curr[i], 5);
});

test("render interpolation mixes quaternions without flipping the previous one into the current hemisphere, so a rotating body would spin the long way around between frames", () => {
    // prev and -prev represent the same rotation; blending toward curr must agree regardless of sign.
    const curr: [number, number, number, number] = [0, Math.SQRT1_2, 0, Math.SQRT1_2];
    const a = nlerpShortest([0, 0, 0, 1], curr, 0.5);
    const b = nlerpShortest([0, 0, 0, -1], curr, 0.5);
    for (let i = 0; i < 4; i++) expect(a[i]).toBeCloseTo(b[i], 5);
});

test("render interpolation leaves the blended quaternion unnormalized, so an interpolated body would shear or scale mid-rotation", () => {
    const q = nlerpShortest([0, 0, 0, 1], [1, 0, 0, 0], 0.3);
    const len = Math.sqrt(q[0] ** 2 + q[1] ** 2 + q[2] ** 2 + q[3] ** 2);
    expect(len).toBeCloseTo(1, 6);
});
