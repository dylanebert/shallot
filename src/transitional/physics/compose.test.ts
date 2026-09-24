import { expect } from "bun:test";
import { check } from "../../harness/check";
import { nlerpShortest, renderScale } from "./compose";
import { ShapeKind } from "./index";

// pure render-interpolation math — no GPU, no physics World.

check(
    "nlerpShortest returns curr exactly at t=1 and prev exactly at t=0",
    {
        claim: "render interpolation misses its endpoints, so a body at rest would show a pose that is neither the previous nor the current physics transform",
    },
    () => {
        const prev: [number, number, number, number] = [0, 0, 0, 1];
        const curr: [number, number, number, number] = [0, Math.SQRT1_2, 0, Math.SQRT1_2];
        expect(nlerpShortest(prev, curr, 0)).toEqual(prev);
        const at1 = nlerpShortest(prev, curr, 1);
        for (let i = 0; i < 4; i++) expect(at1[i]).toBeCloseTo(curr[i], 5);
    },
);

check(
    "nlerpShortest blends the shortest arc",
    {
        claim: "render interpolation mixes quaternions without flipping the previous one into the current hemisphere, so a rotating body would spin the long way around between frames",
    },
    () => {
        // prev and -prev represent the same rotation; blending toward curr must agree regardless of sign.
        const curr: [number, number, number, number] = [0, Math.SQRT1_2, 0, Math.SQRT1_2];
        const a = nlerpShortest([0, 0, 0, 1], curr, 0.5);
        const b = nlerpShortest([0, 0, 0, -1], curr, 0.5);
        for (let i = 0; i < 4; i++) expect(a[i]).toBeCloseTo(b[i], 5);
    },
);

check(
    "nlerpShortest result is always unit-length",
    {
        claim: "render interpolation leaves the blended quaternion unnormalized, so an interpolated body would shear or scale mid-rotation",
    },
    () => {
        const q = nlerpShortest([0, 0, 0, 1], [1, 0, 0, 0], 0.3);
        const len = Math.sqrt(q[0] ** 2 + q[1] ** 2 + q[2] ** 2 + q[3] ** 2);
        expect(len).toBeCloseTo(1, 6);
    },
);

check(
    "renderScale doubles half-extents for box and hull",
    {
        claim: "render scale passes box and hull half-extents through undoubled, so every box would draw at half the size the solver collides with",
    },
    () => {
        expect(renderScale(ShapeKind.Box, [0.5, 1, 1.5], 0)).toEqual([1, 2, 3]);
        expect(renderScale(ShapeKind.Hull, [0.5, 1, 1.5], 0)).toEqual([1, 2, 3]);
    },
);

check(
    "renderScale scales a sphere uniformly to twice its radius",
    {
        claim: "render scale reads a sphere's size from its unused half-extents instead of its radius, so every sphere would draw at zero size",
    },
    () => {
        expect(renderScale(ShapeKind.Sphere, [0, 0, 0], 0.5)).toEqual([1, 1, 1]);
    },
);

check(
    "renderScale gives a capsule (2r, halfHeight + r, 2r)",
    {
        claim: "render scale treats a capsule like a box, so its hemispherical caps would distort under a non-proportional height-to-radius ratio",
    },
    () => {
        expect(renderScale(ShapeKind.Capsule, [0, 1, 0], 0.3)).toEqual([0.6, 1.3, 0.6]);
    },
);
