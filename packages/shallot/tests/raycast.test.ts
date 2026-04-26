import { describe, test, expect } from "bun:test";
import { rayTriangle, rayMesh } from "../src/standard/physics/raycast";
import { getMesh } from "../src/standard/render/mesh";

function near(a: number, b: number, tol = 1e-6): boolean {
    return Math.abs(a - b) < tol;
}

describe("rayTriangle", () => {
    test("hits triangle in XZ plane", () => {
        const r = rayTriangle(0, 1, 0, 0, -1, 0, -1, 0, -1, 1, 0, -1, 0, 0, 1);
        expect(r).not.toBeNull();
        expect(near(r!.t, 1)).toBe(true);
        expect(near(r!.ny, 1)).toBe(true);
    });

    test("misses triangle", () => {
        const r = rayTriangle(5, 1, 0, 0, -1, 0, -1, 0, -1, 1, 0, -1, 0, 0, 1);
        expect(r).toBeNull();
    });

    test("backface still hits (normal flipped)", () => {
        const r = rayTriangle(0, -1, 0, 0, 1, 0, -1, 0, -1, 1, 0, -1, 0, 0, 1);
        expect(r).not.toBeNull();
        expect(near(r!.t, 1)).toBe(true);
        expect(near(r!.ny, -1)).toBe(true);
    });

    test("ray behind triangle returns null", () => {
        const r = rayTriangle(0, -1, 0, 0, -1, 0, -1, 0, -1, 1, 0, -1, 0, 0, 1);
        expect(r).toBeNull();
    });

    test("edge case: ray along triangle plane", () => {
        const r = rayTriangle(0, 0, 0, 1, 0, 0, -1, 0, -1, 1, 0, -1, 0, 0, 1);
        expect(r).toBeNull();
    });
});

describe("rayMesh", () => {
    test("hits built-in box mesh", () => {
        const meshData = getMesh(0)!;
        const r = rayMesh(0, 0, -2, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, meshData);
        expect(r).not.toBeNull();
        expect(r!.t).toBeGreaterThan(0);
        expect(near(r!.nz, -1)).toBe(true);
    });

    test("misses when ray doesn't intersect mesh", () => {
        const meshData = getMesh(0)!;
        const r = rayMesh(5, 5, -2, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, meshData);
        expect(r).toBeNull();
    });

    test("hits rotated mesh", () => {
        const meshData = getMesh(0)!;
        const angle = Math.PI / 2;
        const qy = Math.sin(angle / 2);
        const qw = Math.cos(angle / 2);
        const r = rayMesh(0, 0, -2, 0, 0, 1, 0, 0, 0, 0, qy, 0, qw, 1, 1, 1, meshData);
        expect(r).not.toBeNull();
        expect(r!.t).toBeGreaterThan(0);
    });

    test("hits translated mesh", () => {
        const meshData = getMesh(0)!;
        const r = rayMesh(3, 0, -2, 0, 0, 1, 3, 0, 0, 0, 0, 0, 1, 1, 1, 1, meshData);
        expect(r).not.toBeNull();
        expect(r!.t).toBeGreaterThan(0);
    });

    test("hits scaled mesh", () => {
        const meshData = getMesh(0)!;
        // Unit box is ±0.5, scaled by 4 → ±2. Ray from z=-5 toward +z should hit at z=-2
        const r = rayMesh(0, 0, -5, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 4, 4, 4, meshData);
        expect(r).not.toBeNull();
        expect(near(r!.t, 3)).toBe(true);
        expect(near(r!.nz, -1)).toBe(true);
    });

    test("rotation applies correctly for asymmetric mesh", () => {
        // Tall box (1x4x1) rotated 90° around Z → becomes 4x1x1.
        // Ray from above should miss thin top, ray from side should hit wide face
        const meshData = getMesh(0)!;
        const angle = Math.PI / 2;
        const qz = Math.sin(angle / 2);
        const qw = Math.cos(angle / 2);
        // Ray from above at x=1.5 — unrotated 1x4x1 box extends ±0.5 in x, should miss.
        // Rotated 90° around Z, the 4-unit Y extent maps to X → extends ±2. Should hit.
        const hit = rayMesh(1.5, 3, 0, 0, -1, 0, 0, 0, 0, 0, 0, qz, qw, 1, 4, 1, meshData);
        expect(hit).not.toBeNull();
        // Ray from side at y=1.5 — rotated box extends ±0.5 in Y. Should miss.
        const miss = rayMesh(0, 1.5, -3, 0, 0, 1, 0, 0, 0, 0, 0, qz, qw, 1, 4, 1, meshData);
        expect(miss).toBeNull();
    });

    test("misses outside scaled mesh bounds", () => {
        const meshData = getMesh(0)!;
        // Unit box scaled by 2 → ±1. Ray at x=1.5 should miss
        const r = rayMesh(1.5, 0, -5, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 2, 2, 2, meshData);
        expect(r).toBeNull();
    });
});
