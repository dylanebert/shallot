import { expect, test } from "bun:test";
import * as d from "typegpu/data";
import { perspective } from "../../engine/utils/math";
import { cameraLineQuad, lineQuad, nearTriangleVertex } from "./pipeline";

const resolution = d.vec2f(64, 64);

test("arrow triangles clip every near-plane vertex classification into six corners", () => {
    for (let mask = 0; mask < 8; mask++) {
        const a = d.vec3f(-1, 0, mask & 1 ? -2 : 0);
        const b = d.vec3f(1, 0, mask & 2 ? -2 : 0);
        const c = d.vec3f(0, 1, mask & 4 ? -2 : 0);
        const distance = d.vec3f(-a.z - 1, -b.z - 1, -c.z - 1);
        for (let index = 0; index < 6; index++) {
            const point = nearTriangleVertex(a, b, c, distance, index);
            expect(point.w).toBe(mask === 0 ? 0 : 1);
            if (mask !== 0) {
                expect(point.z).toBeLessThanOrEqual(-1);
                expect(Number.isFinite(point.x + point.y + point.z)).toBe(true);
            }
        }
    }
});

test("infinite reverse-Z lines enforce the separate finite camera bound without changing depth", () => {
    const matrix = perspective(90, 1, 1);
    const project = (x: number, z: number) =>
        d.vec4f(matrix[0]! * x, 0, matrix[10]! * z + matrix[14]!, -z);
    const projection = d.vec4f(1, 10, 1, 0);
    expect(project(0, -12).z).toBe(1);
    for (const reverse of [false, true]) {
        for (const [a, b, expectedX, expectedDepth] of [
            [project(8, -15), project(-2, -5), 0.3, 0.1],
            [project(-0.4, -0.5), project(2, -5), -0.4 + 2.4 / 9, 1],
        ] as const) {
            const quad = cameraLineQuad(
                reverse ? b : a,
                reverse ? a : b,
                resolution,
                reverse ? 1 : 0,
                1,
                3,
                projection,
            );
            expect(quad.clip.x).toBeCloseTo(expectedX, 5);
            expect(quad.clip.z).toBeCloseTo(expectedDepth, 5);
        }
        for (const z of [-0.5, -12]) {
            const a = project(-0.2, z),
                b = project(0.2, z);
            const quad = cameraLineQuad(
                reverse ? b : a,
                reverse ? a : b,
                resolution,
                0,
                1,
                3,
                projection,
            );
            expect(quad.tint.w).toBe(0);
        }
    }
});

test("a long segment clips to the near plane before perspective division in either endpoint order", () => {
    const behind = d.vec4f(1, 3, 0.1, -3300),
        ahead = d.vec4f(1, -3, 0.1, 3300);
    for (const edge of [-1, 1]) {
        for (const quad of [
            lineQuad(behind, ahead, resolution, 0, edge, 3),
            lineQuad(ahead, behind, resolution, 1, edge, 3),
        ]) {
            expect(quad.clip.z).toBe(1);
            expect(quad.clip.w).toBe(1);
            expect(Number.isFinite(quad.clip.x)).toBe(true);
            expect(Number.isFinite(quad.clip.y)).toBe(true);
        }
    }
});

test("a segment crossing the far plane clips at its crossing in either endpoint order, even an endpoint just inside it", () => {
    for (const insideDepth of [1e-7, 0.5]) {
        const outside = d.vec4f(-0.5, 0, -insideDepth, 1);
        const inside = d.vec4f(0.5, 0, insideDepth, 1);
        for (const edge of [-1, 1]) {
            // t selects the outside endpoint's corner in each order
            const forward = lineQuad(outside, inside, resolution, 0, edge, 6);
            const reverse = lineQuad(inside, outside, resolution, 1, edge, 6);
            for (const quad of [forward, reverse]) {
                expect(quad.clip.x).toBeCloseTo(0, 5);
                expect(quad.clip.z).toBeGreaterThanOrEqual(0);
                expect(quad.clip.z).toBeLessThanOrEqual(insideDepth);
            }
        }
    }
});
