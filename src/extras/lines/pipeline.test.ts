import { expect, test } from "bun:test";
import * as d from "typegpu/data";
import { lineQuad } from "./pipeline";

const resolution = d.vec2f(64, 64);

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
