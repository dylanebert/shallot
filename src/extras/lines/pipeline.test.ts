import { expect, test } from "bun:test";
import * as d from "typegpu/data";
import { lineQuad } from "./pipeline";

const resolution = d.vec2f(64, 64);

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
