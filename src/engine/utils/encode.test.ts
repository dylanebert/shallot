import { describe, expect, test } from "bun:test";
import { vec2f, vec3f } from "typegpu/data";
import { octEncode, octEncodeNormal, packColor4, packLdrColor, packUnorm2 } from "./encode";
import { packUnorm2x16 } from "./tgsl";

// The plain-JS mirrors are the only duplicated logic in encode.ts, so each is pinned against the TGSL
// function it mirrors: bit-identical on the packed word, over a sweep dense enough to reach both
// octahedral hemispheres and both segments of the sRGB transfer.
describe("each plain-JS mirror packs the same word as its TGSL source", () => {
    let rng = 0x2545f491;
    const next = () => {
        rng = (rng * 1664525 + 1013904223) >>> 0;
        return rng / 0xffffffff;
    };

    test("octEncode matches octEncodeNormal", () => {
        const edges: [number, number, number][] = [
            [1, 0, 0],
            [-1, 0, 0],
            [0, 1, 0],
            [0, -1, 0],
            [0, 0, 1],
            [0, 0, -1],
            // a zero vector and a -0 lane reach the degenerate-denominator guard; a non-unit input
            // reaches the fold on an unnormalized value.
            [0, 0, 0],
            [0, -0, 0],
            [2, 0, 0],
            [0.3, 0.3, -0.3],
        ];
        for (const [x, y, z] of edges) {
            expect(octEncode(x, y, z)).toBe(octEncodeNormal(vec3f(x, y, z)));
        }
        for (let i = 0; i < 4096; i++) {
            const v = [next() * 2 - 1, next() * 2 - 1, next() * 2 - 1];
            const l = Math.hypot(v[0], v[1], v[2]) || 1;
            const [x, y, z] = [v[0] / l, v[1] / l, v[2] / l];
            expect(octEncode(x, y, z)).toBe(octEncodeNormal(vec3f(x, y, z)));
        }
    });

    test("packUnorm2 matches packUnorm2x16", () => {
        for (const [x, y] of [
            [0, 0],
            [1, 1],
            [2, -1],
        ]) {
            expect(packUnorm2(x, y)).toBe(packUnorm2x16(vec2f(x, y)));
        }
        for (let i = 0; i < 4096; i++) {
            const [x, y] = [next(), next()];
            expect(packUnorm2(x, y)).toBe(packUnorm2x16(vec2f(x, y)));
        }
    });

    test("packColor4 matches packLdrColor", () => {
        for (const [r, g, b, a] of [
            [0, 0, 0, 1],
            [1, 1, 1, 1],
            [0.0031308, 0.0031308, 0.0031308, 0], // the sRGB segment boundary
            [2, -1, 0.5, 0.5],
        ]) {
            expect(packColor4(r, g, b, a)).toBe(packLdrColor(vec3f(r, g, b), a));
        }
        for (let i = 0; i < 2048; i++) {
            const [r, g, b, a] = [next(), next(), next(), next()];
            expect(packColor4(r, g, b, a)).toBe(packLdrColor(vec3f(r, g, b), a));
        }
    });
});
