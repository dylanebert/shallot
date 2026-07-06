import { describe, expect, test } from "bun:test";
import { packColor, unpackColor } from "./color";

describe("unpackColor", () => {
    test("unpacks black (0x000000)", () => {
        const color = unpackColor(0x000000);
        expect(color.r).toBe(0);
        expect(color.g).toBe(0);
        expect(color.b).toBe(0);
    });

    test("unpacks white (0xffffff)", () => {
        const color = unpackColor(0xffffff);
        expect(color.r).toBe(1);
        expect(color.g).toBe(1);
        expect(color.b).toBe(1);
    });

    test("unpacks red (0xff0000)", () => {
        const color = unpackColor(0xff0000);
        expect(color.r).toBe(1);
        expect(color.g).toBe(0);
        expect(color.b).toBe(0);
    });

    test("unpacks green (0x00ff00)", () => {
        const color = unpackColor(0x00ff00);
        expect(color.r).toBe(0);
        expect(color.g).toBe(1);
        expect(color.b).toBe(0);
    });

    test("unpacks blue (0x0000ff)", () => {
        const color = unpackColor(0x0000ff);
        expect(color.r).toBe(0);
        expect(color.g).toBe(0);
        expect(color.b).toBe(1);
    });

    test("unpacks mid-gray (0x808080) as linear", () => {
        const color = unpackColor(0x808080);
        // 0x80/255 = 0.50196 sRGB; the IEC 61966-2-1 EOTF, ((c+0.055)/1.055)^2.4,
        // maps it to this f64 value exactly (gamma expansion, far below the sRGB
        // value), identical on every channel — no rounding, so the match is exact.
        const linearMidGray = 0.21586050011389926;
        expect(color.r).toBe(linearMidGray);
        expect(color.g).toBe(color.r);
        expect(color.b).toBe(color.r);
    });

    test("returns a fresh object per call", () => {
        const a = unpackColor(0xff0000);
        const b = unpackColor(0x00ff00);
        expect(a).not.toBe(b);
        expect(a.r).toBe(1);
        expect(b.g).toBe(1);
    });
});

// The packed word is the CPU→GPU color codec (a shader unpacks it with unpack4x8unorm + sRGB→linear),
// so byte order and the alpha scale are the contract every producer (lines, sprite, text) depends on.
describe("packColor", () => {
    test("keeps sRGB bytes in r,g,b,a order (GPU linearizes on unpack)", () => {
        const c = packColor(0xff8040, 1);
        expect(c & 0xff).toBe(0xff); // r in byte 0 (unpack4x8unorm .x)
        expect((c >> 8) & 0xff).toBe(0x80); // g
        expect((c >> 16) & 0xff).toBe(0x40); // b
        expect((c >>> 24) & 0xff).toBe(0xff); // a
    });

    test("scales opacity into the alpha byte", () => {
        expect((packColor(0xffffff, 0.5) >>> 24) & 0xff).toBe(128);
        expect((packColor(0xffffff, 0) >>> 24) & 0xff).toBe(0);
    });
});
