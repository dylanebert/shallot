import { describe, expect, test } from "bun:test";
import { type PixelProbe, pixelProbePass, probePixels } from "./pixels";

// 4×3 RGBA buffer, row-major, alpha unused by the classifier.
function frame(pixels: [number, number, number][], width: number, height: number): Uint8Array {
    const buf = new Uint8Array(width * height * 4);
    for (let i = 0; i < pixels.length; i++) {
        const [r, g, b] = pixels[i];
        buf[i * 4] = r;
        buf[i * 4 + 1] = g;
        buf[i * 4 + 2] = b;
        buf[i * 4 + 3] = 255;
    }
    return buf;
}

const BLACK: [number, number, number] = [0, 0, 0];
const TAG: [number, number, number] = [200, 4, 200];
const probe: PixelProbe = {
    name: "tag",
    minPixels: 2,
    minSpan: 2,
    r: [180, 255],
    g: [0, 12],
    b: [180, 255],
};

describe("probePixels", () => {
    test("counts only in-band pixels and measures their bounding-box footprint", () => {
        // 4×3 grid, tag block at (1,0)-(2,1) — a 2×2 footprint, 4 pixels
        const px: [number, number, number][] = [
            BLACK,
            BLACK,
            BLACK,
            BLACK,
            BLACK,
            TAG,
            TAG,
            BLACK,
            BLACK,
            TAG,
            TAG,
            BLACK,
        ];
        const result = probePixels(frame(px, 4, 3), 4, 3, probe);
        expect(result).toEqual({ pixels: 4, width: 2, height: 2 });
        expect(pixelProbePass(result, probe)).toBe(true);
    });

    test("an all-black frame — the blank-canvas red proof — matches nothing", () => {
        const px: [number, number, number][] = new Array(12).fill(BLACK);
        const result = probePixels(frame(px, 4, 3), 4, 3, probe);
        expect(result).toEqual({ pixels: 0, width: 0, height: 0 });
        expect(pixelProbePass(result, probe)).toBe(false);
    });

    test("pixelProbePass fails closed below minPixels even with a wide-enough span", () => {
        // a diagonal 2-pixel scatter: 2×2 bounding box (span 2 clears minSpan) but only 2 of the 4
        // required minPixels are tagged — the span alone isn't sufficient.
        const wide: PixelProbe = { ...probe, minPixels: 4 };
        const px: [number, number, number][] = [
            TAG,
            BLACK,
            BLACK,
            BLACK,
            BLACK,
            BLACK,
            BLACK,
            BLACK,
            BLACK,
            BLACK,
            TAG,
            BLACK,
        ];
        const result = probePixels(frame(px, 4, 3), 4, 3, wide);
        expect(result.pixels).toBe(2);
        expect(pixelProbePass(result, wide)).toBe(false);
    });
});
