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

    // the ascii showcase's `glyphInk` probe (`examples/showcase/ascii/test/pixel-probe.playwright.ts`,
    // criterion 5): a near-white grayscale band meant to discriminate the Cells system's glyph draw from
    // the scene's own chromatic colors. A live browser isn't available to this suite (`testing.md`'s GPU
    // ladder), so this pins the discrimination mechanically against synthetic frames standing in for
    // "Cells: true" vs. "Cells: true deleted from shallot.json" (`checks.md`: prove a probe reds with the
    // mechanism it targets removed, not just that it passes on the happy path).
    describe("the ascii showcase's cells-only ink probe discriminates Cells from a bare scene", () => {
        const GlyphInk: PixelProbe = {
            name: "cell glyph ink reaches the compositor",
            minPixels: 80,
            minSpan: 20,
            r: [210, 255],
            g: [210, 255],
            b: [210, 255],
        };
        // the scene's own two colors, per `specs/shallot-tui.md`'s locked shading + the recipe/showcase
        // scenes: a warm box albedo (never near-white — the blue channel alone sits well under 210) and a
        // near-black clear color. Neither is chromatically neutral, so neither can land in a band that
        // demands r, g, and b all high simultaneously.
        const Box: [number, number, number] = [172, 151, 126];
        const Bg: [number, number, number] = [23, 17, 8];
        // `select.ts`'s `selectKernel`: fg is exactly black or exactly white, never the scene's hue.
        const Ink: [number, number, number] = [255, 255, 255];

        test("a bare scene render (no Cells) never reaches the ink band", () => {
            const w = 20;
            const h = 20;
            const px: [number, number, number][] = new Array(w * h)
                .fill(0)
                .map((_, i) => (i < (w * h) / 2 ? Box : Bg));
            const result = probePixels(frame(px, w, h), w, h, GlyphInk);
            expect(result).toEqual({ pixels: 0, width: 0, height: 0 });
            expect(pixelProbePass(result, GlyphInk)).toBe(false);
        });

        test("the same scene with Cells' glyph ink painted in clears the band", () => {
            const w = 40;
            const h = 40;
            const px: [number, number, number][] = new Array(w * h)
                .fill(0)
                .map((_, i) => (i < (w * h) / 2 ? Box : Bg));
            // paint a 25×25 block of ink — an SDF-drawn "|" stroke's worth of coverage, well under
            // 100% of the frame, still clearing GlyphInk's minPixels/minSpan on both axes
            for (let y = 8; y < 33; y++) {
                for (let x = 8; x < 33; x++) px[y * w + x] = Ink;
            }
            const result = probePixels(frame(px, w, h), w, h, GlyphInk);
            expect(pixelProbePass(result, GlyphInk)).toBe(true);
        });
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
