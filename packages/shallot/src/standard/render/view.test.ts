import { describe, expect, test } from "bun:test";
import { backingSize } from "./view";

// backingSize derives a view's render backing (device px) from its CSS display size, a Resolution pin
// (resW/resH, 0 = that axis unset), and the pixelRatio fit-ratio. The aspect derivation + the
// upscale-only `pixelated` flag are the logic worth pinning.
describe("backingSize", () => {
    test("no pin (both 0) renders at display × ratio, smooth at ratio ≥ 1", () => {
        expect(backingSize(0, 0, 1920, 1080, 1)).toEqual({ w: 1920, h: 1080, pixelated: false });
        expect(backingSize(0, 0, 1920, 1080, 2)).toEqual({ w: 3840, h: 2160, pixelated: false });
    });

    test("no pin with ratio < 1 is the pixel-art downscale (pixelated)", () => {
        expect(backingSize(0, 0, 1920, 1080, 0.5)).toEqual({ w: 960, h: 540, pixelated: true });
    });

    test("height pin derives width from the display aspect, point-sampled up", () => {
        // 360 × (1920/1080) = 640 exactly; 640 < 1920 → upscaling → pixelated
        expect(backingSize(0, 360, 1920, 1080, 1)).toEqual({ w: 640, h: 360, pixelated: true });
    });

    test("width pin derives height from the display aspect", () => {
        // 640 × (1080/1920) = 360 exactly
        expect(backingSize(640, 0, 1920, 1080, 1)).toEqual({ w: 640, h: 360, pixelated: true });
    });

    test("both axes pinned are exact, even off the display aspect", () => {
        expect(backingSize(320, 240, 1920, 1080, 1)).toEqual({ w: 320, h: 240, pixelated: true });
    });

    test("aspect derivation rounds to the nearest pixel", () => {
        // 360 × (1366/768) = 640.3125 → 640
        expect(backingSize(0, 360, 1366, 768, 1)).toEqual({ w: 640, h: 360, pixelated: true });
    });

    test("a pin above the display is a smooth supersample, not pixelated", () => {
        // 2160 × (1920/1080) = 3840; backing ≥ display on both axes → no nearest upscale
        expect(backingSize(0, 2160, 1920, 1080, 1)).toEqual({ w: 3840, h: 2160, pixelated: false });
    });
});
