import { describe, expect, test } from "bun:test";
import { commonSize, mipLevels } from "./image";

// the pure half of the image→GPU path — the layer-size + mip-chain math. The decode/upload/blit half is
// GPU-only (validated in the `render`/`sponza` scenarios), so only these are unit-tested.

describe("mipLevels", () => {
    test("full chain down to 1×1 for power-of-two sizes", () => {
        expect(mipLevels(1)).toBe(1);
        expect(mipLevels(2)).toBe(2);
        expect(mipLevels(1024)).toBe(11);
        expect(mipLevels(2048)).toBe(12);
    });
});

describe("commonSize", () => {
    test("picks the max source dimension across the set", () => {
        expect(
            commonSize([
                { w: 1024, h: 1024 },
                { w: 4, h: 4 },
            ]),
        ).toBe(1024);
        expect(
            commonSize([
                { w: 512, h: 256 },
                { w: 256, h: 1024 },
            ]),
        ).toBe(1024);
    });

    test("caps the common size, downscaling larger sources", () => {
        expect(commonSize([{ w: 4096, h: 4096 }], 2048)).toBe(2048);
    });

    test("never returns below 1 for an empty set", () => {
        expect(commonSize([])).toBe(1);
    });
});
