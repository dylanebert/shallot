import { describe, expect, test } from "bun:test";
import { blitWgsl, commonSize, mipLevels } from "./image";

// the pure half of the image→GPU path — the layer-size + mip-chain math. The decode/upload/blit half is
// GPU-only (validated in the `render`/`sponza` scenarios), so only these are unit-tested. The mipmap-blit
// kernel itself gets a device-free structural test below (`blitWgsl`) — the semantics that matter (the
// two entry points, the texture+sampler bindings, the fullscreen-triangle uv derivation) rather than the
// whole emitted text.

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

describe("blitWgsl", () => {
    // structural, not textual — a real shader concatenates this with plumbing outside its control, so
    // the assertions cover the semantics a mutation could silently break: both entry points exist, the
    // texture + sampler bindings are at their expected slots, the fullscreen-triangle uv derivation uses
    // the bit-shift trick (not a lookup table another port shape might reach for), and the sample call
    // reads through both declared bindings.
    test("emits both entry points over a texture_2d + filtering sampler", () => {
        const code = blitWgsl();
        expect(code).toContain("@group(0) @binding(0) var src: texture_2d<f32>;");
        expect(code).toContain("@group(0) @binding(1) var samp: sampler;");
        expect(code).toContain("@vertex fn blitVs(");
        expect(code).toContain("@fragment fn blitFs(");
    });

    test("derives uv from vertex_index via the bit-shift fullscreen-triangle trick", () => {
        const code = blitWgsl();
        expect(code).toContain("@builtin(vertex_index)");
        // the (vi << 1) & 2 / vi & 2 derivation, whatever the emitted local names — a mutated shift or
        // mask (e.g. dropping the flip below) changes triangle coverage or orientation silently on GPU,
        // so this is the seam a CPU test can actually catch.
        expect(code).toMatch(/<<\s*1u?\)\s*&\s*2u?/);
        expect(code).toContain("1f - uv.y");
    });

    test("samples through both declared bindings", () => {
        const code = blitWgsl();
        expect(code).toContain("textureSample(src, samp,");
    });
});
