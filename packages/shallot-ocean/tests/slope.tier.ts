// Real-device/by-path slope cascade reachability arm. It checks that the complete GPU resource path
// (typed buffers, FFT pipelines, storage texture views, and every mip view) can be constructed. The
// CPU slope tests own numerical truth; this tier owns WebGPU API and pipeline validation.
import { expect, test } from "bun:test";
import { requestGPU } from "@dylanebert/shallot/runtime";
import { buildSlopes, getSlopeTexture, SLOPE_MIP_LEVELS, teardownSlopes } from "../src/slope";

test("slope cascade builds a complete GPU mip chain", async () => {
    await requestGPU();
    buildSlopes();
    try {
        const texture = getSlopeTexture();
        expect(texture).not.toBeNull();
        expect(texture!.mipLevelCount).toBe(SLOPE_MIP_LEVELS);
    } finally {
        teardownSlopes();
    }
});
