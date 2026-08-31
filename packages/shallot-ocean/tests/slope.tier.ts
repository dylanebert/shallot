// Real-device/by-path slope cascade reachability arm. It checks that the complete GPU resource path
// (typed buffers, FFT pipelines, storage texture views, and every mip view) can be constructed. The
// CPU slope tests own numerical truth; this tier owns WebGPU API and pipeline validation.
import { expect, test } from "bun:test";
import { requestGPU } from "@dylanebert/shallot/runtime";
import {
    buildSlopes,
    getSlopeTexture,
    readSlopeBuffers,
    runSlopeCpuPipeline,
    SLOPE_CASCADE_CONFIGS,
    SLOPE_MIP_LEVELS,
    teardownSlopes,
} from "../src/slope";
import { generateH0 } from "../src/spectrum";

test("slope cascade builds a complete GPU mip chain", async () => {
    await requestGPU();
    buildSlopes();
    try {
        const texture = getSlopeTexture();
        expect(texture).not.toBeNull();
        expect(texture!.mipLevelCount).toBe(SLOPE_MIP_LEVELS);
        const config = SLOPE_CASCADE_CONFIGS[0];
        const gpu = await readSlopeBuffers(config);
        const cpu = runSlopeCpuPipeline(generateH0(config, 0), config);
        for (const [actual, expected] of [
            [gpu.x, cpu.xField],
            [gpu.z, cpu.zField],
        ] as const) {
            let maxError = 0;
            for (let i = 0; i < actual.length; i++) {
                maxError = Math.max(maxError, Math.abs(actual[i] - expected[i]));
            }
            expect(maxError).toBeLessThan(0.01);
        }
    } finally {
        teardownSlopes();
    }
});
