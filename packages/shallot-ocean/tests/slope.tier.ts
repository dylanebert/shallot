// Real-device/by-path slope cascade reachability arm. It checks that the complete GPU resource path
// (typed buffers, FFT pipelines, storage texture views, and every mip view) can be constructed. The
// CPU slope tests own numerical truth; this tier owns WebGPU API and pipeline validation.
import { expect, test } from "bun:test";
import { requestGPU } from "@dylanebert/shallot/runtime";
import {
    buildSlopes,
    getSlopeTexture,
    readSlopeBuffers,
    readSlopeMips,
    reduceSlopeMip,
    runSlopeCpuPipeline,
    SLOPE_CASCADE_CONFIGS,
    SLOPE_MIP_LEVELS,
    slopeMipSize,
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
        const time = 0.73;
        const gpu = await readSlopeBuffers(config, time);
        const cpu = runSlopeCpuPipeline(generateH0(config, 0), config, time);
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

        const gpuMips = await readSlopeMips(config, time);
        let level0: Float32Array<ArrayBufferLike> = new Float32Array(config.N * config.N * 4);
        for (let i = 0; i < config.N * config.N; i++) {
            const x = cpu.xField[i * 2];
            const z = cpu.zField[i * 2];
            level0[i * 4] = x;
            level0[i * 4 + 1] = z;
            level0[i * 4 + 2] = x * x + z * z;
        }
        const cpuMips: Array<Float32Array<ArrayBufferLike>> = [level0];
        for (let level = 1; level < SLOPE_MIP_LEVELS; level++) {
            level0 = reduceSlopeMip(level0, slopeMipSize(config, level - 1));
            cpuMips.push(level0);
        }
        expect(gpuMips.length).toBe(SLOPE_MIP_LEVELS);
        for (let level = 0; level < SLOPE_MIP_LEVELS; level++) {
            expect(gpuMips[level].length).toBe(cpuMips[level].length);
            let maxNormalizedError = 0;
            for (let i = 0; i < gpuMips[level].length; i++) {
                const expected = cpuMips[level][i];
                const allowed = 0.05 + Math.abs(expected) * 0.01;
                maxNormalizedError = Math.max(
                    maxNormalizedError,
                    Math.abs(gpuMips[level][i] - expected) / allowed,
                );
            }
            expect(maxNormalizedError).toBeLessThan(1);
        }
    } finally {
        teardownSlopes();
    }
});
