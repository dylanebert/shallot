// Trigger: changes to slope-cascade resolution, patch length, declared band, or seeded realization
// can break the N-independent mode placement and represented slope moment checked here.
import { expect, test } from "bun:test";
import {
    rasterSlopeMoment,
    SLOPE_CASCADE_CONFIGS,
    slopeMomentAgreementTolerance,
} from "../src/slope";
import { generateH0, kIndex } from "../src/spectrum";

const [config] = SLOPE_CASCADE_CONFIGS;

/**
 * The N-invariance oracle deliberately compares the same seeded mode set at N and 2N. It does
 * not call production density or duplicate its quadrature; slope.test owns the independent
 * restricted-PSD check.
 */
test("slope N-invariance compares the declared raster moments directly", () => {
    const denser = { ...config, N: config.N * 2 };
    const rasterMoment = rasterSlopeMoment(generateH0(config, 17), config);
    const denserRasterMoment = rasterSlopeMoment(generateH0(denser, 17), denser);
    const tolerance = slopeMomentAgreementTolerance(config);
    console.log(
        `slope N-invariance: N=${config.N} moment=${rasterMoment}, ` +
            `2N=${denser.N} moment=${denserRasterMoment}, bound=${tolerance}`,
    );
    expect(rasterMoment).toBeGreaterThan(0);
    expect(denserRasterMoment).toBeGreaterThan(0);
    expect(Math.abs(denserRasterMoment / rasterMoment - 1)).toBeLessThan(tolerance);

    const base = generateH0(config, 17);
    const high = generateH0(denser, 17);
    const dk = (2 * Math.PI) / config.L;
    for (let y = 0; y < config.N; y++) {
        for (let x = 0; x < config.N; x++) {
            const kx = kIndex(x, config.N) * dk;
            const kz = kIndex(y, config.N) * dk;
            if (Math.hypot(kx, kz) < config.kLo || Math.hypot(kx, kz) > config.kHi) continue;
            const hiX = kIndex(x, config.N) >= 0 ? x : denser.N + kIndex(x, config.N);
            const hiY = kIndex(y, config.N) >= 0 ? y : denser.N + kIndex(y, config.N);
            const a = (y * config.N + x) * 2;
            const b = (hiY * denser.N + hiX) * 2;
            expect(high[b]).toBe(base[a]);
            expect(high[b + 1]).toBe(base[a + 1]);
        }
    }
});
