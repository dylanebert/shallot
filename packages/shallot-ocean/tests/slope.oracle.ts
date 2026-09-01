// Trigger: changes to slope-cascade resolution, patch length, declared band, or seeded realization
// can break the N-independent mode placement and represented slope moment checked here.
import { expect, test } from "bun:test";
import {
    composedSlopePsd,
    rasterSlopeMoment,
    realizedSlopeMss,
    runSlopeCpuPipeline,
    SLOPE_CASCADE_CONFIGS,
    slopeMomentAgreementTolerance,
} from "../src/slope";
import { generateH0, kIndex } from "../src/spectrum";

const [config] = SLOPE_CASCADE_CONFIGS;

/**
 * This by-path oracle compares seeded CPU realizations and their declared radial bands. It does
 * not claim an independent source-density check or implement a second restricted-PSD quadrature;
 * the production composed-PSD helper supplies the expected band moments.
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

test("slope coverage checks every declared radial band against its composed moment", () => {
    const h0 = generateH0(config, 0);
    const bands = [
        [config.kLo, 20],
        [20, 40],
        [40, 50],
        [50, 55],
        [55, config.kHi],
    ] as const;
    for (const [lo, hi] of bands) {
        const band = new Float32Array(h0.length);
        let modes = 0;
        for (let y = 0; y < config.N; y++) {
            for (let x = 0; x < config.N; x++) {
                const k = Math.hypot(
                    kIndex(x, config.N) * ((2 * Math.PI) / config.L),
                    kIndex(y, config.N) * ((2 * Math.PI) / config.L),
                );
                if (k < lo || k > hi) continue;
                const i = (y * config.N + x) * 2;
                band[i] = h0[i];
                band[i + 1] = h0[i + 1];
                if (h0[i] !== 0 || h0[i + 1] !== 0) modes++;
            }
        }
        expect(modes, `no seeded modes in ${lo}..${hi} rad/m`).toBeGreaterThan(0);
        const realized = realizedSlopeMss(runSlopeCpuPipeline(band, config));
        const expected = composedSlopePsd({ ...config, kLo: lo, kHi: hi });
        const ratio = realized / expected;
        const bound = slopeMomentAgreementTolerance({ ...config, kLo: lo, kHi: hi });
        console.log(`slope band ${lo}..${hi} ratio=${ratio}, bound=${bound}`);
        expect(realized).toBeGreaterThan(0);
        expect(Math.abs(ratio - 1)).toBeLessThan(bound);
    }
});
