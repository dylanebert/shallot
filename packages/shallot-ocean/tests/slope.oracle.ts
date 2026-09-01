// Trigger: changes to slope-cascade resolution, patch length, declared band, or seeded realization
// can break the N-independent mode placement and represented slope moment checked here.
import { expect, test } from "bun:test";
import { composedSlopePsd, rasterSlopeMoment, SLOPE_CASCADE_CONFIGS } from "../src/slope";
import { directionalDensity, generateH0, kIndex } from "../src/spectrum";

const [config] = SLOPE_CASCADE_CONFIGS;

function independentSlopeMoment(config: (typeof SLOPE_CASCADE_CONFIGS)[number]): number {
    const radialSteps = 512;
    const angularSteps = 128;
    const dLog = Math.log(config.kHi / config.kLo) / radialSteps;
    const dTheta = (2 * Math.PI) / angularSteps;
    let sum = 0;
    for (let i = 0; i < radialSteps; i++) {
        const k = config.kLo * Math.exp((i + 0.5) * dLog);
        for (let j = 0; j < angularSteps; j++) {
            const theta = (j + 0.5) * dTheta;
            sum +=
                directionalDensity(k * Math.cos(theta), k * Math.sin(theta)) *
                k ** 4 *
                dLog *
                dTheta;
        }
    }
    return sum;
}

test("slope N-invariance preserves the declared mode and rasterized moment", () => {
    const denser = { ...config, N: config.N * 2 };
    const referenceMoment = independentSlopeMoment(config);
    const measuredMoment = composedSlopePsd(config);
    const rasterMoment = rasterSlopeMoment(generateH0(config, 17), config);
    const denserRasterMoment = rasterSlopeMoment(generateH0(denser, 17), denser);
    expect(measuredMoment).toBeGreaterThan(0);
    expect(Math.abs(measuredMoment / referenceMoment - 1)).toBeLessThan(0.005);
    expect(rasterMoment).toBeGreaterThan(0);
    expect(denserRasterMoment).toBeGreaterThan(0);
    expect(Math.abs(rasterMoment / measuredMoment - 1)).toBeLessThan(0.25);
    expect(Math.abs(denserRasterMoment / measuredMoment - 1)).toBeLessThan(0.25);
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
