import { describe, expect, test } from "bun:test";
import { resolve } from "node:path";
import { analyze, load, satisfiesReferenceRelations } from "./look.oracle";
import fixture from "./reference/look-relations.json";

const baselines = [1, 2, 3].map((index) =>
    resolve(import.meta.dir, `baseline/ocean-baseline-${index}.png`),
);

describe("ocean look oracle", () => {
    test("reads the three-capture floor without optional source images", async () => {
        for (const path of baselines) {
            const reading = analyze(await load(path));
            expect(reading.lumaRange).toBeGreaterThan(0.02);
            expect(reading.horizon.transitionWidth).toBeGreaterThan(0);
            expect(reading.horizon.continuity).toBeGreaterThan(0);
            expect(reading.lowerBandBrightSpecks).toBeGreaterThanOrEqual(0);
            expect(reading.farWaterSkyHueDistance).toBeFinite();
        }
    });

    test("committed reference relations retain provenance", () => {
        expect(fixture.oracleRevision).toBe("shallot-ocean-look/S6@91fc585");
        expect(fixture.provenance.horizonRows).toBe("210/720");
        expect(fixture.provenance.sources).toHaveLength(4);
        expect(
            fixture.provenance.sources.every(({ sha256 }) => /^[a-f0-9]{64}$/.test(sha256)),
        ).toBe(true);
    });

    test("a mutated fixture relation rejects an otherwise accepted reading", () => {
        const reading = {
            bands: {},
            farWaterSkyHueDistance: 20,
            lowerBandBrightSpecks: 67,
            horizon: { transitionWidth: 4, continuity: 0.4 },
            foam: { nearCoverage: 0.05, maxLuma: 0.4 },
            lumaRange: 0.4,
        };
        expect(satisfiesReferenceRelations(reading)).toBe(true);
        const mutated = structuredClone(fixture.relations);
        mutated.farWaterSkyHueDistanceMax = 10;
        expect(satisfiesReferenceRelations(reading, mutated)).toBe(false);
    });

    test("rejects a synthetic zero-contrast frame", () => {
        const data = new Uint8Array(64 * 64 * 4).fill(96);
        for (let index = 3; index < data.length; index += 4) data[index] = 255;
        expect(analyze({ width: 64, height: 64, data }).lumaRange).toBeLessThan(0.02);
    });
});
