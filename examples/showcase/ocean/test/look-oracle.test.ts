import { describe, expect, test } from "bun:test";
import { resolve } from "node:path";
import { analyze, load } from "./look.oracle";

const root = resolve(import.meta.dir, "../../../../..");
const baselines = [1, 2, 3].map((index) =>
    resolve(import.meta.dir, `baseline/ocean-baseline-${index}.png`),
);
const references = ["t26.jpg", "t43.jpg"].map((name) =>
    resolve(root, `research/water-surface/oracle-dusk/frames/${name}`),
);

describe("ocean look oracle", () => {
    test("reads the three-capture floor and dusk references", async () => {
        for (const path of [...baselines, ...references]) {
            const reading = analyze(await load(path));
            expect(reading.lumaRange).toBeGreaterThan(0.02);
            expect(reading.horizon.transitionWidth).toBeGreaterThan(0);
            expect(reading.horizon.continuity).toBeGreaterThan(0);
            expect(reading.lowerBandBrightSpecks).toBeGreaterThanOrEqual(0);
            expect(reading.farWaterSkyHueDistance).toBeFinite();
        }
    });

    test("rejects a synthetic zero-contrast frame", () => {
        const data = new Uint8Array(64 * 64 * 4).fill(96);
        for (let index = 3; index < data.length; index += 4) data[index] = 255;
        expect(analyze({ width: 64, height: 64, data }).lumaRange).toBeLessThan(0.02);
    });
});
