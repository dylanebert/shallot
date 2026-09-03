import { describe, expect, test } from "bun:test";
import { resolve } from "node:path";
import {
    analyze,
    assertRegeneratedReferences,
    bandRanges,
    load,
    satisfiesReferenceRelations,
} from "./look.oracle";
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
        }
    });

    test("committed reference readings retain per-image provenance", () => {
        expect(fixture.oracleRevision).toBe("shallot-ocean-look/S14");
        expect(fixture.provenance.absoluteRelationRows).toEqual({ t26: "277/540", t43: "267/540" });
        expect(Object.keys(fixture.provenance.relationMargins)).toHaveLength(10);
        expect(
            fixture.provenance.sources.every(({ sha256 }) => /^[a-f0-9]{64}$/.test(sha256)),
        ).toBe(true);
    });

    test("both marked frames pass every freshly derived relation", () => {
        expect(satisfiesReferenceRelations(fixture.references.t26)).toBe(true);
        expect(satisfiesReferenceRelations(fixture.references.t43)).toBe(true);
        expect(() => assertRegeneratedReferences(fixture.references)).not.toThrow();
    });

    test("a mutated stored reading reds fixture equality", () => {
        const mutated = structuredClone(fixture.references);
        mutated.t26.horizon.row++;
        expect(() => assertRegeneratedReferences(mutated)).toThrow("t26 reading differs");
    });

    test("both rejected S13 witnesses red their named relations", () => {
        expect(satisfiesReferenceRelations(fixture.rejected.s13Default.reading)).toBe(false);
        expect(satisfiesReferenceRelations(fixture.rejected.s13SunFacing.reading)).toBe(false);
        expect(fixture.rejected.s13Default.mustFail).toEqual(["nearMeanChroma"]);
        expect(fixture.rejected.s13Default.reading.normalResponse.nearMeanChroma).toBeGreaterThan(
            fixture.relations.nearMeanChroma.max,
        );
        expect(fixture.rejected.s13SunFacing.mustFail).toEqual(["fadeExtent"]);
        expect(fixture.rejected.s13SunFacing.reading.duskBalance.fadeExtent).toBeLessThan(
            fixture.relations.fadeExtent.min,
        );
    });

    test("cuts every band from the image's detected horizon", () => {
        expect(bandRanges(210, 720).sky[1]).toBe(210 / 720);
        const shifted = bandRanges(260, 720);
        expect(shifted.sky[1]).toBe(260 / 720);
        expect(shifted.horizon[0]).toBe(shifted.sky[1]);
        expect(shifted.nearWater[1]).toBe(1);
    });

    test("rejects a synthetic zero-contrast frame", () => {
        const data = new Uint8Array(64 * 64 * 4).fill(96);
        for (let index = 3; index < data.length; index += 4) data[index] = 255;
        expect(analyze({ width: 64, height: 64, data }).lumaRange).toBeLessThan(0.02);
    });
});
