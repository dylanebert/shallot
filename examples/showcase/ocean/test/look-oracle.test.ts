import { describe, expect, test } from "bun:test";
import { createHash } from "node:crypto";
import { readFile } from "node:fs/promises";
import { resolve } from "node:path";
import witnesses from "./fixtures/look/provenance.json";
import {
    analyze,
    assertRegeneratedReferences,
    bandRanges,
    load,
    type ReferenceRelation,
    referenceRelationResults,
    satisfiesPriorGoodFloor,
    satisfiesReferenceRelations,
    satisfiesS16Floor,
    satisfiesS16Relations,
} from "./look.oracle";
import fixture from "./reference/look-relations.json";

const baselines = [1, 2, 3].map((index) =>
    resolve(import.meta.dir, `baseline/ocean-baseline-${index}.png`),
);
const witnessPaths = Object.fromEntries(
    Object.entries(witnesses.fixtures).map(([name, witness]) => [
        name,
        resolve(import.meta.dir, "fixtures/look", witness.file),
    ]),
) as Record<keyof typeof witnesses.fixtures, string>;
const baselineReadings = await Promise.all(
    baselines.map(async (path) => analyze(await load(path))),
);
const witnessBytes = Object.fromEntries(
    await Promise.all(
        Object.entries(witnessPaths).map(async ([name, path]) => [name, await readFile(path)]),
    ),
) as Record<keyof typeof witnesses.fixtures, Uint8Array>;
const witnessReadings = Object.fromEntries(
    await Promise.all(
        Object.entries(witnessPaths).map(async ([name, path]) => [name, analyze(await load(path))]),
    ),
) as Record<keyof typeof witnesses.fixtures, ReturnType<typeof analyze>>;

describe("ocean look oracle", () => {
    test("reads the three-capture floor without optional source images", () => {
        for (const reading of baselineReadings) {
            expect(reading.lumaRange).toBeGreaterThan(0.02);
            expect(reading.horizon.transitionWidth).toBeGreaterThan(0);
        }
    });

    test("committed reference readings retain per-image provenance", () => {
        expect(fixture.oracleRevision).toBe("shallot-ocean-look/S14");
        expect(fixture.provenance.absoluteRelationRows).toEqual({ t26: "277/540", t43: "267/540" });
        expect(Object.keys(fixture.provenance.relationMargins)).toHaveLength(12);
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

    test("tracked witnesses retain provenance and their declared red and green relations", () => {
        expect(witnesses.oracleRevision).toBe("shallot-ocean-look/S15");
        for (const [name, witness] of Object.entries(witnesses.fixtures)) {
            const digest = createHash("sha256")
                .update(witnessBytes[name as keyof typeof witnesses.fixtures])
                .digest("hex");
            expect(digest).toBe(witness.sha256);
            expect(witness.source).toStartWith("scratch/shallot-ocean-look/");
            expect(witness.capture).toContain("Apple Metal");
        }

        for (const name of ["s13Default", "s13SunFacing"] as const) {
            const witness = witnesses.fixtures[name];
            const results = referenceRelationResults(witnessReadings[name]);
            for (const relation of witness.mustRed)
                expect(results[relation as ReferenceRelation], `${name} must red ${relation}`).toBe(
                    false,
                );
            for (const relation of witness.mustGreen)
                expect(
                    results[relation as ReferenceRelation],
                    `${name} must keep unrelated ${relation} green`,
                ).toBe(true);
        }
    });

    test("S16 gold intervals reject both S13 fixtures and improve from S9", () => {
        const gold = [fixture.references.t26, fixture.references.t43].map((reading) => ({
            normalResponse: reading.normalResponse,
            reflection: {
                signedSkyTracking: (["horizon", "farWater", "midWater", "nearWater"] as const).map(
                    (band) =>
                        reading.bands[band].srgb[2] -
                        reading.bands[band].srgb[0] -
                        (reading.bands.sky.srgb[2] - reading.bands.sky.srgb[0]),
                ) as [number, number, number, number],
                scaleNormalizedStructure:
                    reading.normalResponse.midRowDeviation / (reading.bands.midWater.luma * 255),
            },
        }));
        for (const reading of gold) expect(satisfiesS16Relations(reading)).toBe(true);
        const clampedReflection = structuredClone(gold[0]!);
        clampedReflection.reflection.signedSkyTracking[2] = 0;
        expect(satisfiesS16Relations(clampedReflection)).toBe(false);
        const oldBody = structuredClone(gold[0]!);
        oldBody.normalResponse.nearMeanChroma = 31;
        expect(satisfiesS16Relations(oldBody)).toBe(false);
        expect(satisfiesS16Relations(witnessReadings.s13Default)).toBe(false);
        expect(satisfiesS16Relations(witnessReadings.s13SunFacing)).toBe(false);
        expect(satisfiesS16Floor(gold[0]!, witnessReadings.s9PriorGood)).toBe(true);
    });

    test("the S9 floor stays in-range or moves strictly toward the gold interval", () => {
        const prior = witnessReadings.s9PriorGood;
        const chroma = prior.normalResponse.nearMeanChroma;
        const chromaGold = fixture.relations.nearMeanChroma;
        expect(chroma).toBeGreaterThan(chromaGold.max);
        expect(satisfiesPriorGoodFloor(chroma - 1, chroma, chromaGold)).toBe(true);
        expect(satisfiesPriorGoodFloor(chroma, chroma, chromaGold)).toBe(false);
        expect(satisfiesPriorGoodFloor(chroma + 1, chroma, chromaGold)).toBe(false);

        const hue = prior.farWaterSkyHueDistance;
        const hueGold = fixture.relations.farWaterSkyHueDistance;
        expect(hue).toBeWithin(hueGold.min, hueGold.max);
        expect(satisfiesPriorGoodFloor(hueGold.min, hue, hueGold)).toBe(true);
        expect(satisfiesPriorGoodFloor(hueGold.max + 0.001, hue, hueGold)).toBe(false);
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
