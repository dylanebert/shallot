import { describe, expect, test } from "bun:test";
import {
    CASCADE_CONFIGS,
    directionalDensity,
    fullMeanSquareSlope,
    K_M,
    philips,
    SEA_STATE,
    type SpectrumMutation,
} from "../src/ocean/spectrum";
import { KM, directionalDensity as referenceDensity } from "./elfouhaily-independent";

const K_ROWS = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 4, 8.482, 60, 100, 200, 370];
const WINDS = [5, 10, 15, 20];
const AGES = [0.84, 0.9, 1.5, 3, 5];
const DIRECTIONS = [0.6, 0.6 + Math.PI / 2];

interface Row {
    wind: number;
    age: number;
    k: number;
    theta: number;
}

const rows: Row[] = WINDS.flatMap((wind) =>
    AGES.flatMap((age) =>
        K_ROWS.flatMap((k) => DIRECTIONS.map((theta) => ({ wind, age, k, theta }))),
    ),
);

function production(row: Row, mutation: SpectrumMutation = {}): number {
    return directionalDensity(
        row.k * Math.cos(row.theta),
        row.k * Math.sin(row.theta),
        row.wind,
        0.6,
        row.age,
        mutation,
    );
}

function reference(row: Row): number {
    return referenceDensity(row.k, row.theta, row.wind, row.age, 0.6);
}

function relativeError(actual: number, expected: number): number {
    if (!(expected > 0)) throw new Error("comparison rows require a positive reference density");
    return Math.abs(actual / expected - 1);
}

describe("production Elfouhaily density", () => {
    test("agrees with the independent equations across the published band", () => {
        expect(K_M).toBeCloseTo(KM, 12);
        expect(K_ROWS).toContain(8.482);
        expect(K_ROWS).toContain(60);
        const comparisonRows = rows.filter(
            (row) =>
                ((row.wind === SEA_STATE.windSpeed && row.age === SEA_STATE.omegaC) ||
                    (row.wind === SEA_STATE.windSpeed && row.age === 1.5)) &&
                reference(row) > 0,
        );
        for (const k of K_ROWS) expect(comparisonRows.some((row) => row.k === k)).toBe(true);
        expect(comparisonRows.some((row) => row.theta === 0.6 + Math.PI / 2)).toBe(true);
        for (const row of comparisonRows) {
            expect(relativeError(production(row), reference(row))).toBeLessThan(0.15);
        }
    });

    test("derives significant wave height and records the declared-band truncation", () => {
        console.info("Elfouhaily shipped sea state", {
            significantWaveHeight: SEA_STATE.significantWaveHeight,
            truncationRatio: SEA_STATE.truncationRatio,
        });
        expect(SEA_STATE.significantWaveHeight).toBeGreaterThan(0);
        expect(SEA_STATE.truncationRatio).toBeGreaterThan(0);
        expect(SEA_STATE.truncationRatio).toBeLessThan(1);
    });

    test("cascade compatibility winds cannot diverge from the shared sea state", () => {
        for (const config of CASCADE_CONFIGS) {
            expect(config.windSpeed).toBe(SEA_STATE.windSpeed);
            expect(config.windDir).toBe(SEA_STATE.windDir);
        }
        expect(() =>
            philips(1, 0, { ...CASCADE_CONFIGS[0], windSpeed: SEA_STATE.windSpeed + 1 }),
        ).toThrow("cascade wind must match the shared sea state");
    });

    test("full-tail slope moment stays in the declared Cox–Munk bands", () => {
        for (const wind of WINDS) {
            const coxMunk = 0.003 + 0.00512 * wind;
            const bound = wind === 20 ? 0.3 : 0.25;
            expect(Math.abs(fullMeanSquareSlope(wind) / coxMunk - 1)).toBeLessThan(bound);
        }
    });
});

describe("production mutation matrix", () => {
    const mutations: Array<{
        name: keyof SpectrumMutation;
        threshold: number;
    }> = [
        { name: "constantDensity", threshold: 0.15 },
        { name: "kMinusThreeDensity", threshold: 0.15 },
        { name: "missingJp", threshold: 0.15 },
        { name: "missingFpExponential", threshold: 0.15 },
        { name: "missingBhEnvelope", threshold: 0.15 },
        { name: "oneBranchAlphaM", threshold: 0.15 },
        { name: "useU10ForFriction", threshold: 0.15 },
        { name: "missingAm", threshold: 0.15 },
        { name: "invertedSpread", threshold: 0.01 },
        { name: "lnForLog10", threshold: 0.01 },
    ];

    for (const { name, threshold } of mutations) {
        test(`${name} fails at its maximum-sensitivity row`, () => {
            let maximum = 0;
            let maximumRow: Row | undefined;
            for (const row of rows) {
                if (reference(row) < 1e-250) continue;
                const error = relativeError(production(row, { [name]: true }), reference(row));
                if (error > maximum) {
                    maximum = error;
                    maximumRow = row;
                }
            }
            console.info("Elfouhaily mutation", { name, maximum, row: maximumRow });
            expect(maximumRow).toBeDefined();
            expect(maximum).toBeGreaterThan(threshold);
        });
    }
});
