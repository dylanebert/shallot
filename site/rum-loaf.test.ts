import { describe, expect, test } from "bun:test";
import {
    attributeSlowFrame,
    type LoAFEntry,
    type LoAFScript,
    unsupportedLoafAttribution,
} from "./rum-loaf";

// Red-first: an inclusive-boundary `overlaps` (`<=`/`>=`) failed both boundary-exclusion arms below
// — "an entry ending exactly at the window's start does not overlap" and "...starting exactly at
// the window's end..." each read `received loafEntryCount 1` against `expected 0` — before the
// strict-inequality comparison (`<`/`>`) was restored.

function script(overrides: Partial<LoAFScript> = {}): LoAFScript {
    return {
        sourceURL: "https://example.com/app.js",
        sourceFunctionName: "warm",
        invoker: "https://example.com/app.js",
        duration: 10,
        ...overrides,
    };
}

function entry(overrides: Partial<LoAFEntry> = {}): LoAFEntry {
    return {
        startTime: 0,
        duration: 60,
        blockingDuration: 20,
        scripts: [script()],
        ...overrides,
    };
}

describe("attributeSlowFrame", () => {
    test("zero-overlap path: no entry overlapping the window reports an idle main thread", () => {
        const attribution = attributeSlowFrame(1000, 80, [entry({ startTime: 0, duration: 60 })]);
        expect(attribution.loafSupported).toBe(true);
        expect(attribution.loafEntryCount).toBe(0);
        expect(attribution.loafOverlapDurationMs).toBe(0);
        expect(attribution.loafOverlapBlockingMs).toBe(0);
        expect(attribution.topScript).toBeNull();
    });

    test("an entry ending exactly at the window's start does not overlap", () => {
        // window [100, 180]; entry spans [20, 100]
        const attribution = attributeSlowFrame(100, 80, [entry({ startTime: 20, duration: 80 })]);
        expect(attribution.loafEntryCount).toBe(0);
    });

    test("an entry starting exactly at the window's end does not overlap", () => {
        // window [100, 180]; entry spans [180, 220]
        const attribution = attributeSlowFrame(100, 80, [entry({ startTime: 180, duration: 40 })]);
        expect(attribution.loafEntryCount).toBe(0);
    });

    test("an entry starting exactly at the window's start overlaps", () => {
        // window [100, 180]; entry spans [100, 130]
        const attribution = attributeSlowFrame(100, 80, [entry({ startTime: 100, duration: 30 })]);
        expect(attribution.loafEntryCount).toBe(1);
    });

    test("an entry ending exactly at the window's end overlaps", () => {
        // window [100, 180]; entry spans [150, 180]
        const attribution = attributeSlowFrame(100, 80, [entry({ startTime: 150, duration: 30 })]);
        expect(attribution.loafEntryCount).toBe(1);
    });

    test("multiple overlapping entries: durations and blocking durations sum", () => {
        const entries = [
            entry({
                startTime: 100,
                duration: 40,
                blockingDuration: 10,
                scripts: [script({ duration: 5 })],
            }),
            entry({
                startTime: 130,
                duration: 50,
                blockingDuration: 15,
                scripts: [script({ duration: 30 })],
            }),
        ];
        const attribution = attributeSlowFrame(100, 80, entries);
        expect(attribution.loafEntryCount).toBe(2);
        expect(attribution.loafOverlapDurationMs).toBe(90);
        expect(attribution.loafOverlapBlockingMs).toBe(25);
    });

    test("top script is the longest-running script across every overlapping entry", () => {
        const entries = [
            entry({
                startTime: 100,
                duration: 40,
                scripts: [script({ sourceFunctionName: "small", duration: 5 })],
            }),
            entry({
                startTime: 130,
                duration: 50,
                scripts: [
                    script({ sourceFunctionName: "medium", duration: 12 }),
                    script({ sourceFunctionName: "biggest", duration: 30 }),
                ],
            }),
        ];
        const attribution = attributeSlowFrame(100, 80, entries);
        expect(attribution.topScript?.sourceFunctionName).toBe("biggest");
        expect(attribution.topScript?.duration).toBe(30);
    });

    test("an overlapping entry with an empty scripts array leaves topScript null when no other entry has scripts", () => {
        const attribution = attributeSlowFrame(100, 80, [
            entry({ startTime: 100, duration: 40, scripts: [] }),
        ]);
        expect(attribution.loafEntryCount).toBe(1);
        expect(attribution.topScript).toBeNull();
    });

    test("an empty entries list is the same as no overlap", () => {
        const attribution = attributeSlowFrame(100, 80, []);
        expect(attribution.loafEntryCount).toBe(0);
        expect(attribution.topScript).toBeNull();
    });

    test("unsupportedLoafAttribution carries loafSupported: false and no attribution fields", () => {
        const attribution = unsupportedLoafAttribution();
        expect(attribution.loafSupported).toBe(false);
        expect(attribution.loafEntryCount).toBe(0);
        expect(attribution.loafOverlapDurationMs).toBe(0);
        expect(attribution.loafOverlapBlockingMs).toBe(0);
        expect(attribution.topScript).toBeNull();
    });
});
