import { describe, expect, test } from "bun:test";
import { compileConcurrencyRatio, compileVitalReports } from "./rum-compile-vitals";

// Red-first: witnessed red with `compileVitalReports` stubbed to `return []` unconditionally —
// 6 of 8 tests failed (`received length 0`, `TypeError: undefined is not an object (evaluating
// 'reports[0].duration')`), leaving only the two negative-direction tests ("no report when
// nothing matches", the substring-vs-startsWith test) green, confirming the stub stubbed to the
// wrong side of the filter before the real prefix match and epoch conversion were restored.

const PREFIX = "shallot:pipeline-compile:";

describe("compileVitalReports", () => {
    test("filters to prefixed entries only", () => {
        const reports = compileVitalReports(
            [
                { name: `${PREFIX}sear-forward`, startTime: 100, duration: 40 },
                { name: "some-other-measure", startTime: 200, duration: 10 },
            ],
            PREFIX,
            0,
        );
        expect(reports).toHaveLength(1);
        expect(reports[0].name).toBe(`${PREFIX}sear-forward`);
    });

    test("no report when nothing matches the prefix", () => {
        const reports = compileVitalReports(
            [{ name: "unrelated", startTime: 0, duration: 5 }],
            PREFIX,
            0,
        );
        expect(reports).toHaveLength(0);
    });

    test("label is the measure name with the prefix stripped", () => {
        const reports = compileVitalReports(
            [{ name: `${PREFIX}slab-warm`, startTime: 0, duration: 1 }],
            PREFIX,
            0,
        );
        expect(reports[0].context.label).toBe("slab-warm");
    });

    test("startTime is converted to an epoch timestamp via timeOrigin", () => {
        const reports = compileVitalReports(
            [{ name: `${PREFIX}x`, startTime: 500, duration: 1 }],
            PREFIX,
            1_000_000,
        );
        expect(reports[0].startTime).toBe(1_000_500);
    });

    test("duration passes through unchanged", () => {
        const reports = compileVitalReports(
            [{ name: `${PREFIX}x`, startTime: 0, duration: 73.5 }],
            PREFIX,
            0,
        );
        expect(reports[0].duration).toBe(73.5);
    });

    test("multiple matching entries each produce their own report", () => {
        const reports = compileVitalReports(
            [
                { name: `${PREFIX}a`, startTime: 0, duration: 1 },
                { name: `${PREFIX}b`, startTime: 10, duration: 2 },
            ],
            PREFIX,
            0,
        );
        expect(reports.map((r) => r.context.label)).toEqual(["a", "b"]);
    });

    test("a non-prefixed entry that merely contains the prefix substring mid-string is not a match", () => {
        // the filter is startsWith, not includes — a measure named unrelated to compiles that
        // happens to carry the prefix text later in its name must not forward.
        const reports = compileVitalReports(
            [{ name: `noise-${PREFIX}x`, startTime: 0, duration: 1 }],
            PREFIX,
            0,
        );
        expect(reports).toHaveLength(0);
    });

    test("report.name carries the original entry name, for the caller to clear by", () => {
        const reports = compileVitalReports(
            [{ name: `${PREFIX}clear-me`, startTime: 0, duration: 1 }],
            PREFIX,
            0,
        );
        expect(reports[0].name).toBe(`${PREFIX}clear-me`);
    });
});

// Red-first, `compileConcurrencyRatio`'s own `span` computation: witnessed red with `span` computed
// as `sum` instead of `max(end) - min(start)` (the named mutation this arm exists to catch) — the
// back-to-back and gapped arms both stayed green (their span equals their sum by construction on
// a fully-packed batch, and a gap only lowers a correct span below sum), but "a 3-way overlap
// reads ratio > 1" failed: `sum === sum` forces ratio to exactly 1 on every input, so the
// overlapping-set arm below is the one that discriminates the mutation.
describe("compileConcurrencyRatio", () => {
    test("a fully serial (back-to-back, no gap) batch reads ratio ≈ 1.0", () => {
        // three entries, each starting exactly where the previous one ended.
        const result = compileConcurrencyRatio(
            [
                { name: `${PREFIX}a`, startTime: 0, duration: 10 },
                { name: `${PREFIX}b`, startTime: 10, duration: 15 },
                { name: `${PREFIX}c`, startTime: 25, duration: 5 },
            ],
            PREFIX,
        );
        expect(result.count).toBe(3);
        expect(result.sum).toBe(30);
        expect(result.span).toBe(30);
        expect(result.ratio).toBeCloseTo(1.0, 5);
    });

    test("an overlapping batch reads ratio > 1", () => {
        // three entries all starting at 0, each running the whole 10ms window concurrently —
        // sum (30) triples the span (10) they actually occupy.
        const result = compileConcurrencyRatio(
            [
                { name: `${PREFIX}a`, startTime: 0, duration: 10 },
                { name: `${PREFIX}b`, startTime: 0, duration: 10 },
                { name: `${PREFIX}c`, startTime: 0, duration: 10 },
            ],
            PREFIX,
        );
        expect(result.count).toBe(3);
        expect(result.sum).toBe(30);
        expect(result.span).toBe(10);
        expect(result.ratio).toBeCloseTo(3.0, 5);
    });

    test("a gapped (idle between compiles) batch reads ratio < 1", () => {
        // two 10ms entries with a 30ms idle gap between them — the span includes the gap, the
        // sum doesn't, so the ratio reads below 1 (still correctly "not overlapping").
        const result = compileConcurrencyRatio(
            [
                { name: `${PREFIX}a`, startTime: 0, duration: 10 },
                { name: `${PREFIX}b`, startTime: 40, duration: 10 },
            ],
            PREFIX,
        );
        expect(result.span).toBe(50);
        expect(result.sum).toBe(20);
        expect(result.ratio).toBeCloseTo(0.4, 5);
    });

    test("non-prefixed entries are excluded from count, sum, and span", () => {
        const result = compileConcurrencyRatio(
            [
                { name: `${PREFIX}a`, startTime: 100, duration: 10 },
                { name: "unrelated-measure", startTime: 0, duration: 10_000 },
            ],
            PREFIX,
        );
        expect(result.count).toBe(1);
        expect(result.sum).toBe(10);
        expect(result.span).toBe(10);
    });

    test("an empty prefix-matching batch reads the documented degenerate answer, never NaN", () => {
        const result = compileConcurrencyRatio(
            [{ name: "unrelated-measure", startTime: 0, duration: 5 }],
            PREFIX,
        );
        expect(result).toEqual({ count: 0, sum: 0, span: 0, ratio: 1 });
    });

    test("a single entry reads ratio 1.0 with span equal to its own duration", () => {
        const result = compileConcurrencyRatio(
            [{ name: `${PREFIX}solo`, startTime: 500, duration: 42 }],
            PREFIX,
        );
        expect(result).toEqual({ count: 1, sum: 42, span: 42, ratio: 1 });
    });
});
