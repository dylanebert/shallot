import { describe, expect, test } from "bun:test";
import { compileVitalReports } from "./rum-compile-vitals";

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
