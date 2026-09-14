import { createHash } from "node:crypto";
import { readFileSync } from "node:fs";
import { join } from "node:path";
import { loadConsumerCorpus, type OracleCase, runCommonInput } from "./consumer";

export type StrictResult = {
    id: string;
    status: "pass" | "mismatch";
    mismatchKind?: "value" | "execution-error";
    fingerprint?: string;
    firstDifference?: { path: string; expected: unknown; actual: unknown };
    error?: string;
};

export type StrictReport = {
    schema: "box3d-oracle/strict-report/v1";
    target: string;
    conformsThrough: string;
    bundle: string;
    inventory: { suiteCount: number; caseCount: number; digest: string };
    population: { caseCount: number; executed: number; passed: number; mismatched: number };
    results: StrictResult[];
};

const canonical = (value: unknown): unknown => {
    if (Array.isArray(value)) return value.map(canonical);
    if (value !== null && typeof value === "object")
        return Object.fromEntries(
            Object.entries(value as Record<string, unknown>)
                .sort(([a], [b]) => a.localeCompare(b))
                .map(([key, item]) => [key, canonical(item)]),
        );
    return value;
};
const text = (value: unknown): string => JSON.stringify(canonical(value));
const fingerprint = (expected: unknown, actual: unknown): string =>
    createHash("sha256").update(text({ expected, actual })).digest("hex");

function firstDifference(
    expected: unknown,
    actual: unknown,
    path = "$",
): { path: string; expected: unknown; actual: unknown } | undefined {
    if (Object.is(expected, actual)) return undefined;
    if (typeof expected !== typeof actual || expected === null || actual === null)
        return { path, expected, actual };
    if (Array.isArray(expected) && Array.isArray(actual)) {
        if (expected.length !== actual.length)
            return { path: `${path}.length`, expected: expected.length, actual: actual.length };
        for (let index = 0; index < expected.length; index++) {
            const difference = firstDifference(expected[index], actual[index], `${path}[${index}]`);
            if (difference) return difference;
        }
        return undefined;
    }
    if (typeof expected === "object" && typeof actual === "object") {
        const expectedRecord = expected as Record<string, unknown>;
        const actualRecord = actual as Record<string, unknown>;
        const keys = [
            ...new Set([...Object.keys(expectedRecord), ...Object.keys(actualRecord)]),
        ].sort();
        for (const key of keys) {
            if (!(key in expectedRecord) || !(key in actualRecord))
                return {
                    path: `${path}.${key}`,
                    expected: expectedRecord[key],
                    actual: actualRecord[key],
                };
            const difference = firstDifference(
                expectedRecord[key],
                actualRecord[key],
                `${path}.${key}`,
            );
            if (difference) return difference;
        }
        return undefined;
    }
    return { path, expected, actual };
}

export function compareCase(item: OracleCase, actual: unknown): StrictResult {
    const difference = firstDifference(item.output, actual);
    if (!difference) return { id: item.id, status: "pass" };
    return {
        id: item.id,
        status: "mismatch",
        mismatchKind: "value",
        fingerprint: fingerprint(item.output, actual),
        firstDifference: difference,
    };
}

export type StrictWatch = {
    improvements: string[];
    unchangedMismatches: string[];
    regressions: string[];
};

/**
 * Compare a live strict run with O7's frozen report without using O7 as physics authority.
 * A pass is an improvement, not a regression; a mismatch that remains from O7 must retain its
 * exact execution result while it is still deferred. The ledger is deliberately not consulted.
 */
export function watchStrictProgress(current: StrictReport, baseline: StrictReport): StrictWatch {
    const baselineById = new Map(baseline.results.map((result) => [result.id, result]));
    const currentById = new Map(current.results.map((result) => [result.id, result]));
    const improvements: string[] = [];
    const unchangedMismatches: string[] = [];
    const regressions: string[] = [];

    if (current.results.length !== baseline.results.length) {
        regressions.push("strict population changed");
        return { improvements, unchangedMismatches, regressions };
    }

    for (const baselineResult of baseline.results) {
        const currentResult = currentById.get(baselineResult.id);
        if (!currentResult) {
            regressions.push(`${baselineResult.id}: missing from current report`);
            continue;
        }
        if (baselineResult.status === "pass" && currentResult.status === "mismatch") {
            regressions.push(`${baselineResult.id}: pass became mismatch`);
        } else if (baselineResult.status === "mismatch" && currentResult.status === "pass") {
            improvements.push(baselineResult.id);
        } else if (baselineResult.status === "mismatch") {
            if (
                baselineResult.mismatchKind !== currentResult.mismatchKind ||
                baselineResult.fingerprint !== currentResult.fingerprint
            ) {
                regressions.push(`${baselineResult.id}: deferred mismatch changed`);
            } else {
                unchangedMismatches.push(baselineResult.id);
            }
        }
    }

    for (const currentResult of current.results) {
        if (!baselineById.has(currentResult.id))
            regressions.push(`${currentResult.id}: new result in current report`);
    }

    return { improvements, unchangedMismatches, regressions };
}

export function executeStrictReport(): StrictReport {
    const corpus = loadConsumerCorpus();
    const results: StrictResult[] = [];
    for (const item of corpus.cases) {
        try {
            results.push(compareCase(item, runCommonInput(item)));
        } catch (error) {
            const message = error instanceof Error ? error.message : String(error);
            const expected = item.output;
            results.push({
                id: item.id,
                status: "mismatch",
                mismatchKind: "execution-error",
                fingerprint: fingerprint(expected, { error: message }),
                firstDifference: { path: "$", expected, actual: { error: message } },
                error: message,
            });
        }
    }
    const passed = results.filter((result) => result.status === "pass").length;
    const mismatched = results.length - passed;
    const inventoryPath = join(import.meta.dir, "coverage", "coverage-v6.json");
    const inventory = JSON.parse(readFileSync(inventoryPath, "utf8")) as {
        population: { suiteCount: number; caseCount: number };
        source: { sha: string; tree: string };
    };
    return {
        schema: "box3d-oracle/strict-report/v1",
        target: "47d7f7cc7e091142c08d11dc7d2e493c5d34f536",
        conformsThrough: "29bf523ce7bc4590aba9f17c9db791cdc5c4397e",
        bundle: "47d7f7cc7e091142c08d11dc7d2e493c5d34f536/v6",
        inventory: {
            suiteCount: inventory.population.suiteCount,
            caseCount: inventory.population.caseCount,
            digest: createHash("sha256")
                .update(readFileSync(join(import.meta.dir, "coverage", "inventory-v6.json")))
                .digest("hex"),
        },
        population: { caseCount: results.length, executed: results.length, passed, mismatched },
        results,
    };
}

export function readFrozenStrictReport(
    path = join(import.meta.dir, "reports", "strict-v6.json"),
): StrictReport {
    return JSON.parse(readFileSync(path, "utf8")) as StrictReport;
}

export function strictReportDigest(report: StrictReport): string {
    return createHash("sha256").update(text(report)).digest("hex");
}
