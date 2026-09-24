import { createHash } from "node:crypto";
import type { OracleCase } from "./consumer";

export type StrictResult = {
    id: string;
    status: "pass" | "mismatch";
    mismatchKind?: "value" | "execution-error";
    fingerprint?: string;
    firstDifference?: { path: string; expected: unknown; actual: unknown };
    error?: string;
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
