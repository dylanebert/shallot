import type { StrictReport, StrictResult } from "./strict";

export type KnownDifferenceEntry = {
    caseId: string;
    mismatchKind: string;
    fingerprint: string;
    firstDifference: string;
    reason: string;
    owner: { spec: string; stage: string };
    introduced: string;
    expires: string;
};

export type KnownDifferenceLedger = {
    schema: "box3d-oracle/known-differences/v1";
    target: string;
    bundle: string;
    entries: KnownDifferenceEntry[];
};

export type KnownDifferenceSummary = {
    pass: number;
    expectedDifferences: number;
    unexpected: number;
    errors: string[];
};

const CASE_ID = /^[A-Za-z0-9_.:-]+$/;
const FINGERPRINT = /^[0-9a-f]{64}$/;
const DATE = /^\d{4}-\d{2}-\d{2}$/;

function error(summary: KnownDifferenceSummary, message: string): void {
    summary.errors.push(message);
    summary.unexpected += 1;
}

function validateEntry(
    entry: KnownDifferenceEntry,
    ids: Set<string>,
    now: string,
    summary: KnownDifferenceSummary,
): void {
    if (!CASE_ID.test(entry.caseId) || entry.caseId.includes("*") || entry.caseId.includes("?"))
        error(summary, `wildcard case ID: ${entry.caseId}`);
    if (!ids.has(entry.caseId)) error(summary, `orphan case ID: ${entry.caseId}`);
    if (entry.mismatchKind === "skip" || entry.mismatchKind === "timeout")
        error(summary, `unsupported ledger mismatch kind: ${entry.caseId}`);
    if (!FINGERPRINT.test(entry.fingerprint))
        error(summary, `invalid fingerprint: ${entry.caseId}`);
    if (!DATE.test(entry.introduced) || !DATE.test(entry.expires))
        error(summary, `invalid ledger date: ${entry.caseId}`);
    if (entry.expires <= now) error(summary, `expired ledger entry: ${entry.caseId}`);
    if (
        entry.owner.spec !== "shallot-physics-oracle-foundation" ||
        entry.owner.stage !== "o7-consumer"
    )
        error(summary, `wrong owner: ${entry.caseId}`);
    if (entry.reason.trim() === "" || entry.firstDifference.trim() === "")
        error(summary, `incomplete ledger entry: ${entry.caseId}`);
}

export function evaluateKnownDifferences(
    report: Pick<StrictReport, "target" | "bundle" | "results">,
    ledger: KnownDifferenceLedger,
    inventoryIds: readonly string[],
    now = new Date().toISOString().slice(0, 10),
): KnownDifferenceSummary {
    const summary: KnownDifferenceSummary = {
        pass: 0,
        expectedDifferences: 0,
        unexpected: 0,
        errors: [],
    };
    const ids = new Set(inventoryIds);
    if (ledger.schema !== "box3d-oracle/known-differences/v1")
        error(summary, "invalid known-differences schema");
    if (ledger.target !== report.target || ledger.bundle !== report.bundle)
        error(summary, "ledger target or bundle does not match strict report");
    const byId = new Map<string, KnownDifferenceEntry>();
    for (const entry of ledger.entries) {
        if (byId.has(entry.caseId)) error(summary, `duplicate ledger entry: ${entry.caseId}`);
        byId.set(entry.caseId, entry);
        validateEntry(entry, ids, now, summary);
    }
    const resultIds = new Set<string>();
    for (const result of report.results) {
        if (resultIds.has(result.id)) {
            error(summary, `duplicate strict result: ${result.id}`);
            continue;
        }
        resultIds.add(result.id);
        const entry = byId.get(result.id);
        if (result.status === "pass") {
            if (entry) error(summary, `XPASS: ${result.id}`);
            else summary.pass += 1;
            continue;
        }
        if (!entry) {
            error(summary, `unexpected difference: ${result.id}`);
            continue;
        }
        if (result.error?.toLowerCase().includes("timeout")) {
            error(summary, `execution timeout is not an exact ledger premise: ${result.id}`);
            continue;
        }
        if (
            entry.mismatchKind !== result.mismatchKind ||
            entry.fingerprint !== result.fingerprint
        ) {
            error(summary, `changed fingerprint: ${result.id}`);
            continue;
        }
        summary.expectedDifferences += 1;
    }
    for (const id of ids) if (!resultIds.has(id)) error(summary, `unexecuted case: ${id}`);
    for (const id of byId.keys())
        if (!resultIds.has(id)) error(summary, `orphan ledger result: ${id}`);
    return summary;
}

export function assertKnownDifferencePass(summary: KnownDifferenceSummary): void {
    if (summary.unexpected !== 0 || summary.errors.length !== 0)
        throw new Error(
            `known differences red: ${summary.pass} pass, ${summary.expectedDifferences} expected differences, ${summary.unexpected} unexpected; ${summary.errors.join("; ")}`,
        );
}

export function syntheticResult(
    id: string,
    fingerprint: string,
    kind: "value" | "execution-error" = "value",
): StrictResult {
    return {
        id,
        status: "mismatch",
        mismatchKind: kind,
        fingerprint,
        firstDifference: { path: "$.value", expected: 0, actual: 1 },
    };
}
