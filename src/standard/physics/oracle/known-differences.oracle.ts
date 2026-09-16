import { readFileSync } from "node:fs";
import { check } from "../../../harness/check";
import {
    assertKnownDifferencePass,
    evaluateKnownDifferences,
    type KnownDifferenceEntry,
    type KnownDifferenceLedger,
} from "./known-differences";
import { executeStrictReport, type StrictReport, type StrictResult } from "./strict";

const fingerprint = "a".repeat(64);
const base = (
    caseId: string,
    overrides: Partial<KnownDifferenceEntry> = {},
): KnownDifferenceEntry => ({
    caseId,
    mismatchKind: "value",
    fingerprint,
    firstDifference: "$.value",
    reason: "synthetic adversarial fixture",
    owner: { spec: "shallot-physics-oracle-foundation", stage: "o7-consumer" },
    introduced: "2026-09-14",
    expires: "2099-01-01",
    ...overrides,
});
const ledger = (entries: KnownDifferenceEntry[]): KnownDifferenceLedger => ({
    schema: "box3d-oracle/known-differences/v1",
    target: "target",
    bundle: "bundle",
    entries,
});
const report = (
    status: "pass" | "mismatch" = "mismatch",
    mismatchKind: StrictResult["mismatchKind"] = "value",
    error?: string,
): Pick<StrictReport, "target" | "bundle" | "results"> => ({
    target: "target",
    bundle: "bundle",
    results: [
        status === "mismatch"
            ? {
                  id: "case.v1",
                  status,
                  mismatchKind,
                  fingerprint,
                  firstDifference: { path: "$.value", expected: 0, actual: 1 },
                  ...(error === undefined ? {} : { error }),
              }
            : { id: "case.v1", status },
    ],
});

check(
    "known-difference evaluator rejects ledger adversaries",
    { claim: "box3d-known-difference-adversarial", size: "integration" },
    () => {
        const ids = ["case.v1"];
        const matching = evaluateKnownDifferences(
            report(),
            ledger([base("case.v1")]),
            ids,
            "2026-09-15",
        );
        if (
            matching.expectedDifferences !== 1 ||
            matching.unexpected !== 0 ||
            matching.errors.length !== 0
        )
            throw new Error("matching known difference was not accepted");
        const cases: [string, KnownDifferenceLedger][] = [
            ["changed fingerprint", ledger([base("case.v1", { fingerprint: "b".repeat(64) })])],
            ["expired", ledger([base("case.v1", { expires: "2026-09-14" })])],
            ["orphan", ledger([base("orphan.v1")])],
            ["timeout", ledger([base("case.v1", { mismatchKind: "timeout" })])],
            ["skip", ledger([base("case.v1", { mismatchKind: "skip" })])],
            ["wildcard", ledger([base("case.*")])],
            ["duplicate", ledger([base("case.v1"), base("case.v1")])],
        ];
        for (const [name, candidate] of cases) {
            const result = evaluateKnownDifferences(report(), candidate, ids, "2026-09-15");
            if (result.errors.length === 0) throw new Error(`${name} mutation was accepted`);
        }
        const unexecuted = evaluateKnownDifferences(
            { ...report(), results: [] },
            ledger([base("case.v1")]),
            ids,
            "2026-09-15",
        );
        if (unexecuted.errors.length === 0) throw new Error("unexecuted mutation was accepted");
        const xpass = evaluateKnownDifferences(
            report("pass"),
            ledger([base("case.v1")]),
            ids,
            "2026-09-15",
        );
        if (xpass.errors.length === 0) throw new Error("XPASS mutation was accepted");
        const executionError = evaluateKnownDifferences(
            report("mismatch", "execution-error", "timeout"),
            ledger([base("case.v1")]),
            ids,
            "2026-09-15",
        );
        if (executionError.errors.length === 0)
            throw new Error("timeout execution mutation was accepted");
    },
);

check(
    "Box3D known differences consume the executing ledger",
    { claim: "box3d-known-differences", size: "integration", budget: 20_000 },
    () => {
        const strict = executeStrictReport();
        const known = JSON.parse(
            readFileSync(new URL("./reports/known-differences-v6.json", import.meta.url), "utf8"),
        ) as KnownDifferenceLedger;
        const ids = strict.results.map((result) => result.id);
        const summary = evaluateKnownDifferences(strict, known, ids);
        if (
            summary.pass !== 111 ||
            summary.expectedDifferences !== 0 ||
            summary.unexpected !== 0 ||
            summary.errors.length !== 0
        )
            throw new Error("executing ledger did not accept the exact strict mismatch population");
        assertKnownDifferencePass(summary);
    },
);
