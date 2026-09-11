import { test } from "bun:test";
import { type CheckDeclaration, validateDeclaration } from "./declaration";
import {
    emitVerdict,
    missingRequirement,
    quarantineReason,
    type VerdictMetadata,
    verdictMetadata,
} from "./verdict";

export * from "./declaration";

let currentFile: string | null = null;
const declared = new Map<string, number>();

/** Preload hook: open a check file. Not part of the public call shape. */
export function beginFile(path: string): void {
    currentFile = path;
    declared.set(path, 0);
}

/** Preload hook: refuse a check file that declared nothing. */
export function assertDeclared(path: string): void {
    const seen = currentFile;
    currentFile = null;
    if (seen !== null && seen !== path) {
        throw new Error(`undeclared check file: ${path} loaded while ${seen} was still open`);
    }
    if ((declared.get(path) ?? 0) === 0) {
        throw new Error(
            `undeclared check file: ${path} registers no check(); declare each test with check(name, { claim, size, requires, budget }, body)`,
        );
    }
}

/** Register one check. The declaration supplies the claim, cadence, requirements and timeout. */
export function check(
    name: string,
    declaration: CheckDeclaration,
    body: () => unknown | Promise<unknown>,
): void {
    const decl = validateDeclaration(`check(${JSON.stringify(name)})`, declaration);
    const file = currentFile;
    if (file !== null) declared.set(file, (declared.get(file) ?? 0) + 1);

    // The ordinary package test intentionally excludes integration rows. They remain discovered,
    // but their bodies are not scheduled; test:changed and hosted jobs omit this filter.
    if (
        (process.env.SHALLOT_UNIT_ONLY === "1" && decl.size === "integration") ||
        (process.env.SHALLOT_INTEGRATION_ONLY === "1" && decl.size === "unit")
    ) {
        test.skip(name, () => {}, decl.budget);
        return;
    }

    const refusal =
        (file === null ? null : quarantineReason(file, decl.claim)) ??
        missingRequirement(decl.requires);
    const reports = decl.size === "integration" || refusal !== null;
    if (refusal !== null) {
        if (reports) {
            emitVerdict(decl.claim, decl.size, performance.now(), "refused", { reason: refusal });
        }
        test.skip(name, () => {}, decl.budget);
        return;
    }

    test(
        name,
        async () => {
            const started = performance.now();
            try {
                const value = await body();
                const metadata: VerdictMetadata = verdictMetadata(value);
                if (
                    value !== null &&
                    typeof value === "object" &&
                    (value as Record<string, unknown>).ok === false
                ) {
                    const error = Object.assign(
                        new Error(`browser harness returned a failing verdict for ${decl.claim}`),
                        metadata,
                    );
                    throw error;
                }
                if (reports) emitVerdict(decl.claim, decl.size, started, "pass", metadata);
                return value;
            } catch (error) {
                if (reports) {
                    emitVerdict(decl.claim, decl.size, started, "fail", verdictMetadata(error));
                }
                throw error;
            }
        },
        decl.budget,
    );
}
