import { test } from "bun:test";
import { type CheckDeclaration, validateDeclaration } from "./declaration";
import {
    emitVerdict,
    missingPremise,
    quarantineReason,
    type VerdictMetadata,
    verdictMetadata,
} from "./verdict";

// The one call every check is registered through: Bun's `test` behind a mandatory typed
// declaration, with the declared budget as the runner's timeout.

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
            `undeclared check file: ${path} registers no check(); declare each test with check(name, { claim, class, tier, premises, budget }, body)`,
        );
    }
}

/**
 * Register one check. The declaration is mandatory and its budget is the runner's timeout.
 * @param name what the runner prints.
 * @param declaration the claim, class, tier, premises and budget.
 * @param body the check itself.
 */
export function check(
    name: string,
    declaration: CheckDeclaration,
    body: () => unknown | Promise<unknown>,
): void {
    const decl = validateDeclaration(`check(${JSON.stringify(name)})`, declaration);
    const file = currentFile;
    if (file !== null) declared.set(file, (declared.get(file) ?? 0) + 1);

    const refusal =
        (file === null ? null : quarantineReason(file, decl.claim)) ??
        missingPremise(decl.premises);
    const reports = decl.tier !== "step" || refusal !== null;
    if (refusal !== null) {
        if (reports) {
            emitVerdict(decl.claim, decl.tier, performance.now(), "refused", { reason: refusal });
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
                if (reports) emitVerdict(decl.claim, decl.tier, started, "pass", metadata);
                return value;
            } catch (error) {
                if (reports) {
                    emitVerdict(decl.claim, decl.tier, started, "fail", verdictMetadata(error));
                }
                throw error;
            }
        },
        decl.budget,
    );
}
