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

    // The ordinary unit sweep never schedules integrations. Named oracle files are excluded by the
    // carrier runner; this guard remains so a direct Bun invocation cannot accidentally make an
    // integration row part of the unit population.
    if (
        (process.env.SHALLOT_UNIT_ONLY === "1" && decl.size === "integration") ||
        (process.env.SHALLOT_INTEGRATION_ONLY === "1" && decl.size === "unit") ||
        (process.env.KEX_S3_ROW !== undefined && process.env.KEX_S3_ROW !== decl.claim)
    ) {
        test.skip(name, () => {}, decl.budget);
        return;
    }

    const quarantine = file === null ? null : quarantineReason(file, decl.claim);
    if (quarantine !== null) {
        if (decl.size === "integration") {
            emitVerdict(decl.claim, decl.size, performance.now(), "refused", {
                reason: quarantine,
            });
        }
        test.skip(name, () => {}, decl.budget);
        return;
    }
    const missing = missingRequirement(decl.requires, {
        root: process.env.SHALLOT_PROJECT_ROOT ?? process.cwd(),
        subjects:
            decl.subject === undefined
                ? []
                : typeof decl.subject === "string"
                  ? [decl.subject]
                  : decl.subject,
    });
    if (missing !== null) {
        // A missing premise is refusal, never a green skip. Throw at registration so Bun's exit
        // status carries the refusal through every installed command and hosted runner.
        emitVerdict(decl.claim, decl.size, performance.now(), "refused", { reason: missing });
        throw new Error(`refused check ${decl.claim}: ${missing}`);
    }
    const reports = decl.size === "integration";

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
