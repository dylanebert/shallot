import { test } from "bun:test";
import { type CheckDeclaration, validateDeclaration } from "./declaration";

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
    body: () => void | Promise<void>,
): void {
    const decl = validateDeclaration(`check(${JSON.stringify(name)})`, declaration);
    if (currentFile !== null) declared.set(currentFile, (declared.get(currentFile) ?? 0) + 1);
    test(name, body, decl.budget);
}
