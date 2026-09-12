/** The sizes that set cadence and the wall-clock ceiling for a check. */
export const CHECK_SIZES = ["unit", "integration"] as const;
/** Environment tags a runner may require. */
export const CHECK_REQUIREMENTS = ["chromium", "gpu", "display", "deploy", "cargo"] as const;
export const UNIT_BUDGET_MS = 250;
export const INTEGRATION_BUDGET_MS = 20_000;

export type CheckSize = (typeof CHECK_SIZES)[number];
export type CheckRequirement = (typeof CHECK_REQUIREMENTS)[number];

export interface CheckDeclaration {
    /** unique sentence naming the defect this check would catch. */
    claim: string;
    /** unit by default; integration rows run on the hosted cadence. */
    size?: CheckSize;
    /** external environment tags that must be available to run. */
    requires?: readonly CheckRequirement[];
    /** project-rooted source path(s) whose token changes select an integration row. */
    subject?: string | readonly string[];
    /** wall-clock budget in milliseconds; defaults to the size ceiling. */
    budget?: number;
}

export interface ResolvedCheckDeclaration {
    claim: string;
    size: CheckSize;
    requires: readonly CheckRequirement[];
    subject?: string | readonly string[];
    budget: number;
}

/** Validate a declaration and fill its size-derived values. Shared with the static surface reader. */
export function validateDeclaration(where: string, value: unknown): ResolvedCheckDeclaration {
    if (typeof value !== "object" || value === null || Array.isArray(value)) {
        throw new Error(`invalid declaration: ${where} needs an options object`);
    }
    const decl = value as Record<string, unknown>;
    if (!("claim" in decl)) throw new Error(`invalid declaration: ${where} is missing \`claim\``);
    if (typeof decl.claim !== "string" || decl.claim.trim() === "") {
        throw new Error(`invalid declaration: ${where} needs a non-empty \`claim\``);
    }
    if ("class" in decl || "tier" in decl || "premises" in decl) {
        const field = ["class", "tier", "premises"].find((name) => name in decl);
        throw new Error(`invalid declaration: ${where} has retired field \`${field}\``);
    }
    const size = decl.size === undefined ? "unit" : decl.size;
    if (!CHECK_SIZES.includes(size as CheckSize)) {
        throw new Error(
            `invalid declaration: ${where} has size \`${String(size)}\`, not one of ${CHECK_SIZES.join(", ")}`,
        );
    }
    const requires = decl.requires === undefined ? [] : decl.requires;
    if (!Array.isArray(requires) || requires.some((tag) => typeof tag !== "string")) {
        throw new Error(`invalid declaration: ${where} needs \`requires\` as an array of strings`);
    }
    for (const tag of requires) {
        if (!CHECK_REQUIREMENTS.includes(tag as CheckRequirement)) {
            throw new Error(
                `invalid declaration: ${where} has requirement tag \`${tag}\`, not one of ${CHECK_REQUIREMENTS.join(", ")}`,
            );
        }
    }
    if (
        decl.subject !== undefined &&
        !(
            (typeof decl.subject === "string" && decl.subject.trim() !== "") ||
            (Array.isArray(decl.subject) &&
                decl.subject.length > 0 &&
                decl.subject.every(
                    (subject) => typeof subject === "string" && subject.trim() !== "",
                ))
        )
    ) {
        throw new Error(
            `invalid declaration: ${where} needs a non-empty string \`subject\` or array of strings`,
        );
    }
    if (
        decl.budget !== undefined &&
        (typeof decl.budget !== "number" || !Number.isFinite(decl.budget) || decl.budget <= 0)
    ) {
        throw new Error(
            `invalid declaration: ${where} needs a positive finite \`budget\` in milliseconds`,
        );
    }
    const ceiling = size === "unit" ? UNIT_BUDGET_MS : INTEGRATION_BUDGET_MS;
    const budget = decl.budget ?? ceiling;
    if (budget > ceiling) {
        throw new Error(
            `invalid declaration: ${where} budget ${budget}ms is above the ${size} ceiling of ${ceiling}ms`,
        );
    }
    return {
        claim: decl.claim,
        size: size as CheckSize,
        requires: requires as CheckRequirement[],
        ...(decl.subject === undefined
            ? {}
            : { subject: decl.subject as string | readonly string[] }),
        budget,
    };
}
