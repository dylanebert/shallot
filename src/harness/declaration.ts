/** the four check classes: what a check costs to run, from `checks.md` § Surface Law. */
export const CHECK_CLASSES = ["pure", "process", "seat", "oracle"] as const;
/** the six tiers: which runner and cadence a check belongs to. */
export const CHECK_TIERS = ["step", "gpu", "browser", "headed", "built", "live"] as const;
/** the wall-clock ceiling a `step`-tier check may declare, in milliseconds. Never raised. */
export const STEP_BUDGET_MS = 1000;

export type CheckClass = (typeof CHECK_CLASSES)[number];
export type CheckTier = (typeof CHECK_TIERS)[number];

/** the mandatory options object every check declares. */
export interface CheckDeclaration {
    /** unique sentence naming the defect this check would catch. */
    claim: string;
    /** what running it costs. */
    class: CheckClass;
    /** which runner and cadence owns it. */
    tier: CheckTier;
    /** external things that must exist for it to run; empty means hermetic. */
    premises: readonly string[];
    /** wall-clock ceiling in milliseconds; also the runner's timeout. */
    budget: number;
}

const FIELDS = ["claim", "class", "tier", "premises", "budget"] as const;

/**
 * Validate a declaration, throwing on the first bad field.
 * Shared with `scripts/check-surface.ts`, which reads declarations statically.
 */
export function validateDeclaration(where: string, value: unknown): CheckDeclaration {
    if (typeof value !== "object" || value === null || Array.isArray(value)) {
        throw new Error(`invalid declaration: ${where} needs an options object`);
    }
    const decl = value as Record<string, unknown>;
    for (const field of FIELDS) {
        if (!(field in decl))
            throw new Error(`invalid declaration: ${where} is missing \`${field}\``);
    }
    if (typeof decl.claim !== "string" || decl.claim.trim() === "") {
        throw new Error(`invalid declaration: ${where} needs a non-empty \`claim\``);
    }
    if (!CHECK_CLASSES.includes(decl.class as CheckClass)) {
        throw new Error(
            `invalid declaration: ${where} has class \`${String(decl.class)}\`, not one of ${CHECK_CLASSES.join(", ")}`,
        );
    }
    if (!CHECK_TIERS.includes(decl.tier as CheckTier)) {
        throw new Error(
            `invalid declaration: ${where} has tier \`${String(decl.tier)}\`, not one of ${CHECK_TIERS.join(", ")}`,
        );
    }
    if (!Array.isArray(decl.premises) || decl.premises.some((p) => typeof p !== "string")) {
        throw new Error(`invalid declaration: ${where} needs \`premises\` as an array of strings`);
    }
    if (typeof decl.budget !== "number" || !Number.isFinite(decl.budget) || decl.budget <= 0) {
        throw new Error(
            `invalid declaration: ${where} needs a positive finite \`budget\` in milliseconds`,
        );
    }
    return decl as unknown as CheckDeclaration;
}
