/** the four check classes: what a check costs to run, under the Kex surface law. */
export const CHECK_CLASSES = ["pure", "process", "seat", "oracle"] as const;
/** the six tiers: which runner and cadence a check belongs to. */
export const CHECK_TIERS = ["step", "gpu", "browser", "headed", "built", "live"] as const;
/** the wall-clock ceiling a `step`-tier check may declare, in milliseconds. Never raised. */
export const STEP_BUDGET_MS = 1000;

export type CheckClass = (typeof CHECK_CLASSES)[number];
export type CheckTier = (typeof CHECK_TIERS)[number];

/** Defaults coupled to each tier; `live` has no automatic wall-clock ceiling. */
export const TIER_DEFAULTS = {
    step: { class: "pure", ceiling: STEP_BUDGET_MS },
    gpu: { class: "seat", ceiling: 60000 },
    browser: { class: "process", ceiling: 20000 },
    headed: { class: "seat", ceiling: 60000 },
    built: { class: "process", ceiling: 20000 },
    live: { class: "oracle", ceiling: undefined },
} as const satisfies Record<CheckTier, { class: CheckClass; ceiling: number | undefined }>;

/** The options object every check declares; class, premises and budget derive from its tier. */
export interface CheckDeclaration {
    /** unique sentence naming the defect this check would catch. */
    claim: string;
    /** which runner and cadence owns it. */
    tier: CheckTier;
    /** what running it costs; defaults from `tier`. */
    class?: CheckClass;
    /** external things that must exist for it to run; defaults to hermetic. */
    premises?: readonly string[];
    /** wall-clock ceiling in milliseconds; defaults from `tier` when that tier has one. */
    budget?: number;
}

export interface ResolvedCheckDeclaration {
    claim: string;
    class: CheckClass;
    tier: CheckTier;
    premises: readonly string[];
    budget?: number;
}

/**
 * Validate a declaration and fill its tier-derived values.
 * Shared with `scripts/check-surface.ts`, which reads declarations statically.
 */
export function validateDeclaration(where: string, value: unknown): ResolvedCheckDeclaration {
    if (typeof value !== "object" || value === null || Array.isArray(value)) {
        throw new Error(`invalid declaration: ${where} needs an options object`);
    }
    const decl = value as Record<string, unknown>;
    for (const field of ["claim", "tier"] as const) {
        if (!(field in decl))
            throw new Error(`invalid declaration: ${where} is missing \`${field}\``);
    }
    if (typeof decl.claim !== "string" || decl.claim.trim() === "") {
        throw new Error(`invalid declaration: ${where} needs a non-empty \`claim\``);
    }
    if (!CHECK_TIERS.includes(decl.tier as CheckTier)) {
        throw new Error(
            `invalid declaration: ${where} has tier \`${String(decl.tier)}\`, not one of ${CHECK_TIERS.join(", ")}`,
        );
    }
    const tier = decl.tier as CheckTier;
    const defaults = TIER_DEFAULTS[tier];
    if (decl.class !== undefined && !CHECK_CLASSES.includes(decl.class as CheckClass)) {
        throw new Error(
            `invalid declaration: ${where} has class \`${String(decl.class)}\`, not one of ${CHECK_CLASSES.join(", ")}`,
        );
    }
    if (decl.class !== undefined && decl.class !== defaults.class) {
        throw new Error(
            `invalid declaration: ${where} class \`${String(decl.class)}\` contradicts tier \`${tier}\` (expected \`${defaults.class}\`)`,
        );
    }
    const premises = decl.premises === undefined ? [] : decl.premises;
    if (!Array.isArray(premises) || premises.some((p) => typeof p !== "string")) {
        throw new Error(`invalid declaration: ${where} needs \`premises\` as an array of strings`);
    }
    if (
        decl.budget !== undefined &&
        (typeof decl.budget !== "number" || !Number.isFinite(decl.budget) || decl.budget <= 0)
    ) {
        throw new Error(
            `invalid declaration: ${where} needs a positive finite \`budget\` in milliseconds`,
        );
    }
    const budget = decl.budget ?? defaults.ceiling;
    if (defaults.ceiling !== undefined && budget !== undefined && budget > defaults.ceiling) {
        throw new Error(
            `invalid declaration: ${where} budget ${budget}ms is above the ${tier} ceiling of ${defaults.ceiling}ms`,
        );
    }
    return {
        claim: decl.claim,
        class: defaults.class,
        tier,
        premises,
        budget,
    };
}
