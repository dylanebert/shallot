import { existsSync, readFileSync } from "node:fs";
import { resolve } from "node:path";
import { STEP_BUDGET_MS } from "../src/harness/declaration";
import { collectPopulation } from "./surface";

// `check` arm for the check surface itself: every test-suffix file declares, claims are unique,
// no declaration overruns its tier ceiling, and no quarantine row names a claim nobody has.

const TIER_CEILING_MS: Record<string, number> = { step: STEP_BUDGET_MS };

interface QuarantineRow {
    claim?: unknown;
}

function quarantineClaims(root: string): string[] {
    const path = resolve(root, "quarantine.json");
    if (!existsSync(path)) return [];
    const parsed = JSON.parse(readFileSync(path, "utf8")) as
        | QuarantineRow[]
        | { rows?: QuarantineRow[] };
    const rows = Array.isArray(parsed) ? parsed : (parsed.rows ?? []);
    return rows.map((row) => String(row?.claim ?? ""));
}

/** Read one tree's surface and return every violation, most specific message first. */
export function readSurface(root: string): string[] {
    const population = collectPopulation(root);
    const violations: string[] = [...population.invalid];
    for (const file of population.undeclared) {
        violations.push(`undeclared check file: ${file.file} ${file.reason}`);
    }
    const seen = new Map<string, string>();
    for (const row of population.rows) {
        const first = seen.get(row.claim);
        if (first !== undefined) {
            violations.push(`duplicate claim: "${row.claim}" declared in ${first} and ${row.file}`);
        } else {
            seen.set(row.claim, row.file);
        }
        const ceiling = TIER_CEILING_MS[row.tier];
        if (ceiling !== undefined && row.budget > ceiling) {
            violations.push(
                `over-budget declaration: "${row.claim}" in ${row.file} declares ${row.budget}ms above the ${row.tier} ceiling of ${ceiling}ms`,
            );
        }
    }
    for (const claim of quarantineClaims(root)) {
        if (!seen.has(claim)) {
            violations.push(`orphan quarantine row: "${claim}" names no check in the population`);
        }
    }
    return violations;
}

if (import.meta.main) {
    const args = Bun.argv.slice(2);
    const rootIndex = args.indexOf("--root");
    const root =
        rootIndex === -1 ? resolve(import.meta.dir, "..") : resolve(args[rootIndex + 1] ?? ".");
    const violations = readSurface(root);
    for (const violation of violations) console.error(violation);
    if (violations.length > 0) process.exit(1);
    console.log(`${collectPopulation(root).rows.length} declared checks`);
}
