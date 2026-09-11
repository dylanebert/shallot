import { existsSync, readFileSync } from "node:fs";
import { resolve } from "node:path";
import { CHECK_SIZES, INTEGRATION_BUDGET_MS, UNIT_BUDGET_MS } from "../src/harness/declaration";
import { collectPopulation, readQuarantine, renderWorkflow } from "./surface";

// `check` arm for the check surface itself: every test-suffix file declares, claims are unique,
// no declaration overruns its size ceiling, quarantine rows are live and current, and the hosted
// workflow is exactly the one the population emits.

function isShallotRoot(root: string): boolean {
    const packagePath = resolve(root, "package.json");
    if (!existsSync(packagePath)) return false;
    try {
        return (
            (JSON.parse(readFileSync(packagePath, "utf8")) as { name?: unknown }).name ===
            "@dylanebert/shallot"
        );
    } catch {
        return false;
    }
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
        const ceiling = row.size === "unit" ? UNIT_BUDGET_MS : INTEGRATION_BUDGET_MS;
        if (!CHECK_SIZES.includes(row.size as (typeof CHECK_SIZES)[number])) {
            violations.push(`unknown size: "${row.size}" in ${row.file}`);
        } else if (row.budget > ceiling) {
            violations.push(
                `over-budget declaration: "${row.claim}" in ${row.file} declares ${row.budget}ms above the ${row.size} ceiling of ${ceiling}ms`,
            );
        }
    }

    const quarantine = readQuarantine(root);
    violations.push(...quarantine.errors);
    const files = new Set(population.rows.map((row) => row.file));
    const claims = new Set(population.rows.map((row) => row.claim));
    const exactRows = new Set(population.rows.map((row) => `${row.file}\u0000${row.claim}`));
    const today = new Date().toISOString().slice(0, 10);
    for (const row of quarantine.rows) {
        if (row.expires < today) {
            violations.push(`expired quarantine row: "${row.claim}" expired ${row.expires}`);
        }
        if (!files.has(row.file)) {
            violations.push(
                `orphan quarantine row: file "${row.file}" names no check in the population`,
            );
        }
        if (!claims.has(row.claim)) {
            violations.push(
                `orphan quarantine row: claim "${row.claim}" names no check in the population`,
            );
        }
        if (
            files.has(row.file) &&
            claims.has(row.claim) &&
            !exactRows.has(`${row.file}\u0000${row.claim}`)
        ) {
            violations.push(
                `orphan quarantine row: file "${row.file}" and claim "${row.claim}" do not identify the same check`,
            );
        }
    }

    if (isShallotRoot(root)) {
        const workflowPath = resolve(root, ".github/workflows/test-surface.yml");
        if (!existsSync(workflowPath)) {
            violations.push("missing generated workflow: .github/workflows/test-surface.yml");
        } else if (readFileSync(workflowPath, "utf8") !== renderWorkflow(population)) {
            violations.push(
                "generated workflow drift: .github/workflows/test-surface.yml differs from surface.ts --workflow",
            );
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
    const quarantine = readQuarantine(root);
    console.log(
        `${collectPopulation(root).rows.length} declared checks (${quarantine.rows.length} quarantined)`,
    );
}
