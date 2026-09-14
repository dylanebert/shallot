import { readFileSync } from "node:fs";
import { join } from "node:path";
import { check } from "../../../harness/check";

const root = join(import.meta.dir, "coverage");
const inventory = JSON.parse(readFileSync(join(root, "inventory-v6.json"), "utf8")) as {
    schema: string;
    source: { sha: string; tree: string };
    population: { suiteFiles: string[]; suiteCount: number; caseCount: number };
    suites: unknown[];
    cases: { id: string; registrations?: unknown[] }[];
};
const coverage = JSON.parse(readFileSync(join(root, "coverage-v6.json"), "utf8")) as {
    schema: string;
    inventorySchema: string;
    source: { sha: string; tree: string };
    population: { suiteCount: number; caseCount: number };
    cases: { id: string; status: string; execution: Record<string, unknown> }[];
};

check(
    "Box3D O6b inventory and coverage are consumed exactly",
    { claim: "box3d-inventory-exact-join", size: "integration" },
    () => {
        const sha = "47d7f7cc7e091142c08d11dc7d2e493c5d34f536";
        const tree = "0a6472d03ff61c9af28216443fba341a21cbed96";
        if (
            inventory.schema !== "box3d-oracle/inventory/v1" ||
            coverage.schema !== "box3d-oracle/coverage/v1"
        )
            throw new Error("inventory/coverage schema drifted");
        if (
            coverage.inventorySchema !== inventory.schema ||
            inventory.source.sha !== sha ||
            inventory.source.tree !== tree ||
            coverage.source.sha !== sha ||
            coverage.source.tree !== tree
        )
            throw new Error("inventory source identity drifted");
        if (
            inventory.population.suiteCount !== 25 ||
            inventory.population.caseCount !== 259 ||
            coverage.population.suiteCount !== 25 ||
            coverage.population.caseCount !== 259
        )
            throw new Error("O6b population is not exactly 25 suites and 259 cases");
        if (
            inventory.suites.length !== 25 ||
            inventory.cases.length !== 259 ||
            coverage.cases.length !== 259
        )
            throw new Error("inventory or coverage rows are incomplete");
        const inventoryIds = inventory.cases.map((item) => item.id);
        const coverageIds = coverage.cases.map((item) => item.id);
        if (new Set(inventoryIds).size !== 259 || new Set(coverageIds).size !== 259)
            throw new Error("inventory or coverage contains duplicate IDs");
        if (JSON.stringify(inventoryIds) !== JSON.stringify(coverageIds))
            throw new Error("coverage is not the exact ordered inventory join");
        if (
            coverage.cases.some(
                (item) =>
                    item.status !== "not-applicable" ||
                    item.execution.executable !== "official-upstream-test",
            )
        )
            throw new Error("O7 consumer changed the empty/no-production coverage premise");
    },
);
