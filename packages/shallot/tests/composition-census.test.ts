import { describe, expect, test } from "bun:test";
import { readFile } from "node:fs/promises";
import { join, resolve } from "node:path";
import { $ } from "bun";
import { resolvePlugins } from "../src/engine/app/compose";
import { DEFAULT_PLUGINS } from "../src/standard/defaults";
import { roster } from "./conformance-roster";

const root = resolve(import.meta.dir, "../../..");

type CensusRow = { path: string; gate: string };

/**
 * Existing project gates which execute compositions that cannot be imported in bun without their
 * browser/tooling host. The population itself is discovered below; this table classifies surfaces,
 * never individual compositions, so adding or removing a site cannot silently drift a hand roster.
 */
const PROJECT_GATES: readonly [RegExp, string][] = [
    [/^examples\/flows\//, "bun run flows"],
    [/^examples\/recipes\//, "bun run recipes"],
    [/^examples\/showcase\//, "bun run test:changed --all"],
    [/^examples\/gym\//, "bun bench"],
    [/^evals\/tasks\/[^/]+\/gate\.ts$/, "bun run test"],
    [/^packages\/shallot\/bin\/tui\.ts$/, "bun test ./packages/shallot/bin"],
    [
        /^packages\/shallot\/scripts\/dump-cells-ascii\.ts$/,
        "bun run --cwd packages/shallot dump-cells-ascii",
    ],
];

async function tracked(): Promise<string[]> {
    const out = await $`git ls-files`.cwd(root).quiet().text();
    return out.trim().split("\n").filter(Boolean);
}

async function projectRows(files: readonly string[]): Promise<CensusRow[]> {
    const rows: CensusRow[] = [];
    for (const path of files) {
        const inSurface = PROJECT_GATES.some(([pattern]) => pattern.test(path));
        if (!inSurface || (!path.endsWith(".ts") && !path.endsWith("shallot.json"))) continue;
        const source = await readFile(join(root, path), "utf8");
        const composes =
            path.endsWith("shallot.json") ||
            /^evals\/tasks\/[^/]+\/gate\.ts$/.test(path) ||
            /\bplugins\s*:\s*(?:\[|[A-Za-z_$])/.test(source);
        if (!composes) continue;
        const match = PROJECT_GATES.find(([pattern]) => pattern.test(path));
        if (!match)
            throw new Error(
                `${path}: plugin composition has no device-free census or real-project gate`,
            );
        rows.push({ path, gate: match[1] });
    }
    return rows;
}

describe("plugin composition census", () => {
    test("every importable shared composition resolves device-free", () => {
        const compositions = [
            ["standard defaults", DEFAULT_PLUGINS] as const,
            ...Object.entries(roster).map(
                ([name, entry]) => [`conformance:${name}`, entry.plugins] as const,
            ),
        ];
        expect(compositions.length).toBeGreaterThan(1);
        for (const [name, plugins] of compositions) {
            expect(resolvePlugins(plugins).missing, name).toEqual([]);
        }
    });

    test("derives every hosted composition and names its existing gate", async () => {
        const files = await tracked();
        const rows = await projectRows(files);
        expect(rows.length).toBeGreaterThan(0);

        // Both directions: every discovered site has exactly one gate, and every declared surface
        // owns at least one discovered site. A new unclassified composition or a stale surface reds.
        expect(new Set(rows.map(({ path }) => path)).size).toBe(rows.length);
        for (const [surface, gate] of PROJECT_GATES) {
            expect(
                rows.some((row) => surface.test(row.path) && row.gate === gate),
                `${surface}`,
            ).toBeTrue();
        }
    });
});
