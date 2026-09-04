import { describe, expect, test } from "bun:test";
import { readFile } from "node:fs/promises";
import { join, resolve } from "node:path";
import { $ } from "bun";
import { resolvePlugins } from "../src/engine/app/compose";
import { DEFAULT_PLUGINS } from "../src/standard/defaults";
import { roster } from "./conformance-roster";

const root = resolve(import.meta.dir, "../../..");

type CensusRow = { path: string; gate: string };
type SourceFile = { path: string; source: string };
type Gate = readonly [RegExp, string];

/** Composition-bearing host surfaces. This discovers the population; PROJECT_GATES only classifies it. */
const COMPOSITION_SURFACES: readonly RegExp[] = [
    /^examples\/(?:flows|recipes|showcase|gym)\//,
    /^evals\/tasks\//,
    /^packages\/shallot\/bin\/tui\.ts$/,
    /^packages\/shallot\/scripts\/dump-cells-ascii\.ts$/,
];

/** Existing project gates which execute compositions that cannot be imported in bun. */
const PROJECT_GATES: readonly Gate[] = [
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

function isComposition({ path, source }: SourceFile): boolean {
    return (
        path.endsWith("shallot.json") ||
        /^evals\/tasks\/[^/]+\/gate\.ts$/.test(path) ||
        /\bplugins\s*:\s*(?:\[|[A-Za-z_$])/.test(source)
    );
}

async function discoverCompositions(files: readonly string[]): Promise<SourceFile[]> {
    const candidates = files.filter(
        (path) =>
            COMPOSITION_SURFACES.some((surface) => surface.test(path)) &&
            (path.endsWith(".ts") || path.endsWith("shallot.json")),
    );
    return (
        await Promise.all(
            candidates.map(async (path) => ({
                path,
                source: await readFile(join(root, path), "utf8"),
            })),
        )
    ).filter(isComposition);
}

function classifyCompositions(sites: readonly SourceFile[], gates: readonly Gate[]): CensusRow[] {
    return sites.map(({ path }) => {
        const matches = gates.filter(([pattern]) => pattern.test(path));
        if (matches.length !== 1)
            throw new Error(
                `${path}: plugin composition has ${matches.length} real-project gate classifications`,
            );
        return { path, gate: matches[0][1] };
    });
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
        const sites = await discoverCompositions(await tracked());
        const rows = classifyCompositions(sites, PROJECT_GATES);
        expect(rows.length).toBeGreaterThan(0);

        // Both directions: every independently discovered site has exactly one classification,
        // and every declared gate owns at least one discovered site.
        expect(new Set(rows.map(({ path }) => path)).size).toBe(rows.length);
        for (const [surface, gate] of PROJECT_GATES) {
            expect(
                rows.some((row) => surface.test(row.path) && row.gate === gate),
                `${surface}`,
            ).toBeTrue();
        }
    });

    test("mutation controls reject every omitted gate and an added unclassified site", async () => {
        const sites = await discoverCompositions(await tracked());

        // Derived from the live population and classification table: no composition path is copied
        // into this control, and removing any classification with an owner must fail deterministically.
        for (let omitted = 0; omitted < PROJECT_GATES.length; omitted++) {
            const gate = PROJECT_GATES[omitted];
            if (!sites.some(({ path }) => gate[0].test(path))) continue;
            expect(
                () => classifyCompositions(sites, PROJECT_GATES.toSpliced(omitted, 1)),
                `omitting ${gate[0]}`,
            ).toThrow("real-project gate classifications");
        }

        const added = { path: "evals/tasks/__census/unclassified.ts", source: "plugins: []" };
        expect(COMPOSITION_SURFACES.some((surface) => surface.test(added.path))).toBeTrue();
        expect(isComposition(added)).toBeTrue();
        expect(() => classifyCompositions([...sites, added], PROJECT_GATES)).toThrow(
            "real-project gate classifications",
        );
    });
});
