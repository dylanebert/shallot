import { describe, expect, test } from "bun:test";
import { readFile } from "node:fs/promises";
import { dirname, join, resolve } from "node:path";
import { parse } from "@babel/parser";
import { $ } from "bun";
import { resolvePlugins } from "../../shallot-runtime/src/engine/app/compose";
import { DEFAULT_PLUGINS } from "../../shallot-runtime/src/standard/defaults";
import { roster } from "./conformance-roster";
import touchConfig from "./orbit-touch/playwright.config";

const root = resolve(import.meta.dir, "../../..");

type CensusRow = { path: string; gate: string };
type SourceFile = { path: string; source: string };
type Gate = readonly [RegExp, string];

/** Composition-bearing host surfaces. This discovers the population; PROJECT_GATES only classifies it. */
const COMPOSITION_SURFACES: readonly RegExp[] = [
    /^examples\/(?:recipes|showcase)\//,
    /^bench\//,
    /^packages\/shallot\/tests\/(?:flows|orbit-touch)\//,
    /^evals\/tasks\//,
    /^packages\/shallot-cli\/bin\/tui\.ts$/,
    /^packages\/shallot-cli\/src\/project\/command\.ts$/,
    /^packages\/shallot\/scripts\/dump-cells-ascii\.ts$/,
];

/** Existing project gates which execute compositions that cannot be imported in bun. */
const PROJECT_GATES: readonly Gate[] = [
    [
        /^packages\/shallot\/tests\/flows\/no-walls\//,
        "bun test ./packages/shallot/tests/flow-no-walls.tier.ts",
    ],
    [
        /^packages\/shallot\/tests\/flows\/blank\//,
        "bun test ./packages/shallot-cli/bin/verify-blank.tier.ts",
    ],
    [
        /^packages\/shallot\/tests\/flows\/survive-reload\//,
        "bun test ./packages/shallot/tests/flow-survive-reload.tier.ts",
    ],
    [
        /^packages\/shallot\/tests\/flows\/ui-containment\//,
        "bun test ./packages/shallot/tests/flow-ui-containment.tier.ts",
    ],
    [
        /^packages\/shallot\/tests\/orbit-touch\//,
        "bunx playwright test -c packages/shallot/tests/orbit-touch/playwright.config.ts",
    ],
    [/^examples\/recipes\//, "bun run recipes"],
    [/^examples\/showcase\//, "bun run test:changed --all"],
    [/^bench\//, "bun bench"],
    [/^evals\/tasks\/[^/]+\/gate\.ts$/, "bun run test"],
    [/^packages\/shallot-cli\/bin\/tui\.ts$/, "bun test ./packages/shallot-cli/bin"],
    [
        /^packages\/shallot-cli\/src\/project\/command\.ts$/,
        "bun test ./packages/shallot-cli/src/project",
    ],
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
        // both surface forms of a composed plugin list: the explicit property (`plugins: […]`,
        // `plugins: project.plugins`) and the shorthand a host uses once it holds the list in a
        // variable of that name (`build({ plugins, … })`, which `bin/tui.ts` now does).
        /\bplugins\s*:\s*(?:\[|[A-Za-z_$])|\bplugins\s*,\s*$/m.test(source)
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

// Independent retained host inventory, not a projection of discovery or classification tables.
const recipes =
    "animate-with-clips annotate-the-world billboards-and-sprites breakable-joints build-a-scene compute-and-readback custom-material day-night-sky drive-a-vehicle first-person game-loop gpu-particles import-a-model joints measure-performance moving-platform overlay-ui physics-playground play-sound ragdoll render-to-a-terminal respond-to-input save-and-restore stylize-the-look surface-friction".split(
        " ",
    );
const relocated = [
    ["flows/blank", "src/main.ts", "packages/shallot-cli/bin/verify-blank.tier.ts"],
    ["flows/no-walls", "src/main.ts", "packages/shallot/tests/flow-no-walls.tier.ts"],
    ["flows/survive-reload", "src/lib.ts", "packages/shallot/tests/flow-survive-reload.tier.ts"],
    ["flows/ui-containment", "src/lib.ts", "packages/shallot/tests/flow-ui-containment.tier.ts"],
    ["orbit-touch", "src/main.ts", "packages/shallot/tests/orbit-touch/playwright.config.ts"],
] as const;
const expectedSites = [
    ...recipes.map((name) => `examples/recipes/${name}/shallot.json`),
    "examples/recipes/overlay-ui/src/hud.ts",
    ..."ascii collapse ocean roads sandbox voxel"
        .split(" ")
        .map((name) => `examples/showcase/${name}/shallot.json`),
    "examples/showcase/ocean/test/composition.test.ts",
    "examples/showcase/visualization/src/boot.ts",
    "examples/showcase/visualization/vite.config.ts",
    ..."accel backend cells chain character constraints gltf gpu-diagnostic mesh-fixture motor outline pile queries raining rotation sat sprite stress text"
        .split(" ")
        .map((name) => `bench/src/scenarios/${name}.ts`),
    "bench/vite.config.ts",
    ..."color-on-key falling-box orbit-on-drag persist-color red-box striped-material"
        .split(" ")
        .map((name) => `evals/tasks/${name}/gate.ts`),
    "packages/shallot-cli/bin/tui.ts",
    "packages/shallot-cli/src/project/command.ts",
    "packages/shallot/scripts/dump-cells-ascii.ts",
    ...relocated.flatMap(([fixture, source]) =>
        [source, "vite.config.ts"].map((path) => `packages/shallot/tests/${fixture}/${path}`),
    ),
].sort();

test("retained composition membership includes all 74 lexical sites", async () => {
    expect(expectedSites.length).toBe(74);
    const rows = classifyCompositions(await discoverCompositions(await tracked()), PROJECT_GATES);
    expect(rows.map((row) => row.path).sort()).toEqual(expectedSites);
});

for (const [fixture, source, tier] of relocated) {
    test(`relocated binding ${fixture} names its actual exercising command`, () => {
        // Classify fixed sites directly: discovery omission cannot mask a wrong-command failure.
        const sites = [source, "vite.config.ts"].map((path) => ({
            path: `packages/shallot/tests/${fixture}/${path}`,
            source: "",
        }));
        const command =
            fixture === "orbit-touch" ? `bunx playwright test -c ${tier}` : `bun test ./${tier}`;
        expect(classifyCompositions(sites, PROJECT_GATES)).toEqual(
            sites.map(({ path }) => ({ path, gate: command })),
        );
    });
    if (fixture === "orbit-touch") continue;
    test(`relocated consumer ${fixture} invokes its imported verify once`, async () => {
        const ast = parse(await readFile(join(root, tier), "utf8"), {
            sourceType: "module",
            plugins: ["typescript"],
        });
        const imports = ast.program.body.filter((node) => node.type === "ImportDeclaration");
        const verifyImports = imports.flatMap((node) =>
            node.specifiers
                .filter(
                    (specifier) =>
                        specifier.type === "ImportSpecifier" &&
                        specifier.imported.type === "Identifier" &&
                        specifier.imported.name === "verify",
                )
                .map((specifier) => ({ local: specifier.local.name, from: node.source.value })),
        );
        expect(verifyImports).toHaveLength(1);
        const binding = verifyImports[0];
        expect(resolve(root, dirname(tier), binding.from)).toBe(resolve(root, "scripts/verify"));
        const testImport = imports
            .find((node) => node.source.value === "bun:test")!
            .specifiers.find(
                (specifier) =>
                    specifier.type === "ImportSpecifier" &&
                    specifier.imported.type === "Identifier" &&
                    specifier.imported.name === "test",
            )!;
        const tests = ast.program.body.flatMap((node) =>
            node.type === "ExpressionStatement" &&
            node.expression.type === "CallExpression" &&
            node.expression.callee.type === "Identifier" &&
            node.expression.callee.name === testImport.local.name
                ? [node.expression]
                : [],
        );
        expect(tests).toHaveLength(1);
        const callback = tests[0].arguments[1];
        if (callback.type !== "ArrowFunctionExpression" || callback.body.type !== "BlockStatement")
            throw new Error("expected direct tier callback");
        expect(callback.params).toEqual([]);
        const declarations = callback.body.body.flatMap((node) =>
            node.type === "VariableDeclaration" ? node.declarations : [],
        );
        for (const declaration of declarations) {
            // Reject shadowing in the direct callback scope; nested pixel-loop bindings cannot
            // shadow the preceding top-level verify call.
            const names =
                declaration.id.type === "Identifier"
                    ? [declaration.id.name]
                    : declaration.id.type === "ObjectPattern"
                      ? declaration.id.properties.map((property) => {
                            if (
                                property.type !== "ObjectProperty" ||
                                property.value.type !== "Identifier"
                            )
                                throw new Error("unsupported tier binding");
                            return property.value.name;
                        })
                      : declaration.id.type === "ArrayPattern"
                        ? declaration.id.elements.map((element) => {
                              if (element?.type !== "Identifier")
                                  throw new Error("unsupported tier binding");
                              return element.name;
                          })
                        : [];
            expect(names.length).toBeGreaterThan(0);
            expect(names).not.toContain(binding.local);
        }
        expect(
            callback.body.body.filter(
                (node) => node.type === "FunctionDeclaration" || node.type === "ClassDeclaration",
            ),
        ).toEqual([]);
        const calls = declarations.flatMap((node) =>
            node.init?.type === "AwaitExpression" &&
            node.init.argument.type === "CallExpression" &&
            node.init.argument.callee.type === "Identifier" &&
            node.init.argument.callee.name === binding.local
                ? [node.init.argument]
                : [],
        );
        expect(calls).toHaveLength(1);
        expect(calls[0].arguments[0]).toMatchObject({
            type: "StringLiteral",
            value: `packages/shallot/tests/${fixture}`,
        });
    });
}

test("touch config binds its local test and fixture page", async () => {
    expect(touchConfig.testDir).toBe(".");
    expect(touchConfig.testMatch).toBe("*.playwright.ts");
    expect(touchConfig.webServer).toMatchObject({
        command: "bunx vite --port 3210 --strictPort",
        url: "http://localhost:3210",
    });
    expect(touchConfig.webServer).not.toHaveProperty("cwd");
    expect(touchConfig.use?.baseURL).toBe("http://localhost:3210");
    const fixture = "packages/shallot/tests/orbit-touch";
    const files = await tracked();
    expect(
        files.filter((path) => path.startsWith(`${fixture}/`) && path.endsWith(".playwright.ts")),
    ).toEqual([`${fixture}/touch.playwright.ts`]);
    const ast = parse(await readFile(join(root, fixture, "touch.playwright.ts"), "utf8"), {
        sourceType: "module",
        plugins: ["typescript"],
    });
    const imported = ast.program.body.find(
        (node) => node.type === "ImportDeclaration" && node.source.value === "@playwright/test",
    );
    if (imported?.type !== "ImportDeclaration") throw new Error("missing Playwright import");
    const binding = imported.specifiers.find(
        (node) =>
            node.type === "ImportSpecifier" &&
            node.imported.type === "Identifier" &&
            node.imported.name === "test",
    )!;
    const tests = ast.program.body.flatMap((node) =>
        node.type === "ExpressionStatement" &&
        node.expression.type === "CallExpression" &&
        node.expression.callee.type === "Identifier" &&
        node.expression.callee.name === binding.local.name
            ? [node.expression]
            : [],
    );
    expect(tests).toHaveLength(1);
    const callback = tests[0].arguments[1];
    if (callback.type !== "ArrowFunctionExpression" || callback.body.type !== "BlockStatement")
        throw new Error("missing touch callback");
    expect(callback.params).toHaveLength(1);
    expect(callback.params[0]).toMatchObject({
        type: "ObjectPattern",
        properties: [{ type: "ObjectProperty", key: { name: "page" }, value: { name: "page" } }],
    });
    const navigation = callback.body.body.flatMap((node) =>
        node.type === "ExpressionStatement" &&
        node.expression.type === "AwaitExpression" &&
        node.expression.argument.type === "CallExpression" &&
        node.expression.argument.callee.type === "MemberExpression" &&
        node.expression.argument.callee.object.type === "Identifier" &&
        node.expression.argument.callee.object.name === "page" &&
        node.expression.argument.callee.property.type === "Identifier" &&
        node.expression.argument.callee.property.name === "goto"
            ? [node.expression.argument]
            : [],
    );
    expect(navigation).toHaveLength(1);
    expect(navigation[0].arguments).toMatchObject([{ type: "StringLiteral", value: "/" }]);
    const html = await readFile(join(root, fixture, "index.html"), "utf8");
    expect(html.match(/<script\b[^>]*type="module"[^>]*>/g)).toEqual([
        '<script type="module" src="./src/main.ts">',
    ]);
});

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
        expect(PROJECT_GATES).toHaveLength(12);

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
