// The boundary reader's own arms. Each mutation below is one the moves this reader gates will actually
// make, so the reader is proved against the defect it claims rather than against a synthetic string.
//
// The tree arms build a whole miniature repo — root manifest, engine package with an `exports` map, a
// consumer workspace, a `bin/` tooling dir — because the reader's unit is the tree: a per-string helper
// cannot witness the workspace cone or the two-way seam ledger.

import { afterEach, describe, expect, test } from "bun:test";
import { mkdirSync, mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, resolve } from "node:path";
import {
    checkBoundary,
    type Ledger,
    publishedSurface,
    references,
    stripComments,
} from "./check-boundary";

// the fixture tree declares nothing: the real ledger names real repo files, and every clause below
// supplies its own entries when it needs one.
const EMPTY: Ledger = { toolingSeams: {}, computedLoaders: {}, nonWorkspacePackages: {} };

const trees: string[] = [];
afterEach(() => {
    for (const tree of trees.splice(0)) rmSync(tree, { recursive: true, force: true });
});

const write = (root: string, rel: string, body: string): void => {
    mkdirSync(dirname(resolve(root, rel)), { recursive: true });
    writeFileSync(resolve(root, rel), body);
};

/** A minimal but complete tree: the reader's clauses all read real files, so the fixture has to carry a
 *  workspace manifest, an export map, a consumer and a tooling dir. It must be green as built — a fixture
 *  that starts red cannot attribute any mutation applied to it. */
const make = (): string => {
    const root = mkdtempSync(resolve(tmpdir(), "shallot-check-boundary-"));
    trees.push(root);
    write(
        root,
        "package.json",
        JSON.stringify({ name: "repo", workspaces: ["packages/*", "examples/recipes/*"] }),
    );
    write(
        root,
        "packages/shallot/package.json",
        JSON.stringify({
            name: "@dylanebert/shallot",
            dependencies: { typegpu: "~0.12.4" },
            devDependencies: { vite: "^7.0.0" },
            exports: {
                ".": "./src/index.ts",
                "./render/core": "./src/standard/render/core.ts",
                "./src/*": "./src/*",
            },
        }),
    );
    write(root, "packages/shallot/src/index.ts", "export const engine = 1;\n");
    write(root, "packages/shallot/src/standard/render/core.ts", "export const core = 1;\n");
    write(root, "packages/shallot/src/project/generate.ts", "export const plan = 1;\n");
    write(root, "packages/shallot/tests/oracle.ts", "export const oracle = 1;\n");
    write(
        root,
        "packages/shallot/bin/cli.ts",
        'import { core } from "../src/standard/render/core";\nvoid core;\n',
    );
    write(
        root,
        "examples/recipes/demo/package.json",
        JSON.stringify({ name: "demo", dependencies: { "@dylanebert/shallot": "workspace:*" } }),
    );
    write(root, "examples/recipes/demo/src/main.ts", 'import "@dylanebert/shallot";\n');
    return root;
};

test("the fixture tree is green before any mutation", () => {
    const result = checkBoundary(make(), EMPTY);
    expect(result.violations).toEqual([]);
    expect(result.errors).toEqual([]);
    expect(result.consumers).toBe(1);
});

describe("consumer escapes", () => {
    test("a consumer reaching an unpublished subpath refuses", () => {
        const root = make();
        write(
            root,
            "examples/recipes/demo/src/main.ts",
            'import { plan } from "@dylanebert/shallot/src/project/generate";\nvoid plan;\n',
        );
        expect(checkBoundary(root, EMPTY).violations.map((v) => v.reason)).toContain(
            "reaches an unpublished @dylanebert/shallot internal (not in exports)",
        );
    });

    test("a consumer climbing out of its own project refuses", () => {
        const root = make();
        write(
            root,
            "examples/recipes/demo/src/main.ts",
            'import { plan } from "../../../../packages/shallot/src/project/generate";\nvoid plan;\n',
        );
        expect(checkBoundary(root, EMPTY).violations[0]?.reason).toContain("escapes the project");
    });

    // the pattern the previous reader could not see at all: `export … from` and `import("…")`.
    test("a re-export escape refuses", () => {
        const root = make();
        write(
            root,
            "examples/recipes/demo/src/main.ts",
            'export { plan } from "@dylanebert/shallot/src/project/generate";\n',
        );
        expect(checkBoundary(root, EMPTY).violations).toHaveLength(1);
    });

    test("a literal dynamic-import escape refuses", () => {
        const root = make();
        write(
            root,
            "examples/recipes/demo/src/main.ts",
            'export const load = () => import("@dylanebert/shallot/src/project/generate");\n',
        );
        expect(checkBoundary(root, EMPTY).violations).toHaveLength(1);
    });

    test("a declared published subpath stays allowed", () => {
        const root = make();
        write(
            root,
            "examples/recipes/demo/src/main.ts",
            'import { core } from "@dylanebert/shallot/render/core";\nvoid core;\n',
        );
        expect(checkBoundary(root, EMPTY).violations).toEqual([]);
    });

    // the narrow development-only allowance: the f64 CPU-oracle cross-check seam.
    test("the tests/ oracle seam stays allowed", () => {
        const root = make();
        write(
            root,
            "examples/recipes/demo/src/main.ts",
            'import { oracle } from "../../../../packages/shallot/tests/oracle";\nvoid oracle;\n',
        );
        expect(checkBoundary(root, EMPTY).violations).toEqual([]);
    });

    test("a non-TypeScript consumer file is governed too", () => {
        const root = make();
        write(
            root,
            "examples/recipes/demo/src/setup.mjs",
            'import "@dylanebert/shallot/src/project/generate";\n',
        );
        expect(checkBoundary(root, EMPTY).violations).toHaveLength(1);
    });

    test("a svelte consumer file is governed too", () => {
        const root = make();
        write(
            root,
            "examples/recipes/demo/src/App.svelte",
            '<script lang="ts">\n  import "@dylanebert/shallot/src/project/generate";\n</script>\n',
        );
        expect(checkBoundary(root, EMPTY).violations).toHaveLength(1);
    });
});

describe("tooling reaches", () => {
    test("a new private engine reach from bin/ refuses", () => {
        const root = make();
        write(
            root,
            "packages/shallot/bin/cli.ts",
            'import { plan } from "../src/project/generate";\nvoid plan;\n',
        );
        expect(checkBoundary(root, EMPTY).violations[0]?.reason).toContain("no declared seam");
    });

    test("a computed loader in bin/ with no declared bound refuses", () => {
        const root = make();
        write(root, "packages/shallot/bin/cli.ts", "export const load = (p) => import(p);\n");
        expect(checkBoundary(root, EMPTY).violations[0]?.reason).toContain(
            "builds a module specifier at runtime",
        );
    });

    test("a published engine subpath from bin/ stays allowed", () => {
        expect(checkBoundary(make(), EMPTY).violations).toEqual([]);
    });
});

describe("two-way completeness", () => {
    test("an undeclared workspace package refuses", () => {
        const root = make();
        write(root, "packages/tooling/package.json", JSON.stringify({ name: "tooling" }));
        const pkg = JSON.parse(
            require("node:fs").readFileSync(resolve(root, "package.json"), "utf8"),
        );
        pkg.workspaces = ["examples/recipes/*"];
        write(root, "package.json", JSON.stringify(pkg));
        expect(checkBoundary(root, EMPTY).errors.join("\n")).toContain(
            "neither a declared workspace nor a declared fixture",
        );
    });

    test("a declared workspace with no package.json refuses", () => {
        const root = make();
        rmSync(resolve(root, "examples/recipes/demo/package.json"));
        expect(checkBoundary(root, EMPTY).errors.join("\n")).toContain("has no package.json");
    });

    test("a local production dependency on the published package refuses", () => {
        const root = make();
        write(
            root,
            "packages/shallot/package.json",
            JSON.stringify({
                name: "@dylanebert/shallot",
                dependencies: { widget: "workspace:*" },
                exports: { ".": "./src/index.ts", "./src/*": "./src/*" },
            }),
        );
        expect(checkBoundary(root, EMPTY).errors.join("\n")).toContain(
            "local production dependency: widget@workspace:*",
        );
    });

    test("a file: production dependency refuses too", () => {
        const root = make();
        write(
            root,
            "packages/shallot/package.json",
            JSON.stringify({
                name: "@dylanebert/shallot",
                dependencies: { widget: "file:../widget" },
                exports: { ".": "./src/index.ts", "./src/*": "./src/*" },
            }),
        );
        expect(checkBoundary(root, EMPTY).errors.join("\n")).toContain(
            "local production dependency",
        );
    });
});

describe("published surface", () => {
    test("the ./src/* wildcard is not a published specifier or target", () => {
        const surface = publishedSurface({
            ".": "./src/index.ts",
            "./harness/browser": { types: "./src/harness/browser.ts", default: "./dist/x.js" },
            "./src/*": "./src/*",
        });
        expect(surface.exact.has("@dylanebert/shallot")).toBe(true);
        expect(surface.prefixes).toEqual([]);
        expect(surface.targets.has("src/index.ts")).toBe(true);
        // a conditional export publishes its `types` source, not the compiled projection
        expect(surface.targets.has("src/harness/browser.ts")).toBe(true);
        expect(surface.targets.has("dist/x.js")).toBe(false);
    });
});

describe("reference extraction", () => {
    const specs = (source: string) =>
        references(source)
            .filter((r) => !r.computed)
            .map((r) => r.spec);

    test("every named surface form is read", () => {
        const source = [
            'import a from "one";',
            'import type { B } from "two";',
            'import "three";',
            'export { c } from "four";',
            'export * from "five";',
            'const d = await import("six");',
            'const e = require("seven");',
        ].join("\n");
        expect(specs(source).sort()).toEqual([
            "five",
            "four",
            "one",
            "seven",
            "six",
            "three",
            "two",
        ]);
    });

    test("a multi-line import clause is still one specifier", () => {
        expect(specs('import {\n  a,\n  b,\n} from "one";\n')).toEqual(["one"]);
    });

    test("a computed call is reported without a specifier", () => {
        const found = references("const m = await import(path);\n");
        expect(found).toHaveLength(1);
        expect(found[0].computed).toBe(true);
        expect(found[0].spec).toBeNull();
    });

    test("a specifier quoted in a comment is not an import", () => {
        expect(specs('// import "ghost";\n/* import "phantom"; */\nimport "real";\n')).toEqual([
            "real",
        ]);
    });

    test("a url in a string literal survives comment stripping", () => {
        expect(stripComments('const u = "https://example.com/x";\n')).toContain("https://");
    });
});

describe("the seam ledger cannot outlive its subjects", () => {
    test("a tooling seam naming no live import refuses", () => {
        const root = make();
        const ledger: Ledger = {
            ...EMPTY,
            toolingSeams: { 'packages/shallot/bin/cli.ts "../src/project/generate"': "stale" },
        };
        expect(checkBoundary(root, ledger).errors.join("\n")).toContain(
            "declared tooling seam names no live import",
        );
    });

    test("a live tooling seam is accepted and not reported stale", () => {
        const root = make();
        write(
            root,
            "packages/shallot/bin/cli.ts",
            'import { plan } from "../src/project/generate";\nvoid plan;\n',
        );
        const ledger: Ledger = {
            ...EMPTY,
            toolingSeams: { 'packages/shallot/bin/cli.ts "../src/project/generate"': "declared" },
        };
        const result = checkBoundary(root, ledger);
        expect(result.violations).toEqual([]);
        expect(result.errors).toEqual([]);
    });

    test("a computed-loader declaration naming no live call site refuses", () => {
        const ledger: Ledger = {
            ...EMPTY,
            computedLoaders: { "packages/shallot/bin/cli.ts": "stale" },
        };
        expect(checkBoundary(make(), ledger).errors.join("\n")).toContain(
            "declared computed loader names no live call site",
        );
    });

    test("a non-workspace fixture declaration naming no directory refuses", () => {
        const ledger: Ledger = { ...EMPTY, nonWorkspacePackages: { "scripts/gone": "stale" } };
        expect(checkBoundary(make(), ledger).errors.join("\n")).toContain(
            "declared non-workspace package names no directory",
        );
    });

    test("a declared fixture directory is not reported ungoverned", () => {
        const root = make();
        write(root, "scripts/fixture/package.json", JSON.stringify({ name: "fixture" }));
        const ledger: Ledger = {
            ...EMPTY,
            nonWorkspacePackages: { "scripts/fixture": "packed install fixture" },
        };
        expect(checkBoundary(root, ledger).errors).toEqual([]);
    });
});

// The real repo's own ledger is a live claim, not a fixture: both directions must hold on this tree.
test("the repo's own boundary is clean under its real ledger", () => {
    const result = checkBoundary(resolve(import.meta.dir, ".."));
    expect(result.violations).toEqual([]);
    expect(result.errors).toEqual([]);
    expect(result.consumers).toBeGreaterThan(30);
});
