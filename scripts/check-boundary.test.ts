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

describe("runtime direction with a private tooling owner", () => {
    test("tooling cannot reach a sibling's private source", () => {
        const root = make();
        write(
            root,
            "packages/shallot-tooling/package.json",
            JSON.stringify({ name: "shallot-tooling", private: true }),
        );
        write(
            root,
            "packages/shallot-tooling/src/escape.ts",
            'import "../../../examples/recipes/demo/src/main";',
        );
        const result = checkBoundary(root, EMPTY);
        expect(result.errors).toEqual([]);
        expect(result.violations).toHaveLength(1);
        expect(result.violations[0].reason).toBe(
            "escapes the private tooling owner without a declared seam",
        );
    });
    for (const source of [
        'import "../../shallot-tooling/src/harness/browser";',
        'import "@dylanebert/shallot/harness";',
        'export * from "./harness";',
        'void import("@dylanebert/shallot/harness/browser");',
        'import "@launch";',
    ]) {
        test(source, () => {
            const root = make();
            write(
                root,
                "packages/shallot-tooling/package.json",
                JSON.stringify({ name: "shallot-tooling", private: true }),
            );
            write(
                root,
                "packages/shallot-tooling/src/harness/browser.ts",
                "export const launch = 1;\n",
            );
            write(
                root,
                "packages/shallot/package.json",
                JSON.stringify({
                    name: "@dylanebert/shallot",
                    exports: {
                        ".": "./src/index.ts",
                        "./harness": "./src/harness/index.ts",
                        "./harness/browser": {
                            types: "./src/harness/browser.ts",
                            default: "./dist/harness-browser.js",
                        },
                    },
                }),
            );
            write(
                root,
                "packages/shallot/src/harness/index.ts",
                'export * from "./runtime"; export { launch } from "@dylanebert/shallot/harness/browser";',
            );
            write(
                root,
                "packages/shallot/src/harness/runtime.ts",
                'import "../standard/render/core";',
            );
            write(
                root,
                "tsconfig.json",
                JSON.stringify({
                    compilerOptions: {
                        paths: { "@launch": ["packages/shallot-tooling/src/harness/browser.ts"] },
                    },
                }),
            );
            expect(checkBoundary(root, EMPTY).violations).toEqual([]);
            write(root, "packages/shallot/src/consumer.ts", source);
            const result = checkBoundary(root, EMPTY);
            expect(result.errors).toEqual([]);
            expect(result.violations).toHaveLength(1);
            expect(result.violations[0].reason).toBe(
                "runtime reaches tooling or the distribution's composite harness",
            );
        });
    }
});

describe("canonical runtime ownership", () => {
    for (const [source, allowed] of [
        ['import "@dylanebert/shallot";', true],
        ['export * from "@dylanebert/shallot/render/core";', true],
        ['import "./standard/render/core";', true],
        ['import "../../../examples/recipes/demo/src/main";', false],
        ['void import("@outside");', false],
    ] as const) {
        test(source, () => {
            const root = make();
            for (const owner of ["shallot-runtime", "shallot-tooling"])
                write(
                    root,
                    `packages/${owner}/package.json`,
                    JSON.stringify({ name: owner, private: true }),
                );
            write(root, "packages/shallot-runtime/src/index.ts", "export const engine = 1;");
            write(
                root,
                "packages/shallot-runtime/src/standard/render/core.ts",
                "export const core = 1;",
            );
            write(
                root,
                "tsconfig.json",
                JSON.stringify({
                    compilerOptions: {
                        paths: { "@outside": ["examples/recipes/demo/src/main.ts"] },
                    },
                }),
            );
            expect(checkBoundary(root, EMPTY).violations).toEqual([]);
            write(root, "packages/shallot-runtime/src/consumer.ts", source);
            const result = checkBoundary(root, EMPTY);
            expect(result.errors).toEqual([]);
            expect(result.violations.map((v) => v.reason)).toEqual(
                allowed
                    ? []
                    : ["runtime leaves its canonical owner without a declared runtime export"],
            );
        });
    }
});

describe("runtime unresolved and aliased exits", () => {
    for (const [source, alias] of [
        ['import "@dylanebert/shallot/not-exported";', false],
        ['import "shallot-gpu-particles";', false],
        ["void import(path);", false],
        ['import "@dylanebert/shallot";', true],
    ] as const) {
        test(`${source} alias=${alias}`, () => {
            const root = make();
            for (const owner of ["shallot-runtime", "shallot-tooling", "shallot-gpu-particles"])
                write(
                    root,
                    `packages/${owner}/package.json`,
                    JSON.stringify({ name: owner, private: true }),
                );
            write(root, "packages/shallot-runtime/src/index.ts", "export const engine = 1;");
            if (alias)
                write(
                    root,
                    "tsconfig.json",
                    JSON.stringify({
                        compilerOptions: {
                            paths: {
                                "@dylanebert/shallot": ["examples/recipes/demo/src/main.ts"],
                            },
                        },
                    }),
                );
            expect(checkBoundary(root, EMPTY).violations).toEqual([]);
            write(root, "packages/shallot-runtime/src/consumer.ts", source);
            const result = checkBoundary(root, EMPTY);
            expect(result.errors).toEqual([]);
            expect(
                result.violations.filter((v) => v.file.startsWith("packages/shallot-runtime/")),
            ).toHaveLength(1);
        });
    }
    test("runtime computed disposition is live in both directions", () => {
        const root = make();
        for (const owner of ["shallot-runtime", "shallot-tooling"])
            write(
                root,
                `packages/${owner}/package.json`,
                JSON.stringify({ name: owner, private: true }),
            );
        const file = "packages/shallot-runtime/src/pool.ts";
        write(root, file, 'const spec = "node:worker_threads"; void import(spec);');
        const ledger = { ...EMPTY, computedLoaders: { [file]: "bounded Node host adapter" } };
        expect(checkBoundary(root, ledger).violations).toEqual([]);
        expect(checkBoundary(root, ledger).errors).toEqual([]);
        write(root, file, 'import "node:worker_threads";');
        expect(checkBoundary(root, ledger).errors.join("\n")).toContain(
            "declared computed loader names no live call site",
        );
    });
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

describe("reader gap controls", () => {
    for (const alias of [
        '[{ find: "@private", replacement: "/private" }]',
        '{ "@private": target }',
        '{ [name]: "/private" }',
        '[{ find: /private/, replacement: "/private" }]',
    ]) {
        test(`unresolvable Vite alias refuses: ${alias}`, () => {
            const root = make();
            write(
                root,
                "examples/recipes/demo/vite.config.ts",
                `export default { resolve: { alias: ${alias} } };`,
            );
            expect(checkBoundary(root, EMPTY).errors.join("\n")).toContain("vite.config.ts: alias");
        });
    }
    test("ancestor tsconfig and project-local aliases stay allowed", () => {
        const root = make();
        write(root, "tsconfig.json", "{}");
        write(
            root,
            "examples/recipes/demo/tsconfig.json",
            JSON.stringify({
                extends: "../../../tsconfig.json",
                compilerOptions: { paths: { "@local": ["src/main.ts"] } },
            }),
        );
        write(
            root,
            "examples/recipes/demo/vite.config.ts",
            'export default { resolve: { alias: { "@vite": "./src/main.ts" } } };',
        );
        write(root, "examples/recipes/demo/src/main.ts", 'import "@local"; import "@vite";');
        const result = checkBoundary(root, EMPTY);
        expect(result.errors).toEqual([]);
        expect(result.violations).toEqual([]);
    });
    test("blank tooling dispositions refuse", () => {
        const root = make();
        const file = "packages/shallot/bin/tui.ts";
        write(root, file, 'void import(path); import "../src/project/generate";');
        const result = checkBoundary(root, {
            ...EMPTY,
            computedLoaders: { [file]: " " },
            toolingSeams: { [`${file} "../src/project/generate"`]: " " },
        });
        expect(result.violations).toHaveLength(2);
    });
    test("off-chain tsconfig extends refuses", () => {
        const root = make();
        write(root, "tsconfig.json", JSON.stringify({ extends: "./other.json" }));
        expect(checkBoundary(root, EMPTY).errors.join("\n")).toContain("tsconfig.json: extends");
    });
    test("non-array tsconfig paths refuses explicitly", () => {
        const root = make();
        write(
            root,
            "tsconfig.json",
            JSON.stringify({ compilerOptions: { paths: { "@private": "private" } } }),
        );
        expect(checkBoundary(root, EMPTY).errors.join("\n")).toContain(
            "tsconfig.json: paths @private",
        );
    });
    test("all tooling alias targets are governed", () => {
        const root = make();
        write(
            root,
            "tsconfig.json",
            JSON.stringify({
                compilerOptions: {
                    paths: {
                        "@private": [
                            "packages/shallot/src/standard/render/core.ts",
                            "packages/shallot/src/project/generate.ts",
                        ],
                    },
                },
            }),
        );
        write(root, "packages/shallot/bin/tui.ts", 'import "@private";');
        expect(checkBoundary(root, EMPTY).violations[0]?.reason).toContain("no declared seam");
    });
    test("consumer package imports refuses even without use", () => {
        const root = make();
        write(
            root,
            "examples/recipes/demo/package.json",
            JSON.stringify({ name: "demo", imports: { "#private": "./src/main.ts" } }),
        );
        expect(checkBoundary(root, EMPTY).errors.join("\n")).toContain("package.json: imports");
    });
    test("real root wildcard paths cannot launder private source", () => {
        const root = make();
        write(
            root,
            "tsconfig.json",
            JSON.stringify({
                compilerOptions: {
                    paths: { "@dylanebert/shallot/src/*": ["packages/shallot/src/*"] },
                },
            }),
        );
        write(
            root,
            "examples/recipes/demo/src/main.ts",
            'import "@dylanebert/shallot/src/project/generate";',
        );
        expect(checkBoundary(root, EMPTY).violations[0]?.reason).toContain("escapes the project");
    });
    test("consumer concatenated imports require a bounded disposition", () => {
        const root = make();
        const file = "examples/recipes/demo/src/main.ts";
        write(root, file, 'void import("../../../packages/shallot/" + "src/project/generate");');
        expect(checkBoundary(root, EMPTY).violations[0]?.reason).toContain("no declared bound");
        const result = checkBoundary(root, {
            ...EMPTY,
            computedLoaders: { [file]: "fixture bounded loader" },
        });
        expect(result.violations).toEqual([]);
        expect(result.errors).toEqual([]);
    });

    test("consumer computed imports require a live bounded disposition", () => {
        const root = make();
        const file = "examples/recipes/demo/src/main.ts";
        write(
            root,
            file,
            'const path = "@dylanebert/shallot/src/project/generate"; void import(path);',
        );
        expect(checkBoundary(root, EMPTY).violations[0]?.reason).toContain("no declared bound");
        write(root, file, 'const path = "./plugin.ts"; void import(path);');
        const ledger = {
            ...EMPTY,
            computedLoaders: { [file]: "project-local plugin selected by manifest" },
        };
        expect(checkBoundary(root, ledger).violations).toEqual([]);
        expect(checkBoundary(root, ledger).errors).toEqual([]);
    });

    for (const config of ["tsconfig.json", "examples/recipes/demo/vite.config.ts"]) {
        test(`resolved private alias refuses: ${config}`, () => {
            const root = make();
            write(
                root,
                config,
                config.endsWith("json")
                    ? JSON.stringify({
                          compilerOptions: {
                              baseUrl: ".",
                              paths: { "@private": ["packages/shallot/src/project/generate.ts"] },
                          },
                      })
                    : `export default { resolve: { alias: { "@private": ${JSON.stringify(resolve(root, "packages/shallot/src/project/generate.ts"))} } } };`,
            );
            write(root, "examples/recipes/demo/src/main.ts", 'export { plan } from "@private";');
            expect(checkBoundary(root, EMPTY).violations[0]?.reason).toContain(
                "escapes the project",
            );
        });
    }

    for (const source of [
        'import "@dylanebert/shallot/src/project/generate";',
        'export { plan } from "@dylanebert/shallot/src/project/generate";',
        'void import("@dylanebert/shallot/src/project/generate");',
    ]) {
        test(`TUI alternate private spelling refuses: ${source}`, () => {
            const root = make();
            write(root, "packages/shallot/bin/tui.ts", source);
            expect(checkBoundary(root, EMPTY).violations[0]?.reason).toContain("no declared seam");
        });
    }
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

    for (const source of [
        'void import("/src/" + "edit.ts");',
        'require("/src/" + name)',
        "import(`./${name}`)",
        'import("./x", { with: { type: "json" } })',
    ]) {
        test(`non-literal argument is computed: ${source}`, () => {
            expect(references(source)).toEqual([{ spec: null, line: 1, computed: true }]);
        });
    }
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
