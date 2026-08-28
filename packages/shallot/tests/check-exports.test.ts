import { afterEach, describe, expect, test } from "bun:test";
import { mkdirSync, mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import {
    computeEntryFiles,
    computePublicSurface,
    extractDirectExports,
    extractImports,
    extractReExports,
    findDeadExports,
    isTestFile,
    resolveSpecifier,
    shouldFail,
    stripComments,
} from "../../../scripts/check-exports";

// Fixture trees live under the OS tmpdir, never the repo — `--root`-style isolation so a
// planted dead export never touches a tracked file. Each test gets its own dir; cleaned up after.

const roots: string[] = [];

function fixture(files: Record<string, string>): string {
    const dir = mkdtempSync(join(tmpdir(), "check-exports-"));
    roots.push(dir);
    for (const [rel, content] of Object.entries(files)) {
        const full = join(dir, rel);
        mkdirSync(join(full, ".."), { recursive: true });
        writeFileSync(full, content);
    }
    return dir;
}

afterEach(() => {
    while (roots.length > 0) {
        const dir = roots.pop();
        if (dir) rmSync(dir, { recursive: true, force: true });
    }
});

describe("isTestFile", () => {
    test("identifies test suffixes", () => {
        expect(isTestFile("foo.test.ts")).toBe(true);
        expect(isTestFile("foo.oracle.ts")).toBe(true);
        expect(isTestFile("foo.lab.ts")).toBe(true);
        expect(isTestFile("foo.probes.ts")).toBe(true);
        expect(isTestFile("foo.tier.ts")).toBe(true);
    });

    test("passes non-test files", () => {
        expect(isTestFile("foo.ts")).toBe(false);
        expect(isTestFile("foo.spec.ts")).toBe(false);
    });
});

describe("stripComments", () => {
    test("strips line comments", () => {
        const content = "const x = 1; // comment\nconst y = 2;";
        const stripped = stripComments(content);
        expect(stripped).toBe("const x = 1; \nconst y = 2;");
    });

    test("strips block comments including JSDoc", () => {
        const content =
            "/**\n * @example\n * export const config = 1;\n */\nexport const real = 2;";
        const stripped = stripComments(content);
        expect(stripped).toBe("\n\n\n\nexport const real = 2;");
    });

    test("preserves string literals containing // or /*", () => {
        const content = 'const url = "http://example.com";\nconst path = "a/*b";';
        const stripped = stripComments(content);
        expect(stripped).toBe('const url = "http://example.com";\nconst path = "a/*b";');
    });

    test("does not match export const inside a JSDoc @example block", () => {
        const content = `
/**
 * @example
 * \`\`\`
 * export const config: Config = { plugins: [] };
 * \`\`\`
 */
export const TumblePlugin: Plugin = { name: "Tumble" };
`;
        const stripped = stripComments(content);
        const exports = extractDirectExports(stripped);
        const names = exports.map((e) => e.name);
        expect(names).toContain("TumblePlugin");
        expect(names).not.toContain("config");
    });
});

// Regex literals: `/["']/` carries a quote that the old scanner read as a string opener, and the
// string then ran to end of file — every comment after it was copied through as code, so a
// `@example export const ghost` inside a JSDoc block became a real export. The engine twin
// (`maskTrivia`, `runtime/gpu-labels.test.ts`) was repaired first; these arms are that repair
// here. Mutation: delete the regex branch in `maskTrivia` → `ghost` reappears and this reds.
describe("stripComments — regex literals", () => {
    const source = [
        "const QUOTE = /[\"']/;",
        "/**",
        " * @example",
        " * export const ghost = 1;",
        " */",
        "export const real = 2;",
        "",
    ].join("\n");

    test("a quote inside a regex does not blind the scanner to the comments after it", () => {
        const names = extractDirectExports(stripComments(source)).map((e) => e.name);
        expect(names).toContain("real");
        expect(names).not.toContain("ghost");
    });

    test("division is not read as a regex", () => {
        const stripped = stripComments("const ratio = total / count; // note\nconst y = 2;");
        expect(stripped).toContain("total / count");
        expect(stripped).not.toContain("note");
    });

    test("a non-null assertion before a slash is division, not a regex start", () => {
        const stripped = stripComments("const r = a.get(k)! / b.get(k)!;\nconst y = 2;");
        expect(stripped).toContain("const y = 2;");
    });

    // The strict half: a scanner that runs an unterminated construct to end of file reports a
    // clean pass over code it never read, so the reader every walk goes through refuses instead.
    // Mutation: drop the `unparsed` check in `stripComments` → this arm reds.
    test("throws on a region it could not close", () => {
        expect(() => stripComments("const q = /unterminated\nconst y = 2;\n")).toThrow(
            /unterminated regex literal/,
        );
        expect(() => stripComments("/* unterminated")).toThrow(/unterminated block comment/);
    });
});

describe("extractDirectExports", () => {
    test("captures function, const, class, type, interface, enum", () => {
        const content = `
export function foo() {}
export const bar = 1;
export let baz = 2;
export class Qux {}
export type Foo = number;
export interface Bar {}
export enum Baz { A, B }
`;
        const exports = extractDirectExports(content);
        const names = exports.map((e) => e.name).sort();
        expect(names).toEqual(["Bar", "Baz", "Foo", "Qux", "bar", "baz", "foo"]);
    });

    test("captures export { name } without from (re-export of local symbols)", () => {
        const content = `
const internal = 1;
export { internal };
`;
        const exports = extractDirectExports(content);
        expect(exports.map((e) => e.name)).toEqual(["internal"]);
        expect(exports[0].kind).toBe("re-export");
    });

    test("does not capture export { name } from (that is a re-export, not a direct export)", () => {
        const content = `export { foo } from "./module";`;
        const exports = extractDirectExports(content);
        expect(exports).toHaveLength(0);
    });

    test("does not capture export * from", () => {
        const content = `export * from "./module";`;
        const exports = extractDirectExports(content);
        expect(exports).toHaveLength(0);
    });

    test("does not double-count a name captured by both a keyword export and export { }", () => {
        const content = `
export const foo = 1;
export { foo };
`;
        const exports = extractDirectExports(content);
        expect(exports.filter((e) => e.name === "foo")).toHaveLength(1);
    });

    test("handles export type { name } without from", () => {
        const content = `
type Foo = number;
export type { Foo };
`;
        const exports = extractDirectExports(content);
        expect(exports.map((e) => e.name)).toEqual(["Foo"]);
    });
});

describe("extractReExports", () => {
    test("captures export * from", () => {
        const reExports = extractReExports('export * from "./module";');
        expect(reExports).toHaveLength(1);
        expect(reExports[0].names).toBe("*");
        expect(reExports[0].source).toBe("./module");
    });

    test("captures export { name } from", () => {
        const reExports = extractReExports('export { foo, bar } from "./module";');
        expect(reExports).toHaveLength(1);
        expect(reExports[0].names).toEqual(["foo", "bar"]);
        expect(reExports[0].source).toBe("./module");
    });

    test("captures export type { name } from", () => {
        const reExports = extractReExports('export type { Foo } from "./module";');
        expect(reExports).toHaveLength(1);
        expect(reExports[0].names).toEqual(["Foo"]);
    });

    test("handles multiline export { } from", () => {
        const content = `export {
    foo,
    bar,
} from "./module";`;
        const reExports = extractReExports(content);
        expect(reExports).toHaveLength(1);
        expect(reExports[0].names).toEqual(["foo", "bar"]);
    });

    test("strips type modifier on individual names", () => {
        const reExports = extractReExports('export { foo, type Bar } from "./module";');
        expect(reExports[0].names).toEqual(["foo", "Bar"]);
    });
});

describe("extractImports", () => {
    test("captures named imports", () => {
        const imports = extractImports('import { foo, bar } from "./module";');
        expect(imports).toHaveLength(1);
        expect(imports[0].names).toEqual(["foo", "bar"]);
        expect(imports[0].specifier).toBe("./module");
    });

    test("captures type-only named imports", () => {
        const imports = extractImports('import type { Foo } from "./module";');
        expect(imports[0].names).toEqual(["Foo"]);
    });

    test("strips type modifier on individual names", () => {
        const imports = extractImports('import { foo, type Bar } from "./module";');
        expect(imports[0].names).toEqual(["foo", "Bar"]);
    });

    test("captures default import", () => {
        const imports = extractImports('import foo from "./module";');
        expect(imports[0].names).toEqual(["foo"]);
    });

    test("captures mixed default + named import", () => {
        const imports = extractImports('import foo, { bar } from "./module";');
        expect(imports[0].names).toEqual(["foo", "bar"]);
    });

    test("captures namespace import with namespace name", () => {
        const imports = extractImports('import * as ns from "./module";');
        expect(imports).toHaveLength(1);
        expect(imports[0].names).toEqual(["*"]);
        expect(imports[0].namespace).toBe("ns");
    });

    test("strips alias from imported name (source name before `as`)", () => {
        const imports = extractImports('import { foo as bar } from "./module";');
        expect(imports[0].names).toEqual(["foo"]);
    });
});

describe("resolveSpecifier", () => {
    test("resolves a relative import to a .ts file", () => {
        const root = fixture({
            "packages/shallot/src/module.ts": "export const foo = 1;",
            "packages/shallot/src/consumer.ts": 'import { foo } from "./module";',
            "packages/shallot/package.json": JSON.stringify({ exports: {} }),
        });
        const resolved = resolveSpecifier("packages/shallot/src/consumer.ts", "./module", root, {});
        expect(resolved).toBe("packages/shallot/src/module.ts");
    });

    test("resolves a relative import to an index.ts", () => {
        const root = fixture({
            "packages/shallot/src/mod/index.ts": "export const foo = 1;",
            "packages/shallot/src/consumer.ts": 'import { foo } from "./mod";',
            "packages/shallot/package.json": JSON.stringify({ exports: {} }),
        });
        const resolved = resolveSpecifier("packages/shallot/src/consumer.ts", "./mod", root, {});
        expect(resolved).toBe("packages/shallot/src/mod/index.ts");
    });

    test("resolves a package import through the exports map", () => {
        const root = fixture({
            "packages/shallot/src/index.ts": "export const foo = 1;",
            "packages/shallot/package.json": JSON.stringify({
                exports: { ".": "./src/index.ts" },
            }),
        });
        const resolved = resolveSpecifier("examples/app/src/app.ts", "@dylanebert/shallot", root, {
            ".": "./src/index.ts",
        });
        expect(resolved).toBe("packages/shallot/src/index.ts");
    });

    test("resolves a package subpath through the exports map", () => {
        const root = fixture({
            "packages/shallot/src/extras/index.ts": "export const foo = 1;",
            "packages/shallot/package.json": JSON.stringify({
                exports: { "./extras": "./src/extras/index.ts" },
            }),
        });
        const resolved = resolveSpecifier(
            "examples/app/src/app.ts",
            "@dylanebert/shallot/extras",
            root,
            { "./extras": "./src/extras/index.ts" },
        );
        expect(resolved).toBe("packages/shallot/src/extras/index.ts");
    });

    test("resolves a wildcard ./src/* export", () => {
        const root = fixture({
            "packages/shallot/src/engine/ecs/reflection.ts": "export const foo = 1;",
            "packages/shallot/package.json": JSON.stringify({
                exports: { "./src/*": "./src/*" },
            }),
        });
        const resolved = resolveSpecifier(
            "packages/shallot/tests/test.test.ts",
            "@dylanebert/shallot/src/engine/ecs/reflection",
            root,
            { "./src/*": "./src/*" },
        );
        expect(resolved).toBe("packages/shallot/src/engine/ecs/reflection.ts");
    });

    test("returns null for non-package, non-relative specifiers", () => {
        const root = fixture({
            "packages/shallot/package.json": JSON.stringify({ exports: {} }),
        });
        const resolved = resolveSpecifier("packages/shallot/src/consumer.ts", "typegpu", root, {});
        expect(resolved).toBeNull();
    });
});

describe("computeEntryFiles", () => {
    test("resolves named entry points to file paths", () => {
        const root = fixture({
            "packages/shallot/src/index.ts": "export const foo = 1;",
            "packages/shallot/src/extras/index.ts": "export const bar = 2;",
            "packages/shallot/package.json": JSON.stringify({
                exports: {
                    ".": "./src/index.ts",
                    "./extras": "./src/extras/index.ts",
                    "./src/*": "./src/*",
                },
            }),
        });
        const entries = computeEntryFiles(root, {
            ".": "./src/index.ts",
            "./extras": "./src/extras/index.ts",
            "./src/*": "./src/*",
        });
        expect(entries.sort()).toEqual(
            ["packages/shallot/src/index.ts", "packages/shallot/src/extras/index.ts"].sort(),
        );
    });

    test("ignores the ./src/* wildcard escape hatch", () => {
        const root = fixture({
            "packages/shallot/src/index.ts": "export const foo = 1;",
            "packages/shallot/package.json": JSON.stringify({
                exports: { ".": "./src/index.ts", "./src/*": "./src/*" },
            }),
        });
        const entries = computeEntryFiles(root, {
            ".": "./src/index.ts",
            "./src/*": "./src/*",
        });
        expect(entries).toEqual(["packages/shallot/src/index.ts"]);
    });
});

describe("computePublicSurface", () => {
    test("star re-export makes all direct exports of the source public", () => {
        const directExports = new Map([
            ["packages/shallot/src/index.ts", new Set(["reExportedFn"])],
            ["packages/shallot/src/module.ts", new Set(["reExportedFn", "notReExported"])],
        ]);
        const reExports = new Map<string, { names: string[] | "*"; sourceFile: string }[]>([
            [
                "packages/shallot/src/index.ts",
                [{ names: "*" as const, sourceFile: "packages/shallot/src/module.ts" }],
            ],
        ]);
        const surface = computePublicSurface(
            ["packages/shallot/src/index.ts"],
            directExports,
            reExports,
        );
        // star re-export: all direct exports of module.ts are public
        expect(surface.has("packages/shallot/src/module.ts::reExportedFn")).toBe(true);
        expect(surface.has("packages/shallot/src/module.ts::notReExported")).toBe(true);
        // the barrel's own direct export is also public
        expect(surface.has("packages/shallot/src/index.ts::reExportedFn")).toBe(true);
    });

    test("named re-export makes only the named symbol public, not siblings", () => {
        const directExports = new Map([
            ["packages/shallot/src/index.ts", new Set<string>()],
            ["packages/shallot/src/module.ts", new Set(["foo", "bar"])],
        ]);
        const reExports = new Map([
            [
                "packages/shallot/src/index.ts",
                [{ names: ["foo"], sourceFile: "packages/shallot/src/module.ts" }],
            ],
        ]);
        const surface = computePublicSurface(
            ["packages/shallot/src/index.ts"],
            directExports,
            reExports,
        );
        expect(surface.has("packages/shallot/src/module.ts::foo")).toBe(true);
        expect(surface.has("packages/shallot/src/module.ts::bar")).toBe(false);
    });
});

describe("findDeadExports — red-first proof", () => {
    test("flags a planted zero-consumer export and does not flag a live one", async () => {
        const root = fixture({
            "packages/shallot/package.json": JSON.stringify({
                exports: { ".": "./src/index.ts", "./src/*": "./src/*" },
            }),
            "packages/shallot/src/module.ts": `
export function deadFn(): number { return 0; }
export function liveFn(): number { return 1; }
export function inFileFn(): number { return inFileFn(); }
`,
            "packages/shallot/src/consumer.ts": `
import { liveFn } from "./module";
liveFn();
`,
            // index re-exports liveFn (public surface) but NOT deadFn or inFileFn
            "packages/shallot/src/index.ts": `export { liveFn } from "./module";`,
        });

        const dead = await findDeadExports(root);
        const names = dead.map((d) => d.name);

        // deadFn has zero consumers and is not referenced in-file → zero-consumer
        expect(names).toContain("deadFn");
        const deadFn = dead.find((d) => d.name === "deadFn")!;
        expect(deadFn.category).toBe("zero-consumer");

        // liveFn is consumed by consumer.ts → not flagged
        expect(names).not.toContain("liveFn");

        // inFileFn is referenced in-file but not imported externally → in-file-only
        expect(names).toContain("inFileFn");
        const inFile = dead.find((d) => d.name === "inFileFn")!;
        expect(inFile.category).toBe("in-file-only");
    });

    test("flags a test-only export (consumed only by a .test.ts file)", async () => {
        const root = fixture({
            "packages/shallot/package.json": JSON.stringify({
                exports: { ".": "./src/index.ts", "./src/*": "./src/*" },
            }),
            "packages/shallot/src/module.ts": `
export function testOnlyFn(): number { return 0; }
export function prodFn(): number { return 1; }
`,
            "packages/shallot/src/consumer.ts": `
import { prodFn } from "./module";
prodFn();
`,
            "packages/shallot/tests/test.test.ts": `
import { testOnlyFn } from "../src/module";
testOnlyFn();
`,
            // index re-exports prodFn (public) but NOT testOnlyFn
            "packages/shallot/src/index.ts": `export { prodFn } from "./module";`,
        });

        const dead = await findDeadExports(root);
        const testOnly = dead.find((d) => d.name === "testOnlyFn");
        expect(testOnly).toBeDefined();
        expect(testOnly!.category).toBe("test-only");
        expect(testOnly!.testConsumers).toHaveLength(1);
        expect(testOnly!.testConsumers[0].file).toContain("test.test.ts");

        // prodFn is consumed by a production file → not flagged
        expect(dead.find((d) => d.name === "prodFn")).toBeUndefined();
    });

    test("follows barrel re-export chains: a symbol consumed through a barrel is not flagged", async () => {
        const root = fixture({
            "packages/shallot/package.json": JSON.stringify({
                exports: { ".": "./src/index.ts", "./src/*": "./src/*" },
            }),
            "packages/shallot/src/module.ts": `
export function barrelFn(): number { return 0; }
`,
            // barrelFn is on the public surface (star re-exported from entry point) AND consumed
            "packages/shallot/src/index.ts": `export * from "./module";`,
            "packages/shallot/src/consumer.ts": `
import { barrelFn } from "./index";
barrelFn();
`,
        });

        const dead = await findDeadExports(root);
        expect(dead.find((d) => d.name === "barrelFn")).toBeUndefined();
    });

    test("namespace import only consumes exports actually accessed through the namespace", async () => {
        const root = fixture({
            "packages/shallot/package.json": JSON.stringify({
                exports: { ".": "./src/index.ts", "./src/*": "./src/*" },
            }),
            "packages/shallot/src/module.ts": `
export function usedNs(): number { return 0; }
export function unusedNs(): number { return 0; }
`,
            "packages/shallot/src/consumer.ts": `
import * as ns from "./module";
ns.usedNs();
`,
            // index re-exports usedNs (public) but NOT unusedNs
            "packages/shallot/src/index.ts": `export { usedNs } from "./module";`,
        });

        const dead = await findDeadExports(root);
        // usedNs is accessed through ns.usedNs() → not flagged
        expect(dead.find((d) => d.name === "usedNs")).toBeUndefined();
        // unusedNs is NOT accessed through the namespace → flagged as zero-consumer
        const unused = dead.find((d) => d.name === "unusedNs");
        expect(unused).toBeDefined();
        expect(unused!.category).toBe("zero-consumer");
    });

    test("allowlist suppresses a flagged export", async () => {
        const root = fixture({
            "packages/shallot/package.json": JSON.stringify({
                exports: { ".": "./src/index.ts", "./src/*": "./src/*" },
            }),
            "packages/shallot/src/module.ts": `
export function allowedDead(): number { return 0; }
export function notAllowed(): number { return 0; }
`,
            // neither is re-exported from the entry point
            "packages/shallot/src/index.ts": `export const other = 1;`,
        });

        const dead = await findDeadExports(root, [
            { file: "packages/shallot/src/module.ts", name: "allowedDead" },
        ]);
        expect(dead.find((d) => d.name === "allowedDead")).toBeUndefined();
        expect(dead.find((d) => d.name === "notAllowed")).toBeDefined();
    });

    test("a clean fixture with no dead exports reports empty", async () => {
        const root = fixture({
            "packages/shallot/package.json": JSON.stringify({
                exports: { ".": "./src/index.ts", "./src/*": "./src/*" },
            }),
            "packages/shallot/src/module.ts": `
export function liveFn(): number { return 1; }
`,
            "packages/shallot/src/consumer.ts": `
import { liveFn } from "./module";
liveFn();
`,
            // liveFn is both consumed AND on the public surface
            "packages/shallot/src/index.ts": `export * from "./module";`,
        });

        const dead = await findDeadExports(root);
        expect(dead).toHaveLength(0);
    });

    // --- New semantics: public surface exclusion ---

    test("a symbol with zero in-repo consumers but re-exported from a declared entry point is NOT flagged", async () => {
        const root = fixture({
            "packages/shallot/package.json": JSON.stringify({
                exports: { ".": "./src/index.ts", "./src/*": "./src/*" },
            }),
            "packages/shallot/src/module.ts": `
export function publicFn(): number { return 0; }
export function deadFn(): number { return 0; }
`,
            // index.ts re-exports publicFn but NOT deadFn
            "packages/shallot/src/index.ts": `export { publicFn } from "./module";`,
        });

        const dead = await findDeadExports(root);
        const names = dead.map((d) => d.name);

        // publicFn is on the public surface (re-exported from the `.` entry point) → NOT flagged
        expect(names).not.toContain("publicFn");

        // deadFn is NOT on the public surface → flagged as zero-consumer
        expect(names).toContain("deadFn");
        expect(dead.find((d) => d.name === "deadFn")!.category).toBe("zero-consumer");
    });

    test("a symbol re-exported through a star barrel chain from an entry point is NOT flagged", async () => {
        const root = fixture({
            "packages/shallot/package.json": JSON.stringify({
                exports: {
                    ".": "./src/index.ts",
                    "./extras": "./src/extras/index.ts",
                    "./src/*": "./src/*",
                },
            }),
            "packages/shallot/src/mod.ts": `
export function deepFn(): number { return 0; }
`,
            "packages/shallot/src/extras/index.ts": `export * from "../mod";`,
            "packages/shallot/src/index.ts": `export * from "./mod";`,
        });

        const dead = await findDeadExports(root);
        // deepFn is reachable from both `.` and `./extras` entry points via star re-exports → NOT flagged
        expect(dead.find((d) => d.name === "deepFn")).toBeUndefined();
    });

    test("the ./src/* wildcard is ignored: a symbol only reachable through it IS flagged", async () => {
        const root = fixture({
            "packages/shallot/package.json": JSON.stringify({
                exports: { ".": "./src/index.ts", "./src/*": "./src/*" },
            }),
            "packages/shallot/src/module.ts": `
export function wildcardOnly(): number { return 0; }
`,
            // index.ts does NOT re-export from module — module is only reachable via ./src/*
            "packages/shallot/src/index.ts": `export const other = 1;`,
        });

        const dead = await findDeadExports(root);
        // wildcardOnly is only reachable through ./src/* which is ignored → flagged
        const found = dead.find((d) => d.name === "wildcardOnly");
        expect(found).toBeDefined();
        expect(found!.category).toBe("zero-consumer");
    });

    test("JSDoc @example reference is not counted as a consumer or export", async () => {
        const root = fixture({
            "packages/shallot/package.json": JSON.stringify({
                exports: { ".": "./src/index.ts", "./src/*": "./src/*" },
            }),
            "packages/shallot/src/module.ts": `
/**
 * @example
 * \`\`\`
 * export const config = { foo: 1 };
 * \`\`\`
 */
export function realFn(): number { return 0; }
export function deadFn(): number { return 0; }
`,
            "packages/shallot/src/consumer.ts": `
// realFn is used here, but deadFn is only mentioned in a comment
import { realFn } from "./module";
realFn();
// deadFn is mentioned here but not imported
`,
            // index re-exports realFn (public) but NOT deadFn
            "packages/shallot/src/index.ts": `export { realFn } from "./module";`,
        });

        const dead = await findDeadExports(root);
        const names = dead.map((d) => d.name);

        // realFn is consumed → not flagged
        expect(names).not.toContain("realFn");

        // deadFn is not consumed (comment mention doesn't count) → flagged
        expect(names).toContain("deadFn");

        // `config` from the JSDoc @example is not a real export → not in the output at all
        expect(names).not.toContain("config");
    });
});

describe("exit-condition split — only zero-consumer is fatal", () => {
    test("shouldFail returns true when zero-consumer is non-empty", () => {
        const dead = [
            {
                file: "a.ts",
                name: "dead",
                line: 1,
                category: "zero-consumer" as const,
                testConsumers: [],
            },
        ];
        expect(shouldFail(dead)).toBe(true);
    });

    test("shouldFail returns false when only advisory buckets are non-empty", () => {
        const dead = [
            {
                file: "a.ts",
                name: "inFile",
                line: 1,
                category: "in-file-only" as const,
                testConsumers: [],
            },
            {
                file: "b.ts",
                name: "testOnly",
                line: 2,
                category: "test-only" as const,
                testConsumers: [{ file: "t.test.ts", line: 3 }],
            },
        ];
        expect(shouldFail(dead)).toBe(false);
    });

    test("shouldFail returns false on an empty list", () => {
        expect(shouldFail([])).toBe(false);
    });

    test("fixture: advisory buckets non-empty, zero-consumer empty → shouldFail is false (passing case)", async () => {
        const root = fixture({
            "packages/shallot/package.json": JSON.stringify({
                exports: { ".": "./src/index.ts", "./src/*": "./src/*" },
            }),
            "packages/shallot/src/module.ts": `
export function inFileFn(): number { return inFileFn(); }
export function testOnlyFn(): number { return 0; }
`,
            "packages/shallot/tests/test.test.ts": `
import { testOnlyFn } from "../src/module";
testOnlyFn();
`,
            // index re-exports neither — both are advisory, zero-consumer is empty
            "packages/shallot/src/index.ts": `export const other = 1;`,
        });

        const dead = await findDeadExports(root);
        // both are flagged but neither is zero-consumer
        const zeroConsumer = dead.filter((d) => d.category === "zero-consumer");
        const inFileOnly = dead.filter((d) => d.category === "in-file-only");
        const testOnly = dead.filter((d) => d.category === "test-only");
        expect(zeroConsumer).toHaveLength(0);
        expect(inFileOnly.length).toBeGreaterThan(0);
        expect(testOnly.length).toBeGreaterThan(0);
        // the exit decision: advisory-only → pass
        expect(shouldFail(dead)).toBe(false);
    });

    test("fixture: non-empty zero-consumer → shouldFail is true (failing case)", async () => {
        const root = fixture({
            "packages/shallot/package.json": JSON.stringify({
                exports: { ".": "./src/index.ts", "./src/*": "./src/*" },
            }),
            "packages/shallot/src/module.ts": `
export function deadFn(): number { return 0; }
export function inFileFn(): number { return inFileFn(); }
`,
            // index re-exports neither
            "packages/shallot/src/index.ts": `export const other = 1;`,
        });

        const dead = await findDeadExports(root);
        const zeroConsumer = dead.filter((d) => d.category === "zero-consumer");
        expect(zeroConsumer.length).toBeGreaterThan(0);
        // the exit decision: zero-consumer present → fail
        expect(shouldFail(dead)).toBe(true);
    });
});
