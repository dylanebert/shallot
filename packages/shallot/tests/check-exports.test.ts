import { afterEach, describe, expect, test } from "bun:test";
import { mkdirSync, mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import {
    extractDirectExports,
    extractImports,
    extractReExports,
    findDeadExports,
    isTestFile,
    resolveSpecifier,
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

    test("does not match export const inside a JSDoc @example block", () => {
        // A known limitation of regex-based parsing: `export const config` inside a JSDoc
        // @example code block is matched as a real export. This test documents the limitation
        // so a future fix (AST-based parsing) can flip it to false.
        const content = `
/**
 * @example
 * \`\`\`
 * export const config: Config = { plugins: [] };
 * \`\`\`
 */
export const TumblePlugin: Plugin = { name: "Tumble" };
`;
        const exports = extractDirectExports(content);
        const names = exports.map((e) => e.name);
        expect(names).toContain("TumblePlugin");
        // limitation: `config` is falsely captured from the JSDoc example
        expect(names).toContain("config");
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
            "packages/shallot/src/index.ts": `export * from "./module";`,
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
            "packages/shallot/src/index.ts": `export * from "./module";`,
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
            "packages/shallot/src/index.ts": `export * from "./module";`,
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
            "packages/shallot/src/index.ts": `export * from "./module";`,
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
            "packages/shallot/src/index.ts": `export * from "./module";`,
        });

        const dead = await findDeadExports(root);
        expect(dead).toHaveLength(0);
    });
});
