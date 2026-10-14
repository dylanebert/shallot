import { expect, setDefaultTimeout, test } from "bun:test";
import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { parse } from "@babel/parser";
import { TIMEOUT } from "@dylanebert/shallot/testing";
import { Glob } from "bun";
import { normalizePath } from "../src/project/paths";

setDefaultTimeout(TIMEOUT.node);

const ROOT = resolve(import.meta.dir, "..");
type AstNode = Record<string, any>;
type UrlOrigin = "file" | "http" | "other";

function walk(
    node: unknown,
    visit: (node: AstNode, ancestors: AstNode[]) => void,
    ancestors: AstNode[] = [],
): void {
    if (!node || typeof node !== "object") return;
    if (Array.isArray(node)) {
        for (const child of node) walk(child, visit, ancestors);
        return;
    }
    const object = node as AstNode;
    const typed = typeof object.type === "string";
    if (typed) visit(object, ancestors);
    const below = typed ? [...ancestors, object] : ancestors;
    for (const [key, value] of Object.entries(object)) {
        if (key !== "loc" && key !== "start" && key !== "end" && key !== "extra")
            walk(value, visit, below);
    }
}

function propertyName(node: AstNode): string | undefined {
    if (node.type === "Identifier") return node.name;
    if (node.type === "StringLiteral") return node.value;
}

function isImportMetaUrl(node: AstNode): boolean {
    return (
        node.type === "MemberExpression" &&
        !node.computed &&
        propertyName(node.property) === "url" &&
        node.object.type === "MetaProperty" &&
        node.object.meta.name === "import" &&
        node.object.property.name === "meta"
    );
}

function containsImportMetaUrl(node: unknown): boolean {
    let found = false;
    walk(node, (child) => {
        if (isImportMetaUrl(child)) found = true;
    });
    return found;
}

function isUrlConstructor(node: AstNode | null | undefined): node is AstNode {
    return (
        !!node &&
        node.type === "NewExpression" &&
        node.callee.type === "Identifier" &&
        node.callee.name === "URL"
    );
}

function isHttpUrl(node: AstNode | null | undefined): boolean {
    if (!isUrlConstructor(node)) return false;
    const args = node.arguments as AstNode[];
    if (
        args.some((arg) => arg.type === "StringLiteral" && /^https?:\/\//i.test(arg.value)) ||
        args.slice(1).some((arg) => arg.type === "StringLiteral" && /^https?:\/\//i.test(arg.value))
    )
        return true;

    // Vite's server.url is already an HTTP URL; parsing it for its route is not a file conversion.
    const input = args[0];
    return input?.type === "MemberExpression" && propertyName(input.property) === "url";
}

function urlOrigin(node: AstNode | null | undefined): UrlOrigin | undefined {
    if (!isUrlConstructor(node)) return;
    if (containsImportMetaUrl(node)) return "file";
    return isHttpUrl(node) ? "http" : "other";
}

function fileUrlPathErrors(file: string, source: string): string[] {
    if (!source.includes(".pathname")) return [];

    const ast = parse(source, { sourceType: "unambiguous", plugins: ["typescript"] });
    const origins = new Map<string, UrlOrigin>();
    walk(ast.program, (node) => {
        if (node.type !== "VariableDeclarator" || node.id.type !== "Identifier") return;
        const origin = urlOrigin(node.init);
        if (origin) origins.set(node.id.name, origin);
    });

    const errors: string[] = [];
    walk(ast.program, (node) => {
        if (node.type !== "MemberExpression" || propertyName(node.property) !== "pathname") return;

        const origin =
            urlOrigin(node.object) ??
            (node.object.type === "Identifier" ? origins.get(node.object.name) : undefined);
        if (origin === undefined || origin === "http") return;
        errors.push(
            `${file}:${node.loc.start.line}: file URL .pathname is not a filesystem path; use fileURLToPath() from node:url`,
        );
    });
    return errors;
}

const HELPER = "src/project/paths.ts";
const WRAPPERS = new Set([
    "TSAsExpression",
    "TSNonNullExpression",
    "TSSatisfiesExpression",
    "ParenthesizedExpression",
]);

function isNamed(node: AstNode | undefined, name: string): boolean {
    return node?.type === "Identifier" && node.name === name;
}

// The call that receives `node`'s value, past type assertions, and the argument it is.
function receiver(node: AstNode, ancestors: AstNode[]): { call?: AstNode; index: number } {
    let child = node;
    for (let i = ancestors.length - 1; i >= 0; i--) {
        const parent = ancestors[i]!;
        if (WRAPPERS.has(parent.type)) {
            child = parent;
            continue;
        }
        if (parent.type !== "CallExpression") return { index: -1 };
        return { call: parent, index: (parent.arguments as AstNode[]).indexOf(child) };
    }
    return { index: -1 };
}

function isSeparator(node: AstNode | undefined): boolean {
    if (!node) return false;
    if (isNamed(node, "sep")) return true;
    if (node.type === "MemberExpression" && propertyName(node.property) === "sep") return true;
    if (node.type === "StringLiteral") return node.value === "\\";
    // `/\\/` or a class of only separators, such as `/[\\/]/`; an escape class that holds `\\` is not one.
    return node.type === "RegExpLiteral" && /^(?:\\\\|\[(?:\\\\|\\?\/)+\])\+?$/.test(node.pattern);
}

type PathBindings = { relative: Set<string>; modules: Set<string> };

function pathBindings(ast: AstNode): PathBindings {
    const relative = new Set<string>();
    const modules = new Set<string>();
    for (const statement of ast.program.body as AstNode[]) {
        if (
            statement.type !== "ImportDeclaration" ||
            !["node:path", "path"].includes(statement.source.value)
        )
            continue;
        for (const specifier of statement.specifiers as AstNode[]) {
            if (specifier.importKind === "type") continue;
            if (specifier.type === "ImportSpecifier") {
                if (propertyName(specifier.imported) === "relative")
                    relative.add(specifier.local.name);
            } else if (
                specifier.type === "ImportNamespaceSpecifier" ||
                specifier.type === "ImportDefaultSpecifier"
            ) {
                modules.add(specifier.local.name);
            }
        }
    }
    walk(ast.program, (node) => {
        const left =
            node.type === "AssignmentPattern"
                ? node.left
                : node.type === "VariableDeclarator"
                  ? node.id
                  : undefined;
        const right =
            node.type === "AssignmentPattern"
                ? node.right
                : node.type === "VariableDeclarator"
                  ? node.init
                  : undefined;
        if (left?.type === "Identifier" && right?.type === "Identifier" && modules.has(right.name))
            modules.add(left.name);
    });
    return { relative, modules };
}

function isPathRelative(node: AstNode, bindings: PathBindings): boolean {
    const callee = node.callee;
    return (
        (callee.type === "Identifier" && bindings.relative.has(callee.name)) ||
        (callee.type === "MemberExpression" &&
            propertyName(callee.property) === "relative" &&
            callee.object.type === "Identifier" &&
            bindings.modules.has(callee.object.name))
    );
}

// These are the producers and separator spellings used by hosted Windows reds that compared slash paths.
function producer(node: AstNode, bindings: PathBindings): string | undefined {
    if (node.type !== "CallExpression") return;
    const callee = node.callee;
    const name = callee.type === "MemberExpression" ? propertyName(callee.property) : callee.name;
    if (isPathRelative(node, bindings)) return "relative() path";
    if (callee.type === "MemberExpression" && (name === "scan" || name === "scanSync"))
        return "Glob entry";
    if (name !== "readdir" && name !== "readdirSync") return;
    const recursive = (node.arguments as AstNode[]).some(
        (arg) =>
            arg.type === "ObjectExpression" &&
            arg.properties.some(
                (property: AstNode) =>
                    property.type === "ObjectProperty" &&
                    propertyName(property.key) === "recursive" &&
                    property.value.value === true,
            ),
    );
    if (recursive) return "recursive readdir entry";
}

// A producer enters through `normalizePath(...)` or `Array.from(entries, normalizePath)`, so its result has `/` separators.
function entersNormalized(node: AstNode, ancestors: AstNode[]): boolean {
    const { call, index } = receiver(node, ancestors);
    if (!call) return false;
    if (isNamed(call.callee, "normalizePath")) return index === 0;
    const callee = call.callee;
    return (
        index === 0 &&
        callee.type === "MemberExpression" &&
        isNamed(callee.object, "Array") &&
        ["from", "fromAsync"].includes(propertyName(callee.property) ?? "") &&
        isNamed(call.arguments[1], "normalizePath")
    );
}

function separatorErrors(file: string, source: string): string[] {
    const ast = parse(source, { sourceType: "unambiguous", plugins: ["typescript"] });
    const bindings = pathBindings(ast);
    const errors: string[] = [];
    walk(ast.program, (node, ancestors) => {
        if (node.type !== "CallExpression") return;
        const at = `${file}:${node.loc.start.line}`;
        const method =
            node.callee.type === "MemberExpression"
                ? propertyName(node.callee.property)
                : undefined;
        if (
            file !== HELPER &&
            (method === "split" || method === "replace" || method === "replaceAll") &&
            isSeparator(node.arguments[0])
        )
            errors.push(`${at}: separators handled outside normalizePath (${HELPER})`);
        const produced = producer(node, bindings);
        if (produced && !entersNormalized(node, ancestors))
            errors.push(`${at}: ${produced} enters without normalizePath (${HELPER})`);
    });
    return errors;
}

// Tooling, which reaches normalizePath. Game modules and examples never import it (CONTRIBUTING, Layout and
// Examples), and run where no host path enters.
function toolingFiles(): string[] {
    const patterns = [
        "src/{project,cli,native,types}/**/*.{ts,js}",
        "{scripts,diagnostics,bin}/**/*.{ts,js,mjs}",
        "crates/*/scripts/**/*.ts",
    ];
    return [
        ...new Set(
            patterns.flatMap((pattern) =>
                Array.from(new Glob(pattern).scanSync(ROOT), normalizePath),
            ),
        ),
    ].sort();
}

function nodeSourceFiles(): string[] {
    const patterns = [
        "src/**/*.node.ts",
        "src/cli/**/*.ts",
        "src/native/**/*.ts",
        "src/project/**/*.ts",
        "src/project/**/*.js",
        "scripts/**/*.ts",
        "scripts/**/*.js",
        "scripts/**/*.mjs",
        "diagnostics/**/*.ts",
        "diagnostics/**/*.js",
        "diagnostics/**/*.mjs",
        "bin/**/*.ts",
        "bin/**/*.js",
    ];
    const files = new Set(
        patterns.flatMap((pattern) => Array.from(new Glob(pattern).scanSync(ROOT), normalizePath)),
    );
    return [...files]
        .filter(
            (file) =>
                !file.endsWith(".e2e.ts") &&
                !file.endsWith(".entry.ts") &&
                !file.endsWith(".fixture.ts") &&
                file !== "diagnostics/first-person-allocation/display.ts" &&
                file !== "diagnostics/first-person-allocation/display-seat.ts",
        )
        .sort();
}

test("file URL paths are converted with fileURLToPath", () => {
    expect(
        fileUrlPathErrors(
            "src/planted.node.ts",
            "const path = new URL(import.meta.url).pathname;\nreadFileSync(path);\n",
        ),
    ).toEqual([
        "src/planted.node.ts:1: file URL .pathname is not a filesystem path; use fileURLToPath() from node:url",
    ]);
    expect(
        fileUrlPathErrors(
            "src/planted.node.ts",
            "const url = new URL('./asset', import.meta.url);\nreadFileSync(url.pathname);\n",
        ),
    ).toEqual([
        "src/planted.node.ts:2: file URL .pathname is not a filesystem path; use fileURLToPath() from node:url",
    ]);
});

test("HTTP URL pathnames remain available for routing", () => {
    expect(
        fileUrlPathErrors(
            "src/cli/dev.node.ts",
            'new URL(server.url).pathname;\nnew URL(req.url, "http://localhost").pathname;\n',
        ),
    ).toEqual([]);
});

test("Node-side source has no file URL pathnames", () => {
    const violations = nodeSourceFiles().flatMap((file) =>
        fileUrlPathErrors(file, readFileSync(resolve(ROOT, file), "utf8")),
    );
    expect(violations).toEqual([]);
});

test("relative(), Glob scans, recursive readdir, and separator spellings normalize at entry", () => {
    expect(
        separatorErrors(
            "scripts/planted.ts",
            [
                'import { relative } from "node:path";',
                'const a = relative (".", file);',
                'const b = relative(root, file).split(sep).join("/");',
                'import { relative as rel } from "path";',
                'const c = rel(".", file);',
                'import * as path from "node:path";',
                'const d = path.relative(".", file);',
                'const e = [...new Glob("**").scanSync(root)];',
                "const f = readdirSync(dir, { recursive: true });",
                'const g = path.replace(/\\\\/g, "/");',
                "const h = normalizePath(path.relative(root, file));",
                "const i = Array.from(glob.scanSync(root) as string[], normalizePath);",
                "const j = await Array.fromAsync(glob.scan(root), normalizePath);",
                'function relativeModule(pathModule = path) { return pathModule.relative(".", file); }',
            ].join("\n"),
        ),
    ).toEqual([
        "scripts/planted.ts:2: relative() path enters without normalizePath (src/project/paths.ts)",
        "scripts/planted.ts:3: separators handled outside normalizePath (src/project/paths.ts)",
        "scripts/planted.ts:3: relative() path enters without normalizePath (src/project/paths.ts)",
        "scripts/planted.ts:5: relative() path enters without normalizePath (src/project/paths.ts)",
        "scripts/planted.ts:7: relative() path enters without normalizePath (src/project/paths.ts)",
        "scripts/planted.ts:8: Glob entry enters without normalizePath (src/project/paths.ts)",
        "scripts/planted.ts:9: recursive readdir entry enters without normalizePath (src/project/paths.ts)",
        "scripts/planted.ts:10: separators handled outside normalizePath (src/project/paths.ts)",
        "scripts/planted.ts:14: relative() path enters without normalizePath (src/project/paths.ts)",
    ]);
});

test("tooling holds relative(), Glob scans, recursive readdir, and separator spellings used by hosted Windows reds", () => {
    const violations = toolingFiles().flatMap((file) =>
        separatorErrors(file, readFileSync(resolve(ROOT, file), "utf8")),
    );
    expect(violations).toEqual([]);
});
