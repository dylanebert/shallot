import { expect, test } from "bun:test";
import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { parse } from "@babel/parser";
import { Glob } from "bun";

const ROOT = resolve(import.meta.dir, "..");
type AstNode = Record<string, any>;
type UrlOrigin = "file" | "http" | "other";

function walk(node: unknown, visit: (node: AstNode) => void): void {
    if (!node || typeof node !== "object") return;
    if (Array.isArray(node)) {
        for (const child of node) walk(child, visit);
        return;
    }
    const object = node as AstNode;
    if (typeof object.type === "string") visit(object);
    for (const [key, value] of Object.entries(object)) {
        if (key !== "loc" && key !== "start" && key !== "end" && key !== "extra")
            walk(value, visit);
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
    const files = new Set(patterns.flatMap((pattern) => [...new Glob(pattern).scanSync(ROOT)]));
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

// This only checks URL-to-path conversion, not separator assumptions. Windows separator handling is injected
// through path.win32 in diagnostics/first-person-allocation/allocation.test.ts.
