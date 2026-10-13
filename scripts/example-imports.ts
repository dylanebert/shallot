import { existsSync, statSync } from "node:fs";
import { dirname, extname, resolve } from "node:path";
import { parse } from "@babel/parser";

const SOURCE_EXTENSIONS = [
    ".d.ts",
    ".ts",
    ".tsx",
    ".mts",
    ".cts",
    ".js",
    ".jsx",
    ".mjs",
    ".cjs",
    ".json",
    ".css",
    ".svg",
    ".wgsl",
];

/** Collect static module specifiers and module loads whose targets cannot be proved statically. */
export function importsIn(
    source: string,
    file: string,
): { specifiers: string[]; unresolved: string[] } {
    const ast = parse(source, { sourceType: "unambiguous", plugins: ["typescript", "jsx"] });
    const imports = new Set<string>();
    const unresolved = new Set<string>();
    function visit(node: any): void {
        if (!node || typeof node !== "object") return;
        if (Array.isArray(node)) {
            for (const child of node) visit(child);
            return;
        }
        if (
            ["ImportDeclaration", "ExportNamedDeclaration", "ExportAllDeclaration"].includes(
                node.type,
            ) &&
            typeof node.source?.value === "string"
        )
            imports.add(node.source.value);
        if (node.type === "TSImportType" && typeof node.source?.value === "string")
            imports.add(node.source.value);
        if (node.type === "ImportExpression") {
            if (typeof node.source?.value === "string") imports.add(node.source.value);
            else unresolved.add("non-literal dynamic import");
        }
        if (
            node.type === "CallExpression" &&
            (node.callee.type === "Import" ||
                (node.callee.type === "Identifier" && node.callee.name === "require"))
        ) {
            if (node.arguments[0]?.type === "StringLiteral") imports.add(node.arguments[0].value);
            else unresolved.add("non-literal dynamic import or require");
        }
        if (node.type === "TSExternalModuleReference" && typeof node.expression?.value === "string")
            imports.add(node.expression.value);
        for (const [key, child] of Object.entries(node))
            if (!["loc", "start", "end", "tokens", "comments"].includes(key)) visit(child);
    }
    try {
        visit(ast.program);
    } catch (error) {
        throw new Error(`could not scan imports in ${file}`, { cause: error });
    }
    return { specifiers: [...imports], unresolved: [...unresolved] };
}

/** Resolve a relative TypeScript or asset import to its source file. */
export function resolveRelative(from: string, specifier: string): string | undefined {
    const base = resolve(dirname(from), specifier.split(/[?#]/, 1)[0]!);
    const extension = extname(base);
    const substitutions: Record<string, string[]> = {
        ".js": [".ts", ".tsx", ".d.ts"],
        ".jsx": [".tsx", ".ts"],
        ".mjs": [".mts"],
        ".cjs": [".cts"],
    };
    const candidates = [base];
    for (const replacement of substitutions[extension] ?? [])
        candidates.push(`${base.slice(0, -extension.length)}${replacement}`);
    candidates.push(...SOURCE_EXTENSIONS.map((suffix) => `${base}${suffix}`));
    for (const candidate of candidates)
        if (existsSync(candidate) && statSync(candidate).isFile()) return candidate;
    if (existsSync(base) && statSync(base).isDirectory()) {
        for (const suffix of SOURCE_EXTENSIONS) {
            const candidate = resolve(base, `index${suffix}`);
            if (existsSync(candidate) && statSync(candidate).isFile()) return candidate;
        }
    }
    return undefined;
}
