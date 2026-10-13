import { existsSync, statSync } from "node:fs";
import { dirname, extname, resolve } from "node:path";
import { parse } from "@babel/parser";
import { parse as parseComponent } from "svelte/compiler";

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

type ModuleLoads = { specifiers: string[]; unresolved: string[] };

/** The string value of a Babel `StringLiteral` or an ESTree `Literal`. */
function literal(node: any): string | undefined {
    return (node?.type === "StringLiteral" || node?.type === "Literal") &&
        typeof node.value === "string"
        ? node.value
        : undefined;
}

/** Walk a Babel or ESTree tree for static module specifiers and loads whose targets it cannot prove. */
function moduleLoads(root: unknown): ModuleLoads {
    const imports = new Set<string>();
    const unresolved = new Set<string>();
    const visited = new WeakSet<object>();
    function visit(node: any): void {
        if (!node || typeof node !== "object" || visited.has(node)) return;
        visited.add(node);
        if (Array.isArray(node)) {
            for (const child of node) visit(child);
            return;
        }
        const source = literal(node.source);
        if (
            ["ImportDeclaration", "ExportNamedDeclaration", "ExportAllDeclaration"].includes(
                node.type,
            ) &&
            source !== undefined
        )
            imports.add(source);
        if (node.type === "TSImportType") {
            const argument = literal(node.source) ?? literal(node.argument?.literal);
            if (argument !== undefined) imports.add(argument);
        }
        if (node.type === "ImportExpression") {
            if (source !== undefined) imports.add(source);
            else unresolved.add("non-literal dynamic import");
        }
        if (
            node.type === "CallExpression" &&
            (node.callee.type === "Import" ||
                (node.callee.type === "Identifier" && node.callee.name === "require"))
        ) {
            const argument = literal(node.arguments[0]);
            if (argument !== undefined) imports.add(argument);
            else unresolved.add("non-literal dynamic import or require");
        }
        if (node.type === "TSExternalModuleReference" && literal(node.expression) !== undefined)
            imports.add(literal(node.expression)!);
        for (const [key, child] of Object.entries(node))
            if (!["loc", "start", "end", "tokens", "comments"].includes(key)) visit(child);
    }
    visit(root);
    return { specifiers: [...imports], unresolved: [...unresolved] };
}

/** Collect static module specifiers and module loads whose targets cannot be proved statically. */
export function importsIn(source: string, file: string): ModuleLoads {
    try {
        const ast = parse(source, { sourceType: "unambiguous", plugins: ["typescript", "jsx"] });
        return moduleLoads(ast.program);
    } catch (error) {
        throw new Error(`could not scan imports in ${file}`, { cause: error });
    }
}

/** The same, over a whole Svelte component: its scripts and every expression in its markup. */
export function componentImportsIn(source: string, file: string): ModuleLoads {
    try {
        return moduleLoads(parseComponent(source, { modern: true }));
    } catch (error) {
        throw new Error(`could not scan imports in ${file}`, { cause: error });
    }
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
