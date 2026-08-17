// ts-prune-shaped: for every symbol exported from `packages/shallot/src/**`, determine whether
// any *external* consumer imports it across `src/`, `examples/`, `scripts/`, and `tests/`.
// Reports the zero-consumer ones (exported, nobody imports them, not even referenced in-file),
// the in-file-only ones (exported, referenced within the defining file but not imported
// externally — exports.md's internal-stays-internal), and the test-only ones (consumed only by
// test files, zero production consumers). Re-export chains (barrel files) are followed to the
// original exporter, so a symbol consumed through a barrel is not falsely flagged.
//
// `--root <dir>` points the check at an alternate tree (fixture-driven proof; check-scripts.ts
// and check-boundary.ts carry the same flag for the same reason).

import { existsSync } from "node:fs";
import { dirname, join, relative, resolve } from "node:path";
import { Glob } from "bun";

const PKG = "@dylanebert/shallot";

// --- Types ------------------------------------------------------------------

export type ExportEntry = {
    file: string; // path relative to rootDir, forward slashes
    name: string;
    line: number;
    kind: string;
};

export type DeadExport = {
    file: string;
    name: string;
    line: number;
    category: "zero-consumer" | "in-file-only" | "test-only";
    testConsumers: { file: string; line: number }[];
};

// --- Helpers ----------------------------------------------------------------

export function isTestFile(path: string): boolean {
    return /\.(test|oracle|lab|probes|tier)\.ts$/.test(path);
}

export function escapeRegExp(s: string): string {
    return s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

function lineOf(content: string, offset: number): number {
    return content.slice(0, offset).split("\n").length;
}

// --- Export extraction ------------------------------------------------------

// Direct exports: symbols declared with `export` in the file, including `export { name }`
// without `from` (re-export of local symbols). Re-exports with `from` are handled by
// extractReExports — they are pass-throughs, not new symbols.
export function extractDirectExports(content: string): ExportEntry[] {
    const exports: ExportEntry[] = [];

    const patterns: [RegExp, string][] = [
        [/export\s+(?:async\s+)?function\s+(\w+)/g, "function"],
        [/export\s+const\s+(\w+)/g, "const"],
        [/export\s+let\s+(\w+)/g, "let"],
        [/export\s+class\s+(\w+)/g, "class"],
        [/export\s+type\s+(\w+)/g, "type"],
        [/export\s+interface\s+(\w+)/g, "interface"],
        [/export\s+enum\s+(\w+)/g, "enum"],
    ];

    for (const [pattern, kind] of patterns) {
        for (const m of content.matchAll(pattern)) {
            if (m.index === undefined) continue;
            exports.push({ file: "", name: m[1], line: lineOf(content, m.index), kind });
        }
    }

    // export { name1, name2 } (without `from` — re-export of local symbols)
    // also handles export type { name1, name2 } without `from`
    for (const m of content.matchAll(/export\s+(?:type\s+)?\{([^}]+)\}(?!\s*from\b)/g)) {
        if (m.index === undefined) continue;
        const names = m[1]
            .split(",")
            .map((s) =>
                s
                    .trim()
                    .replace(/^type\s+/, "")
                    .split(/\s+as\s+/)[0]
                    .trim(),
            )
            .filter(Boolean);
        for (const name of names) {
            if (!exports.some((e) => e.name === name)) {
                exports.push({ file: "", name, line: lineOf(content, m.index), kind: "re-export" });
            }
        }
    }

    return exports;
}

export type ReExportEntry = {
    names: string[] | "*"; // "*" for export * from
    source: string; // module specifier as written
};

// Re-exports: export { name } from "..." and export * from "..."
export function extractReExports(content: string): ReExportEntry[] {
    const reExports: ReExportEntry[] = [];

    for (const m of content.matchAll(/export\s+\*\s+from\s+["']([^"']+)["']/g)) {
        reExports.push({ names: "*", source: m[1] });
    }

    for (const m of content.matchAll(
        /export\s+(?:type\s+)?\{([^}]+)\}\s*from\s+["']([^"']+)["']/g,
    )) {
        const names = m[1]
            .split(",")
            .map((s) =>
                s
                    .trim()
                    .replace(/^type\s+/, "")
                    .split(/\s+as\s+/)[0]
                    .trim(),
            )
            .filter(Boolean);
        reExports.push({ names, source: m[2] });
    }

    return reExports;
}

// --- Import extraction ------------------------------------------------------

export type ImportEntry = {
    names: string[]; // source names (before `as`); ["*"] for namespace imports
    specifier: string;
    line: number;
    namespace: string | null; // namespace name for `import * as ns from`
};

export function extractImports(content: string): ImportEntry[] {
    const imports: ImportEntry[] = [];

    const parseNames = (str: string) =>
        str
            .split(",")
            .map((s) =>
                s
                    .trim()
                    .replace(/^type\s+/, "")
                    .split(/\s+as\s+/)[0]
                    .trim(),
            )
            .filter(Boolean);

    // import { name1, name2 } from "..." / import type { ... } from "..."
    for (const m of content.matchAll(
        /import\s+(?:type\s+)?\{([^}]+)\}\s*from\s+["']([^"']+)["']/g,
    )) {
        if (m.index === undefined) continue;
        imports.push({
            names: parseNames(m[1]),
            specifier: m[2],
            line: lineOf(content, m.index),
            namespace: null,
        });
    }

    // import defaultName, { name1, name2 } from "..."
    for (const m of content.matchAll(
        /import\s+(\w+)\s*,\s*(?:type\s+)?\{([^}]+)\}\s*from\s+["']([^"']+)["']/g,
    )) {
        if (m.index === undefined) continue;
        imports.push({
            names: [m[1], ...parseNames(m[2])],
            specifier: m[3],
            line: lineOf(content, m.index),
            namespace: null,
        });
    }

    // import defaultName from "..." (not type-only, not mixed with named)
    for (const m of content.matchAll(/import\s+(?!type\s)(\w+)\s+from\s+["']([^"']+)["']/g)) {
        if (m.index === undefined) continue;
        imports.push({
            names: [m[1]],
            specifier: m[2],
            line: lineOf(content, m.index),
            namespace: null,
        });
    }

    // import type defaultName from "..."
    for (const m of content.matchAll(/import\s+type\s+(\w+)\s+from\s+["']([^"']+)["']/g)) {
        if (m.index === undefined) continue;
        imports.push({
            names: [m[1]],
            specifier: m[2],
            line: lineOf(content, m.index),
            namespace: null,
        });
    }

    // import * as name from "..." / import type * as name from "..."
    for (const m of content.matchAll(
        /import\s+(?:type\s+)?\*\s+as\s+(\w+)\s+from\s+["']([^"']+)["']/g,
    )) {
        if (m.index === undefined) continue;
        imports.push({
            names: ["*"],
            specifier: m[2],
            line: lineOf(content, m.index),
            namespace: m[1],
        });
    }

    return imports;
}

// --- Specifier resolution ---------------------------------------------------

export function resolveSpecifier(
    fromFile: string,
    specifier: string,
    rootDir: string,
    packageExports: Record<string, unknown>,
): string | null {
    if (specifier.startsWith(".")) {
        const fromDir = dirname(resolve(rootDir, fromFile));
        const abs = resolve(fromDir, specifier);
        for (const candidate of [abs + ".ts", join(abs, "index.ts"), abs]) {
            if (existsSync(candidate)) {
                return relative(rootDir, candidate).replace(/\\/g, "/");
            }
        }
        return null;
    }

    if (specifier === PKG || specifier.startsWith(PKG + "/")) {
        const subpath = specifier === PKG ? "." : `./${specifier.slice(PKG.length + 1)}`;
        const pkgDir = resolve(rootDir, "packages/shallot");

        const target = (entry: unknown) =>
            typeof entry === "string" ? entry : (entry as { types?: string })?.types;

        // exact match
        const exact = packageExports[subpath];
        if (exact) {
            const t = target(exact);
            if (t) {
                const resolved = resolve(pkgDir, t);
                if (existsSync(resolved)) return relative(rootDir, resolved).replace(/\\/g, "/");
            }
        }

        // wildcard match (e.g. ./src/*)
        for (const [key, value] of Object.entries(packageExports)) {
            if (!key.includes("*")) continue;
            const t = target(value);
            if (typeof t !== "string" || !t.includes("*")) continue;

            const keyStar = key.indexOf("*");
            const keyPrefix = key.slice(0, keyStar);
            const keySuffix = key.slice(keyStar + 1);
            if (subpath.startsWith(keyPrefix) && subpath.endsWith(keySuffix)) {
                const captured = subpath.slice(keyPrefix.length, subpath.length - keySuffix.length);
                const tStar = t.indexOf("*");
                const resolvedPath = t.slice(0, tStar) + captured + t.slice(tStar + 1);
                for (const candidate of [
                    resolve(pkgDir, resolvedPath + ".ts"),
                    join(resolve(pkgDir, resolvedPath), "index.ts"),
                    resolve(pkgDir, resolvedPath),
                ]) {
                    if (existsSync(candidate))
                        return relative(rootDir, candidate).replace(/\\/g, "/");
                }
            }
        }
    }

    return null;
}

// --- Re-export chain resolution ---------------------------------------------

function resolveExport(
    file: string,
    name: string,
    directExports: Map<string, Set<string>>,
    reExports: Map<string, { names: string[] | "*"; sourceFile: string }[]>,
    visited: Set<string>,
): string | null {
    if (visited.has(file)) return null;
    visited.add(file);

    if (directExports.get(file)?.has(name)) return file;

    for (const re of reExports.get(file) ?? []) {
        if (re.names === "*" || re.names.includes(name)) {
            const result = resolveExport(re.sourceFile, name, directExports, reExports, visited);
            if (result) return result;
        }
    }

    return null;
}

// --- In-file usage check ----------------------------------------------------

function isInFileOnly(content: string, name: string): boolean {
    const pattern = new RegExp(`\\b${escapeRegExp(name)}\\b`, "g");
    return (content.match(pattern) ?? []).length > 1;
}

// --- Consumer tracking ------------------------------------------------------

type Consumer = { isTest: boolean; consumerFile: string; consumerLine: number };

function addConsumer(
    consumed: Map<string, Map<string, Consumer[]>>,
    file: string,
    name: string,
    consumer: Consumer,
): void {
    if (!consumed.has(file)) consumed.set(file, new Map());
    if (!consumed.get(file)!.has(name)) consumed.get(file)!.set(name, []);
    consumed.get(file)!.get(name)!.push(consumer);
}

// --- Main -------------------------------------------------------------------

export async function findDeadExports(
    rootDir: string,
    allowlist: { file: string; name: string }[] = [],
): Promise<DeadExport[]> {
    const srcDir = resolve(rootDir, "packages/shallot/src");
    const pkgPath = resolve(rootDir, "packages/shallot/package.json");
    if (!existsSync(pkgPath)) return [];
    const pkg = (await Bun.file(pkgPath).json()) as { exports?: Record<string, unknown> };
    const packageExports = pkg.exports ?? {};

    const allowSet = new Set(allowlist.map((a) => `${a.file}::${a.name}`));

    // Step 1: walk source files, extract direct exports and re-exports
    const directExportsMap = new Map<string, Set<string>>();
    const reExportsMap = new Map<string, { names: string[] | "*"; sourceFile: string }[]>();
    const allExports: ExportEntry[] = [];
    const fileContents = new Map<string, string>();

    const srcGlob = new Glob("**/*.ts");
    for await (const path of srcGlob.scan({ cwd: srcDir })) {
        if (isTestFile(path)) continue;
        const full = resolve(srcDir, path);
        const content = await Bun.file(full).text();
        const relPath = relative(rootDir, full).replace(/\\/g, "/");
        fileContents.set(relPath, content);

        const exports = extractDirectExports(content);
        for (const e of exports) {
            e.file = relPath;
            allExports.push(e);
        }
        directExportsMap.set(relPath, new Set(exports.map((e) => e.name)));

        const reExports = extractReExports(content);
        const resolved: { names: string[] | "*"; sourceFile: string }[] = [];
        for (const re of reExports) {
            const sourceFile = resolveSpecifier(relPath, re.source, rootDir, packageExports);
            if (sourceFile) resolved.push({ names: re.names, sourceFile });
        }
        reExportsMap.set(relPath, resolved);
    }

    // Step 2: walk consumer files, extract imports, resolve to original exporters
    const consumed = new Map<string, Map<string, Consumer[]>>();

    const consumerDirs = [
        "packages/shallot/src",
        "packages/shallot/tests",
        "packages/shallot/bin",
        "packages/shallot/scripts",
        "scripts",
        "examples",
        "evals",
    ];

    for (const dir of consumerDirs) {
        const full = resolve(rootDir, dir);
        if (!existsSync(full)) continue;
        const glob = new Glob("**/*.ts");
        for await (const path of glob.scan({ cwd: full })) {
            if (path.includes("node_modules") || path.includes("dist/")) continue;

            const fullPath = resolve(full, path);
            const content = await Bun.file(fullPath).text();
            const relPath = relative(rootDir, fullPath).replace(/\\/g, "/");
            const isTest = isTestFile(relPath);

            const imports = extractImports(content);
            for (const imp of imports) {
                const targetFile = resolveSpecifier(
                    relPath,
                    imp.specifier,
                    rootDir,
                    packageExports,
                );
                if (!targetFile) continue;
                if (targetFile === relPath) continue; // self-import is not external

                for (const name of imp.names) {
                    if (name === "*") {
                        // namespace import — only exports actually accessed through the namespace
                        // (nsName.exportName in the file content) are consumed. A bare `import * as ns`
                        // that never references ns.foo does not consume any export.
                        const nsName = imp.namespace;
                        if (nsName) {
                            const nsPattern = new RegExp(
                                `\\b${escapeRegExp(nsName)}\\.(\\w+)`,
                                "g",
                            );
                            for (const m of content.matchAll(nsPattern)) {
                                const accessedName = m[1];
                                const original = resolveExport(
                                    targetFile,
                                    accessedName,
                                    directExportsMap,
                                    reExportsMap,
                                    new Set(),
                                );
                                if (original) {
                                    addConsumer(consumed, original, accessedName, {
                                        isTest,
                                        consumerFile: relPath,
                                        consumerLine: imp.line,
                                    });
                                }
                            }
                        }
                    } else {
                        const original = resolveExport(
                            targetFile,
                            name,
                            directExportsMap,
                            reExportsMap,
                            new Set(),
                        );
                        if (original) {
                            addConsumer(consumed, original, name, {
                                isTest,
                                consumerFile: relPath,
                                consumerLine: imp.line,
                            });
                        }
                    }
                }
            }
        }
    }

    // Step 3: find dead exports
    const dead: DeadExport[] = [];
    for (const e of allExports) {
        if (allowSet.has(`${e.file}::${e.name}`)) continue;

        const consumers = consumed.get(e.file)?.get(e.name) ?? [];
        const productionConsumers = consumers.filter((c) => !c.isTest);
        const testConsumers = consumers.filter((c) => c.isTest);

        if (productionConsumers.length > 0) continue;

        if (testConsumers.length > 0) {
            dead.push({
                file: e.file,
                name: e.name,
                line: e.line,
                category: "test-only",
                testConsumers: testConsumers.map((c) => ({
                    file: c.consumerFile,
                    line: c.consumerLine,
                })),
            });
        } else {
            const content = fileContents.get(e.file) ?? "";
            const inFile = isInFileOnly(content, e.name);
            dead.push({
                file: e.file,
                name: e.name,
                line: e.line,
                category: inFile ? "in-file-only" : "zero-consumer",
                testConsumers: [],
            });
        }
    }

    return dead;
}

export async function run(rootDir: string): Promise<{ dead: DeadExport[]; total: number }> {
    const dead = await findDeadExports(rootDir);
    return { dead, total: dead.length };
}

if (import.meta.main) {
    const rootArgIdx = process.argv.indexOf("--root");
    const rootDir = resolve(
        rootArgIdx >= 0 ? process.argv[rootArgIdx + 1] : resolve(import.meta.dir, ".."),
    );

    const { dead, total } = await run(rootDir);

    if (total > 0) {
        const byCategory = {
            "zero-consumer": dead.filter((d) => d.category === "zero-consumer"),
            "in-file-only": dead.filter((d) => d.category === "in-file-only"),
            "test-only": dead.filter((d) => d.category === "test-only"),
        };

        console.error(`✗ ${total} dead export(s):\n`);

        for (const [category, items] of Object.entries(byCategory)) {
            if (items.length === 0) continue;
            console.error(`  [${category}] (${items.length}):\n`);
            for (const d of items) {
                console.error(`    ${d.file}:${d.line} ${d.name}`);
                if (d.category === "test-only") {
                    for (const c of d.testConsumers) {
                        console.error(`      consumed by: ${c.file}:${c.line}`);
                    }
                }
            }
            console.error();
        }

        console.error(
            "An exported symbol with zero production consumers is dead — demote to internal\n" +
                "(exports.md internal-stays-internal) or delete. Test-only consumers should reach\n" +
                "the symbol via the module path, not through a public export.",
        );
        process.exit(1);
    }

    console.log("✓ no dead exports");
}
