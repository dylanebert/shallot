import { expect, setDefaultTimeout, test } from "bun:test";

import { CEILING } from "@dylanebert/shallot/testing";

setDefaultTimeout(CEILING.node);

import { mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { basename, dirname, join, relative, resolve } from "node:path";
import { parse } from "@babel/parser";
import { normalizePath } from "../src/project/paths";

const ROOT = resolve(import.meta.dir, "..");
const PACKAGE = JSON.parse(readFileSync(join(ROOT, "package.json"), "utf8")) as {
    exports: Record<string, string | Record<string, string>>;
};

type Entry = { name: string; source: string; publishedDeclaration?: string };
function selectedTarget(target: string | Record<string, string>): string {
    if (typeof target === "string") return target;
    const selected = target.types ?? target.import ?? target.default ?? Object.values(target)[0];
    if (!selected) throw new Error("package export has no declaration target");
    return selected;
}
const entries: Entry[] = Object.entries(PACKAGE.exports)
    .map(([name, target]) => {
        const path = selectedTarget(target);
        if (path.endsWith(".json")) return undefined; // The tsconfig subpath is data, not a code API.
        if (path.endsWith(".d.ts"))
            return {
                name,
                source: resolve(ROOT, "src/project", basename(path).replace(/\.d\.ts$/, ".js")),
                publishedDeclaration: resolve(ROOT, path),
            };
        if (path.endsWith(".js")) return { name, source: resolve(ROOT, path) };
        if (!path.endsWith(".ts")) return undefined;
        return { name, source: resolve(ROOT, path) };
    })
    .filter((entry): entry is Entry => entry !== undefined);

function emittedPath(outDir: string, source: string): string {
    return join(outDir, normalizePath(relative(ROOT, source)).replace(/\.(?:tsx?|js)$/, ".d.ts"));
}
function isHandleDeclaration(file: string): boolean {
    return /[/\\]src[/\\]standard[/\\]physics[/\\]api[/\\](?:body|joint|joints|shape|world)\.d\.ts$/.test(
        file,
    );
}
function childNodes(value: unknown): object[] {
    if (Array.isArray(value))
        return value.filter((item): item is object => !!item && typeof item === "object");
    if (value && typeof value === "object") return [value];
    return [];
}
function visit(node: object, callback: (node: Record<string, any>) => void): void {
    const current = node as Record<string, any>;
    callback(current);
    for (const [key, value] of Object.entries(current)) {
        if (
            [
                "loc",
                "start",
                "end",
                "leadingComments",
                "innerComments",
                "trailingComments",
            ].includes(key)
        )
            continue;
        for (const child of childNodes(value)) visit(child, callback);
    }
}
function idName(node: any): string | undefined {
    if (node?.type === "Identifier") return node.name;
    if (node?.type === "TSQualifiedName") return idName(node.left);
    if (node?.type === "MemberExpression") return idName(node.object);
    return undefined;
}

test("public declarations do not expose private solver handles", () => {
    const scratch = mkdtempSync(join(tmpdir(), "shallot-public-declarations-"));
    const outDir = join(scratch, "out");
    const configPath = join(scratch, "tsconfig.json");
    const tempConfig = {
        extends: join(ROOT, "tsconfig.json"),
        compilerOptions: {
            declaration: true,
            emitDeclarationOnly: true,
            noEmit: false,
            declarationMap: false,
            incremental: false,
            composite: false,
            rootDir: ROOT,
            outDir,
            skipLibCheck: true,
            typeRoots: [join(ROOT, "node_modules/@types"), join(ROOT, "node_modules")],
        },
        files: entries.map((entry) => entry.source),
        include: [],
    };
    writeFileSync(configPath, JSON.stringify(tempConfig));

    try {
        const emitted = Bun.spawnSync(
            ["node", resolve(ROOT, "node_modules/typescript/bin/tsc"), "-p", configPath],
            {
                cwd: ROOT,
                stdout: "pipe",
                stderr: "pipe",
            },
        );
        if (!emitted.success)
            throw new Error(
                `package declaration emit failed:\n${emitted.stdout.toString()}${emitted.stderr.toString()}`,
            );

        const roots = entries.map((entry) => emittedPath(outDir, entry.source));
        const packagePaths = new Map<string, string>();
        for (const entry of entries) {
            const subpath = entry.name === "." ? "" : entry.name.slice(2);
            packagePaths.set(
                `@dylanebert/shallot${subpath ? `/${subpath}` : ""}`,
                emittedPath(outDir, entry.source),
            );
        }
        const parsedFiles = new Map<string, any>();
        const exportedCache = new Map<string, Map<string, { file: string; name: string }[]>>();
        const visitedTypes = new Set<string>();
        const errors = new Set<string>();

        function load(file: string): any | undefined {
            file = resolve(file);
            const cached = parsedFiles.get(file);
            if (cached) return cached;
            try {
                const ast = parse(readFileSync(file, "utf8"), {
                    sourceType: "module",
                    plugins: ["typescript"],
                });
                parsedFiles.set(file, ast);
                return ast;
            } catch {
                return undefined;
            }
        }

        function resolveModule(from: string, specifier: string): string | undefined {
            const packagePath = packagePaths.get(specifier);
            if (packagePath) return packagePath;
            if (!specifier.startsWith(".")) return undefined;
            const base = resolve(dirname(from), specifier);
            for (const candidate of [
                base,
                `${base}.d.ts`,
                `${base}.d.mts`,
                `${base}.d.cts`,
                join(base, "index.d.ts"),
            ]) {
                try {
                    readFileSync(candidate);
                    return resolve(candidate);
                } catch {}
            }
            return undefined;
        }

        function declarations(file: string): Map<string, object[]> {
            const ast = load(file);
            const result = new Map<string, object[]>();
            if (!ast) return result;
            for (const statement of ast.program.body as any[]) {
                if (statement.type === "ExportDefaultDeclaration") {
                    result.set("default", [statement.declaration]);
                    continue;
                }
                const declaration =
                    statement.type === "ExportNamedDeclaration" ? statement.declaration : statement;
                if (!declaration) continue;
                const ids: string[] = [];
                if (
                    [
                        "TSInterfaceDeclaration",
                        "TSTypeAliasDeclaration",
                        "ClassDeclaration",
                        "FunctionDeclaration",
                        "VariableDeclaration",
                        "TSEnumDeclaration",
                    ].includes(declaration.type)
                ) {
                    if (declaration.type === "VariableDeclaration") {
                        for (const item of declaration.declarations)
                            if (item.id?.type === "Identifier") ids.push(item.id.name);
                    } else if (declaration.id?.name) ids.push(declaration.id.name);
                }
                for (const id of ids) {
                    const list = result.get(id) ?? [];
                    list.push(declaration);
                    result.set(id, list);
                }
            }
            return result;
        }

        function exports(file: string): Map<string, { file: string; name: string }[]> {
            file = resolve(file);
            const cached = exportedCache.get(file);
            if (cached) return cached;
            const result = new Map<string, { file: string; name: string }[]>();
            exportedCache.set(file, result);
            const ast = load(file);
            if (!ast) return result;
            const localDeclarations = declarations(file);
            for (const statement of ast.program.body as any[]) {
                if (
                    statement.type !== "ExportNamedDeclaration" &&
                    statement.type !== "ExportAllDeclaration" &&
                    statement.type !== "ExportDefaultDeclaration"
                )
                    continue;
                if (statement.type === "ExportDefaultDeclaration") {
                    result.set("default", [{ file, name: "default" }]);
                    continue;
                }
                const source = statement.source?.value;
                const target = source ? resolveModule(file, source) : undefined;
                if (statement.type === "ExportAllDeclaration") {
                    if (target) {
                        for (const name of exports(target).keys()) {
                            if (name === "default") continue;
                            result.set(name, [{ file: target, name }]);
                        }
                    }
                    continue;
                }
                if (statement.declaration) {
                    for (const [name] of declarationsForNode(statement.declaration))
                        result.set(name, [{ file, name }]);
                }
                for (const specifier of statement.specifiers ?? []) {
                    if (specifier.type === "ExportNamespaceSpecifier") {
                        if (target)
                            result.set(specifier.exported.name, [{ file: target, name: "*" }]);
                        continue;
                    }
                    const local = specifier.local?.name;
                    const name = specifier.exported?.name ?? local;
                    if (!name || !local) continue;
                    if (target) result.set(name, [{ file: target, name: local }]);
                    else if (localDeclarations.has(local))
                        result.set(name, [{ file, name: local }]);
                    else {
                        const astImports = (ast.program.body as any[]).filter(
                            (node) => node.type === "ImportDeclaration",
                        );
                        for (const imported of astImports) {
                            for (const importedSpecifier of imported.specifiers ?? []) {
                                if (importedSpecifier.local.name !== local) continue;
                                const importTarget = resolveModule(file, imported.source.value);
                                const importedName =
                                    importedSpecifier.type === "ImportDefaultSpecifier"
                                        ? "default"
                                        : importedSpecifier.type === "ImportNamespaceSpecifier"
                                          ? "*"
                                          : (importedSpecifier.imported?.name ?? local);
                                if (importTarget)
                                    result.set(name, [{ file: importTarget, name: importedName }]);
                            }
                        }
                    }
                }
            }
            return result;
        }

        function declarationsForNode(node: any): [string, object[]][] {
            if (!node) return [];
            if (node.type === "VariableDeclaration")
                return node.declarations
                    .filter((item: any) => item.id?.type === "Identifier")
                    .map((item: any) => [item.id.name, [node]]);
            if (node.id?.name) return [[node.id.name, [node]]];
            return [];
        }

        function inspect(file: string, name: string, via: string): void {
            const key = `${resolve(file)}#${name}`;
            if (visitedTypes.has(key)) return;
            visitedTypes.add(key);
            if (name === "*") {
                for (const targets of exports(file).values())
                    for (const target of targets) inspect(target.file, target.name, via);
                return;
            }
            const publicTargets = exports(file).get(name);
            const targets = publicTargets ?? [{ file, name }];
            for (const target of targets) {
                const targetFile = resolve(target.file);
                const targetName = target.name;
                if (isHandleDeclaration(targetFile)) {
                    errors.add(
                        `${via} exposes solver handle ${targetName} from ${normalizePath(relative(ROOT, targetFile))}`,
                    );
                    continue;
                }
                const ast = load(targetFile);
                if (!ast) continue;
                const decls = declarations(targetFile).get(targetName) ?? [];
                for (const declaration of decls) {
                    visit(declaration, (node) => {
                        if (node.type === "TSImportType") {
                            const specifier = node.argument?.value;
                            const module =
                                typeof specifier === "string"
                                    ? resolveModule(targetFile, specifier)
                                    : undefined;
                            const importedName = idName(node.qualifier);
                            if (module && importedName) inspect(module, importedName, via);
                        }
                        if (
                            node.type === "TSTypeReference" ||
                            node.type === "TSExpressionWithTypeArguments"
                        ) {
                            const reference = node.typeName ?? node.expression;
                            const local = idName(reference);
                            if (!local) return;
                            const currentAst = load(targetFile);
                            const importNode = currentAst?.program.body.find(
                                (item: any) =>
                                    item.type === "ImportDeclaration" &&
                                    item.specifiers?.some((s: any) => s.local.name === local),
                            );
                            if (importNode) {
                                const specifier = importNode.source.value;
                                const module = resolveModule(targetFile, specifier);
                                const binding = importNode.specifiers.find(
                                    (s: any) => s.local.name === local,
                                );
                                const importedName =
                                    binding.type === "ImportDefaultSpecifier"
                                        ? "default"
                                        : binding.type === "ImportNamespaceSpecifier"
                                          ? "*"
                                          : (binding.imported?.name ?? local);
                                if (module) inspect(module, importedName, via);
                            } else {
                                const localDecl = declarations(targetFile).get(local);
                                if (localDecl) inspect(targetFile, local, via);
                            }
                        }
                        if (node.type === "TSTypeQuery") {
                            const local = idName(node.exprName);
                            if (local) {
                                const localDecl = declarations(targetFile).get(local);
                                if (localDecl) inspect(targetFile, local, via);
                            }
                        }
                    });
                }
            }
        }

        for (let i = 0; i < entries.length; i++) {
            const entry = entries[i];
            const declarationRoots = [
                roots[i],
                ...(entry.publishedDeclaration ? [entry.publishedDeclaration] : []),
            ];
            for (const root of declarationRoots) {
                if (!load(root))
                    throw new Error(
                        `could not read declaration for package export ${entry.name}: ${root}`,
                    );
                for (const targets of exports(root).values())
                    for (const target of targets) inspect(target.file, target.name, entry.name);
            }
        }
        expect(
            [...errors],
            `solver handles leak through ${entries.length} emitted package declarations`,
        ).toEqual([]);
    } finally {
        rmSync(scratch, { recursive: true, force: true });
    }
});
