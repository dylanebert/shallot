import { expect, setDefaultTimeout, test } from "bun:test";

import { TIMEOUT } from "@dylanebert/shallot/testing";

setDefaultTimeout(TIMEOUT.node);

import { readdirSync, readFileSync } from "node:fs";
import { dirname, isAbsolute, relative, resolve } from "node:path";
import { packageManifest } from "../src/cli/create";
import { normalizePath } from "../src/project/paths";
import {
    importsIn as collectImports,
    componentImportsIn,
    resolveRelative,
} from "./example-imports";

const ROOT = resolve(import.meta.dir, "..");
const EXAMPLES = resolve(ROOT, "examples");
const ENGINE_MANIFEST = JSON.parse(readFileSync(resolve(ROOT, "package.json"), "utf8")) as {
    name: string;
    exports: Record<string, unknown>;
};

function sourceFiles(directory: string): string[] {
    return readdirSync(directory, { withFileTypes: true }).flatMap((entry) => {
        const path = resolve(directory, entry.name);
        if (entry.isDirectory()) {
            if (["build", "dist", "node_modules"].includes(entry.name)) return [];
            return sourceFiles(path);
        }
        return entry.isFile() && /\.(?:[cm]?[jt]sx?|html|svelte)$/.test(entry.name) ? [path] : [];
    });
}

test("interact carries unchanged copies of first-person's controller over published imports", () => {
    for (const file of ["player.ts", "motion.ts", "follow.ts"]) {
        expect(readFileSync(resolve(EXAMPLES, "interact/src", file), "utf8")).toBe(
            readFileSync(resolve(EXAMPLES, "first-person/src", file), "utf8"),
        );
    }
});

function moduleScripts(source: string): string[] {
    const scripts = source.matchAll(/<script\b([^>]*)>([\s\S]*?)<\/script\s*>/gi);
    return [...scripts]
        .filter(([, attributes]) =>
            /\btype\s*=\s*(?:"module"|'module'|module\b)/i.test(attributes!),
        )
        .map(([, , script]) => script!);
}

function fileImports(file: string): { specifiers: string[]; unresolved: string[] } {
    const source = readFileSync(file, "utf8");
    if (file.endsWith(".svelte")) return componentImportsIn(source, file);
    const sources = file.endsWith(".html") ? moduleScripts(source) : [source];
    const scans = sources.map((code) => collectImports(code, file));
    return {
        specifiers: [...new Set(scans.flatMap((scan) => scan.specifiers))],
        unresolved: [...new Set(scans.flatMap((scan) => scan.unresolved))],
    };
}

function packageName(specifier: string): string {
    const parts = specifier.split("/");
    return specifier.startsWith("@") ? parts.slice(0, 2).join("/") : parts[0]!;
}

function hasEngineExport(specifier: string): boolean {
    const subpath =
        specifier === ENGINE_MANIFEST.name
            ? "."
            : `./${specifier.slice(ENGINE_MANIFEST.name.length + 1)}`;
    return Object.keys(ENGINE_MANIFEST.exports).some((entry) => {
        if (entry === subpath) return true;
        const wildcard = entry.indexOf("*");
        return (
            wildcard >= 0 &&
            subpath.startsWith(entry.slice(0, wildcard)) &&
            subpath.endsWith(entry.slice(wildcard + 1))
        );
    });
}

function createdDependencies(exampleRoot: string, name: string): Set<string> {
    const created = JSON.parse(
        packageManifest(resolve(ROOT, ".created", name), exampleRoot, ROOT),
    ) as {
        dependencies: Record<string, string>;
        devDependencies: Record<string, string>;
    };
    return new Set([...Object.keys(created.dependencies), ...Object.keys(created.devDependencies)]);
}

function importViolations(): string[] {
    const violations: string[] = [];
    for (const entry of readdirSync(EXAMPLES, { withFileTypes: true }).filter((item) =>
        item.isDirectory(),
    )) {
        const exampleRoot = resolve(EXAMPLES, entry.name);
        const files = sourceFiles(exampleRoot);
        const importsByFile = new Map(files.map((file) => [file, fileImports(file)]));
        const dependencies = createdDependencies(exampleRoot, entry.name);

        for (const file of files) {
            const relativeFile = normalizePath(relative(ROOT, file));
            const imports = importsByFile.get(file)!;
            for (const unresolved of imports.unresolved)
                violations.push(`${relativeFile} -> ${unresolved}`);
            for (const specifier of imports.specifiers) {
                if (specifier.startsWith("./") || specifier.startsWith("../")) {
                    const base = resolve(dirname(file), specifier.split(/[?#]/, 1)[0]!);
                    const withinExample = normalizePath(relative(exampleRoot, base));
                    if (
                        withinExample === ".." ||
                        withinExample.startsWith("../") ||
                        isAbsolute(withinExample)
                    ) {
                        violations.push(
                            `${relativeFile} -> ${specifier} resolves outside its example`,
                        );
                        continue;
                    }
                    const target = resolveRelative(file, specifier);
                    if (!target) {
                        violations.push(
                            `${relativeFile} -> ${specifier} does not resolve inside its example`,
                        );
                        continue;
                    }
                    const targetRelative = normalizePath(relative(exampleRoot, target));
                    if (
                        targetRelative === ".." ||
                        targetRelative.startsWith("../") ||
                        isAbsolute(targetRelative)
                    )
                        violations.push(
                            `${relativeFile} -> ${specifier} resolves outside its example`,
                        );
                    continue;
                }
                if (specifier.startsWith("node:") || specifier.startsWith("bun:")) {
                    try {
                        if (Bun.resolveSync(specifier, ROOT) !== specifier)
                            violations.push(`${relativeFile} -> ${specifier} is not a builtin`);
                    } catch {
                        violations.push(`${relativeFile} -> ${specifier} is not a builtin`);
                    }
                    continue;
                }
                if (specifier.startsWith("/") || specifier.startsWith("#")) {
                    violations.push(`${relativeFile} -> ${specifier} is not a published import`);
                    continue;
                }
                const dependency = packageName(specifier);
                if (!dependencies.has(dependency))
                    violations.push(
                        `${relativeFile} -> ${specifier} is not declared by the created manifest`,
                    );
                if (dependency === ENGINE_MANIFEST.name && !hasEngineExport(specifier))
                    violations.push(
                        `${relativeFile} -> ${specifier} is not an engine package export`,
                    );
            }
        }
    }
    return [...new Set(violations)].sort();
}

test("every example import stays inside its copy or uses a created public dependency", () => {
    expect(importViolations()).toEqual([]);
});
