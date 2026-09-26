import { existsSync, readFileSync } from "node:fs";
import { builtinModules } from "node:module";
import { dirname, relative, resolve, sep } from "node:path";
import { parse } from "@babel/parser";
import { Glob } from "bun";

const GAME_TIERS = ["engine", "core", "standard", "extras"] as const;
const TIER_ORDER = new Map(GAME_TIERS.map((tier, index) => [tier, index]));
const TOOLING = new Set(["project", "cli", "native", "harness", "types"]);
const MODULE_TIERS = new Set(["core", "standard", "extras"]);
// The runtime floor is a host-only leaf; Vite is public only through this package subpath.
const DIRECT_LEAVES = new Set(["engine/runtime/floor.ts"]);
const PUBLIC_ENTRIES = new Map([["@dylanebert/shallot/vite", "project/vite.ts"]]);
const SOURCE_ROOT = "src";
const TSC = resolve(import.meta.dir, "../node_modules/.bin/tsc");
const RUNTIME_MODULES = new Set([
    ...builtinModules.flatMap((name) => [name, name.startsWith("node:") ? name : `node:${name}`]),
    "bun:ffi",
    "bun:sqlite",
    "bun:test",
]);

interface ImportReference {
    readonly specifier: string;
    readonly line: number;
}

interface ModuleResolution {
    readonly target: string | null;
    readonly complete: boolean;
}

interface Module {
    readonly tier: string;
    readonly name: string;
    readonly directory: string;
    readonly kind: "game" | "tooling" | "transitional";
}

function walk(node: unknown, visit: (node: any) => void): void {
    if (node === null || typeof node !== "object") return;
    if (Array.isArray(node)) {
        for (const child of node) walk(child, visit);
        return;
    }
    visit(node);
    for (const [key, value] of Object.entries(node)) {
        if (key === "loc" || key === "start" || key === "end" || key === "extra") continue;
        walk(value, visit);
    }
}

/** Read module specifiers from declarations, re-exports, type queries and dynamic imports. */
export function references(source: string, file = "<source>"): ImportReference[] {
    let ast: any;
    try {
        ast = parse(source, {
            sourceType: "unambiguous",
            plugins: [
                ["typescript", { dts: /\.d\.(?:ts|mts|cts)$/.test(file) }],
                ...(file.endsWith(".tsx") ? (["jsx"] as const) : []),
                "decorators-legacy" as const,
            ],
        });
    } catch (error) {
        throw new Error(`${file}: cannot parse: ${(error as Error).message}`);
    }

    const found: ImportReference[] = [];
    const add = (node: any, literal: any): void => {
        if (typeof literal?.value === "string")
            found.push({ specifier: literal.value, line: node.loc?.start.line ?? 1 });
    };
    walk(ast.program, (node) => {
        if (
            (node.type === "ImportDeclaration" ||
                node.type === "ExportNamedDeclaration" ||
                node.type === "ExportAllDeclaration") &&
            node.source
        ) {
            add(node, node.source);
        } else if (node.type === "TSImportType") {
            add(node, node.source ?? node.argument);
        } else if (node.type === "ImportExpression") {
            add(node, node.source);
        } else if (
            node.type === "CallExpression" &&
            node.callee?.type === "Import" &&
            node.arguments?.length > 0
        ) {
            add(node, node.arguments[0]);
        }
    });
    return found.sort((a, b) => a.line - b.line);
}

function moduleAt(src: string, file: string): Module | undefined {
    const path = relative(src, file).split(sep).join("/");
    const parts = path.split("/");
    const top = parts[0];
    if (TOOLING.has(top))
        return { tier: top, name: top, directory: resolve(src, top), kind: "tooling" };
    if (top === "transitional" && parts.length > 1)
        return {
            tier: top,
            name: parts[1],
            directory: resolve(src, top, parts[1]),
            kind: "transitional",
        };
    if ((GAME_TIERS as readonly string[]).includes(top) && parts.length > 1) {
        if (parts.length === 2 && /^index\.tsx?$/.test(parts[1])) return undefined;
        return {
            tier: top,
            name: parts[1],
            directory: resolve(src, top, parts[1]),
            kind: "game",
        };
    }
    return undefined;
}

function gameTierAt(src: string, file: string): string | undefined {
    const top = relative(src, file).split(sep)[0];
    return TIER_ORDER.has(top as (typeof GAME_TIERS)[number]) ? top : undefined;
}

// TypeScript 7 no longer exports the legacy JS resolver; its compiler trace uses the same resolver and project options.
function moduleResolutions(root: string): Map<string, ModuleResolution> {
    const project = resolve(root, "tsconfig.json");
    const result = Bun.spawnSync(
        [TSC, "--project", project, "--noEmit", "--traceResolution", "--pretty", "false"],
        { cwd: root, stdout: "pipe", stderr: "pipe" },
    );
    const resolutions = new Map<string, ModuleResolution>();
    let current: { key: string; specifier: string } | undefined;
    for (const line of result.stdout.toString().split("\n")) {
        const start = line.match(/^======== Resolving module '(.+)' from '(.+)'\. ========$/);
        if (start) {
            const key = `${resolve(start[2])}\u0000${start[1]}`;
            if (!resolutions.has(key)) resolutions.set(key, { target: null, complete: false });
            current = { key, specifier: start[1] };
            continue;
        }
        const resolved = line.match(
            /^======== Module name '(.+)' was successfully resolved to '(.+?)'(?: with Package ID .*)?\. ========$/,
        );
        if (current && resolved && resolved[1] === current.specifier) {
            resolutions.set(current.key, { target: resolve(resolved[2]), complete: true });
            current = undefined;
        } else {
            const failed = line.match(/^======== Module name '(.+)' was not resolved\. ========$/);
            if (current && failed && failed[1] === current.specifier) {
                resolutions.set(current.key, { target: null, complete: true });
                current = undefined;
            }
        }
    }
    return resolutions;
}

function modulePath(module: Module): string {
    return module.kind === "tooling" ? module.name : `${module.tier}/${module.name}`;
}

function isModuleEntry(module: Module, target: string): boolean {
    return (
        target === resolve(module.directory, "index.ts") ||
        target === resolve(module.directory, "index.tsx")
    );
}

function transitionalRed(src: string, module: Module): string {
    const index = resolve(module.directory, "index.ts");
    const source = existsSync(index) ? readFileSync(index, "utf8") : "";
    const lines = source.split("\n");
    const line = lines.findIndex((entry) => entry.startsWith("// Destination:"));
    const destination = line < 0 ? "// Destination: not recorded." : lines[line];
    return `${relative(dirname(src), index).split(sep).join("/")}:${line < 0 ? 1 : line + 1}: ${destination}`;
}

/** Return all import-boundary reds in a source tree. */
export function checkImports(root: string): string[] {
    const src = resolve(root, SOURCE_ROOT);
    const files = [...new Glob("**/*.{ts,tsx,mts,cts}").scanSync(src)]
        .filter((file) => !/\.test\.(?:ts|tsx|mts|cts)$/.test(file))
        .map((file) => resolve(src, file))
        .sort();
    const resolutions = moduleResolutions(root);
    const violations: string[] = [];
    const transitional = new Map<string, Module>();

    for (const file of files) {
        const sourceModule = moduleAt(src, file);
        const transition = sourceModule?.kind === "transitional";
        if (sourceModule?.kind === "transitional")
            transitional.set(sourceModule.directory, sourceModule);
        const sourceTier = gameTierAt(src, file);
        const path = relative(root, file).split(sep).join("/");
        for (const reference of references(readFileSync(file, "utf8"), path)) {
            const key = `${file}\u0000${reference.specifier}`;
            const resolution = resolutions.get(key);
            if (
                !resolution?.complete ||
                (!resolution.target && !RUNTIME_MODULES.has(reference.specifier))
            ) {
                violations.push(
                    `${path}:${reference.line}: unresolved import "${reference.specifier}"`,
                );
                continue;
            }
            if (transition) continue;
            const target = resolution.target;
            if (!target || (target !== src && !target.startsWith(`${src}${sep}`))) continue;
            const targetModule = moduleAt(src, target);
            const targetTier = gameTierAt(src, target);
            const location = `${path}:${reference.line}`;

            if (sourceTier && targetModule?.kind === "tooling") {
                violations.push(
                    `${location}: game module ${sourceModule ? modulePath(sourceModule) : sourceTier} imports tooling module ${targetModule.name}`,
                );
                continue;
            }
            if (
                sourceTier &&
                targetTier &&
                TIER_ORDER.get(targetTier as (typeof GAME_TIERS)[number])! >
                    TIER_ORDER.get(sourceTier as (typeof GAME_TIERS)[number])!
            ) {
                violations.push(
                    `${location}: ${sourceTier} imports outward to ${targetTier}/${targetModule?.name ?? "index"}`,
                );
                continue;
            }
            if (
                sourceModule?.kind === "game" &&
                targetModule?.kind === "game" &&
                MODULE_TIERS.has(sourceModule.tier) &&
                sourceModule.tier === targetModule.tier &&
                sourceModule.name !== targetModule.name
            ) {
                violations.push(
                    `${location}: sibling import ${sourceModule.tier}/${sourceModule.name} → ${targetModule.tier}/${targetModule.name}`,
                );
                continue;
            }
            if (
                sourceModule?.kind === "game" &&
                sourceModule.name === "physics" &&
                targetModule?.kind === "game" &&
                targetModule.name === "rendering"
            ) {
                violations.push(
                    `${location}: physics module ${sourceModule.tier}/physics imports rendering module ${targetModule.tier}/rendering`,
                );
                continue;
            }
            const targetPath = relative(src, target).split(sep).join("/");
            if (
                sourceModule &&
                targetModule &&
                sourceModule.directory !== targetModule.directory &&
                !isModuleEntry(targetModule, target) &&
                !DIRECT_LEAVES.has(targetPath) &&
                PUBLIC_ENTRIES.get(reference.specifier) !== targetPath
            ) {
                violations.push(
                    `${location}: import past ${modulePath(targetModule)}/index.ts → ${relative(src, target).split(sep).join("/")}`,
                );
            }
        }
    }

    for (const module of [...transitional.values()].sort((a, b) => a.name.localeCompare(b.name)))
        violations.push(transitionalRed(src, module));
    return violations;
}

if (import.meta.main) {
    const args = Bun.argv.slice(2);
    const rootArg = args.indexOf("--root");
    const root = resolve(rootArg < 0 ? process.cwd() : (args[rootArg + 1] ?? process.cwd()));
    const violations = checkImports(root);
    for (const violation of violations) console.error(violation);
    if (violations.length > 0) process.exit(1);
    console.log("imports pass");
}
