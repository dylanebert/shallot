import { Glob } from "bun";
import { dirname, join, relative, resolve } from "path";

const src = resolve(import.meta.dir, "../packages/shallot/src");
const pkg = await Bun.file(resolve(import.meta.dir, "../packages/shallot/package.json")).json();

// Extract allowed subpath targets from package.json exports
// e.g. "./render/core" → "standard/render/core" (relative to src/)
const allowedSubpaths = new Set<string>();
for (const [, target] of Object.entries(pkg.exports as Record<string, string>)) {
    if (target.startsWith("./src/") && !target.endsWith("index.ts") && !target.includes("*")) {
        allowedSubpaths.add(target.replace("./src/", "").replace(".ts", ""));
    }
}

// Module = directory directly under engine/, standard/, extras/, or document/
// e.g. "standard/render", "engine/ecs", "extras/orbit"
function getModule(fileRelative: string): string | null {
    const parts = fileRelative.split("/");
    if (parts.length < 2) return null;
    const group = parts[0];
    if (["engine", "standard", "extras", "document"].includes(group) && parts.length >= 2) {
        return `${group}/${parts[1]}`;
    }
    return null;
}

function resolveImport(fromFile: string, specifier: string): string | null {
    if (!specifier.startsWith(".")) return null;
    const abs = resolve(src, dirname(fromFile), specifier);
    return relative(src, abs);
}

const importRe = /from\s+["']([^"']+)["']/g;

const violations: { file: string; line: number; import: string; target: string }[] = [];

const glob = new Glob("**/*.ts");
for await (const path of glob.scan({ cwd: src })) {
    // tests sit beside their source and legitimately cross modules
    if (path.endsWith(".test.ts")) continue;
    const full = join(src, path);
    const content = await Bun.file(full).text();
    const lines = content.split("\n");

    const fileModule = getModule(path);

    for (let i = 0; i < lines.length; i++) {
        for (const match of lines[i].matchAll(importRe)) {
            const specifier = match[1];
            const resolved = resolveImport(path, specifier);
            if (!resolved) continue;

            const targetModule = getModule(resolved);
            if (!targetModule) continue;
            if (targetModule === fileModule) continue;

            // Cross-module import — check if it targets a barrel or allowed subpath
            if (resolved.endsWith("/index") || resolved === targetModule) continue;
            if (allowedSubpaths.has(resolved)) continue;

            violations.push({
                file: path,
                line: i + 1,
                import: specifier,
                target: resolved,
            });
        }
    }
}

if (violations.length > 0) {
    console.warn(`⚠ ${violations.length} deep cross-module import(s):\n`);
    for (const v of violations) {
        console.warn(`  ${v.file}:${v.line}`);
        console.warn(`    import from "${v.import}" → ${v.target}`);
    }
    console.warn(
        "\nCross-module imports must use barrel exports (index.ts) or named subpaths from package.json exports.",
    );
}
