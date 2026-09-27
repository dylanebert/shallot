import { existsSync, readFileSync } from "node:fs";
import { dirname, relative, resolve, sep } from "node:path";
import { Glob } from "bun";

const SKIP = new Set([".git", ".cache", "node_modules", "fixtures", "target", "dist", "coverage"]);
const DEEP_RECIPE_IMPORT = /(?:from\s+|import\s*\(\s*)["'][^"']*\/src(?:\/|["'])/;
const PHYSICS_WORLD_ESCAPE = /\bPhysics\.world\b|\bphysicsWorld\s*\(/;

function relativeFile(root: string, path: string): string {
    return relative(root, path).split(sep).join("/");
}

function recipeSourceViolations(root: string): string[] {
    const violations: string[] = [];
    const manifests = new Glob("**/shallot.json").scanSync({ cwd: root, dot: false });
    for (const relativeManifest of manifests) {
        if (relativeManifest.split("/").some((part) => SKIP.has(part))) continue;
        const manifestPath = resolve(root, relativeManifest);
        let manifest: unknown;
        try {
            manifest = JSON.parse(readFileSync(manifestPath, "utf8"));
        } catch {
            continue;
        }
        if (
            !manifest ||
            typeof manifest !== "object" ||
            (manifest as { kind?: unknown }).kind !== "recipe"
        )
            continue;

        const sourceRoot = resolve(dirname(manifestPath), "src");
        if (!existsSync(sourceRoot)) continue;
        for (const match of new Glob("**/*.ts").scanSync({ cwd: sourceRoot, dot: false })) {
            const path = resolve(sourceRoot, match);
            const file = relativeFile(root, path);
            const source = readFileSync(path, "utf8");
            if (DEEP_RECIPE_IMPORT.test(source))
                violations.push(
                    `recipe source uses a deep engine import: ${file}; import only from package exports`,
                );
            if (PHYSICS_WORLD_ESCAPE.test(source))
                violations.push(
                    `recipe source uses Physics.world/physicsWorld: ${file}; use the State-scoped public seam`,
                );
        }
    }
    return violations;
}

function shallotPackage(name: string): boolean {
    return name === "@dylanebert/shallot" || name.startsWith("@dylanebert/shallot-");
}

type ForbiddenSpecifier = "link" | "file" | "git" | "github" | "URL" | "workspace" | "portal";
const FULL_COMMIT = /^[0-9a-f]{40}$/i;
const PUBLISHED_RANGE = /^(?:[vV]?\d|[~^<>=*|])/;
const REMOTE_TARBALL = /^https?:\/\/[^\s]+\.(?:tgz|tar\.gz)(?:[?#].*)?$/i;
const LOCAL_TARBALL = /\.(?:tgz|tar\.gz)$/i;

function forbiddenSpecifier(spec: string): ForbiddenSpecifier | null {
    if (spec.startsWith("link:")) return "link";
    if (spec.startsWith("file:")) return "file";
    if (spec.startsWith("workspace:")) return "workspace";
    if (spec.startsWith("portal:")) return "portal";
    if (spec.startsWith("github:")) return "github";
    if (spec.startsWith("git")) return "git";
    try {
        if (new URL(spec).protocol) return "URL";
    } catch {}
    return null;
}

function gitCommit(spec: string): string | null {
    const hash = spec.slice(spec.lastIndexOf("#") + 1);
    return spec.includes("#") && FULL_COMMIT.test(hash) ? hash : null;
}

function lockText(root: string): string | null {
    const path = resolve(root, "bun.lock");
    return existsSync(path) ? readFileSync(path, "utf8") : null;
}

function lockHas(lock: string | null, spec: string, integrity: boolean): boolean {
    if (lock === null) return false;
    const start = lock.indexOf(spec);
    if (start < 0) return false;
    if (!integrity) return true;
    return /sha(?:256|512)-[A-Za-z0-9+/=_-]+/.test(lock.slice(start, start + 2048));
}

function checkedTarballViolation(root: string, file: string, spec: string): string | null {
    const relativePath = spec.slice("file:".length);
    if (relativePath.startsWith("/") || relativePath.split("/").includes(".."))
        return `${file}: local Shallot tarball path escapes the project: ${JSON.stringify(spec)}`;
    if (!LOCAL_TARBALL.test(relativePath))
        return `${file}: local Shallot source directory is not a sanctioned artifact: ${JSON.stringify(spec)}`;
    const tarball = resolve(root, relativePath);
    if (!existsSync(tarball))
        return `${file}: checked-in Shallot tarball is missing: ${relativePath}`;
    const digest = ["sha256", "sha512"]
        .map((algorithm) => `${tarball}.${algorithm}`)
        .find((sidecar) => existsSync(sidecar));
    if (
        digest === undefined ||
        !new RegExp("^[0-9a-f]{" + (digest.endsWith("sha256") ? 64 : 128) + "}", "im").test(
            readFileSync(digest, "utf8"),
        )
    )
        return `${file}: checked-in Shallot tarball needs a hexadecimal digest sidecar: ${relativePath}`;
    const provenance = `${tarball}.source-commit`;
    if (!existsSync(provenance) || !FULL_COMMIT.test(readFileSync(provenance, "utf8").trim()))
        return `${file}: checked-in Shallot tarball needs a full source-commit sidecar: ${relativePath}`;
    return null;
}

function dependencyViolations(root: string): string[] {
    const violations: string[] = [];
    const lock = lockText(root);
    const files = [...new Glob("**/package.json").scanSync({ cwd: root, dot: false })]
        .filter((file) => !file.split("/").some((part) => SKIP.has(part)))
        .sort();
    for (const file of files) {
        const path = resolve(root, file);
        let manifest: Record<string, unknown>;
        try {
            const parsed: unknown = JSON.parse(readFileSync(path, "utf8"));
            if (parsed === null || typeof parsed !== "object" || Array.isArray(parsed)) continue;
            manifest = parsed as Record<string, unknown>;
        } catch (error) {
            violations.push(`invalid package manifest: ${file}: ${(error as Error).message}`);
            continue;
        }
        const packageName = typeof manifest.name === "string" ? manifest.name : null;
        for (const [table, raw] of Object.entries(manifest)) {
            if (!/dependencies$/i.test(table) || raw === null || typeof raw !== "object") continue;
            for (const [name, value] of Object.entries(raw as Record<string, unknown>)) {
                if (!shallotPackage(name) || typeof value !== "string") continue;
                if (name === packageName && value === "link:.") continue;
                const kind = forbiddenSpecifier(value);
                if (kind === "link" || kind === "workspace" || kind === "portal") {
                    violations.push(
                        `${file}: ${table}.${name} uses forbidden ${kind} specifier ${JSON.stringify(value)}`,
                    );
                    continue;
                }
                if (kind === "file") {
                    const violation = checkedTarballViolation(root, file, value);
                    if (violation !== null) violations.push(violation);
                    continue;
                }
                if (kind === "git" || kind === "github") {
                    const commit = gitCommit(value);
                    if (commit === null)
                        violations.push(
                            `${file}: ${table}.${name} requires a full 40-hex Git commit, got ${JSON.stringify(value)}`,
                        );
                    else if (!lockHas(lock, value, false))
                        violations.push(
                            `${file}: ${table}.${name} Git identity is not recorded in bun.lock: ${JSON.stringify(value)}`,
                        );
                    continue;
                }
                if (kind === "URL") {
                    if (!REMOTE_TARBALL.test(value))
                        violations.push(
                            `${file}: ${table}.${name} uses an unqualified Shallot URL: ${JSON.stringify(value)}`,
                        );
                    else if (!lockHas(lock, value, true))
                        violations.push(
                            `${file}: ${table}.${name} remote tarball needs matching lock integrity: ${JSON.stringify(value)}`,
                        );
                    continue;
                }
                if (!PUBLISHED_RANGE.test(value))
                    violations.push(
                        `${file}: ${table}.${name} uses mutable dist-tag ${JSON.stringify(value)}`,
                    );
            }
        }
    }
    return violations;
}

export function readProjectPolicy(root: string): string[] {
    return [...recipeSourceViolations(root), ...dependencyViolations(root)];
}

