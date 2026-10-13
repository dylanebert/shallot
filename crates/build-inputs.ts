import { createHash } from "node:crypto";
import { existsSync, readFileSync, readdirSync } from "node:fs";
import { resolve } from "node:path";

/**
 * Fingerprints tree inputs, not machine-local toolchains or build outputs. Of `Cargo.lock`, only the
 * locked packages `lockRoot` reaches are inputs, as Cargo fingerprints a package by its own resolved
 * dependencies.
 */
export function buildInputHash(
    root: string,
    files: string[],
    sourceDirs: string[],
    lockRoot: string,
): string {
    const inputs = [...files];
    function sources(dir: string): void {
        for (const entry of readdirSync(resolve(root, dir), { withFileTypes: true })) {
            const path = `${dir}/${entry.name}`;
            if (entry.isDirectory()) sources(path);
            else inputs.push(path);
        }
    }
    for (const dir of sourceDirs) sources(dir);

    const hash = createHash("sha256");
    for (const path of inputs.sort()) {
        const bytes = readFileSync(resolve(root, path));
        hash.update(`${path}\0${bytes.length}\0`);
        hash.update(bytes);
    }
    // Of the Bun manifests, only Binaryen's locked package is a build input (scripts/wasm-opt.ts), so
    // an unrelated package.json or bun.lock edit leaves the artifacts fresh.
    const binaryen = readFileSync(resolve(root, "bun.lock"), "utf8").match(
        /^\s*"binaryen": \[.*$/m,
    )?.[0];
    if (!binaryen) throw new Error("bun.lock locks no binaryen package, which pins wasm-opt");
    hash.update(`bun.lock binaryen\0${binaryen.trim()}`);
    for (const entry of cargoLockClosure(readFileSync(resolve(root, "Cargo.lock"), "utf8"), lockRoot))
        hash.update(`Cargo.lock\0${entry}`);
    return hash.digest("hex");
}

/**
 * The `[[package]]` entries of a Cargo.lock that `rootName` reaches, sorted. A dependency is named
 * `name` when one version is locked, or `name version` when several are.
 */
export function cargoLockClosure(lock: string, rootName: string): string[] {
    const packages = lock
        .split(/^\[\[package\]\]$/m)
        .slice(1)
        .map((entry) => {
            const text = entry.trim();
            const field = (key: string) => text.match(new RegExp(`^${key} = "([^"]+)"`, "m"))?.[1];
            const dependencies = [...(text.match(/^dependencies = \[([^\]]*)\]/m)?.[1] ?? "").matchAll(/"([^"]+)"/g)].map(
                (match) => match[1].split(" "),
            );
            return { text, name: field("name"), version: field("version"), dependencies };
        });
    const reached = new Set<(typeof packages)[number]>();
    const visit = (name: string, version?: string) => {
        const matches = packages.filter((pkg) => pkg.name === name && (!version || pkg.version === version));
        if (matches.length !== 1) throw new Error(`Cargo.lock locks ${matches.length} packages for ${name} ${version ?? ""}`.trim());
        const [pkg] = matches;
        if (reached.has(pkg)) return;
        reached.add(pkg);
        for (const [depName, depVersion] of pkg.dependencies) visit(depName, depVersion);
    };
    visit(rootName);
    return [...reached].map((pkg) => pkg.text).sort();
}

export function buildArtifactIsFresh(
    root: string,
    artifact: string,
    stampFile: string,
    inputHash: string,
    stampLine = 1,
): boolean {
    if (!existsSync(resolve(root, artifact))) return false;
    try {
        const stamp = `// Build inputs: sha256:${inputHash}`;
        return readFileSync(resolve(root, stampFile), "utf8").split("\n", stampLine + 1)[stampLine] === stamp;
    } catch (error) {
        if ((error as NodeJS.ErrnoException).code === "ENOENT") return false;
        throw error;
    }
}
