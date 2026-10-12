import { createHash } from "node:crypto";
import { existsSync, readFileSync, readdirSync } from "node:fs";
import { resolve } from "node:path";

/** Fingerprints tree inputs, not machine-local toolchains or build outputs. */
export function buildInputHash(root: string, files: string[], sourceDirs: string[]): string {
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
    return hash.digest("hex");
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
