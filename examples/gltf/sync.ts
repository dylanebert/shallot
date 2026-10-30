import { createHash } from "node:crypto";
import { mkdir, readFile, rename, rm, writeFile } from "node:fs/promises";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import manifest from "./assets.json";

export interface AssetFile {
    path: string;
    sha256: string;
}

function verify(file: AssetFile, bytes: Uint8Array): void {
    const actual = createHash("sha256").update(bytes).digest("hex");
    if (actual !== file.sha256)
        throw new Error(
            `${file.path}: SHA-256 mismatch: expected ${file.sha256}, received ${actual}`,
        );
}

/** Reuses only verified files; a mismatch in cached or fetched bytes fails without replacing the destination. */
export async function syncFiles(
    files: readonly AssetFile[],
    source: URL,
    destination: string,
): Promise<void> {
    for (const file of files) {
        const path = join(destination, file.path);
        let bytes: Uint8Array | undefined;
        try {
            bytes = await readFile(path);
        } catch (error) {
            if ((error as NodeJS.ErrnoException).code !== "ENOENT") throw error;
        }
        if (bytes) {
            verify(file, bytes);
            continue;
        }
        const url = new URL(file.path, source);
        if (url.protocol === "file:") {
            bytes = await readFile(url);
        } else {
            const response = await fetch(url);
            if (!response.ok) throw new Error(`${file.path}: HTTP ${response.status}`);
            bytes = new Uint8Array(await response.arrayBuffer());
        }
        verify(file, bytes);
        await mkdir(dirname(path), { recursive: true });
        const partial = `${path}.${process.pid}.part`;
        try {
            await writeFile(partial, bytes);
            await rename(partial, path);
        } finally {
            await rm(partial, { force: true });
        }
    }
}

if (import.meta.main) {
    await syncFiles(
        manifest.models.flatMap((model) => model.files),
        new URL(`${manifest.commit}/`, manifest.source),
        fileURLToPath(new URL("./public/assets/", import.meta.url)),
    );
}
