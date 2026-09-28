import { mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";

const ROOT = resolve(import.meta.dir, "..");

export function buildViteDeclaration(): string {
    const scratch = mkdtempSync(join(tmpdir(), "shallot-vite-declaration-"));
    const outdir = join(scratch, "out");
    const config = join(scratch, "tsconfig.json");
    writeFileSync(
        config,
        JSON.stringify({
            extends: resolve(ROOT, "tsconfig.json"),
            compilerOptions: {
                declaration: true,
                emitDeclarationOnly: true,
                noEmit: false,
                outDir: outdir,
                rootDir: resolve(ROOT, "src/project"),
                skipLibCheck: true,
                typeRoots: [resolve(ROOT, "node_modules/@types"), resolve(ROOT, "node_modules")],
            },
            files: [resolve(ROOT, "src/project/vite.ts")],
            include: [],
        }),
    );

    try {
        const result = Bun.spawnSync(
            ["node", resolve(ROOT, "node_modules/typescript/bin/tsc"), "-p", config],
            { cwd: ROOT, stdout: "pipe", stderr: "pipe" },
        );
        if (result.exitCode !== 0) {
            throw new Error(
                `build-vite-types: declaration emit failed:\n${result.stdout.toString()}${result.stderr.toString()}`,
            );
        }
        const declaration = readFileSync(join(outdir, "vite.d.ts"), "utf8");
        const imports = [
            ...declaration.matchAll(
                /(?:\bfrom\s+|\bimport\s*\(\s*|<reference\s+(?:path|types)=)["']([^"']+)["']/g,
            ),
        ].map(([, specifier]) => specifier);
        const unexpected = imports.filter((specifier) => specifier !== "vite");
        if (unexpected.length) {
            throw new Error(
                `build-vite-types: dist/vite.d.ts may depend only on Vite types, found: ${[...new Set(unexpected)].join(", ")}`,
            );
        }
        return declaration;
    } finally {
        rmSync(scratch, { recursive: true, force: true });
    }
}
