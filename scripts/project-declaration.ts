import { mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";

const ROOT = resolve(import.meta.dir, "..");
type ProjectTool = "testing" | "vite";

export function buildProjectDeclaration(entry: ProjectTool): string {
    const scratch = mkdtempSync(join(tmpdir(), `shallot-${entry}-declaration-`));
    const outdir = join(scratch, "out");
    const config = join(scratch, "tsconfig.json");
    const source = resolve(ROOT, `src/project/${entry}.js`);
    writeFileSync(
        config,
        JSON.stringify({
            extends: resolve(ROOT, "tsconfig.json"),
            compilerOptions: {
                allowJs: true,
                checkJs: true,
                declaration: true,
                emitDeclarationOnly: true,
                noEmit: false,
                outDir: outdir,
                rootDir: resolve(ROOT, "src"),
                skipLibCheck: true,
                typeRoots: [resolve(ROOT, "node_modules/@types"), resolve(ROOT, "node_modules")],
            },
            files: [source],
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
                `build-tooling: declaration emit for ${source} failed:\n${result.stdout.toString()}${result.stderr.toString()}`,
            );
        }
        const declaration = readFileSync(join(outdir, "project", `${entry}.d.ts`), "utf8");
        const imports = [
            ...declaration.matchAll(
                /(?:\bfrom\s+|\bimport\s*\(\s*|<reference\s+(?:path|types)=)["']([^"']+)["']/g,
            ),
        ].map(([, specifier]) => specifier);
        const allowed = entry === "vite" ? ["vite"] : ["playwright/test", "@dylanebert/shallot"];
        const unexpected = imports.filter((specifier) => !allowed.includes(specifier));
        if (unexpected.length) {
            throw new Error(
                `build-tooling: dist/${entry}.d.ts may depend only on ${allowed.join(", ")} types, found: ${[...new Set(unexpected)].join(", ")}`,
            );
        }
        return declaration;
    } finally {
        rmSync(scratch, { recursive: true, force: true });
    }
}
