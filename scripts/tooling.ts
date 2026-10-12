import { mkdirSync, rmSync, writeFileSync } from "node:fs";
import { resolve } from "node:path";
import { buildProjectDeclaration } from "./project-declaration";

const ROOT = resolve(import.meta.dir, "..");
const OUT = resolve(ROOT, "dist");

// Recreate dist/ so a retired declaration cannot survive a later prepare.
rmSync(OUT, { recursive: true, force: true });
mkdirSync(OUT, { recursive: true });

for (const entry of ["testing", "vite"] as const) {
    writeFileSync(resolve(OUT, `${entry}.d.ts`), buildProjectDeclaration(entry));
}

console.log("build-tooling: generated dist/testing.d.ts, dist/vite.d.ts");
