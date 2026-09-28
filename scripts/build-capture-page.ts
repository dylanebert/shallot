import { mkdirSync } from "node:fs";
import { resolve } from "node:path";

const root = resolve(import.meta.dir, "..");
const outdir = resolve(root, ".artifacts");
mkdirSync(outdir, { recursive: true });

const result = await Bun.build({
    entrypoints: [resolve(import.meta.dir, "capture-page.ts")],
    outdir,
    target: "browser",
    format: "iife",
    naming: "capture-page.js",
    metafile: true,
});
if (!result.success) {
    for (const log of result.logs) console.error(log);
    throw new Error("failed to bundle the public capture export for the browser check");
}

const builtins = Object.values(result.metafile!.inputs)
    .flatMap((input) => input.imports)
    .map((edge) => edge.path)
    .filter((path) => /^(?:node:|bun:|fs$|path$)/.test(path));
if (builtins.length > 0) {
    throw new Error(`browser capture entry imports host-only modules: ${builtins.join(", ")}`);
}
