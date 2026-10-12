import { existsSync, mkdtempSync, readFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import { buildViteDeclaration } from "./vite-declaration";

const root = resolve(import.meta.dir, "..");
const outdir = mkdtempSync(join(tmpdir(), "shallot-tooling-check-"));
const external = [
    "vite",
    "unplugin-typegpu",
    "unplugin-typegpu/vite",
    "node:fs",
    "node:path",
    "fs",
    "path",
];
const entries = [
    { source: "src/project/vite.ts", name: "vite" },
    { source: "src/project/testing.ts", name: "testing" },
];

try {
    for (const { source, name } of entries) {
        const result = await Bun.build({
            entrypoints: [resolve(root, source)],
            outdir,
            target: "node",
            format: "esm",
            naming: `${name}.js`,
            external,
            minify: false,
        });
        if (!result.success) {
            for (const log of result.logs) console.error(log);
            throw new Error(`check-tooling: failed to compile ${source}`);
        }

        const built = resolve(root, `dist/${name}.js`);
        const fresh = readFileSync(resolve(outdir, `${name}.js`), "utf8");
        if (!existsSync(built) || readFileSync(built, "utf8") !== fresh) {
            console.error(`✗ dist/${name}.js is stale; run \`bun run build\` to regenerate it.`);
            process.exitCode = 1;
        } else {
            console.log(`dist/${name}.js is fresh`);
        }
    }

    const builtTypes = resolve(root, "dist/vite.d.ts");
    const freshTypes = buildViteDeclaration();
    if (!existsSync(builtTypes) || readFileSync(builtTypes, "utf8") !== freshTypes) {
        console.error("✗ dist/vite.d.ts is stale; run `bun run build` to regenerate it.");
        process.exitCode = 1;
    } else {
        console.log("dist/vite.d.ts is fresh");
    }
} finally {
    rmSync(outdir, { recursive: true, force: true });
}
