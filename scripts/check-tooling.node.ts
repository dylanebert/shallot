import { setDefaultTimeout, test } from "bun:test";
import { existsSync, mkdtempSync, readFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
// This check validates dist/testing.js, so its own timeout policy must come from source rather than
// the generated package export it is responsible for checking.
import { CEILING } from "../src/project/testing";
import { buildViteDeclaration } from "./vite-declaration";

setDefaultTimeout(CEILING.node);

test("published tooling bundles and declarations are current", async () => {
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
            if (!existsSync(built)) {
                throw new Error(
                    `dist/${name}.js is missing; run \`bun run prepare\` to generate it.`,
                );
            }
            if (readFileSync(built, "utf8") !== fresh) {
                throw new Error(`dist/${name}.js is stale; run \`bun run prepare\` to refresh it.`);
            }
            console.log(`dist/${name}.js is fresh`);
        }

        const builtTypes = resolve(root, "dist/vite.d.ts");
        const freshTypes = buildViteDeclaration();
        if (!existsSync(builtTypes)) {
            throw new Error("dist/vite.d.ts is missing; run `bun run prepare` to generate it.");
        }
        if (readFileSync(builtTypes, "utf8") !== freshTypes) {
            throw new Error("dist/vite.d.ts is stale; run `bun run prepare` to refresh it.");
        }
        console.log("dist/vite.d.ts is fresh");
    } finally {
        rmSync(outdir, { recursive: true, force: true });
    }
});
