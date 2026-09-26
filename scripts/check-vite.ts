import { existsSync, mkdtempSync, readFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";

const root = resolve(import.meta.dir, "..");
const outdir = mkdtempSync(join(tmpdir(), "shallot-vite-check-"));
const external = [
    "vite",
    "unplugin-typegpu",
    "unplugin-typegpu/vite",
    "node:fs",
    "node:path",
    "fs",
    "path",
];

try {
    const result = await Bun.build({
        entrypoints: [resolve(root, "src/project/vite.ts")],
        outdir,
        target: "node",
        format: "esm",
        naming: "vite.js",
        external,
        minify: false,
    });
    if (!result.success) {
        for (const log of result.logs) console.error(log);
        throw new Error("check-vite: failed to compile src/project/vite.ts");
    }

    const committed = resolve(root, "dist/vite.js");
    const fresh = readFileSync(resolve(outdir, "vite.js"), "utf8");
    if (!existsSync(committed) || readFileSync(committed, "utf8") !== fresh) {
        console.error("✗ dist/vite.js is stale; run `bun run build` to regenerate it.");
        process.exitCode = 1;
    } else {
        console.log("dist/vite.js is fresh");
    }
} finally {
    rmSync(outdir, { recursive: true, force: true });
}
