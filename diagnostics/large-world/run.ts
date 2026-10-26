import { copyFileSync, mkdtempSync, symlinkSync } from "node:fs";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import { shallot } from "@dylanebert/shallot/bun";
import { removeTestDirectory } from "../../scripts/test-directory";

// bun diagnostics/large-world/run.ts [grid count, default 1000] [cpu|draw|shared] [workerCount: 1 shared, 0 unshared]
const root = resolve(import.meta.dir, "../..");
const dir = mkdtempSync(join(tmpdir(), "shallot-large-world-"));
try {
    const built = await Bun.build({
        entrypoints: [join(import.meta.dir, "cost.ts")],
        target: "node", format: "esm", outdir: dir, naming: "cost.mjs",
        external: ["webgpu"], plugins: [shallot({ root })],
    });
    if (!built.success) throw new Error(built.logs.map(String).join("\n"));
    symlinkSync(join(root, "node_modules"), join(dir, "node_modules"), "junction");
    copyFileSync(join(root, "src/extras/text/Inter-Regular.ttf"), join(dir, "Inter-Regular.ttf"));
    const child = Bun.spawn(["node", join(dir, "cost.mjs"), ...process.argv.slice(2)], { stdin: "inherit", stdout: "inherit", stderr: "inherit" });
    process.exitCode = await child.exited;
} finally { removeTestDirectory(dir); }
