import { copyFileSync, mkdtempSync, symlinkSync } from "node:fs";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import { shallot } from "@dylanebert/shallot/bun";
import { removeTestDirectory } from "../../scripts/test-directory";

const root = resolve(import.meta.dir, "../..");
const directory = mkdtempSync(join(tmpdir(), "shallot-benchmark-profile-"));
try {
    const built = await Bun.build({
        entrypoints: [join(root, "examples/physics/tests/benchmark-profile.oracle.ts")],
        target: "node",
        format: "esm",
        outdir: directory,
        naming: "profile.mjs",
        external: ["webgpu"],
        plugins: [shallot({ root })],
    });
    if (!built.success) throw new Error(built.logs.map(String).join("\n"));
    symlinkSync(join(root, "node_modules"), join(directory, "node_modules"), "junction");
    copyFileSync(join(root, "src/extras/text/Inter-Regular.ttf"), join(directory, "Inter-Regular.ttf"));
    const child = Bun.spawn(["node", join(directory, "profile.mjs")], {
        stdin: "inherit", stdout: "inherit", stderr: "inherit",
    });
    process.exitCode = await child.exited;
} finally {
    removeTestDirectory(directory);
}
