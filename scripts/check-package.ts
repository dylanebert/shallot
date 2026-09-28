import { existsSync, mkdtempSync, readFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";

const root = resolve(import.meta.dir, "..");
const manifest = JSON.parse(readFileSync(resolve(root, "package.json"), "utf8"));
const packageFile = `${manifest.name.replace(/^@/, "").replace("/", "-")}-${manifest.version}.tgz`;
const scratch = mkdtempSync(join(tmpdir(), "shallot-package-lint-"));
const tarball = join(scratch, packageFile);

function run(label: string, command: string[]): string {
    const start = performance.now();
    const result = Bun.spawnSync(command, {
        cwd: root,
        stdout: "pipe",
        stderr: "pipe",
    });
    const seconds = ((performance.now() - start) / 1000).toFixed(1);
    if (!result.success) {
        process.stdout.write(result.stdout);
        process.stderr.write(result.stderr);
        throw new Error(`${label} failed (${seconds}s)`);
    }
    return seconds;
}

try {
    const packSeconds = run("bun pm pack", [
        "bun",
        "pm",
        "pack",
        "--destination",
        scratch,
        "--quiet",
    ]);
    if (!existsSync(tarball)) throw new Error(`packed tarball is missing: ${tarball}`);

    const publintSeconds = run("publint", [
        "publint",
        "run",
        tarball,
        "--strict",
        "--pack",
        "false",
    ]);
    // ATTW's ignore is global: it accepts the TypeScript-source Node16 gap owned by publish.md,
    // but could hide a broken bundler import too. The packed-install test typechecks every export
    // under bundler resolution to cover that loss.
    const attwSeconds = run("@arethetypeswrong/cli", ["attw", tarball]);

    console.log(
        `✓ one packed tarball shared by publint and @arethetypeswrong/cli (${packSeconds}s)`,
    );
    console.log(`✓ publint (${publintSeconds}s)`);
    console.log(`✓ @arethetypeswrong/cli (${attwSeconds}s)`);
} finally {
    rmSync(scratch, { recursive: true, force: true });
}
