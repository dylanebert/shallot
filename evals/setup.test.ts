// S3 arm — evals/setup.ts stripTarball
//
// Invariant: the --bare arm strips AGENTS.md and MIGRATION.md from the tarball (not just
// examples/). The S2 fix added rmSync calls for both files inside stripTarball.
//
// stripTarball is hermetic (untar → delete → re-tar in a temp dir), so this arm builds a
// fixture tree containing package/AGENTS.md, package/MIGRATION.md, and package/examples/,
// tars it, runs the real exported function, untars the result, and asserts the stripped
// files are gone. No grep over source text — this is a behavioral test of the actual function.

import { expect, test } from "bun:test";
import { existsSync, mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { stripTarball } from "./setup";

function buildFixtureTarball(): string {
    // Build a fixture tree: package/ with AGENTS.md, MIGRATION.md, examples/, and a code file.
    const ex = mkdtempSync(join(tmpdir(), "shallot-setup-arm-untar-"));
    const pkg = join(ex, "package");
    mkdirSync(join(pkg, "examples"), { recursive: true });
    writeFileSync(join(pkg, "AGENTS.md"), "# Agents\n");
    writeFileSync(join(pkg, "MIGRATION.md"), "# Migration\n");
    writeFileSync(join(pkg, "examples", "recipe.ts"), "// recipe\n");
    writeFileSync(join(pkg, "index.ts"), "export const x = 1;\n");

    // Tar it into a .tgz at the same path stripTarball expects.
    const tgz = join(ex, "engine.tgz");
    const tar = Bun.spawnSync(["tar", "-czf", tgz, "-C", ex, "package"], { cwd: ex });
    if (tar.exitCode !== 0) throw new Error(`fixture tar failed: ${tar.stderr}`);
    return tgz;
}

function untarTo(tgz: string, dest: string): void {
    mkdirSync(dest, { recursive: true });
    const tar = Bun.spawnSync(["tar", "-xzf", tgz, "-C", dest], { cwd: dest });
    if (tar.exitCode !== 0) throw new Error(`fixture untar failed: ${tar.stderr}`);
}

test("stripTarball — removes AGENTS.md from the tarball", () => {
    const tgz = buildFixtureTarball();
    try {
        stripTarball(tgz);
        const dest = mkdtempSync(join(tmpdir(), "shallot-setup-arm-check-"));
        try {
            untarTo(tgz, dest);
            expect(existsSync(join(dest, "package/AGENTS.md"))).toBe(false);
        } finally {
            rmSync(dest, { recursive: true, force: true });
        }
    } finally {
        rmSync(tgz, { force: true });
        rmSync(join(tmpdir(), "shallot-setup-arm-untar-"), { recursive: true, force: true });
    }
});

test("stripTarball — removes MIGRATION.md from the tarball", () => {
    const tgz = buildFixtureTarball();
    try {
        stripTarball(tgz);
        const dest = mkdtempSync(join(tmpdir(), "shallot-setup-arm-check-"));
        try {
            untarTo(tgz, dest);
            expect(existsSync(join(dest, "package/MIGRATION.md"))).toBe(false);
        } finally {
            rmSync(dest, { recursive: true, force: true });
        }
    } finally {
        rmSync(tgz, { force: true });
        rmSync(join(tmpdir(), "shallot-setup-arm-untar-"), { recursive: true, force: true });
    }
});

test("stripTarball — removes examples/ from the tarball", () => {
    const tgz = buildFixtureTarball();
    try {
        stripTarball(tgz);
        const dest = mkdtempSync(join(tmpdir(), "shallot-setup-arm-check-"));
        try {
            untarTo(tgz, dest);
            expect(existsSync(join(dest, "package/examples"))).toBe(false);
        } finally {
            rmSync(dest, { recursive: true, force: true });
        }
    } finally {
        rmSync(tgz, { force: true });
        rmSync(join(tmpdir(), "shallot-setup-arm-untar-"), { recursive: true, force: true });
    }
});

test("stripTarball — preserves the code files (does not over-strip)", () => {
    const tgz = buildFixtureTarball();
    try {
        stripTarball(tgz);
        const dest = mkdtempSync(join(tmpdir(), "shallot-setup-arm-check-"));
        try {
            untarTo(tgz, dest);
            expect(existsSync(join(dest, "package/index.ts"))).toBe(true);
            expect(readFileSync(join(dest, "package/index.ts"), "utf8").trim()).toBe(
                "export const x = 1;",
            );
        } finally {
            rmSync(dest, { recursive: true, force: true });
        }
    } finally {
        rmSync(tgz, { force: true });
        rmSync(join(tmpdir(), "shallot-setup-arm-untar-"), { recursive: true, force: true });
    }
});
