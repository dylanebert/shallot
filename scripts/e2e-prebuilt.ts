#!/usr/bin/env bun
// End-to-end proof that the prebuilt download+extract path needs no Rust toolchain.
//
// Flow:
//   1. Download artifacts from a green dry run (`gh run download`).
//   2. Create a draft prerelease tag and upload the six archives + SHA256SUMS (`gh release create` + `upload`).
//   3. Download them back from the draft (`gh release download`) — proves the assets are accessible.
//   4. Verify every archive's SHA256 against SHA256SUMS.
//   5. Extract the linux-portable archive into a temp cache dir (via `XDG_CACHE_HOME`, the existing seam)
//      and write the `.complete` sentinel the resolver checks for a cache hit.
//   6. Pack the engine, scaffold a temp project, and install the packed tarball (a real node_modules
//      layout, so `getPackageVersion()` detects the version and the resolver engages).
//   7. Create a `cargo` stub that exits non-zero and PATH-inject it ahead of the real cargo.
//   8. Run `shallot build --target linux --portable --release` — the resolver finds the cache hit,
//      skips cargo entirely, and the build copies the prebuilt binary + CEF runtime.
//
// The script's exit code is the gate: 0 = pass, 1 = fail. Idempotent — deletes any existing draft at
// the test tag before starting. Safe — failure never leaves a draft release or a polluted cache behind
// (the draft is deleted and the temp dirs are removed in a `finally`, including on failure).
//
// Run: `bun run scripts/e2e-prebuilt.ts` (attended seat with `gh` authed, workflow scope).

import { execSync } from "node:child_process";
import { createHash } from "node:crypto";
import {
    chmodSync,
    existsSync,
    mkdirSync,
    mkdtempSync,
    readdirSync,
    readFileSync,
    rmSync,
    writeFileSync,
} from "node:fs";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";

// GitHub run artifacts expire (~90 days). If this run's artifacts are gone, substitute a fresh
// workflow_dispatch run id from `gh run list --workflow release.yml`.
const GREEN_RUN_ID = "32253745114";
const REPO = "dylanebert/shallot";
const LINUX_TARGET = "x86_64-unknown-linux-gnu";
const ENGINE_DIR = resolve(import.meta.dir, "../packages/shallot");
const CREATE_SHALLOT = resolve(import.meta.dir, "../packages/create-shallot/index.ts");
const ENGINE_PKG = JSON.parse(readFileSync(resolve(ENGINE_DIR, "package.json"), "utf8"));
const VERSION = ENGINE_PKG.version;
const TAG = `v${VERSION}-e2e-prebuilt`;
const PORTABLE_ARCHIVE = `shallot-window-${LINUX_TARGET}-portable.tar.gz`;
const EXPECTED_ARCHIVE_COUNT = 6;

/** Run a `gh` subcommand, throwing on non-zero exit. Returns trimmed stdout. */
function gh(args: string[], opts?: { cwd?: string; stdio?: "pipe" | "inherit" }): string {
    const p = Bun.spawnSync(["gh", ...args], {
        cwd: opts?.cwd,
        stdout: opts?.stdio === "inherit" ? "inherit" : "pipe",
        stderr: opts?.stdio === "inherit" ? "inherit" : "pipe",
    });
    if (p.exitCode !== 0) {
        const err = `${p.stdout?.toString() ?? ""}\n${p.stderr?.toString() ?? ""}`;
        throw new Error(`gh ${args.join(" ")} failed (exit ${p.exitCode}):\n${err}`);
    }
    return p.stdout?.toString().trim() ?? "";
}

/** Recursively find files whose name matches `pattern` under `dir`. */
function findFiles(dir: string, pattern: RegExp): string[] {
    const results: string[] = [];
    function walk(d: string) {
        for (const entry of readdirSync(d, { withFileTypes: true })) {
            const full = join(d, entry.name);
            if (entry.isDirectory()) walk(full);
            else if (pattern.test(entry.name)) results.push(full);
        }
    }
    walk(dir);
    return results;
}

async function main() {
    console.log("e2e-prebuilt: starting end-to-end prebuilt proof");
    console.log(`  version=${VERSION}  tag=${TAG}  target=${LINUX_TARGET}  portable  release`);

    // --- Temp dirs ---
    const work = mkdtempSync(join(tmpdir(), "shallot-e2e-"));
    const cacheHome = join(work, "cache");
    const artifactsDir = join(work, "artifacts");
    const downloadedDir = join(work, "downloaded");
    const projectParent = join(work, "scaffold");
    const stubDir = join(work, "stubs");
    const packDir = join(work, "pack");
    for (const d of [cacheHome, artifactsDir, downloadedDir, projectParent, stubDir, packDir])
        mkdirSync(d, { recursive: true });

    let draftCreated = false;

    try {
        // --- 1. Idempotency: delete any existing draft at the test tag ---
        console.log("e2e-prebuilt: cleaning up any existing draft at test tag…");
        try {
            gh(["release", "delete", TAG, "--repo", REPO, "--yes"], {
                stdio: "pipe",
            });
            console.log("  deleted existing draft");
        } catch {
            // No existing draft — expected on a clean run.
        }

        // --- 2. Download artifacts from the green dry run ---
        console.log(`e2e-prebuilt: downloading artifacts from run ${GREEN_RUN_ID}…`);
        gh(["run", "download", GREEN_RUN_ID, "--repo", REPO, "--dir", artifactsDir], {
            stdio: "inherit",
        });

        const archives = findFiles(artifactsDir, /\.tar\.gz$/);
        const sumsFiles = findFiles(artifactsDir, /^SHA256SUMS$/);
        if (archives.length !== EXPECTED_ARCHIVE_COUNT) {
            throw new Error(
                `expected ${EXPECTED_ARCHIVE_COUNT} archives, found ${archives.length}: ${archives.join(", ")}`,
            );
        }
        if (sumsFiles.length !== 1) {
            throw new Error(`expected 1 SHA256SUMS file, found ${sumsFiles.length}`);
        }
        console.log(`  found ${archives.length} archives + SHA256SUMS`);

        // --- 3. Create a draft prerelease and upload artifacts ---
        console.log(`e2e-prebuilt: creating draft prerelease at ${TAG}…`);
        gh(
            [
                "release",
                "create",
                TAG,
                "--repo",
                REPO,
                "--draft",
                "--prerelease",
                "--title",
                "E2E prebuilt test (auto-cleanup)",
                "--notes",
                "Temporary draft for e2e prebuilt testing. Auto-deleted by the e2e script.",
            ],
            { stdio: "pipe" },
        );
        draftCreated = true;

        console.log("e2e-prebuilt: uploading artifacts to draft…");
        gh(["release", "upload", TAG, "--repo", REPO, ...archives, ...sumsFiles], {
            stdio: "pipe",
        });
        console.log("  uploaded");

        // --- 4. Download from the draft (proves assets are accessible) ---
        console.log("e2e-prebuilt: downloading from draft…");
        gh(["release", "download", TAG, "--repo", REPO, "--dir", downloadedDir], {
            stdio: "pipe",
        });
        console.log("  downloaded");

        // --- 5. Verify SHA256SUMS ---
        console.log("e2e-prebuilt: verifying SHA256SUMS…");
        const sumsContent = readFileSync(join(downloadedDir, "SHA256SUMS"), "utf8");
        const sumsLines = sumsContent.trim().split("\n");
        if (sumsLines.length !== EXPECTED_ARCHIVE_COUNT) {
            throw new Error(
                `SHA256SUMS has ${sumsLines.length} lines, expected ${EXPECTED_ARCHIVE_COUNT}`,
            );
        }
        for (const line of sumsLines) {
            const parts = line.trim().split(/\s+/);
            if (parts.length < 2) continue;
            const expectedHash = parts[0];
            const archiveName = parts[1].replace(/^\*/, "");
            const archivePath = join(downloadedDir, archiveName);
            if (!existsSync(archivePath)) {
                throw new Error(`SHA256SUMS references ${archiveName} but it was not downloaded`);
            }
            const actualHash = createHash("sha256").update(readFileSync(archivePath)).digest("hex");
            if (actualHash !== expectedHash) {
                throw new Error(
                    `checksum mismatch for ${archiveName}: expected ${expectedHash}, got ${actualHash}`,
                );
            }
        }
        console.log("  all checksums verified");

        // --- 6. Extract the linux-portable archive into the temp cache ---
        console.log("e2e-prebuilt: extracting linux-portable archive into cache…");
        const cacheTarget = resolve(
            cacheHome,
            "shallot",
            "prebuilt",
            VERSION,
            `${LINUX_TARGET}-portable`,
        );
        mkdirSync(cacheTarget, { recursive: true });
        const portableArchivePath = join(downloadedDir, PORTABLE_ARCHIVE);
        if (!existsSync(portableArchivePath)) {
            throw new Error(`linux-portable archive not found at ${portableArchivePath}`);
        }
        execSync(`tar -xzf "${portableArchivePath}" -C "${cacheTarget}"`, { stdio: "pipe" });
        // The resolver's cache-hit sentinel — without this, a partial extraction is never read as a hit.
        writeFileSync(join(cacheTarget, ".complete"), "");
        const binaryPath = join(cacheTarget, "shallot-window");
        if (!existsSync(binaryPath)) {
            throw new Error(
                `shallot-window binary not found in extracted archive at ${binaryPath}`,
            );
        }
        const cefDir = join(cacheTarget, "cef");
        if (!existsSync(join(cefDir, "libcef.so"))) {
            throw new Error(`libcef.so not found in extracted archive at ${cefDir}`);
        }
        console.log("  extracted + sentinel written");

        // --- 7. Pack the engine ---
        console.log("e2e-prebuilt: packing engine…");
        const packResult = Bun.spawnSync(["bun", "pm", "pack", "--destination", packDir], {
            cwd: ENGINE_DIR,
            stdout: "pipe",
            stderr: "pipe",
        });
        if (packResult.exitCode !== 0) {
            throw new Error(
                `bun pm pack failed (exit ${packResult.exitCode}):\n${packResult.stdout?.toString()}\n${packResult.stderr?.toString()}`,
            );
        }
        const tarball = readdirSync(packDir).find((f) => f.endsWith(".tgz"));
        if (!tarball) throw new Error("no tarball produced by bun pm pack");
        const tarballPath = join(packDir, tarball);
        console.log(`  packed: ${tarball}`);

        // --- 8. Scaffold a temp project and install the packed tarball ---
        console.log("e2e-prebuilt: scaffolding project…");
        const scaffoldResult = Bun.spawnSync(["bun", CREATE_SHALLOT, "starter-app"], {
            cwd: projectParent,
            stdout: "pipe",
            stderr: "pipe",
        });
        if (scaffoldResult.exitCode !== 0) {
            throw new Error(
                `bun create shallot failed (exit ${scaffoldResult.exitCode}):\n${scaffoldResult.stdout?.toString()}\n${scaffoldResult.stderr?.toString()}`,
            );
        }
        const projectDir = join(projectParent, "starter-app");
        if (!existsSync(join(projectDir, "package.json"))) {
            throw new Error("scaffold did not produce a package.json");
        }

        // Rewrite the engine dependency to the packed tarball (a real node_modules install, not a link).
        const pkg = JSON.parse(readFileSync(join(projectDir, "package.json"), "utf8"));
        pkg.dependencies["@dylanebert/shallot"] = `file:${tarballPath}`;
        writeFileSync(join(projectDir, "package.json"), `${JSON.stringify(pkg, null, 2)}\n`);

        console.log("e2e-prebuilt: installing…");
        const installResult = Bun.spawnSync(["bun", "install"], {
            cwd: projectDir,
            stdout: "pipe",
            stderr: "pipe",
        });
        if (installResult.exitCode !== 0) {
            throw new Error(
                `bun install failed (exit ${installResult.exitCode}):\n${installResult.stdout?.toString()}\n${installResult.stderr?.toString()}`,
            );
        }
        console.log("  installed");

        // Verify the CLI is in node_modules (the path that makes getPackageVersion() detect the version).
        const cliPath = join(projectDir, "node_modules/@dylanebert/shallot/bin/cli.ts");
        if (!existsSync(cliPath)) {
            throw new Error(`CLI not found at ${cliPath}`);
        }

        // --- 9. Create a cargo stub that exits non-zero ---
        const cargoStub = join(stubDir, "cargo");
        writeFileSync(
            cargoStub,
            "#!/bin/sh\necho 'cargo: stubbed out for e2e prebuilt test' >&2\nexit 1\n",
        );
        chmodSync(cargoStub, 0o755);

        // --- 10. Run the build with cargo shimmed ---
        console.log(
            "e2e-prebuilt: running shallot build --target linux --portable --release with cargo shimmed…",
        );
        const buildEnv = {
            ...process.env,
            PATH: `${stubDir}:${process.env.PATH}`,
            XDG_CACHE_HOME: cacheHome,
        };
        const buildResult = Bun.spawnSync(
            ["bun", cliPath, "build", "--target", "linux", "--portable", "--release"],
            { cwd: projectDir, env: buildEnv, stdout: "inherit", stderr: "inherit" },
        );
        if (buildResult.exitCode !== 0) {
            throw new Error(
                `build failed with exit code ${buildResult.exitCode} — the cargo shim should not have been reached; the prebuilt cache hit was missed`,
            );
        }

        // --- 11. Verify the output ---
        const outputDir = join(projectDir, "build", "linux", "release-portable");
        const outputBinary = join(outputDir, "starter-app");
        if (!existsSync(outputBinary)) {
            throw new Error(`expected output binary not found at ${outputBinary}`);
        }
        const outputCefDir = join(outputDir, "cef");
        if (!existsSync(join(outputCefDir, "libcef.so"))) {
            throw new Error(`expected libcef.so not found at ${outputCefDir}/libcef.so`);
        }

        console.log("e2e-prebuilt: PASS — prebuilt used, cargo never invoked, CEF runtime copied");
    } finally {
        // --- 12. Clean up: delete draft release + tag (always, even on failure) ---
        if (draftCreated) {
            console.log("e2e-prebuilt: cleaning up draft release…");
            try {
                // Draft releases don't create a git tag, so --cleanup-tag would fail with
                // "Reference does not exist" — delete the release object only.
                gh(["release", "delete", TAG, "--repo", REPO, "--yes"], {
                    stdio: "pipe",
                });
                console.log("  draft deleted");
            } catch (e) {
                console.error(
                    `e2e-prebuilt: WARNING — failed to delete draft release ${TAG}: ${e}`,
                );
            }
        }

        // Clean up any pack projections left in the engine dir (prepack creates them; postpack
        // removes them on success, but a mid-pack failure can leave them behind).
        rmSync(join(ENGINE_DIR, "examples"), { recursive: true, force: true });
        rmSync(join(ENGINE_DIR, "dist"), { recursive: true, force: true });

        // Remove temp dirs.
        rmSync(work, { recursive: true, force: true });
    }
}

main().catch((e) => {
    console.error(`e2e-prebuilt: FAIL — ${e instanceof Error ? e.message : String(e)}`);
    process.exit(1);
});
