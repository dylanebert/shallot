// S3 arm — evals/harness/wsl.ts detectDisplay() + stageOnWindows
//
// Two invariants pinned:
//   (1) detectDisplay() — an unsupported/undetectable platform returns false (skip is honest),
//       and the WSL branch delegates to a host probe (not an unconditional true). The WSL and win32
//       branches were shipped logic-level and are unwitnessable at runtime on this darwin seat, so
//       the arm exercises the injectable pure seam (detectDisplayForPlatform) that detectDisplay()
//       delegates to, reaching every branch decision without needing that platform.
//   (2) stageOnWindows() — a failed staging throws (does not fall through into the gate run).
//       On this darwin seat powershell.exe is absent, so windowsTempPaths() throws before the
//       staging spawn is reached, and the throw the original arm observed was not the guard's.
//       The staging exit-code guard is extracted into checkStagingResult(), a pure seam the arm
//       exercises directly: a nonzero exit removes the staging dir and throws.

import { expect, test } from "bun:test";
import { existsSync, mkdirSync, mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { checkStagingResult, detectDisplayForPlatform, stageOnWindows } from "./wsl";

// ── detectDisplay: branch decisions ──────────────────────────────────────────

test("detectDisplay — win32 (non-WSL) returns false, not true", () => {
    // The fix: an unsupported platform returns false, not true. Before the fix, every non-linux
    // platform returned true unconditionally, so a headless win32 box never skipped.
    expect(detectDisplayForPlatform("win32", false, () => true)).toBe(false);
});

test("detectDisplay — any unrecognised platform returns false", () => {
    expect(detectDisplayForPlatform("freebsd", false, () => true)).toBe(false);
    expect(detectDisplayForPlatform("sunos", false, () => true)).toBe(false);
});

test("detectDisplay — WSL delegates to the host probe, not an unconditional true", () => {
    // The fix: WSL probes host reachability instead of returning true unconditionally.
    // A reachable host → true; an unreachable host → false (honest skip).
    expect(detectDisplayForPlatform("linux", true, () => true)).toBe(true);
    expect(detectDisplayForPlatform("linux", true, () => false)).toBe(false);
});

test("detectDisplay — WSL probe is the decision, not the platform check", () => {
    // Even on a non-linux platform, if wsl=true the probe decides — the WSL interop is the gate,
    // not process.platform. This is the branch the S2 fix moved the decision to.
    expect(detectDisplayForPlatform("win32", true, () => true)).toBe(true);
    expect(detectDisplayForPlatform("win32", true, () => false)).toBe(false);
});

// ── stageOnWindows: failed staging throws (behavioral via the pure seam) ──────

test("checkStagingResult — a nonzero exit removes the staging dir and throws", () => {
    // The guard: a nonzero staging exit removes the populated-but-incomplete staging dir and
    // throws — it does NOT fall through into the gate run. This exercises the actual guard
    // behaviorally: a real temp dir is created, the guard fires, the dir is removed, and it throws.
    const tmp = mkdtempSync(join(tmpdir(), "shallot-wsl-staging-arm-"));
    writeFileSync(join(tmp, "marker"), "staged");
    expect(existsSync(tmp)).toBe(true);

    expect(() => checkStagingResult(1, tmp)).toThrow(/staging failed/);

    // The staging dir must be removed — a leftover with a half-installed node_modules is a trap.
    expect(existsSync(tmp)).toBe(false);
});

test("checkStagingResult — a zero exit does not throw and preserves the staging dir", () => {
    // The green direction: a successful staging does not throw and the staging dir survives.
    const tmp = mkdtempSync(join(tmpdir(), "shallot-wsl-staging-green-"));
    writeFileSync(join(tmp, "marker"), "staged");

    expect(() => checkStagingResult(0, tmp)).not.toThrow();
    expect(existsSync(tmp)).toBe(true);

    rmSync(tmp, { recursive: true, force: true });
});

test("checkStagingResult — a null exit code also throws (spawn failed entirely)", () => {
    const tmp = mkdtempSync(join(tmpdir(), "shallot-wsl-staging-null-"));
    expect(() => checkStagingResult(null, tmp)).toThrow(/staging failed/);
    expect(existsSync(tmp)).toBe(false);
});

// ── stageOnWindows: the full function still throws on a darwin seat ────────────

test("stageOnWindows — throws on a darwin seat (powershell.exe absent)", () => {
    // On this darwin seat powershell.exe is absent, so windowsTempPaths() throws before the
    // staging spawn is reached. The function does not return a WindowsPaths object — it does
    // NOT fall through into the gate run. This is kept as a second check, but the behavioral
    // arm above (checkStagingResult) is the one that exercises the actual guard.
    const tmp = mkdtempSync(join(tmpdir(), "shallot-wsl-arm-"));
    const srcDir = join(tmp, "src");
    const stageName = join(tmp, "stage");
    mkdirSync(srcDir, { recursive: true });
    writeFileSync(join(srcDir, "file.txt"), "test");

    expect(() => stageOnWindows(srcDir, stageName, ["file.txt"])).toThrow();

    rmSync(tmp, { recursive: true, force: true });
});
