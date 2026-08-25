// S3 arm — evals/harness/wsl.ts detectDisplay() + stageOnWindows
//
// Two invariants pinned:
//   (1) detectDisplay() — an unsupported/undetectable platform returns false (skip is honest),
//       and the WSL branch delegates to a host probe (not an unconditional true). The WSL and win32
//       branches were shipped logic-level and are unwitnessable at runtime on this darwin seat, so
//       the arm exercises the injectable pure seam (detectDisplayForPlatform) that detectDisplay()
//       delegates to, reaching every branch decision without needing that platform.
//   (2) stageOnWindows() — a failed staging throws (does not fall through into the gate run).
//       On this darwin seat powershell.exe is absent, so the staging spawn fails and the throw fires.
//
// The seam (detectDisplayForPlatform) is the minimal production-code change this stage makes;
// detectDisplay() now delegates to it. Named in the stage report.

import { expect, test } from "bun:test";
import { mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import { detectDisplayForPlatform, stageOnWindows } from "./wsl";

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

// ── stageOnWindows: failed staging throws ─────────────────────────────────────

test("stageOnWindows — a failed staging throws, does not fall through", () => {
    // On this darwin seat powershell.exe is absent. stageOnWindows calls windowsTempPaths (which
    // spawns powershell.exe) and then the staging spawn (also powershell.exe). Either call fails
    // because powershell.exe is not on PATH, and the function throws rather than returning
    // successfully — it does NOT fall through into the gate run.
    //
    // The throw may come from windowsTempPaths (before the staging check) or from the staging
    // check itself. Either way, the function does not return a WindowsPaths object, which is the
    // invariant: a failed staging does not fall through.
    const tmp = mkdtempSync(join(tmpdir(), "shallot-wsl-arm-"));
    const srcDir = join(tmp, "src");
    const stageName = join(tmp, "stage");
    mkdirSync(srcDir, { recursive: true });
    writeFileSync(join(srcDir, "file.txt"), "test");

    expect(() => stageOnWindows(srcDir, stageName, ["file.txt"])).toThrow();

    // cleanup — stageOnWindows throws, but be defensive
    rmSync(tmp, { recursive: true, force: true });
});

test("stageOnWindows — the staging exit-code check is present (structural pin)", () => {
    // The S1 fix added `if (staging.exitCode !== 0) { rmSync(...); throw ... }` in stageOnWindows.
    // On a darwin seat the throw fires from windowsTempPaths (powershell.exe absent) before
    // reaching the staging spawn, so the behavioral test above can't isolate the staging check.
    // This structural pin ensures the check is present — it reds if the fix is reverted.
    const wslSrc = readFileSync(resolve(import.meta.dir, "wsl.ts"), "utf8");
    expect(wslSrc).toMatch(/staging\.exitCode\s*!==\s*0/);
    expect(wslSrc).toMatch(/Playwright dependency staging failed/);
});
