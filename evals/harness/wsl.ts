import { cpSync, existsSync, mkdirSync, rmSync } from "fs";
import { dirname, join } from "path";

export const isWSL =
    process.platform === "linux" && existsSync("/proc/sys/fs/binfmt_misc/WSLInterop");

export function detectDisplay(): boolean {
    if (isWSL) {
        // The WSL eval path stages onto the Windows host and runs Playwright there via PowerShell —
        // it never touches the WSL display, so DISPLAY/WAYLAND_DISPLAY are irrelevant here. Probe
        // host reachability instead: powershell.exe resolvable AND a bun or node on the Windows host
        // (AGENTS.md: these wrappers "skip only when the host lacks the node/bun it needs").
        return wslHostReachable();
    }
    if (process.platform === "darwin") {
        // WindowServer runs in a GUI session; absent on a headless mac (CI, SSH).
        const r = Bun.spawnSync(["pgrep", "-x", "WindowServer"], {
            stdout: "ignore",
            stderr: "ignore",
        });
        return r.exitCode === 0;
    }
    if (process.platform === "linux") {
        return !!(process.env.DISPLAY || process.env.WAYLAND_DISPLAY);
    }
    // An undetectable or unsupported platform returns false, not true — a skip is an honest outcome
    // (grade.ts reports skipped: true), a false run is not, and a browser gate launched with no
    // display reds for the wrong reason.
    return false;
}

// Bounded, non-hanging probe for the Windows host from WSL — a spawnSync with a timeout, no
// interactive shell. Never runs on a non-WSL platform (the caller gates on isWSL).
function wslHostReachable(): boolean {
    const Timeout = 5000;
    // powershell.exe resolvable AND a bun or node reachable on the host — one bounded call.
    // If powershell.exe is not on PATH the spawn fails (exitCode !== 0); if it is but neither bun
    // nor node is on the host, the command exits 1. Either way the gate skips honestly.
    const probe = Bun.spawnSync(
        [
            "powershell.exe",
            "-NoProfile",
            "-NonInteractive",
            "-Command",
            "if ((Get-Command bun -ErrorAction SilentlyContinue) -or (Get-Command node -ErrorAction SilentlyContinue)) { exit 0 } else { exit 1 }",
        ],
        { stdout: "ignore", stderr: "ignore", timeout: Timeout },
    );
    return probe.exitCode === 0;
}

export interface WindowsPaths {
    win: string;
    wsl: string;
}

export function windowsTempPaths(name: string): WindowsPaths {
    const winTempProc = Bun.spawnSync(
        ["powershell.exe", "-Command", "Write-Host -NoNewline $env:TEMP"],
        { stdout: "pipe" },
    );
    const winTempPath = new TextDecoder().decode(winTempProc.stdout).trim().replace(/\r/g, "");
    const wslTempProc = Bun.spawnSync(["wslpath", winTempPath], { stdout: "pipe" });
    const wslTemp = new TextDecoder().decode(wslTempProc.stdout).trim();
    return {
        win: `${winTempPath}\\${name}`,
        wsl: join(wslTemp, name),
    };
}

// Mirror the caller's Playwright files to a fresh Windows TEMP directory (`<name>`) so Playwright can
// run from the host, then install its deps there. `files` are paths relative to `srcDir`, subdirs
// allowed (the parent is created). Returns both path views — `win` for PowerShell `cd`, `wsl` for
// reading artifacts (screenshots) back. Used by the eval gate driver (grade.ts).
export function stageOnWindows(srcDir: string, name: string, files: string[]): WindowsPaths {
    const paths = windowsTempPaths(name);

    rmSync(paths.wsl, { recursive: true, force: true });
    mkdirSync(paths.wsl, { recursive: true });

    for (const file of files) {
        const dest = join(paths.wsl, file);
        mkdirSync(dirname(dest), { recursive: true });
        cpSync(join(srcDir, file), dest);
    }

    console.log("Installing Playwright dependencies...");
    const staging = Bun.spawnSync(
        [
            "powershell.exe",
            "-Command",
            `cd '${paths.win}'; bun install --silent; bunx playwright install chromium`,
        ],
        { stdout: "inherit", stderr: "inherit" },
    );
    if (staging.exitCode !== 0) {
        // Remove the populated-but-incomplete staging dir before throwing — a leftover dir with a
        // half-installed node_modules is a debugging trap on the next run.
        rmSync(paths.wsl, { recursive: true, force: true });
        throw new Error(`Playwright dependency staging failed (exit ${staging.exitCode})`);
    }

    return paths;
}
