import { dirname, join } from "path";
import { cpSync, existsSync, mkdirSync, rmSync } from "fs";

export const isWSL =
    process.platform === "linux" && existsSync("/proc/sys/fs/binfmt_misc/WSLInterop");

export function detectDisplay(): boolean {
    if (isWSL) return true;
    if (process.platform !== "linux") return true;
    return !!(process.env.DISPLAY || process.env.WAYLAND_DISPLAY);
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

// Mirror a launcher's Playwright files to a fresh Windows TEMP directory (`<name>`) so Playwright can
// run from the host, then install its deps there. `files` are paths relative to `srcDir`, subdirs
// allowed (the parent is created). Returns both path views — `win` for PowerShell `cd`, `wsl` for
// reading artifacts (screenshots) back. Used by both launchers: the gym (page.ts) and capture (flows).
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
    Bun.spawnSync(
        [
            "powershell.exe",
            "-Command",
            `cd '${paths.win}'; bun install --silent; bunx playwright install chromium`,
        ],
        { stdout: "inherit", stderr: "inherit" },
    );

    return paths;
}
