/** Pure branch-decision seam — takes the platform and answers whether a headed browser can launch
 *  here. Exported separately from {@link detectDisplay} so every branch is exercisable without
 *  being on that platform (the win32 and unsupported branches are unwitnessable on a linux seat).
 *
 *  Invariant: an unsupported or undetectable platform returns false. A skip is an honest outcome —
 *  `grade.ts` reports `skipped: true` — while a false run is not, and a browser gate launched with
 *  no display reds for the wrong reason. */
export function detectDisplayForPlatform(platform: string): boolean {
    if (platform === "darwin") {
        // WindowServer runs in a GUI session; absent on a headless mac (CI, SSH).
        const r = Bun.spawnSync(["pgrep", "-x", "WindowServer"], {
            stdout: "ignore",
            stderr: "ignore",
        });
        return r.exitCode === 0;
    }
    if (platform === "linux") {
        return !!(process.env.DISPLAY || process.env.WAYLAND_DISPLAY);
    }
    return false;
}

export function detectDisplay(): boolean {
    return detectDisplayForPlatform(process.platform);
}
