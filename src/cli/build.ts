import { basename, relative } from "node:path";
import { bundleNativeLinux, bundleNativeMac, bundleNativeWindows, nativeOutDir } from "../native";
import { buildWeb, requireBackend } from "../project";

export async function buildProject(
    projectDir: string,
    opts: { target?: string; release?: boolean; portable?: boolean },
) {
    const target = opts.target;
    if (target === "windows" || target === "mac" || target === "linux") {
        const release = opts.release ?? false;
        const portable = opts.portable ?? false;

        await requireBackend(projectDir, target, portable);

        const outputDir = nativeOutDir(projectDir, target, release, portable);
        const label = relative(projectDir, outputDir);
        console.log(`\n  building ${basename(projectDir)} → ${label}/\n`);
        const bundleOpts = { release, portable };
        if (target === "windows") await bundleNativeWindows(projectDir, outputDir, bundleOpts);
        else if (target === "mac") await bundleNativeMac(projectDir, outputDir, bundleOpts);
        else await bundleNativeLinux(projectDir, outputDir, bundleOpts);
        console.log(`\n  done. ${label}/\n`);
        return;
    }

    if (target && target !== "web") {
        console.error(`unknown target: ${target}`);
        console.error("See `shallot build --help` for available targets and options.");
        process.exit(1);
    }

    await buildWeb(projectDir);
}
