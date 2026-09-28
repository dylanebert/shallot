import { basename, relative } from "node:path";
import { bundleNativeLinux, bundleNativeMac, bundleNativeWindows, nativeOutDir } from "../native";
import { buildWeb, requireBackend, runVite } from "../project";

export async function buildProject(
    projectDir: string,
    opts: {
        target?: string;
        release?: boolean;
        portable?: boolean;
        args?: string[];
        dev?: boolean;
    },
): Promise<number> {
    const target = opts.target;
    if (target !== "windows" && target !== "mac" && target !== "linux") {
        return runVite(projectDir, "build", opts.args);
    }

    const release = opts.release ?? false;
    const portable = opts.portable ?? false;
    await requireBackend(projectDir, target, portable);

    if (!opts.dev) await buildWeb(projectDir, {}, opts.args);

    const outputDir = nativeOutDir(projectDir, target, release, portable);
    const label = relative(projectDir, outputDir);
    console.log(`\n  building ${basename(projectDir)} → ${label}/\n`);
    const bundleOpts = { release, portable, dev: opts.dev };
    if (target === "windows") await bundleNativeWindows(projectDir, outputDir, bundleOpts);
    else if (target === "mac") await bundleNativeMac(projectDir, outputDir, bundleOpts);
    else await bundleNativeLinux(projectDir, outputDir, bundleOpts);
    console.log(`\n  done. ${label}/\n`);
    return 0;
}
