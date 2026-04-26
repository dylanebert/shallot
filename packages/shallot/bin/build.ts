import { resolve, basename } from "node:path";
import { existsSync } from "node:fs";
import { execSync } from "node:child_process";
import { bundleNative, bundleNativeMac, bundleNativeLinux } from "./native";

export async function buildWeb(projectDir: string): Promise<void> {
    if (!existsSync(resolve(projectDir, "index.html"))) {
        console.error(`\n  No index.html found at ${projectDir}\n`);
        process.exit(1);
    }

    console.log(`\n  building ${basename(projectDir)} → dist/\n`);

    execSync("bunx vite build", { cwd: projectDir, stdio: "inherit" });
    console.log(`\n  done.\n`);
}

export async function buildProject(
    projectDir: string,
    opts: { target?: string; release?: boolean },
) {
    if (opts.target === "windows") {
        const profile = opts.release ? "release" : "debug";
        const outputDir = resolve(projectDir, "build/windows", profile);
        console.log(`\n  building ${basename(projectDir)} → build/windows/${profile}/\n`);
        await bundleNative(projectDir, outputDir, { release: opts.release });
        console.log(`\n  done. build/windows/${profile}/${basename(projectDir)}.exe\n`);
        return;
    }

    if (opts.target === "mac") {
        const profile = opts.release ? "release" : "debug";
        const outputDir = resolve(projectDir, "build/mac", profile);
        console.log(`\n  building ${basename(projectDir)} → build/mac/${profile}/\n`);
        await bundleNativeMac(projectDir, outputDir, { release: opts.release });
        console.log(`\n  done. build/mac/${profile}/${basename(projectDir)}.app\n`);
        return;
    }

    if (opts.target === "linux") {
        const profile = opts.release ? "release" : "debug";
        const outputDir = resolve(projectDir, "build/linux", profile);
        console.log(`\n  building ${basename(projectDir)} → build/linux/${profile}/\n`);
        await bundleNativeLinux(projectDir, outputDir, { release: opts.release });
        console.log(`\n  done. build/linux/${profile}/${basename(projectDir)}\n`);
        return;
    }

    if (opts.target && opts.target !== "web") {
        console.error(`unknown target: ${opts.target}`);
        process.exit(1);
    }

    await buildWeb(projectDir);
}
