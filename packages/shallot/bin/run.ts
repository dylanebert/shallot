import { execSync } from "node:child_process";
import { basename, resolve } from "node:path";
import { preview } from "vite";
import { CROSS_ORIGIN_ISOLATION } from "../src/project/vite";
import { buildWeb } from "./build";
import { bundleNativeLinux, bundleNativeMac, bundleNativeWindows, nativeOutDir } from "./native";

export async function runProject(
    projectDir: string,
    opts: { target?: string; port?: number; release?: boolean; portable?: boolean },
) {
    const target = opts.target ?? "web";
    const release = opts.release ?? false;
    const portable = opts.portable ?? false;

    if (target === "web") {
        await buildWeb(projectDir);
        const server = await preview({
            root: projectDir,
            // cross-origin isolation so tumble physics multithreads (COOP/COEP → shared WebAssembly.Memory)
            preview: { port: opts.port, open: true, headers: CROSS_ORIGIN_ISOLATION },
        });
        server.printUrls();
        console.log();
        return;
    }

    if (target === "mac") {
        const outputDir = nativeOutDir(projectDir, "mac", release, portable);
        console.log(`\n  building ${basename(projectDir)}...\n`);

        await bundleNativeMac(projectDir, outputDir, { release, portable });

        const appDir = resolve(outputDir, `${basename(projectDir)}.app`);
        console.log(`\n  running ${basename(projectDir)}...\n`);

        const result = Bun.spawnSync(["open", "-W", appDir]);
        process.stdout.write(result.stdout);
        process.stderr.write(result.stderr);
        process.exit(result.exitCode);
    }

    if (target === "linux") {
        const outputDir = nativeOutDir(projectDir, "linux", release, portable);
        console.log(`\n  building ${basename(projectDir)}...\n`);

        await bundleNativeLinux(projectDir, outputDir, { release, portable });

        const bin = resolve(outputDir, basename(projectDir));
        console.log(`\n  running ${basename(projectDir)}...\n`);

        const env = { ...process.env };
        // portable resolves libcef.so from the sibling cef/ dir; the system build uses host WebKitGTK.
        if (portable) {
            const cefLibDir = resolve(outputDir, "cef");
            env.LD_LIBRARY_PATH = `${cefLibDir}:${env.LD_LIBRARY_PATH || ""}`;
        }

        const result = Bun.spawnSync([bin], { env });
        process.stdout.write(result.stdout);
        process.stderr.write(result.stderr);
        process.exit(result.exitCode);
    }

    if (target !== "windows") {
        console.error(`unknown target: ${target}`);
        process.exit(1);
    }

    const outputDir = nativeOutDir(projectDir, "windows", release, portable);
    console.log(`\n  building ${basename(projectDir)}...\n`);

    await bundleNativeWindows(projectDir, outputDir, { release, portable });

    console.log(`\n  running ${basename(projectDir)}...\n`);

    const winPath = execSync(`wslpath -w "${outputDir}"`, { encoding: "utf-8" }).trim();
    const cmd = `cd '${winPath}'; .\\${basename(projectDir)}.exe`;
    const result = Bun.spawnSync(["powershell.exe", "-Command", cmd]);

    process.stdout.write(result.stdout);
    process.stderr.write(result.stderr);
    process.exit(result.exitCode);
}
