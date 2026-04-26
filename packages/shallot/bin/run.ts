import { resolve, basename } from "node:path";
import { execSync } from "node:child_process";
import { preview } from "vite";
import { bundleNative, bundleNativeMac, bundleNativeLinux } from "./native";
import { buildWeb } from "./build";

export async function runProject(
    projectDir: string,
    opts: { target?: string; port?: number; release?: boolean },
) {
    const target = opts.target ?? "web";

    if (target === "web") {
        await buildWeb(projectDir);
        const server = await preview({
            root: projectDir,
            preview: { port: opts.port, open: true },
        });
        server.printUrls();
        console.log();
        return;
    }

    if (target === "mac") {
        const profile = opts.release ? "release" : "debug";
        const outputDir = resolve(projectDir, "build/mac", profile);
        console.log(`\n  building ${basename(projectDir)} (${profile})...\n`);

        await bundleNativeMac(projectDir, outputDir, { release: opts.release });

        const appDir = resolve(outputDir, `${basename(projectDir)}.app`);
        console.log(`\n  running ${basename(projectDir)}...\n`);

        const result = Bun.spawnSync(["open", "-W", appDir]);
        process.stdout.write(result.stdout);
        process.stderr.write(result.stderr);
        process.exit(result.exitCode);
    }

    if (target === "linux") {
        const profile = opts.release ? "release" : "debug";
        const outputDir = resolve(projectDir, "build/linux", profile);
        console.log(`\n  building ${basename(projectDir)} (${profile})...\n`);

        await bundleNativeLinux(projectDir, outputDir, { release: opts.release });

        const bin = resolve(outputDir, basename(projectDir));
        console.log(`\n  running ${basename(projectDir)}...\n`);

        const cefLibDir = resolve(outputDir, "cef");
        const env = { ...process.env };
        env.LD_LIBRARY_PATH = `${cefLibDir}:${env.LD_LIBRARY_PATH || ""}`;

        const result = Bun.spawnSync([bin], { env });
        process.stdout.write(result.stdout);
        process.stderr.write(result.stderr);
        process.exit(result.exitCode);
    }

    if (target !== "windows") {
        console.error(`unknown target: ${target}`);
        process.exit(1);
    }

    const profile = opts.release ? "release" : "debug";
    const outputDir = resolve(projectDir, "build/windows", profile);
    console.log(`\n  building ${basename(projectDir)} (${profile})...\n`);

    await bundleNative(projectDir, outputDir, { release: opts.release });

    console.log(`\n  running ${basename(projectDir)}...\n`);

    const winPath = execSync(`wslpath -w "${outputDir}"`, { encoding: "utf-8" }).trim();
    const cmd = `cd '${winPath}'; .\\${basename(projectDir)}.exe`;
    const result = Bun.spawnSync(["powershell.exe", "-Command", cmd]);

    process.stdout.write(result.stdout);
    process.stderr.write(result.stderr);
    process.exit(result.exitCode);
}
