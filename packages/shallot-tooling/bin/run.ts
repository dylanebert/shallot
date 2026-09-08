import { execSync } from "node:child_process";
import { basename, resolve } from "node:path";
import { preview } from "vite";
import { CROSS_ORIGIN_ISOLATION } from "../src/project/vite";
import { buildWeb } from "./build";
import { bundleNativeLinux, bundleNativeMac, bundleNativeWindows, nativeOutDir } from "./native";

export type RunTarget =
    | { kind: "web" }
    | { kind: "mac" }
    | { kind: "linux" }
    | { kind: "windows" }
    | { kind: "unknown"; target: string };

/** which of `runProject`'s branches a `--target` value selects. Pure — `opts.target` defaults "web". */
export function resolveRunTarget(target = "web"): RunTarget {
    if (target === "web") return { kind: "web" };
    if (target === "mac") return { kind: "mac" };
    if (target === "linux") return { kind: "linux" };
    if (target === "windows") return { kind: "windows" };
    return { kind: "unknown", target };
}

/**
 * the linux launch env: `LD_LIBRARY_PATH` prefixed with the bundle's `cef/` dir under `--portable`
 * (which resolves `libcef.so` from there), unchanged otherwise (the system build uses host WebKitGTK).
 */
export function linuxRunEnv(
    outputDir: string,
    portable: boolean,
    baseEnv: Record<string, string | undefined>,
): Record<string, string | undefined> {
    const env = { ...baseEnv };
    if (portable) {
        const cefLibDir = resolve(outputDir, "cef");
        env.LD_LIBRARY_PATH = `${cefLibDir}:${env.LD_LIBRARY_PATH || ""}`;
    }
    return env;
}

/** the powershell.exe command line that launches a windows build from its WSL-resolved host path. */
export function windowsRunCommand(winPath: string, projectName: string): string {
    return `cd '${winPath}'; .\\${projectName}.exe`;
}

export async function runProject(
    projectDir: string,
    opts: { target?: string; port?: number; release?: boolean; portable?: boolean },
) {
    const runTarget = resolveRunTarget(opts.target);
    const release = opts.release ?? false;
    const portable = opts.portable ?? false;

    if (runTarget.kind === "web") {
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

    if (runTarget.kind === "mac") {
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

    if (runTarget.kind === "linux") {
        const outputDir = nativeOutDir(projectDir, "linux", release, portable);
        console.log(`\n  building ${basename(projectDir)}...\n`);

        await bundleNativeLinux(projectDir, outputDir, { release, portable });

        const bin = resolve(outputDir, basename(projectDir));
        console.log(`\n  running ${basename(projectDir)}...\n`);

        const env = linuxRunEnv(outputDir, portable, process.env);

        const result = Bun.spawnSync([bin], { env });
        process.stdout.write(result.stdout);
        process.stderr.write(result.stderr);
        process.exit(result.exitCode);
    }

    if (runTarget.kind === "unknown") {
        console.error(`unknown target: ${runTarget.target}`);
        process.exit(1);
    }

    const outputDir = nativeOutDir(projectDir, "windows", release, portable);
    console.log(`\n  building ${basename(projectDir)}...\n`);

    await bundleNativeWindows(projectDir, outputDir, { release, portable });

    console.log(`\n  running ${basename(projectDir)}...\n`);

    const winPath = execSync(`wslpath -w "${outputDir}"`, { encoding: "utf-8" }).trim();
    const cmd = windowsRunCommand(winPath, basename(projectDir));
    const result = Bun.spawnSync(["powershell.exe", "-Command", cmd]);

    process.stdout.write(result.stdout);
    process.stderr.write(result.stderr);
    process.exit(result.exitCode);
}
