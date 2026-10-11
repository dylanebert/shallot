import { execSync } from "node:child_process";
import { existsSync } from "node:fs";
import { basename, relative, resolve } from "node:path";
import { nativeOutDir } from "../native";

export type PreviewTarget =
    | { kind: "mac" }
    | { kind: "linux" }
    | { kind: "windows" }
    | { kind: "unknown"; target: string };

export function resolvePreviewTarget(target: string): PreviewTarget {
    if (target === "mac") return { kind: "mac" };
    if (target === "linux") return { kind: "linux" };
    if (target === "windows") return { kind: "windows" };
    return { kind: "unknown", target };
}

/** the linux launch env: portable builds resolve `libcef.so` from their bundled `cef/` dir. */
export function linuxPreviewEnv(
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
export function windowsPreviewCommand(winPath: string, projectName: string): string {
    return `cd '${winPath}'; .\\${projectName}.exe`;
}

function requireBuild(projectDir: string, target: string, artifact: string): void {
    if (existsSync(artifact)) return;
    throw new Error(
        `no ${target} build found at ${relative(projectDir, artifact)}; run \`shallot build --target ${target}\` first.`,
    );
}

export async function previewProject(
    projectDir: string,
    opts: { target: string; release?: boolean; portable?: boolean; devUrl?: string },
): Promise<number> {
    const target = resolvePreviewTarget(opts.target);
    const release = opts.release ?? false;
    const portable = opts.portable ?? false;

    if (target.kind === "unknown") {
        console.error(`unknown target: ${target.target}`);
        console.error("See `shallot preview --help` for available targets and options.");
        return 1;
    }

    const name = basename(projectDir);
    const outputDir = nativeOutDir(projectDir, target.kind, release, portable);
    const artifact =
        target.kind === "mac"
            ? resolve(outputDir, `${name}.app`)
            : resolve(outputDir, `${name}${target.kind === "windows" ? ".exe" : ""}`);
    requireBuild(projectDir, target.kind, artifact);
    const env = {
        ...process.env,
        ...(opts.devUrl ? { SHALLOT_DEV_URL: opts.devUrl } : {}),
    };

    if (target.kind === "mac") {
        console.log(`\n  running ${name}...\n`);
        const command = opts.devUrl
            ? [resolve(artifact, "Contents", "MacOS", name)]
            : ["open", "-W", artifact];
        const result = Bun.spawnSync(command, { env });
        process.stdout.write(result.stdout);
        process.stderr.write(result.stderr);
        return result.exitCode;
    }

    if (target.kind === "linux") {
        console.log(`\n  running ${name}...\n`);
        const launchEnv = linuxPreviewEnv(outputDir, portable, env);
        const result = Bun.spawnSync([artifact], { env: launchEnv });
        process.stdout.write(result.stdout);
        process.stderr.write(result.stderr);
        return result.exitCode;
    }

    console.log(`\n  running ${name}...\n`);
    const winPath = execSync(`wslpath -w "${outputDir}"`, { encoding: "utf-8" }).trim();
    const cmd = windowsPreviewCommand(winPath, name);
    const result = Bun.spawnSync(["powershell.exe", "-Command", cmd], { env });
    process.stdout.write(result.stdout);
    process.stderr.write(result.stderr);
    return result.exitCode;
}
