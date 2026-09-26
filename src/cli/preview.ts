import { execSync } from "node:child_process";
import { existsSync } from "node:fs";
import { basename, relative, resolve } from "node:path";
import { type PreviewServer, preview, resolveConfig } from "vite";
import { nativeOutDir } from "../native";
import { CROSS_ORIGIN_ISOLATION } from "../project";

export type PreviewTarget =
    | { kind: "web" }
    | { kind: "mac" }
    | { kind: "linux" }
    | { kind: "windows" }
    | { kind: "unknown"; target: string };

/** which of `previewProject`'s branches a `--target` value selects. Pure — `opts.target` defaults "web". */
export function resolvePreviewTarget(target = "web"): PreviewTarget {
    if (target === "web") return { kind: "web" };
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

export class MissingBuildError extends Error {}

function requireBuild(projectDir: string, target: string, artifact: string): void {
    if (existsSync(artifact)) return;
    const command = target === "web" ? "shallot build" : `shallot build --target ${target}`;
    throw new MissingBuildError(
        `no ${target} build found at ${relative(projectDir, artifact)}; run \`${command}\` first.`,
    );
}

/** Serve the current web output, without building it. The caller owns closing the returned server. */
export async function startWebPreview(
    projectDir: string,
    opts: { port?: number; open?: boolean; host?: string } = {},
): Promise<PreviewServer> {
    const config = await resolveConfig(
        { root: projectDir },
        "serve",
        "production",
        undefined,
        true,
    );
    const outDir = resolve(projectDir, config.build.outDir);
    requireBuild(projectDir, "web", resolve(outDir, "index.html"));
    return preview({
        root: projectDir,
        preview: {
            port: opts.port,
            open: opts.open ?? true,
            host: opts.host,
            headers: CROSS_ORIGIN_ISOLATION,
        },
    });
}

export async function previewProject(
    projectDir: string,
    opts: { target?: string; port?: number; release?: boolean; portable?: boolean; open?: boolean },
): Promise<void> {
    const target = resolvePreviewTarget(opts.target);
    const release = opts.release ?? false;
    const portable = opts.portable ?? false;

    if (target.kind === "unknown") {
        console.error(`unknown target: ${target.target}`);
        console.error("See `shallot preview --help` for available targets and options.");
        process.exit(1);
    }

    if (target.kind === "web") {
        const server = await startWebPreview(projectDir, { port: opts.port, open: opts.open });
        server.printUrls();
        console.log();
        return;
    }

    const name = basename(projectDir);
    const outputDir = nativeOutDir(projectDir, target.kind, release, portable);
    const artifact =
        target.kind === "mac"
            ? resolve(outputDir, `${name}.app`)
            : resolve(outputDir, `${name}${target.kind === "windows" ? ".exe" : ""}`);
    requireBuild(projectDir, target.kind, artifact);

    if (target.kind === "mac") {
        console.log(`\n  running ${name}...\n`);
        const result = Bun.spawnSync(["open", "-W", artifact]);
        process.stdout.write(result.stdout);
        process.stderr.write(result.stderr);
        process.exit(result.exitCode);
    }

    if (target.kind === "linux") {
        console.log(`\n  running ${name}...\n`);
        const env = linuxPreviewEnv(outputDir, portable, process.env);
        const result = Bun.spawnSync([artifact], { env });
        process.stdout.write(result.stdout);
        process.stderr.write(result.stderr);
        process.exit(result.exitCode);
    }

    console.log(`\n  running ${name}...\n`);
    const winPath = execSync(`wslpath -w "${outputDir}"`, { encoding: "utf-8" }).trim();
    const cmd = windowsPreviewCommand(winPath, name);
    const result = Bun.spawnSync(["powershell.exe", "-Command", cmd]);
    process.stdout.write(result.stdout);
    process.stderr.write(result.stderr);
    process.exit(result.exitCode);
}
