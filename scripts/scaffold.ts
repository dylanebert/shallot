import { resolve, join } from "path";
import { existsSync, mkdirSync, readFileSync, rmSync, writeFileSync } from "fs";
import { tmpdir } from "os";

const projectDir = resolve(import.meta.dir, "..");

export const isWSL =
    process.platform === "linux" && existsSync("/proc/sys/fs/binfmt_misc/WSLInterop");

export function detectDisplay(): boolean {
    if (isWSL) return true;
    if (process.platform !== "linux") return true;
    return !!(process.env.DISPLAY || process.env.WAYLAND_DISPLAY);
}

function exec(cmd: string[], cwd: string) {
    const result = Bun.spawnSync(cmd, { cwd, stdout: "pipe", stderr: "pipe" });
    if (result.exitCode !== 0) {
        const stderr = new TextDecoder().decode(result.stderr);
        throw new Error(`${cmd.join(" ")} failed (exit ${result.exitCode}): ${stderr}`);
    }
    return new TextDecoder().decode(result.stdout);
}

export interface ScaffoldResult {
    dir: string;
    tmpBase: string;
    cleanup: () => void;
}

export function scaffold(name: string, port: number): ScaffoldResult {
    const tmpBase = join(tmpdir(), `shallot-${name}-${Date.now()}`);
    mkdirSync(tmpBase, { recursive: true });
    const dir = join(tmpBase, name);

    console.log(`Scaffolding in ${dir}`);
    const createShallot = resolve(projectDir, "packages/create-shallot/index.ts");
    exec(["bun", createShallot, name], tmpBase);

    writeFileSync(
        join(dir, "vite.config.ts"),
        `import { defineConfig } from "vite";

export default defineConfig({
    server: { port: ${port} },
    build: { target: "esnext" },
});
`,
    );

    const pkgDir = resolve(projectDir, "packages/shallot");
    exec(["bun", "pm", "pack", "--destination", dir, "--quiet"], pkgDir);
    const tgz = new TextDecoder()
        .decode(Bun.spawnSync(["ls"], { cwd: dir, stdout: "pipe" }).stdout)
        .split("\n")
        .find((f) => f.endsWith(".tgz"));
    if (!tgz) throw new Error("bun pm pack produced no tarball");

    const pkgJsonPath = join(dir, "package.json");
    const pkgJson = JSON.parse(readFileSync(pkgJsonPath, "utf-8"));
    pkgJson.dependencies["@dylanebert/shallot"] = `./${tgz}`;
    writeFileSync(pkgJsonPath, JSON.stringify(pkgJson, null, 2) + "\n");
    exec(["bun", "install"], dir);

    console.log("Scaffold complete");

    return {
        dir,
        tmpBase,
        cleanup: () => rmSync(tmpBase, { recursive: true, force: true }),
    };
}

export interface ServerHandle {
    process: ReturnType<typeof Bun.spawn>;
    cleanup: () => void;
}

export async function startServer(
    cmd: string[],
    cwd: string,
    port: number,
    onCleanup?: () => void,
): Promise<ServerHandle> {
    try {
        await fetch(`http://localhost:${port}`);
        Bun.spawnSync(["fuser", "-k", `${port}/tcp`], { stdout: "ignore", stderr: "ignore" });
        await Bun.sleep(500);
    } catch {}

    const server = Bun.spawn(cmd, {
        cwd,
        stdout: "inherit",
        stderr: "inherit",
        env: { ...process.env, BROWSER: "none" },
    });

    const cleanup = () => {
        server.kill();
        onCleanup?.();
    };
    process.on("exit", cleanup);
    process.on("SIGINT", () => {
        cleanup();
        process.exit(1);
    });
    process.on("SIGTERM", () => {
        cleanup();
        process.exit(1);
    });

    for (let i = 0; i < 60; i++) {
        if (server.exitCode !== null) {
            throw new Error(`Server exited early (code ${server.exitCode})`);
        }
        try {
            await fetch(`http://localhost:${port}`);
            console.log(`Server ready on port ${port}`);
            break;
        } catch {
            if (i === 59) throw new Error(`Server failed to start on port ${port}`);
            await Bun.sleep(500);
        }
    }

    return { process: server, cleanup };
}
