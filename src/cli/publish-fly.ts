import { spawnSync } from "node:child_process";
import { writeFile } from "node:fs/promises";
import { join } from "node:path";

export async function buildFlyServer(
    project: string,
    directory: string,
    app: string,
): Promise<void> {
    function bun(args: string[]): string {
        const result = spawnSync("bun", args, {
            cwd: project,
            encoding: "utf8",
            stdio: ["ignore", "pipe", "pipe"],
        });
        if (result.error) throw result.error;
        if (result.status !== 0)
            throw new Error(`bun ${args.join(" ")} failed:\n${result.stdout}${result.stderr}`);
        return result.stdout.trim();
    }
    const version = bun(["--version"]);
    bun([
        "build",
        join(project, "server.ts"),
        "--target",
        "bun",
        "--outfile",
        join(directory, "server.js"),
    ]);
    await writeFile(
        join(directory, "Dockerfile"),
        `FROM oven/bun:${version}\nWORKDIR /app\nCOPY server.js .\nCMD ["bun", "server.js"]\n`,
    );
    await writeFile(join(directory, "fly.toml"), flyConfig(app));
}

export function flyConfig(app: string): string {
    return `app = ${JSON.stringify(app)}
[env]
  PORT = "8080"
[http_service]
  internal_port = 8080
  force_https = true
  auto_stop_machines = "stop"
  auto_start_machines = true
  min_machines_running = 0
[[vm]]
  size = "shared-cpu-1x"
  memory = "256mb"
`;
}
