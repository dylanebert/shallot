import { join } from "node:path";

export async function buildFlyServer(
    project: string,
    directory: string,
    app: string,
): Promise<void> {
    const build = await Bun.build({
        entrypoints: [join(project, "server.ts")],
        target: "bun",
        outdir: directory,
        naming: "server.js",
    });
    if (!build.success) throw new AggregateError(build.logs, "Server bundle failed");
    await Bun.write(
        join(directory, "Dockerfile"),
        `FROM oven/bun:${Bun.version}\nWORKDIR /app\nCOPY server.js .\nCMD ["bun", "server.js"]\n`,
    );
    await Bun.write(join(directory, "fly.toml"), flyConfig(app));
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
