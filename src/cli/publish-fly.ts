import { spawnSync } from "node:child_process";
import { createHash } from "node:crypto";
import {
    chmodSync,
    existsSync,
    mkdirSync,
    mkdtempSync,
    renameSync,
    rmSync,
    writeFileSync,
} from "node:fs";
import { writeFile } from "node:fs/promises";
import { homedir } from "node:os";
import { dirname, join } from "node:path";
import { extractZip } from "./publish-itch";

export interface FlyLink {
    target: "fly";
    app: string;
}

export const FLY_VERSION = "0.5.12";
export const FLY_ARCHIVES: Record<string, string> = Object.fromEntries([
    ["macOS_arm64", "a3cbaa5b089e851150443806ecc33f9ad2dca236666db48edbd1059fc4b8d8b5"],
    ["macOS_x86_64", "b2dce56b1ef4f2e31fb603dfdb5999368015953bdd5b010216bc583365264633"],
    ["Linux_arm64", "c9a16392dc229f9abf471ba82be6928a4de9490735193d60fb3270ec7e61e70d"],
    ["Linux_x86_64", "41e3194dde966648e0943c13230c7a9c9874a7a235ac507dbf4ad4f575fa54fc"],
    ["Windows_arm64", "781f6fe0c6cc227eccbb7ff51d7c7e5139dd36db0b94369b79b89dda2a63e92f"],
    ["Windows_x86_64", "8a2170aee866282e8b21d4f278ffeb10c82838fbef9211279fedbde2cb87d278"],
]);

export function validFlyApp(app: string): boolean {
    return /^[a-z][a-z0-9-]{0,61}[a-z0-9]$/.test(app);
}

/** Release checksums from superfly/flyctl v0.5.12's checksums.txt. */
export async function ensureFly(
    platform = `${{ darwin: "macOS", linux: "Linux", win32: "Windows" }[process.platform as string]}_${{ arm64: "arm64", x64: "x86_64" }[process.arch as string]}`,
    source = `https://github.com/superfly/flyctl/releases/download/v${FLY_VERSION}/flyctl_${FLY_VERSION}_${platform}.${platform.startsWith("Windows") ? "zip" : "tar.gz"}`,
): Promise<string> {
    if (!FLY_ARCHIVES[platform]) throw new Error(`Unsupported flyctl platform: ${platform}`);
    const directory = join(
        process.env.XDG_CACHE_HOME ?? join(homedir(), ".cache"),
        "shallot",
        "flyctl",
        FLY_VERSION,
        platform,
    );
    const executable = platform.startsWith("Windows") ? "flyctl.exe" : "flyctl";
    const path = join(directory, executable);
    if (existsSync(join(directory, ".complete"))) return path;
    const response = await fetch(source);
    if (!response.ok) throw new Error(`Fetching flyctl failed: ${response.status}`);
    const archive = Buffer.from(await response.arrayBuffer());
    const hash = createHash("sha256").update(archive).digest("hex");
    if (hash !== FLY_ARCHIVES[platform])
        throw new Error(
            `flyctl SHA-256 ${hash} differs from ${FLY_ARCHIVES[platform]}; refusing to run it.`,
        );
    mkdirSync(dirname(directory), { recursive: true });
    const staging = mkdtempSync(`${directory}.partial-`);
    try {
        if (platform.startsWith("Windows")) extractZip(archive, staging);
        else {
            const tar = join(staging, "archive.tar.gz");
            writeFileSync(tar, archive);
            const result = spawnSync("tar", ["-xzf", tar, "-C", staging, executable]);
            if (result.status !== 0) throw new Error("Could not extract flyctl");
            rmSync(tar);
            chmodSync(join(staging, executable), 0o755);
        }
        writeFileSync(join(staging, ".complete"), "");
        if (!existsSync(join(directory, ".complete"))) {
            rmSync(directory, { recursive: true, force: true });
            renameSync(staging, directory);
        }
    } finally {
        rmSync(staging, { recursive: true, force: true });
    }
    return path;
}

export function runFly(cli: string, args: string[], cwd: string, terminal = false): string {
    const result = spawnSync(cli, args, {
        cwd,
        encoding: "utf8",
        stdio: terminal ? ["inherit", 2, 2] : ["ignore", "pipe", "pipe"],
    });
    if (result.error) throw result.error;
    if (result.status !== 0)
        throw new Error(
            `fly ${args.join(" ")} failed:\n${result.stderr ?? ""}${result.stdout ?? ""}`,
        );
    return result.stdout?.trim() ?? "";
}

/** Refuse project-owned configuration before authentication or a remote mutation. */
export function refuseFlyConfig(project: string): void {
    for (const name of ["fly.toml", "fly.json", "fly.yaml", "fly.yml"]) {
        if (existsSync(join(project, name)))
            throw new Error(
                `Fly config found: ${name}; publish refuses to replace project-owned Fly configuration.`,
            );
    }
}

export function flyOrganizations(output: string): string[] {
    const orgs: unknown = JSON.parse(output);
    if (
        !orgs ||
        typeof orgs !== "object" ||
        Array.isArray(orgs) ||
        !Object.keys(orgs).length ||
        Object.values(orgs).some((name) => typeof name !== "string")
    )
        throw new Error("Fly reported no accessible organizations");
    return Object.keys(orgs);
}

export async function deployFlyServer(
    project: string,
    work: string,
    link: FlyLink,
    interactive: boolean,
    authorized: boolean,
    ask: (question: string) => Promise<string>,
    cli: string,
): Promise<void> {
    refuseFlyConfig(project);
    if (!process.env.FLY_API_TOKEN) {
        if (!interactive)
            throw new Error("Fly login is missing. Set FLY_API_TOKEN for non-interactive publish.");
        try {
            runFly(cli, ["auth", "whoami"], work);
        } catch {
            runFly(cli, ["auth", "login"], work, true);
        }
    }
    const apps: Record<"Name", string>[] = JSON.parse(
        runFly(cli, ["apps", "list", "--json"], work),
    );
    if (!Array.isArray(apps) || apps.some((row) => typeof row.Name !== "string"))
        throw new Error("Unrecognized fly apps list output");
    const exists = apps.some((row) => row.Name === link.app);
    if (exists && !authorized)
        throw new Error(
            `Fly app "${link.app}" already exists; refusing a first link. A hand-written server link authorizes updating it.`,
        );
    await buildFlyServer(project, work, link.app);
    if (!exists) {
        const orgs = flyOrganizations(runFly(cli, ["orgs", "list", "--json"], work));
        if (orgs.length > 1 && !interactive)
            throw new Error(
                "More than one Fly organization is accessible; publish in a terminal to choose one.",
            );
        const org =
            orgs.length === 1 ? orgs[0]! : await ask(`Fly organization (${orgs.join(", ")})`);
        if (!orgs.includes(org)) throw new Error("Fly organization is not accessible");
        runFly(cli, ["apps", "create", link.app, "--org", org], work);
        runFly(cli, ["ips", "allocate-v4", "--shared", "--app", link.app], work);
    }
    runFly(
        cli,
        [
            "deploy",
            "--app",
            link.app,
            "--config",
            join(work, "fly.toml"),
            "--remote-only",
            "--ha=false",
            "--yes",
        ],
        work,
    );
}

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
primary_region = "iad"
[env]
  PORT = "8080"
[http_service]
  internal_port = 8080
  force_https = true
  auto_stop_machines = "stop"
  auto_start_machines = true
  min_machines_running = 0
[http_service.concurrency]
  type = "connections"
  soft_limit = 20
[[vm]]
  size = "shared-cpu-1x"
  memory = "256mb"
[[restart]]
  policy = "never"
`;
}
