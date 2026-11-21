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

const flyRegion = "iad";
const volumeName = "shallot_data";
const volumeSize = 1;
const dataPath = "/data";
const volumeUSDPerGBMonth = 0.15;

function volumeMonthlyCost(size: number): string {
    return `$${(size * volumeUSDPerGBMonth).toFixed(2)}/month`;
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

export function flyCommands(cli: string, app: string): string {
    return `Fly CLI: "${cli}"\nServer logs: "${cli}" logs --app ${app}\nServer rollback: check out an earlier commit, then bunx shallot publish\nConfirm with the person before deleting the server.\nServer deletion: "${cli}" apps destroy ${app}`;
}

export function flyRetainedApp(cli: string, app: string): string {
    return `Fly app "${app}" is not deleted by publish. Its ${volumeSize} GB volume, if provisioned, costs ${volumeMonthlyCost(volumeSize)} even while stopped. Confirm with the person before deleting it: "${cli}" apps destroy ${app}`;
}

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

export function runFly(
    cli: string,
    args: string[],
    cwd: string,
    terminal = false,
    timeout?: number,
): string {
    const result = spawnSync(cli, args, {
        cwd,
        timeout,
        killSignal: "SIGKILL",
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

interface FlyMachine {
    id: string;
    region?: string;
    state?: string;
    config?: {
        guest?: Record<string, unknown>;
        metadata?: Record<string, unknown>;
        mounts?: { volume?: string; path?: string }[];
    };
}

interface FlyVolume {
    id: string;
    name?: string;
    ["size_gb"]: number;
    region?: string;
    ["auto_backup_enabled"]?: boolean;
    ["attached_machine_id"]?: string | null;
    ["attached_alloc_id"]?: string | null;
    ["host_status"]?: string;
}

function record(value: unknown): value is Record<string, unknown> {
    return typeof value === "object" && value !== null && !Array.isArray(value);
}

function integerAtLeast(value: unknown, minimum: number): boolean {
    return typeof value === "number" && Number.isSafeInteger(value) && value >= minimum;
}

function isFlyVolume(row: unknown): row is FlyVolume {
    return (
        record(row) &&
        typeof row.id === "string" &&
        /^vol_[a-z0-9]{16}$/.test(row.id) &&
        integerAtLeast(row.size_gb, 1)
    );
}

function isFlyMachine(row: unknown): row is FlyMachine {
    if (!record(row) || typeof row.id !== "string" || !/^[a-f0-9]{14}$/.test(row.id)) return false;
    if (row.region !== undefined && typeof row.region !== "string") return false;
    if (row.config === undefined) return true;
    if (!record(row.config)) return false;
    const guest = row.config.guest;
    if (guest === undefined) return true;
    return (
        record(guest) &&
        (guest.cpu_kind === undefined || typeof guest.cpu_kind === "string") &&
        (guest.cpus === undefined || integerAtLeast(guest.cpus, 1)) &&
        (guest.memory_mb === undefined || integerAtLeast(guest.memory_mb, 1)) &&
        (guest.gpus === undefined || integerAtLeast(guest.gpus, 0)) &&
        (guest.gpu_kind === undefined || typeof guest.gpu_kind === "string")
    );
}

function readFlyInventory(cli: string, work: string, app: string) {
    const errors: string[] = [];
    const timeout = 15_000;
    function list<T>(kind: string, valid: (row: unknown) => row is T) {
        const parsed: T[] = [];
        const args = [kind, "list", "--app", app, "--json"];
        try {
            const rows: unknown = JSON.parse(runFly(cli, args, work, false, timeout));
            if (rows === null) return { rows: parsed, count: 0 };
            if (!Array.isArray(rows))
                throw new Error(`expected an inventory array from ${args.join(" ")}`);
            for (const [index, row] of rows.entries()) {
                if (valid(row)) parsed.push(row);
                else errors.push(`${kind} list: invalid row ${index + 1}`);
            }
            return { rows: parsed, count: parsed.length === rows.length ? rows.length : undefined };
        } catch (error) {
            errors.push(
                `${kind} list: ${error instanceof Error && "code" in error && error.code === "ETIMEDOUT" ? `timed out after ${timeout / 1000} seconds` : String(error)}`,
            );
            return { rows: parsed, count: undefined };
        }
    }
    const volumes = list("volumes", isFlyVolume);
    const machines = list("machine", isFlyMachine);
    return { volumes, machines, errors };
}

function singleSharedCPU(guest: Record<string, unknown> | undefined): boolean {
    return guest?.cpu_kind === "shared" && guest.cpus === 1 && !guest.gpus && !guest.gpu_kind;
}

function refuseFlyInventory(
    cli: string,
    app: string,
    { volumes, machines, errors }: ReturnType<typeof readFlyInventory>,
    phase: "before deploy" | "after deploy",
    differences: string[],
): never {
    const resource = (id: string, kind: string, cost: string) =>
        `  ${id}: ${cost}. Delete with "${cli}" ${kind} destroy ${id} --app ${app}`;
    const count = (n: number | undefined, noun: string) => {
        if (n === undefined) return `${noun} count unknown`;
        if (n === 0 && phase === "after deploy") return `missing ${noun}`;
        return `${n} ${noun}${n === 1 ? "" : "s"}`;
    };
    const report = [
        `Fly app "${app}" failed its inventory check ${phase}: ${count(volumes.count, "volume")}, ${count(machines.count, "Machine")}. Publish requires one of each after deploy.`,
        ...errors,
        ...differences,
        "Confirm with the person which resources to keep before deleting any. Publish did not delete anything.",
    ];
    for (const volume of volumes.rows) {
        const cost = `${volumeMonthlyCost(volume.size_gb)} for provisioned storage${volume.auto_backup_enabled === true ? "; snapshot charges depend on usage" : ""}`;
        report.push(resource(volume.id, "volumes", cost));
    }
    for (const machine of machines.rows) {
        const guest = machine.config?.guest;
        const memory = guest?.memory_mb;
        const cost =
            machine.region === flyRegion &&
            singleSharedCPU(guest) &&
            typeof memory === "number" &&
            memory >= 256
                ? `about $${(2.19 + ((memory - 256) * 6) / 1024).toFixed(2)}/month if running continuously, plus storage and transfer`
                : "monthly compute depends on size, region and uptime; estimate at https://fly.io/calculator/";
        report.push(resource(machine.id, "machine", cost));
    }
    throw new Error(report.join("\n"));
}

function verifyFlyEnvelope(
    cli: string,
    work: string,
    app: string,
    phase: "before deploy" | "after deploy",
    pending = false,
): boolean {
    const inventory = readFlyInventory(cli, work, app);
    const { machines, volumes } = inventory;
    const initializing = phase === "before deploy" && pending && machines.count === 0;
    const differences: string[] = [];
    const machine = machines.rows[0];
    const volume = volumes.rows[0];
    if (machines.count === 0 && !initializing) differences.push("Machine count 0; expected one");
    if (machines.count === 1 && machine) {
        const guest = machine.config?.guest;
        if (!singleSharedCPU(guest) || guest?.memory_mb !== 256)
            differences.push(
                `${machine.id}: Machine size ${JSON.stringify(guest) ?? "unknown"}; expected shared-cpu-1x 256 MB without GPUs.`,
            );
        if (machine.region !== flyRegion)
            differences.push(
                `${machine.id}: Machine region ${JSON.stringify(machine.region) ?? "unknown"}; expected ${flyRegion}.`,
            );
        const metadata = machine.config?.metadata;
        if (
            metadata?.fly_platform_version !== "v2" ||
            metadata.fly_process_group !== "app" ||
            metadata.process_group === "release_command" ||
            machine.state === "destroying" ||
            machine.state === "destroyed"
        )
            differences.push(
                `${machine.id}: Machine is not managed by Fly Launch in the app process group or is being destroyed.`,
            );
    }
    const expectedVolume = `expected one ${volumeSize} GB ${volumeName} volume in ${flyRegion}`;
    if (volumes.count === 0 && !initializing) differences.push(expectedVolume);
    if (volumes.count === 1 && volume) {
        if (volume.auto_backup_enabled !== false)
            differences.push(`${volume.id}: scheduled snapshots must be off.`);
        if (
            volume.name !== volumeName ||
            volume.size_gb !== volumeSize ||
            volume.region !== flyRegion
        )
            differences.push(
                `${volume.id}: found ${volume.size_gb} GB ${JSON.stringify(volume.name) ?? "unnamed"} volume in ${JSON.stringify(volume.region) ?? "unknown region"}; ${expectedVolume}.`,
            );
        if (initializing) {
            if (
                volume.attached_machine_id !== null ||
                volume.attached_alloc_id !== null ||
                volume.host_status !== "ok"
            )
                differences.push(
                    `${volume.id}: expected an unattached volume on a healthy host before initialization.`,
                );
        } else if (machines.count === 1 && machine) {
            const mounts = machine.config?.mounts;
            if (
                volume.attached_machine_id !== machine.id ||
                !Array.isArray(mounts) ||
                mounts.length !== 1 ||
                mounts[0]?.volume !== volume.id ||
                mounts[0]?.path !== dataPath
            )
                differences.push(
                    `${volume.id}: expected it attached to Machine ${machine.id} as its only mount at ${dataPath}.`,
                );
        }
    }
    if (
        machines.count === undefined ||
        volumes.count === undefined ||
        machines.count > 1 ||
        volumes.count > 1 ||
        differences.length
    )
        refuseFlyInventory(cli, app, inventory, phase, differences);
    return initializing;
}

export async function deployFlyServer(
    project: string,
    work: string,
    link: FlyLink,
    interactive: boolean,
    authorized: boolean,
    ask: (question: string) => Promise<string>,
    cli: string,
    recovery?: { pending: boolean; created: () => void },
): Promise<void> {
    refuseFlyConfig(project);
    if (!process.env.FLY_API_TOKEN) {
        try {
            runFly(cli, ["auth", "whoami"], work);
        } catch {
            if (!interactive)
                throw new Error(
                    "Fly login is missing. Run fly auth login in a terminal or set FLY_API_TOKEN.",
                );
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
    let needsSharedIP = !exists;
    if (exists) {
        const ipArgs = ["ips", "list", "--app", link.app, "--json"];
        const refuse = (reason: string, args: string[]): never => {
            throw new Error(
                `Fly app "${link.app}" is outside publish's spend envelope: ${reason}. Inspect with "${cli}" ${args.join(" ")}. Publish did not change the app.`,
            );
        };
        const initializing = verifyFlyEnvelope(
            cli,
            work,
            link.app,
            "before deploy",
            recovery?.pending,
        );
        const ips: Record<"Type", string>[] = JSON.parse(runFly(cli, ipArgs, work)) ?? [];
        if (!Array.isArray(ips)) refuse("unrecognized IP inventory", ipArgs);
        const outside = ips.filter((ip) => ip?.Type !== "shared_v4" && ip?.Type !== "v6");
        if (outside.length)
            refuse(
                `IP inventory contains ${outside.map((ip) => ip?.Type ?? "unknown").join(", ")}; only shared IPv4 and free IPv6 are supported`,
                ipArgs,
            );
        const shared = ips.filter((ip) => ip?.Type === "shared_v4").length;
        needsSharedIP = !!initializing && shared === 0;
        if (!needsSharedIP && shared !== 1)
            refuse("IP inventory must contain one shared IPv4", ipArgs);
    }
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
        recovery?.created();
    }
    if (needsSharedIP) runFly(cli, ["ips", "allocate-v4", "--shared", "--app", link.app], work);
    let deployFailure: unknown;
    try {
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
    } catch (error) {
        deployFailure = error;
    }
    try {
        verifyFlyEnvelope(cli, work, link.app, "after deploy");
    } catch (error) {
        if (deployFailure)
            throw new Error(`${String(deployFailure)}\n${String(error)}`, { cause: deployFailure });
        throw error;
    }
    if (deployFailure) throw deployFailure;
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
primary_region = "${flyRegion}"
kill_signal = "SIGTERM"
[env]
  PORT = "8080"
  SHALLOT_DATA_DIR = "${dataPath}"
[mounts]
  source = "${volumeName}"
  destination = "${dataPath}"
  initial_size = "${volumeSize}GB"
  scheduled_snapshots = false
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
