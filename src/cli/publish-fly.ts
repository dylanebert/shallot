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
const serverPort = 8080;
const machineMemory = 256;
const connectionLimit = 20;
const stopSignal = "SIGTERM";
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

function runFly(
    cli: string,
    args: string[],
    cwd: string,
    terminal = false,
    timeout?: number,
): { stdout: string; stderr: string } {
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
    return { stdout: result.stdout?.trim() ?? "", stderr: result.stderr ?? "" };
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
    ["image_ref"]?: unknown;
    config?: {
        image?: unknown;
        guest?: Record<string, unknown>;
    };
}

interface FlyVolume {
    id: string;
    ["size_gb"]: number;
    ["auto_backup_enabled"]?: boolean;
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
            const rows: unknown = JSON.parse(runFly(cli, args, work, false, timeout).stdout);
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
        if (n === 0) return `missing ${noun}`;
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
            memory >= machineMemory
                ? `about $${(2.19 + ((memory - machineMemory) * 6) / 1024).toFixed(2)}/month if running continuously, plus storage and transfer`
                : "monthly compute depends on size, region and uptime; estimate at https://fly.io/calculator/";
        report.push(resource(machine.id, "machine", cost));
    }
    throw new Error(report.join("\n"));
}

const flyAssignedFields = new Set([
    // Machine identity and network address.
    "machine.id",
    "machine.name",
    "machine.private_ip",
    // Lifecycle and allocation observations, not Machine configuration.
    "machine.instance_id",
    "machine.version",
    "machine.created_at",
    "machine.updated_at",
    "machine.events",
    "machine.state",
    // Fly Launch stamps this metadata; the platform and process group remain checked.
    "machine.config.metadata.fly_builder_id",
    "machine.config.metadata.fly_flyctl_version",
    "machine.config.metadata.fly_release_id",
    "machine.config.metadata.fly_release_version",
    // Volume identity, creation time and physical placement within its checked region.
    "volume.id",
    "volume.created_at",
    "volume.zone",
]);

function compareFlyFields(
    expected: unknown,
    actual: unknown,
    path: string,
    report: (difference: string) => void,
): void {
    if (flyAssignedFields.has(path) || expected === actual) return;
    if (record(expected) && record(actual)) {
        for (const key of new Set([...Object.keys(expected), ...Object.keys(actual)])) {
            const field = /^[a-zA-Z_][a-zA-Z_0-9]*$/.test(key)
                ? `.${key}`
                : `[${JSON.stringify(key)}]`;
            compareFlyFields(
                Object.hasOwn(expected, key) ? expected[key] : undefined,
                Object.hasOwn(actual, key) ? actual[key] : undefined,
                `${path}${field}`,
                report,
            );
        }
    } else if (
        Array.isArray(expected) &&
        Array.isArray(actual) &&
        expected.length === actual.length
    ) {
        for (const [index, value] of expected.entries())
            compareFlyFields(value, actual[index], `${path}[${index}]`, report);
    } else {
        const display = (value: unknown) => {
            if (value === undefined) return "absent";
            if (/^machine\.config\.env(?:[.[]|$)/.test(path) && value === actual)
                return "present (value withheld)";
            return JSON.stringify(value);
        };
        report(`${path}: expected ${display(expected)}, got ${display(actual)}`);
    }
}

// Machine config emitted by Fly Launch for flyConfig with the pinned flyctl.
function canonicalFlyMachine(volume: FlyVolume) {
    return {
        env: {
            FLY_PROCESS_GROUP: "app",
            PRIMARY_REGION: flyRegion,
            PORT: String(serverPort),
            SHALLOT_DATA_DIR: dataPath,
        },
        init: {},
        guest: { ["cpu_kind"]: "shared", cpus: 1, ["memory_mb"]: machineMemory },
        metadata: { ["fly_platform_version"]: "v2", ["fly_process_group"]: "app" },
        mounts: [
            {
                encrypted: true,
                path: dataPath,
                ["size_gb"]: volumeSize,
                volume: volume.id,
                name: volumeName,
            },
        ],
        services: [
            {
                protocol: "tcp",
                ["internal_port"]: serverPort,
                autostop: true,
                autostart: true,
                ["min_machines_running"]: 0,
                ports: [
                    { port: 80, handlers: ["http"], ["force_https"]: true },
                    { port: 443, handlers: ["http", "tls"] },
                ],
                concurrency: { type: "connections", ["soft_limit"]: connectionLimit },
                ["force_instance_key"]: null,
            },
        ],
        restart: { policy: "no" },
        ["stop_config"]: { signal: stopSignal },
    };
}

function canonicalFlyVolume(machine: FlyMachine | undefined) {
    return {
        name: volumeName,
        state: "created",
        ["size_gb"]: volumeSize,
        region: flyRegion,
        encrypted: true,
        ["attached_machine_id"]: machine?.id ?? null,
        ["attached_alloc_id"]: null,
        ["host_dedication_id"]: "",
        ["snapshot_retention"]: 5,
        ["auto_backup_enabled"]: false,
        ["host_status"]: "ok",
    };
}

function deployedFlyTag(stderr: string, app: string): string {
    // Pinned flyctl's deploy build summary prints the same img.Tag passed to Machine deployment.
    const prefix = `image: registry.fly.io/${app}:`;
    const tags = stderr.split(/\r?\n/).filter((line) => line.startsWith(prefix));
    const tag = tags.length === 1 ? tags[0]?.slice(prefix.length) : undefined;
    if (!tag || !/^deployment-[0-9A-HJKMNP-TV-Z]{26}$/.test(tag))
        throw new Error(`Fly deploy did not report one release image for "${app}".`);
    return tag;
}

function compareFlyImage(
    app: string,
    tag: string,
    machine: FlyMachine,
    report: (difference: string) => void,
): void {
    const registry = "registry.fly.io";
    const image = `${registry}/${app}:${tag}`;
    compareFlyFields(image, machine.config?.image, "machine.config.image", report);
    const ref = machine.image_ref;
    const expected = { registry, repository: app, tag };
    compareFlyFields(
        expected,
        record(ref) ? { registry: ref.registry, repository: ref.repository, tag: ref.tag } : ref,
        "machine.image_ref",
        report,
    );
    if (
        record(ref) &&
        (typeof ref.digest !== "string" || !/^sha256:[a-f0-9]{64}$/.test(ref.digest))
    )
        report(
            `machine.image_ref.digest: expected a resolved SHA-256 digest, got ${JSON.stringify(ref.digest) ?? "absent"}`,
        );
}

// flyctl 0.5.12 uses fly-go 0.12.1's Machine states. Proxy autostart wakes stopped or suspended Machines.
const routableMachineStates = ["started", "stopped", "suspended"];

function verifyFlyEnvelope(
    cli: string,
    work: string,
    app: string,
    gate: { phase: "before deploy"; pending?: boolean } | { phase: "after deploy"; tag?: string },
): boolean {
    const inventory = readFlyInventory(cli, work, app);
    const { machines, volumes } = inventory;
    const initializing = gate.phase === "before deploy" && !!gate.pending && machines.count === 0;
    const differences: string[] = [];
    const machine = machines.rows[0];
    const volume = volumes.rows[0];
    if (machines.count === 1 && machine) {
        const report = (difference: string) => differences.push(`${machine.id}: ${difference}`);
        if (!routableMachineStates.includes(machine.state ?? ""))
            report(
                `machine.state: expected one of ${routableMachineStates.join(", ")}, got ${JSON.stringify(machine.state) ?? "absent"}`,
            );
        if (volumes.count === 1 && volume) {
            const { image_ref: _ref, config, ...fields } = machine;
            const { image: _image, ...settings } = config ?? {};
            compareFlyFields(
                {
                    region: flyRegion,
                    ["host_status"]: "ok",
                    cordoned: false,
                    config: canonicalFlyMachine(volume),
                },
                { ...fields, config: config && settings },
                "machine",
                report,
            );
        }
        if (gate.phase === "after deploy" && gate.tag)
            compareFlyImage(app, gate.tag, machine, report);
    }
    if (volumes.count === 1 && volume && (initializing || machines.count === 1))
        compareFlyFields(canonicalFlyVolume(machine), volume, "volume", (difference) =>
            differences.push(`${volume.id}: ${difference}`),
        );
    if (
        machines.count === undefined ||
        volumes.count === undefined ||
        (initializing ? volumes.count > 1 : machines.count !== 1 || volumes.count !== 1) ||
        differences.length
    )
        refuseFlyInventory(cli, app, inventory, gate.phase, differences);
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
        runFly(cli, ["apps", "list", "--json"], work).stdout,
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
        const initializing = verifyFlyEnvelope(cli, work, link.app, {
            phase: "before deploy",
            pending: recovery?.pending,
        });
        const ips: Record<"Type", string>[] = JSON.parse(runFly(cli, ipArgs, work).stdout) ?? [];
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
        const orgs = flyOrganizations(runFly(cli, ["orgs", "list", "--json"], work).stdout);
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
    let tag: string | undefined;
    try {
        const { stderr } = runFly(
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
        tag = deployedFlyTag(stderr, link.app);
    } catch (error) {
        deployFailure = error;
    }
    try {
        verifyFlyEnvelope(cli, work, link.app, { phase: "after deploy", tag });
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
kill_signal = "${stopSignal}"
[env]
  PORT = "${serverPort}"
  SHALLOT_DATA_DIR = "${dataPath}"
[mounts]
  source = "${volumeName}"
  destination = "${dataPath}"
  initial_size = "${volumeSize}GB"
  scheduled_snapshots = false
[http_service]
  internal_port = ${serverPort}
  force_https = true
  auto_stop_machines = "stop"
  auto_start_machines = true
  min_machines_running = 0
[http_service.concurrency]
  type = "connections"
  soft_limit = ${connectionLimit}
[[vm]]
  size = "shared-cpu-1x"
  memory = "${machineMemory}mb"
[[restart]]
  policy = "never"
`;
}
