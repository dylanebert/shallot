import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { type App, createApp, DiagnosticsStore } from "@dylanebert/shallot";
import { client } from "@dylanebert/shallot/multiplayer";
import { chromium } from "playwright";
import { ClientSession, GAME } from "./src/network";

const prefix = "shallot-tier-";
const abort = new AbortController();
const interrupt = () => abort.abort(new Error("Fly oracle interrupted"));
process.once("SIGINT", interrupt);
process.once("SIGTERM", interrupt);
const watchdog = setTimeout(interrupt, 12 * 60_000);
let directory: string | undefined;
let owned = false;
let dockerHost: string | undefined;
const appName = `${prefix}${Date.now()}`;
const apps: App[] = [];
const metrics = ["round_trip", "time_since_receive", "confirmed_tick_lag"];

async function command(cmd: string[], cwd?: string, cleanup = false, env = process.env) {
    console.log(`$ ${cmd.join(" ")}`);
    const child = Bun.spawn({
        cmd,
        cwd,
        env,
        stdout: "pipe",
        stderr: "inherit",
        signal: cleanup ? undefined : abort.signal,
        timeout: 180_000,
    });
    const output = await new Response(child.stdout).text();
    const code = await child.exited;
    if (code !== 0) throw new Error(`${cmd.join(" ")} exited ${code}`);
    return output;
}
async function fly(args: string[], cleanup = false) {
    return command(["fly", ...args], directory, cleanup, {
        ...process.env,
        ...(dockerHost === undefined ? {} : { DOCKER_HOST: dockerHost }),
    });
}
async function inventory(cleanup = false) {
    const rows: Record<"Name", string>[] = JSON.parse(
        await fly(["apps", "list", "--json"], cleanup),
    );
    if (!Array.isArray(rows) || rows.some((row) => typeof row.Name !== "string"))
        throw new Error("Unrecognized fly apps list output");
    return rows.filter((row) => row.Name.startsWith(prefix));
}
type Machine = {
    id: string;
    state: string;
    config: { guest: Record<"cpu_kind", string> & Record<"cpus" | "memory_mb", number> };
};
async function machine() {
    const rows: Machine[] = JSON.parse(await fly(["machine", "list", "--app", appName, "--json"]));
    const row = rows[0];
    if (rows.length !== 1 || !row) throw new Error(`Expected one Machine, got ${rows.length}`);
    const guest = row.config.guest;
    if (guest.cpu_kind !== "shared" || guest.cpus !== 1 || guest.memory_mb !== 256)
        throw new Error(`Unexpected Machine size: ${JSON.stringify(guest)}`);
    return row;
}
async function pause(ms: number) {
    abort.signal.throwIfAborted();
    await Bun.sleep(ms);
    abort.signal.throwIfAborted();
}
function sample(app: App) {
    return {
        confirmed: app.world.resource(ClientSession).client?.prediction.receiver.confirmed ?? 0,
        ...Object.fromEntries(
            metrics.map((name) => [
                name,
                app.world.resource(DiagnosticsStore).get(`network/${name}`)?.value() ?? null,
            ]),
        ),
    };
}
async function bunClients(url: string, count: number, seconds: number) {
    let error: Error | undefined;
    const started = performance.now();
    try {
        for (let i = 0; i < count; i++)
            apps.push(
                await createApp({
                    defaults: false,
                    plugins: [
                        client(GAME, {
                            url,
                            onError: (value) => {
                                error = value;
                            },
                        }),
                    ],
                }),
            );
        let previous = performance.now();
        const progress = apps.map(() => 0);
        const changed = apps.map(() => previous);
        let firstState: number | undefined;
        while (performance.now() - started < seconds * 1000) {
            await pause(5);
            if (error) throw error;
            const now = performance.now();
            for (const [i, app] of apps.entries()) {
                app.world.step((now - previous) / 1000);
                const row = sample(app);
                if (row.confirmed > progress[i]!) {
                    progress[i] = row.confirmed;
                    changed[i] = now;
                    firstState ??= now - started;
                }
                if (now - changed[i]! > 15_000)
                    throw new Error(`Bun client ${i} stopped confirming`);
            }
            previous = now;
        }
        if (progress.some((tick) => tick < 60))
            throw new Error("Bun clients did not confirm 60 ticks");
        console.log(JSON.stringify({ bunClients: apps.map(sample), firstStateMs: firstState }));
        return firstState;
    } catch (error) {
        console.error(
            JSON.stringify({
                failure: String(error),
                elapsedMs: performance.now() - started,
                bunClients: apps.map(sample),
            }),
        );
        throw error;
    } finally {
        for (const app of apps.splice(0)) app.dispose();
    }
}

try {
    dockerHost =
        process.env.DOCKER_HOST ??
        (
            await command([
                "docker",
                "context",
                "inspect",
                "--format",
                "{{.Endpoints.docker.Host}}",
            ])
        ).trim();
    if (!dockerHost) throw new Error("Docker endpoint is empty");
    const existing = await inventory();
    if (existing.length)
        throw new Error(
            `Refusing while verification apps exist. Remove each leftover explicitly:\n${existing.map((row) => `fly apps destroy ${row.Name} --yes`).join("\n")}`,
        );
    directory = await mkdtemp(join(tmpdir(), "shallot-fly-"));
    const build = await Bun.build({
        entrypoints: [join(import.meta.dir, "server.ts")],
        target: "bun",
        outdir: directory,
        naming: "server.js",
    });
    if (!build.success) throw new AggregateError(build.logs, "Server bundle failed");
    await Bun.write(
        join(directory, "start.ts"),
        `setTimeout(() => process.exit(0), 10 * 60_000);\nawait import("./server.js");\n`,
    );
    await Bun.write(
        join(directory, "Dockerfile"),
        `FROM oven/bun:${Bun.version}\nWORKDIR /app\nCOPY . .\nCMD ["bun", "start.ts", "--port", "8080"]\n`,
    );
    const config = `app = "${appName}"
primary_region = "iad"
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
    await Bun.write(join(directory, "fly.toml"), config);
    await command(["docker", "-H", dockerHost, "info"]);
    // Ownership starts before launch: even a partially failed launch must reach destroy.
    owned = true;
    await fly([
        "launch",
        "--name",
        appName,
        "--region",
        "iad",
        "--copy-config",
        "--no-deploy",
        "--no-db",
        "--no-redis",
        "--no-object-storage",
        "--no-github-workflow",
        "--yes",
    ]);
    // Launch may rewrite configuration; deploy only the generated limits.
    await Bun.write(join(directory, "fly.toml"), config);
    const deployed = performance.now();
    await fly([
        "deploy",
        "--app",
        appName,
        "--local-only",
        "--ha=false",
        "--vm-size",
        "shared-cpu-1x",
        "--vm-memory",
        "256",
        "--yes",
    ]);
    console.log(JSON.stringify({ deployMs: performance.now() - deployed }));
    const addresses: Record<"Type" | "Address", string>[] = JSON.parse(
        await fly(["ips", "list", "--app", appName, "--json"]),
    );
    console.log(JSON.stringify({ addresses }));
    if (
        !Array.isArray(addresses) ||
        addresses.some((address) => !["shared_v4", "v6"].includes(address.Type))
    )
        throw new Error("Only shared IPv4 and IPv6 addresses are allowed");
    await machine();
    const url = `wss://${appName}.fly.dev`;
    await bunClients(url, 2, 20);

    // GAME has two player slots; the page follows the two-Bun-client phase.
    const vite = Bun.spawn({
        cmd: [
            process.execPath,
            "x",
            "vite",
            import.meta.dir,
            "--host",
            "127.0.0.1",
            "--port",
            "31988",
            "--strictPort",
        ],
        stdout: "ignore",
        stderr: "inherit",
    });
    try {
        let ready = false;
        for (let i = 0; i < 100 && !ready; i++) {
            await pause(100);
            ready = await fetch("http://127.0.0.1:31988").then(
                (r) => r.ok,
                () => false,
            );
        }
        if (!ready) throw new Error("Vite did not start");
        const browser = await chromium.launch({
            args: ["--enable-unsafe-webgpu", "--use-angle=swiftshader", "--enable-features=Vulkan"],
        });
        try {
            const page = await browser.newPage({ acceptDownloads: true });
            const errors: string[] = [];
            page.on("pageerror", (error) => errors.push(error.message));
            await page.goto(
                `http://127.0.0.1:31988/tests/stepped.html?record&server=${encodeURIComponent(url)}`,
            );
            await page.waitForFunction(() => window.ready);
            let first: { box: number[]; confirmed: number } | undefined;
            let latest: { box: number[]; confirmed: number } | undefined;
            const end = performance.now() + 20_000;
            while (performance.now() < end) {
                abort.signal.throwIfAborted();
                latest = await page.evaluate(async () => {
                    window.game.key("KeyW", true);
                    return window.game.frames();
                });
                if (latest!.confirmed > 0) first ??= latest;
                await pause(16);
            }
            if (
                !first ||
                !latest ||
                latest.confirmed <= first.confirmed + 120 ||
                latest.box[2]! >= first.box[2]! - 1
            )
                throw new Error(
                    `Chromium did not confirm and push: ${JSON.stringify({ first, latest })}`,
                );
            console.log(JSON.stringify({ chromium: latest }));
            const download = page.waitForEvent("download");
            await page.keyboard.down("F3");
            await page.keyboard.press("r");
            await page.keyboard.up("F3");
            const recording = join(directory, "recording.json");
            await (await download).saveAs(recording);
            await command([process.execPath, join(import.meta.dir, "tests/replay.ts"), recording]);
            if (errors.length) throw new Error(errors.join("\n"));
        } finally {
            await browser.close();
        }
    } finally {
        vite.kill();
        await vite.exited;
    }
    const idle = performance.now();
    while ((await machine()).state !== "stopped") {
        if (performance.now() - idle > 7 * 60_000)
            throw new Error("Machine did not autostop within 7 minutes");
        await pause(5000);
    }
    console.log(JSON.stringify({ autostopMs: performance.now() - idle }));
    const cold = await bunClients(url, 1, 30);
    console.log(JSON.stringify({ coldFirstStateMs: cold }));
} finally {
    clearTimeout(watchdog);
    try {
        if (owned) {
            await fly(["apps", "destroy", appName, "--yes"], true);
            owned = false;
            const remaining = await inventory(true);
            if (remaining.length) {
                console.error(`Verification apps remain: ${JSON.stringify(remaining)}`);
                process.exitCode = 1;
            }
        }
    } finally {
        if (owned) console.error(`CLEANUP FAILED: run fly apps destroy ${appName} --yes`);
        if (directory) await rm(directory, { recursive: true, force: true });
        process.removeListener("SIGINT", interrupt);
        process.removeListener("SIGTERM", interrupt);
    }
}
