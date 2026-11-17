import { createInterface } from "node:readline";
import { Readable } from "node:stream";
import { type App, createApp, DiagnosticsStore, Time } from "@dylanebert/shallot";
import { client } from "@dylanebert/shallot/multiplayer";
import { ClientSession, GAME } from "./src/network";

function tc(...args: string[]) {
    const result = Bun.spawnSync(["tc", ...args]);
    if (result.exitCode !== 0) throw new Error(result.stderr.toString());
    return result.stdout.toString();
}
const qdisc = tc("qdisc", "show", "dev", "lo");
if (!qdisc.includes("netem")) throw new Error("Loopback netem is required inside unshare -rn");
const port = 31987;
const server = Bun.spawn({
    cmd: [
        process.execPath,
        `${import.meta.dir}/server.ts`,
        "--host",
        "127.0.0.1",
        "--port",
        String(port),
    ],
    stdout: "pipe",
    stderr: "inherit",
});
const apps: App[] = [];
let socketError: Error | undefined;
const names = ["round_trip", "time_since_receive", "replayed_ticks", "confirmed_tick_lag"];
const progress = [0, 0];
const changed = [performance.now(), performance.now()];
const baseline = [0, 0];
let previous = performance.now();
function sample() {
    return apps.map((app) =>
        Object.fromEntries(
            names.map((name) => [
                name,
                app.world.resource(DiagnosticsStore).get(`network/${name}`)?.value() ?? null,
            ]),
        ),
    );
}
async function step() {
    await Bun.sleep(2);
    const now = performance.now();
    const delta = (now - previous) / 1000;
    previous = now;
    if (socketError) throw socketError;
    if (server.exitCode !== null) throw new Error(`Server exited: ${server.exitCode}`);
    for (const [i, app] of apps.entries()) {
        app.world.step(delta);
        const prediction = app.world.resource(ClientSession).client?.prediction;
        if (!prediction) continue;
        const confirmed = prediction.receiver.confirmed;
        if (confirmed > progress[i]!) {
            progress[i] = confirmed;
            changed[i] = now;
        }
        if (app.world.time.fixedTick - confirmed > prediction.maxAhead)
            throw new Error(`Client ${i} prediction exceeded ${prediction.maxAhead}`);
        if (now - changed[i]! > 5000)
            throw new Error(
                `Client ${i} stopped confirming: tick=${confirmed}, idle=${now - changed[i]!}ms`,
            );
    }
}
async function phase(name: string, seconds: number) {
    const start = performance.now();
    const first = [...progress];
    const maxima = apps.map(() => Object.fromEntries(names.map((name) => [name, 0])));
    while (performance.now() - start < seconds * 1000) {
        await step();
        for (const [i, row] of sample().entries())
            for (const key of names) maxima[i]![key] = Math.max(maxima[i]![key]!, row[key] ?? 0);
    }
    console.log(
        JSON.stringify({
            phase: name,
            seconds,
            latest: sample(),
            maxima,
            confirmed: progress,
            advanced: progress.map((tick, i) => tick - first[i]!),
        }),
    );
}
try {
    for await (const line of createInterface({ input: Readable.from(server.stdout) })) {
        console.log(line);
        if (line.includes("Multiplayer server:")) break;
    }
    for (let i = 0; i < 2; i++)
        apps.push(
            await createApp({
                defaults: false,
                plugins: [
                    client(GAME, {
                        url: `ws://127.0.0.1:${port}`,
                        onError: (error) => {
                            socketError = error;
                        },
                    }),
                ],
            }),
        );
    tc("qdisc", "change", "dev", "lo", "root", "netem", "delay", "0ms", "loss", "0%");
    previous = performance.now();
    await phase("baseline", 10);
    for (const [i, row] of sample().entries()) baseline[i] = row.confirmed_tick_lag ?? 0;
    for (const loss of [1, 5, 20]) {
        tc("qdisc", "change", "dev", "lo", "root", "netem", "delay", "20ms", "loss", `${loss}%`);
        await phase(`${loss}%`, 20);
    }
    tc("qdisc", "change", "dev", "lo", "root", "netem", "delay", "0ms", "loss", "0%");
    const lifted = performance.now();
    const atLift = [...progress];
    const recovered = [false, false];
    while (!recovered.every(Boolean) && performance.now() - lifted <= 5000) {
        await step();
        for (const [i, row] of sample().entries())
            if (
                progress[i]! > atLift[i]! &&
                row.confirmed_tick_lag !== null &&
                row.confirmed_tick_lag! <= baseline[i]! &&
                row.time_since_receive !== null &&
                row.time_since_receive! < Time.FIXED_DT * 2000
            )
                recovered[i] = true;
    }
    console.log(
        JSON.stringify({
            phase: "lifted",
            recoveryMs: performance.now() - lifted,
            recovered,
            latest: sample(),
        }),
    );
    if (!recovered.every(Boolean)) throw new Error("Recovery exceeded 5s");
} catch (error) {
    console.error(
        JSON.stringify({ failure: String(error), latest: sample(), confirmed: progress }),
    );
    throw error;
} finally {
    for (const app of apps) app.dispose();
    server.kill();
    await server.exited;
}
