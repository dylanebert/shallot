import { createInterface } from "node:readline";
import { Readable } from "node:stream";
import { parseArgs } from "node:util";
import { bunChild } from "../../../scripts/bun-child";
import { type App, createApp, Time } from "../../engine";
import { IntentSender, MultiplayerPlugin, ReplicationReceiver } from ".";
import { observeNetwork } from "./diagnostics";
import { collectionSize, counters, LoadRow, loadPlugin } from "./load.fixture";
import { WebSocketConnection } from "./websocket";

const { values } = parseArgs({
    options: {
        peers: { type: "string", default: "2" },
        seconds: { type: "string", default: "60" },
        interval: { type: "string", default: "10" },
        "heap-growth": { type: "string", default: "8388608" },
        "history-growth": { type: "string", default: "32" },
    },
});
const peers = Number(values.peers),
    seconds = Number(values.seconds),
    interval = Number(values.interval);
const heapGrowth = Number(values["heap-growth"]),
    historyGrowth = Number(values["history-growth"]);
for (const value of [peers, seconds, interval, heapGrowth, historyGrowth])
    if (!Number.isSafeInteger(value) || value <= 0)
        throw new Error("Options must be positive integers");
const child = Bun.spawn({
    ...bunChild([
        new URL("./load-server.fixture.ts", import.meta.url).pathname,
        String(peers),
        String(seconds),
        String(interval),
    ]),
    stdout: "pipe",
    stderr: "inherit",
});
const clients: {
    app: App;
    connection: WebSocketConnection;
    receiver: ReplicationReceiver;
    input: IntentSender;
    diagnostics: ReturnType<typeof observeNetwork>;
    full: boolean;
}[] = [];
let timer: ReturnType<typeof setInterval> | undefined;
let start = 0,
    rounds = 0,
    maxLag = 0;
let baseline:
    | {
          heap: number;
          history: number;
          queue: number;
          clientHeap: number;
          clientHistory: number;
          clientQueue: number;
      }
    | undefined;
let completed = false;
console.log(
    JSON.stringify({
        peers,
        full: Math.ceil(peers / 2),
        thin: Math.floor(peers / 2),
        seconds,
        interval,
        heapGrowth,
        historyGrowth,
        catchUpTicks: Time.MAX_FIXED_STEPS,
    }),
);
try {
    for await (const line of createInterface({ input: Readable.from(child.stdout) })) {
        const message = JSON.parse(line);
        if (message.kind === "ready") {
            for (let i = 0; i < peers; i++) {
                const socket = new WebSocket(`ws://127.0.0.1:${message.port}`);
                const connection = new WebSocketConnection(socket);
                await new Promise<void>((resolve, reject) => {
                    socket.addEventListener("open", () => resolve(), { once: true });
                    socket.addEventListener(
                        "error",
                        () => reject(new Error("Client socket failed")),
                        { once: true },
                    );
                });
                const app = await createApp({
                    defaults: false,
                    plugins: [loadPlugin, MultiplayerPlugin],
                });
                clients.push({
                    app,
                    connection,
                    receiver: new ReplicationReceiver(app.world, [LoadRow], connection),
                    input: new IntentSender(connection),
                    diagnostics: observeNetwork(app.world, connection),
                    full: i < Math.ceil(peers / 2),
                });
            }
            start = performance.now();
            timer = setInterval(() => {
                const now = performance.now();
                let steps = 0;
                while (
                    (rounds + 1) * Time.FIXED_DT * 1000 <= now - start &&
                    steps++ < Time.MAX_FIXED_STEPS
                ) {
                    for (const client of clients) {
                        client.input.flush(client.receiver.flush());
                        if (client.full) client.app.world.step(Time.FIXED_DT);
                    }
                    rounds++;
                }
                maxLag = Math.max(
                    maxLag,
                    performance.now() - start - rounds * Time.FIXED_DT * 1000,
                );
            }, 1);
        } else if (message.kind === "sample") {
            Bun.gc(true);
            const clientHeap = process.memoryUsage().heapUsed;
            let clientHistory = 0,
                clientQueue = 0;
            for (const client of clients) {
                clientHistory = Math.max(
                    clientHistory,
                    collectionSize(client.receiver, "_versions"),
                );
                clientQueue = Math.max(
                    clientQueue,
                    collectionSize(client.receiver, "_pending"),
                    collectionSize(client.connection, "_messages"),
                );
            }
            console.log(
                JSON.stringify({
                    ...message,
                    clientHeap,
                    clientHistory,
                    clientQueue,
                    clientMaxLag: maxLag,
                    clientBytes: clients.map((client) => counters(client.diagnostics)),
                }),
            );
            if (message.lag > Time.MAX_FIXED_STEPS * Time.FIXED_DT * 1000)
                throw new Error(`Server behind bounded catch-up: ${message.lag}ms`);
            if (maxLag > Time.MAX_FIXED_STEPS * Time.FIXED_DT * 1000)
                throw new Error(
                    `Client driver cannot hold real time: ${maxLag}ms; split clients across processes`,
                );
            if (
                baseline &&
                (message.heap - baseline.heap > heapGrowth ||
                    clientHeap - baseline.clientHeap > heapGrowth ||
                    message.history - baseline.history > historyGrowth ||
                    clientHistory - baseline.clientHistory > historyGrowth ||
                    message.queue - baseline.queue > historyGrowth ||
                    clientQueue - baseline.clientQueue > historyGrowth)
            )
                throw new Error("Growth threshold exceeded");
            baseline ??= {
                heap: message.heap,
                history: message.history,
                queue: message.queue,
                clientHeap,
                clientHistory,
                clientQueue,
            };
            if (message.done) completed = true;
        }
    }
    if ((await child.exited) !== 0 || !completed)
        throw new Error("Server exited without completing load run");
} finally {
    clearInterval(timer);
    child.kill();
    await child.exited;
    for (const client of clients) {
        client.connection.close();
        client.app.dispose();
    }
}
