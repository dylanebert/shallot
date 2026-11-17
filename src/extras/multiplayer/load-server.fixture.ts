import { Time } from "../../engine";
import { serveGame } from "./bun";
import { ServerSession } from "./game";
import { collectionSize, counters, loadGame, percentiles } from "./load.fixture";

const [peers, seconds, interval] = process.argv.slice(2).map(Number) as [number, number, number];
const server = await serveGame(loadGame(peers), { hostname: "127.0.0.1", port: 0 });
const authority = server.world.resource(ServerSession).server!;
const samples: number[] = [];
const tick = server.world.tick.bind(server.world);
server.world.tick = () => {
    const before = performance.now();
    tick();
    if (start !== undefined) samples.push(performance.now() - before);
};
let start: number | undefined;
let firstTick = 0;
let nextReport = interval * 1000;
console.log(JSON.stringify({ kind: "ready", port: server.port }));
function report(elapsed: number, lag: number, done: boolean) {
    Bun.gc(true);
    let history = 0,
        queue = 0;
    const bytes = [];
    for (const peer of authority.peers.values()) {
        history = Math.max(
            history,
            collectionSize(peer.sender, "_pending"),
            peer.intent.inputs.size,
            peer.intent.records.size,
        );
        queue = Math.max(queue, collectionSize(peer.connection, "_messages"));
        bytes.push(counters(peer.diagnostics));
    }
    console.log(
        JSON.stringify({
            kind: "sample",
            elapsed: elapsed / 1000,
            ticks: server.world.time.fixedTick - firstTick,
            ...percentiles(samples),
            heap: process.memoryUsage().heapUsed,
            history,
            queue,
            lag,
            bytes,
            done,
        }),
    );
    samples.length = 0;
}
const timer = setInterval(() => {
    if (start === undefined) {
        if (authority.peers.size !== peers) return;
        start = performance.now();
        firstTick = server.world.time.fixedTick;
        console.log(JSON.stringify({ kind: "started" }));
        return;
    }
    const elapsed = performance.now() - start;
    const lag = elapsed - (server.world.time.fixedTick - firstTick) * Time.FIXED_DT * 1000;
    const behind = lag > Time.MAX_FIXED_STEPS * Time.FIXED_DT * 1000;
    const done = elapsed >= seconds * 1000 || behind;
    if (elapsed >= nextReport || done) {
        report(elapsed, lag, done);
        nextReport += interval * 1000;
    }
    if (done) {
        clearInterval(timer);
        server.stop();
        if (behind) process.exitCode = 1;
    }
}, 1);
