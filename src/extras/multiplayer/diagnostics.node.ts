import { expect, setDefaultTimeout, test } from "bun:test";
import { TIMEOUT } from "@dylanebert/shallot/testing";
import { VisualCorrection } from "../../core/rendering";
import { GlobalTransform } from "../../core/transform";
import { createApp, DiagnosticsStore, type World } from "../../engine";
import {
    ClientPrediction,
    type Game,
    IntentSender,
    PredictionPlugin,
    PredictionTarget,
    Replicate,
    ReplicationSender,
    ServerSession,
    seededLink,
    server,
    WebSocketConnection,
} from ".";
import { serveWebSocket } from "./bun";
import { observeNetwork } from "./diagnostics";
import { createProxy } from "./proxy";
import { createSession } from "./testing";

setDefaultTimeout(TIMEOUT.node);
const game: Game = {
    plugins: [],
    schema: [],
    maxClients: 4,
    rules: () => [],
    validateIntent: () => true,
    applyIntent() {},
};
const row = (world: World, path: string) => world.resource(DiagnosticsStore).get(path)!;

for (const delay of [0, 100]) {
    test(`idle and thin WebSocket clients measure symmetric ping round trip with ${delay} ms proxy delay`, async () => {
        const session = await createSession(game, {
            clients: 1,
            thinClients: 1,
            transport: "websocket",
            proxy: true,
        });
        const runtime = session.server.world.resource(ServerSession).server!;
        try {
            for (const peer of [...session.clients, ...session.thinClients]) {
                peer.proxy!.upstream.latency = delay / 2;
                peer.proxy!.downstream.latency = delay / 2;
            }
            // Let the pre-fault join traffic drain, then require fresh samples from every end.
            const pump = async (milliseconds: number) => {
                const end = performance.now() + milliseconds;
                do {
                    await session.advance();
                    await Bun.sleep(4);
                } while (performance.now() < end);
            };
            await pump(250);
            const rows = [
                ...[...session.clients, ...session.thinClients].map((peer) =>
                    row(peer.world, "network/round_trip"),
                ),
                ...[...runtime.peers.keys()].map((eid) =>
                    row(session.server.world, `network/connections/${eid}/round_trip`),
                ),
            ];
            for (const diagnostic of rows) diagnostic.clearHistory();
            await pump(350);
            for (const diagnostic of rows) {
                expect(diagnostic.historyLength).toBeGreaterThan(0);
                for (const sample of diagnostic.measurements()) {
                    if (delay) expect(sample.value).toBeGreaterThanOrEqual(100);
                    else expect(sample.value).toBeLessThan(100);
                }
            }
            expect(session.thinClients[0]!.world.time.fixedTick).toBe(0);
            const eid = runtime.peers.keys().next().value!;
            const age = row(session.server.world, `network/connections/${eid}/time_since_receive`);
            session.clients[0]!.proxy!.upstream.stalled = true;
            await pump(180);
            expect(age.value()).toBeGreaterThanOrEqual(150);
            session.clients[0]!.proxy!.upstream.stalled = false;
            const release = performance.now();
            do {
                await pump(5);
            } while (age.value()! >= 50 && performance.now() - release < delay + 200);
            expect(age.value()).toBeLessThan(50);
        } finally {
            await session.dispose();
        }
    });
}

test("departed connection rows are removed before its eid is recycled", async () => {
    const app = await createApp({ defaults: false, plugins: [server(game)] });
    const runtime = app.world.resource(ServerSession).server!;
    const faults = { minDelay: 0, maxDelay: 0, loss: 0, duplication: 0 };
    try {
        const first = runtime.connect(seededLink(1, faults).a)!;
        const prefix = `network/connections/${first}/`;
        expect(row(app.world, prefix + "bytes_sent").value()).toBeGreaterThan(0);
        expect(row(app.world, "network/peer_count").value()).toBe(1);
        runtime.disconnect(runtime.peers.get(first)!.clientId);
        expect(
            [...app.world.resource(DiagnosticsStore).iter()].some((d) =>
                d.path.path.startsWith(prefix),
            ),
        ).toBe(false);
        expect(row(app.world, "network/peer_count").value()).toBe(0);
        const second = runtime.connect(seededLink(2, faults).a)!;
        expect(second).toBe(first);
        expect(row(app.world, prefix + "round_trip").historyLength).toBe(0);
        expect(row(app.world, prefix + "bytes_received").value()).toBe(0);
        expect(row(app.world, prefix + "bytes_sent").historyLength).toBe(3);
    } finally {
        app.dispose();
    }
});

test("correction diagnostics measure the largest applied translation and shortest rotation error", async () => {
    const a = await createApp({ defaults: false, plugins: [PredictionPlugin] });
    const b = await createApp({ defaults: false, plugins: [PredictionPlugin] });
    const link = seededLink(1, { minDelay: 0, maxDelay: 0, loss: 0, duplication: 0 });
    try {
        const remote = a.world.create(),
            local = b.world.create();
        a.world.add(remote, GlobalTransform, {
            translation: [3, 4, 0],
            rotation: [0, 0, Math.SQRT1_2, Math.SQRT1_2],
        });
        a.world.add(remote, Replicate);
        b.world.add(local, GlobalTransform);
        b.world.add(local, PredictionTarget);
        const prediction = new ClientPrediction(
            b.world,
            [GlobalTransform],
            link.b,
            () => local,
            () => {},
        );
        prediction.receiver.entities.set(1, local);
        const sender = new ReplicationSender(
            a.world,
            [GlobalTransform],
            [new Map([[1, [GlobalTransform]]])],
            1,
            link.a,
        );
        sender.record(0);
        sender.flush();
        prediction.receiver.flush();
        expect(row(b.world, "network/correction_translation").value()).toBeCloseTo(5, 5);
        expect(row(b.world, "network/correction_rotation").value()).toBeCloseTo(Math.PI / 2, 5);
        const offset = b.world.get(local, VisualCorrection)!;
        expect(Math.hypot(...offset.translation)).toBeCloseTo(5, 5);
        expect(2 * Math.acos(Math.abs(offset.rotation[3]!))).toBeCloseTo(Math.PI / 2, 5);
    } finally {
        a.dispose();
        b.dispose();
    }
});

test("loss raises peak replayed ticks per drain and its cumulative counter equals actual repeated ticks", async () => {
    const totals: number[] = [];
    for (const loss of [0, 0.2]) {
        let highWater = 0;
        let repeated = 0;
        let prediction: ClientPrediction;
        let input: IntentSender;
        const session = await createSession(
            {
                ...game,
                plugins: [PredictionPlugin],
                onClient(world, connection) {
                    input = new IntentSender(connection);
                    prediction = new ClientPrediction(
                        world,
                        [],
                        connection,
                        () => undefined,
                        () => {},
                        (w, _outputs, confirmed) => {
                            if (confirmed) return;
                            if (w.time.fixedTick <= highWater) repeated++;
                            highWater = Math.max(highWater, w.time.fixedTick);
                        },
                    );
                    prediction.onMessages = (messages) => {
                        input.flush(messages);
                    };
                },
            },
            { clients: 1, seed: 27, faults: { minDelay: 17, maxDelay: 17, loss, duplication: 0 } },
        );
        try {
            await session.advance();
            let peak = 0;
            let previous = 0;
            for (let tick = 0; tick < 400; tick++) {
                const record = input!.record(
                    session.clients[0]!.world.time.fixedTick + 1,
                    "input",
                    [1],
                );
                prediction!.accept([record]);
                input!.flush();
                await session.advance();
                const current = row(session.clients[0]!.world, "network/replayed_ticks").value()!;
                peak = Math.max(peak, current - previous);
                previous = current;
            }
            const measured = row(session.clients[0]!.world, "network/replayed_ticks").value()!;
            expect(measured).toBe(repeated);
            totals.push(peak);
        } finally {
            await session.dispose();
        }
    }
    expect(totals[1]).toBeGreaterThan(totals[0]!);
});

test("WebSocket diagnostic counters exactly match independent proxy payload counts", async () => {
    const authority = await createApp({ defaults: false, plugins: [] });
    const client = await createApp({ defaults: false, plugins: [] });
    let accepted: import("./bun").ServerWebSocketConnection | undefined;
    const endpoint = serveWebSocket({
        hostname: "127.0.0.1",
        port: 0,
        connect(connection) {
            accepted = connection;
        },
        disconnect() {},
    });
    const proxy = await createProxy(endpoint.port!, 1);
    const socket = new WebSocket(`ws://127.0.0.1:${proxy.port}`);
    try {
        await new Promise<void>((resolve, reject) => {
            socket.onopen = () => resolve();
            socket.onerror = () => reject(new Error("socket failed"));
        });
        const connection = new WebSocketConnection(socket);
        const a = observeNetwork(authority.world, accepted!, "network/connections/1/");
        const b = observeNetwork(client.world, connection);
        // Includes extended WebSocket length headers, masking and both directions' control traffic.
        accepted!.send(new Uint8Array(70000));
        connection.send(new Uint8Array(300));
        for (let i = 0; i < 30; i++) {
            a.poll();
            b.poll();
            await Bun.sleep(1);
        }
        for (const [world, prefix, sent, received] of [
            [
                authority.world,
                "network/connections/1/",
                proxy.counts.downstream,
                proxy.counts.upstream,
            ],
            [client.world, "network/", proxy.counts.upstream, proxy.counts.downstream],
        ] as const) {
            expect(row(world, prefix + "bytes_sent").value()).toBe(sent.bytes);
            expect(row(world, prefix + "messages_sent").value()).toBe(sent.messages);
            expect(row(world, prefix + "bytes_received").value()).toBe(received.bytes);
            expect(row(world, prefix + "messages_received").value()).toBe(received.messages);
        }
    } finally {
        socket.close();
        proxy.stop();
        await endpoint.stop(true);
        proxy.forceClose();
        authority.dispose();
        client.dispose();
    }
});
