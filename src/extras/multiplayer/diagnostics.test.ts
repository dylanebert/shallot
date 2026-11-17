import { expect, test } from "bun:test";
import { createApp, DiagnosticsStore } from "../../engine";
import { ReplicationReceiver, ReplicationSender, seededLink } from ".";
import { observeNetwork, stampReceipt, transportClock } from "./diagnostics";
import { clientStartup, type Game } from "./game";

const value = (world: import("../../engine").World, path: string) =>
    world.resource(DiagnosticsStore).get(path)?.value();

test("link diagnostic counters equal independently counted payloads including loss and duplication", async () => {
    const a = await createApp({ defaults: false, plugins: [] });
    const b = await createApp({ defaults: false, plugins: [] });
    const link = seededLink(17, { minDelay: 2, maxDelay: 8, loss: 0.2, duplication: 0.3 });
    const counts = [link.a, link.b].map((end) => {
        const count = { sent: 0, sentBytes: 0, received: 0, receivedBytes: 0 };
        const send = end.send.bind(end),
            receive = end.receive.bind(end);
        end.send = (bytes) => {
            count.sent++;
            count.sentBytes += bytes.length;
            send(bytes);
        };
        end.receive = () => {
            const messages = receive();
            for (const bytes of messages) {
                count.received++;
                count.receivedBytes += bytes.length;
            }
            return messages;
        };
        return count;
    });
    observeNetwork(a.world, link.a, "network/connections/1/");
    observeNetwork(b.world, link.b);
    const sender = new ReplicationSender(a.world, [], [], 1, link.a);
    const receiver = new ReplicationReceiver(b.world, [], link.b);
    try {
        for (let tick = 0; tick < 300; tick++) {
            link.advance(tick * 17);
            sender.record(tick);
            sender.flush();
            receiver.flush();
        }
        for (const [i, world, prefix] of [
            [0, a.world, "network/connections/1/"],
            [1, b.world, "network/"],
        ] as const) {
            const count = counts[i]!;
            expect(value(world, prefix + "bytes_sent")).toBe(count.sentBytes);
            expect(value(world, prefix + "messages_sent")).toBe(count.sent);
            expect(value(world, prefix + "bytes_received")).toBe(count.receivedBytes);
            expect(value(world, prefix + "messages_received")).toBe(count.received);
        }
        expect(counts[0]!.sent + counts[1]!.sent).toBe(link.counts.sent);
        expect(link.counts.dropped).toBeGreaterThan(0);
        expect(link.counts.duplicated).toBeGreaterThan(0);
    } finally {
        a.dispose();
        b.dispose();
    }
});

test("ping IDs retain older pongs, subtract remote hold, reject duplicate and expired pongs", async () => {
    const app = await createApp({ defaults: false, plugins: [] });
    let now = 0;
    let incoming: Uint8Array[] = [];
    const sent: Uint8Array[] = [];
    const connection = {
        [transportClock]: () => now,
        close() {},
        send(bytes: Uint8Array) {
            sent.push(bytes.slice());
        },
        receive() {
            const messages = incoming;
            incoming = [];
            return messages;
        },
    };
    const observer = observeNetwork(app.world, connection);
    const pong = (id: number, held: number, arrival: number) => {
        const bytes = new Uint8Array(16),
            view = new DataView(bytes.buffer);
        view.setUint32(0, 7, true);
        view.setUint32(4, id, true);
        view.setFloat64(8, held, true);
        return stampReceipt(bytes, arrival);
    };
    try {
        observer.poll();
        now = 100;
        observer.poll();
        now = 400;
        incoming.push(pong(1, 30, 230));
        observer.poll();
        expect(value(app.world, "network/round_trip")).toBe(100);
        incoming.push(pong(0, 150, 350));
        observer.poll();
        expect(value(app.world, "network/round_trip")).toBe(200);
        const row = app.world.resource(DiagnosticsStore).get("network/round_trip")!;
        const length = row.historyLength;
        incoming.push(pong(0, 0, 400), pong(999, 0, 400));
        observer.poll();
        expect(row.historyLength).toBe(length);
        const ping = new Uint8Array(8),
            view = new DataView(ping.buffer);
        view.setUint32(0, 6, true);
        view.setUint32(4, 42, true);
        incoming.push(stampReceipt(ping, 250));
        observer.poll();
        const answer = new DataView(sent.at(-1)!.buffer);
        expect(answer.getUint32(0, true)).toBe(7);
        expect(answer.getUint32(4, true)).toBe(42);
        expect(answer.getFloat64(8, true)).toBe(150);
        for (let i = 0; i < 1025; i++) {
            now += 100;
            observer.poll();
        }
        incoming.push(pong(2, 0, now));
        observer.poll();
        expect(row.historyLength).toBe(length);
    } finally {
        app.dispose();
    }
});

test("a replacement client transport starts fresh counters and owns its clock", async () => {
    const app = await createApp({ defaults: false, plugins: [] });
    const faults = { minDelay: 0, maxDelay: 0, loss: 0, duplication: 0 };
    const old = seededLink(1, faults),
        fresh = seededLink(2, faults);
    try {
        const previous = observeNetwork(app.world, old.a);
        previous.poll();
        old.a.close();
        fresh.advance(500);
        const current = observeNetwork(app.world, fresh.a);
        expect(current.connection).toBe(fresh.a);
        expect(current).not.toBe(previous);
        fresh.a.send(new Uint8Array(123));
        expect(value(app.world, "network/bytes_sent")).toBe(123);
        expect(value(app.world, "network/messages_sent")).toBe(1);
        expect(value(app.world, "network/bytes_received")).toBe(0);
        expect(app.world.resource(DiagnosticsStore).get("network/round_trip")!.historyLength).toBe(
            0,
        );
        expect(
            app.world.resource(DiagnosticsStore).get("network/bytes_sent")!.measurement()!.time,
        ).toBe(0.5);
        previous.poll();
        previous.record("confirmed_tick_lag", 999);
        previous.remove();
        expect(value(app.world, "network/bytes_sent")).toBe(123);
        expect(value(app.world, "network/confirmed_tick_lag")).toBeUndefined();
    } finally {
        old.a.close();
        fresh.a.close();
        app.dispose();
    }
});

for (const remote of [false, true]) {
    test(`client rows are released on ${remote ? "observed remote" : "local"} close`, async () => {
        const app = await createApp({ defaults: false, plugins: [] });
        const link = seededLink(1, { minDelay: 0, maxDelay: 0, loss: 0, duplication: 0 });
        try {
            const observer = observeNetwork(app.world, link.a);
            observer.poll();
            (remote ? link.b : link.a).close();
            if (remote) observer.poll();
            expect(
                [...app.world.resource(DiagnosticsStore).iter()].filter((d) =>
                    d.path.path.startsWith("network/"),
                ),
            ).toEqual([]);
        } finally {
            link.a.close();
            app.dispose();
        }
    });
}

test("clientStartup follows its replacement connection provider", async () => {
    const faults = { minDelay: 0, maxDelay: 0, loss: 0, duplication: 0 };
    const old = seededLink(1, faults),
        fresh = seededLink(2, faults);
    let connection = old.a;
    const game: Game = {
        plugins: [],
        schema: [],
        maxClients: 1,
        rules: () => [],
        validateIntent: () => true,
        applyIntent() {},
    };
    const app = await createApp({
        defaults: false,
        plugins: [{ name: "Reconnect", systems: [clientStartup(game, () => connection)] }],
    });
    try {
        app.world.step(0);
        old.a.close();
        connection = fresh.a;
        fresh.advance(500);
        app.world.step(0);
        fresh.a.send(new Uint8Array(123));
        expect(value(app.world, "network/bytes_sent")).toBe(131);
        expect(value(app.world, "network/messages_sent")).toBe(2);
        expect(
            app.world.resource(DiagnosticsStore).get("network/bytes_sent")!.measurement()!.time,
        ).toBe(0.5);
    } finally {
        old.a.close();
        fresh.a.close();
        app.dispose();
    }
});

test("impossible and malformed pongs preserve the outstanding ID without recording a sample", async () => {
    const app = await createApp({ defaults: false, plugins: [] });
    let now = 0;
    let incoming: Uint8Array[] = [];
    const connection = {
        [transportClock]: () => now,
        close() {},
        send() {},
        receive() {
            const messages = incoming;
            incoming = [];
            return messages;
        },
    };
    const pong = (id: number, hold: number) => {
        const bytes = new Uint8Array(16),
            view = new DataView(bytes.buffer);
        view.setUint32(0, 7, true);
        view.setUint32(4, id, true);
        view.setFloat64(8, hold, true);
        return stampReceipt(bytes, now);
    };
    try {
        const observer = observeNetwork(app.world, connection);
        observer.poll();
        const row = app.world.resource(DiagnosticsStore).get("network/round_trip")!;
        now = 50;
        for (const hold of [1e300, 51, -1, NaN, Infinity]) {
            incoming.push(pong(0, hold));
            observer.poll();
            expect(row.historyLength).toBe(0);
        }
        incoming.push(pong(999, 0), pong(0, 0).subarray(0, 12));
        observer.poll();
        expect(row.historyLength).toBe(0);
        now = 80;
        incoming.push(pong(0, 20));
        observer.poll();
        expect(row.historyLength).toBe(1);
        expect(row.value()).toBe(60);
        incoming.push(pong(0, 0));
        observer.poll();
        expect(row.historyLength).toBe(1);
        expect(row.value()).toBe(60);
    } finally {
        app.dispose();
    }
});

test("receive age uses transport receipt even when the first drain starts late", async () => {
    const app = await createApp({ defaults: false, plugins: [] });
    let now = 500;
    let incoming = [stampReceipt(new Uint8Array([99]), 100)];
    const connection = {
        [transportClock]: () => now,
        close() {},
        send() {},
        receive() {
            const messages = incoming;
            incoming = [];
            return messages;
        },
    };
    try {
        observeNetwork(app.world, connection);
        connection.receive();
        expect(value(app.world, "network/time_since_receive")).toBe(400);
        now = 700;
        connection.receive();
        expect(value(app.world, "network/time_since_receive")).toBe(600);
        incoming.push(stampReceipt(new Uint8Array([99]), 700));
        connection.receive();
        expect(value(app.world, "network/time_since_receive")).toBe(0);
    } finally {
        app.dispose();
    }
});

test("client confirmed lag is the gap behind the newest observed tick, not its local prediction clock", async () => {
    const a = await createApp({ defaults: false, plugins: [] });
    const b = await createApp({ defaults: false, plugins: [] });
    const link = seededLink(1, { minDelay: 0, maxDelay: 0, loss: 0, duplication: 0 });
    const send = link.a.send.bind(link.a);
    let held: Uint8Array | undefined;
    link.a.send = (bytes) => {
        const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
        if (bytes.length >= 20 && view.getUint32(8, true) === 1) held = bytes.slice();
        else send(bytes);
    };
    observeNetwork(a.world, link.a, "network/connections/1/");
    observeNetwork(b.world, link.b);
    const sender = new ReplicationSender(a.world, [], [], 1, link.a);
    const receiver = new ReplicationReceiver(b.world, [], link.b);
    try {
        receiver.flush();
        expect(value(b.world, "network/confirmed_tick")).toBeUndefined();
        sender.record(0);
        sender.flush();
        receiver.flush();
        sender.flush();
        sender.record(1);
        sender.flush();
        sender.record(2);
        sender.flush();
        receiver.flush();
        expect(value(b.world, "network/confirmed_tick")).toBe(0);
        expect(value(b.world, "network/confirmed_tick_lag")).toBe(2);
        expect(value(a.world, "network/connections/1/confirmed_tick_lag")).toBe(2);
        send(held!);
        receiver.flush();
        sender.flush();
        expect(value(b.world, "network/confirmed_tick")).toBe(2);
        expect(value(b.world, "network/confirmed_tick_lag")).toBe(0);
        expect(value(a.world, "network/connections/1/confirmed_tick_lag")).toBe(0);
    } finally {
        a.dispose();
        b.dispose();
    }
});
