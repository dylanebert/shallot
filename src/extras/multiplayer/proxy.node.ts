import { expect, setDefaultTimeout, spyOn, test } from "bun:test";
import { once } from "node:events";
import { createConnection, createServer, Server, type Socket } from "node:net";
import { TIMEOUT } from "@dylanebert/shallot/testing";
import type { Connection, Game } from ".";
import { createProxy, listenerReleased } from "./proxy";
import { createSession } from "./testing";

setDefaultTimeout(TIMEOUT.node);

async function traffic(seed = 1234, clients = 1) {
    const accepted: Connection[] = [];
    const game: Game = {
        plugins: [],
        schema: [],
        maxClients: 2,
        rules: () => [],
        validateIntent: () => true,
        applyIntent() {},
        onJoin(_world, _eid, connection) {
            accepted.push(connection);
        },
    };
    const session = await createSession(game, {
        clients,
        transport: "websocket",
        proxy: true,
        seed,
    });
    const peer = session.clients[0]!;
    const client = peer.connection!;
    // Drain the join image before injecting faults on established sockets.
    while (!client.receive().length) await Bun.sleep(1);
    return { session, client, server: accepted[0]!, proxy: peer.proxy! };
}

async function receive(connection: Connection, count: number) {
    const bytes: number[] = [];
    while (bytes.length < count) {
        for (const message of connection.receive()) bytes.push(...message);
        if (bytes.length < count) await Bun.sleep(1);
    }
    return bytes;
}

async function tcpTraffic() {
    const listener = createServer({ allowHalfOpen: true });
    listener.listen(0, "127.0.0.1");
    await once(listener, "listening");
    const proxy = await createProxy((listener.address() as { port: number }).port, 1234);
    const accepted = once(listener, "connection");
    const client = createConnection({ host: "127.0.0.1", port: proxy.port, allowHalfOpen: true });
    await once(client, "connect");
    const [server] = (await accepted) as [Socket];
    const observe = (socket: Socket) => {
        const chunks: Buffer[] = [];
        const errors: Error[] = [];
        socket.on("data", (bytes: Buffer) => chunks.push(bytes));
        socket.on("error", (error) => errors.push(error));
        return { socket, chunks, errors, end: once(socket, "end") };
    };
    return {
        proxy,
        client: observe(client),
        server: observe(server),
        async dispose() {
            client.destroy();
            server.destroy();
            proxy.stop();
            listener.close();
            expect(await listenerReleased(proxy.port)).toBe(true);
        },
    };
}

for (const direction of ["upstream", "downstream"] as const) {
    for (const fault of ["none", "latency", "bandwidth", "stall"] as const) {
        test(`${direction} crossing FINs retain queued bytes under ${fault}`, async () => {
            const traffic = await tcpTraffic();
            const { proxy, client, server } = traffic;
            const [source, target] = direction === "upstream" ? [client, server] : [server, client];
            const payload = Buffer.alloc(16384, 42);
            try {
                if (fault === "latency") proxy.controls[direction].latency = 100;
                if (fault === "bandwidth") proxy.controls[direction].bandwidth = 131072;
                if (fault === "stall") proxy.controls[direction].stalled = true;
                source.socket.end(payload);
                await Bun.sleep(10);
                target.socket.end();
                await Bun.sleep(30);
                proxy.controls[direction].stalled = false;
                await target.end;
                await source.end;
                const delivered = Buffer.concat(target.chunks);
                expect(delivered.length).toBe(payload.length);
                expect(delivered).toEqual(payload);
                expect([...source.errors, ...target.errors]).toEqual([]);
                while (proxy.pendingSockets) await Bun.sleep(1);
                expect(proxy.pendingSockets).toBe(0);
            } finally {
                await traffic.dispose();
            }
        });
    }

    test(`${direction} FIN leaves the opposite direction flowing`, async () => {
        const traffic = await tcpTraffic();
        const { proxy, client, server } = traffic;
        const [source, target] = direction === "upstream" ? [client, server] : [server, client];
        try {
            proxy.controls[direction].latency = 30;
            source.socket.end(Buffer.from([42]));
            await target.end;
            expect(Buffer.concat(target.chunks)).toEqual(Buffer.from([42]));
            for (let i = 1; i <= 3; i++) {
                target.socket.write(Buffer.from([i]));
                while (Buffer.concat(source.chunks).length < i) await Bun.sleep(1);
                expect(source.socket.readableEnded).toBe(false);
                expect(Buffer.concat(source.chunks)).toEqual(
                    Buffer.from(Array.from({ length: i }, (_, j) => j + 1)),
                );
            }
            target.socket.end();
            await source.end;
            expect([...source.errors, ...target.errors]).toEqual([]);
            while (proxy.pendingSockets) await Bun.sleep(1);
        } finally {
            await traffic.dispose();
        }
    });
}

test("proxy delay sets a round-trip floor on established WebSockets", async () => {
    const { session, client, server, proxy } = await traffic();
    try {
        proxy.upstream.latency = 40;
        proxy.downstream.latency = 60;
        const start = performance.now();
        client.send(new Uint8Array([7]));
        expect(await receive(server, 1)).toEqual([7]);
        server.send(new Uint8Array([8]));
        expect(await receive(client, 1)).toEqual([8]);
        expect(performance.now() - start).toBeGreaterThanOrEqual(100);
    } finally {
        await session.dispose();
    }
});

for (const direction of ["upstream", "downstream"] as const) {
    test(`seeded ${direction} jitter delays traffic even with zero base latency`, async () => {
        const { session, client, server, proxy } = await traffic();
        try {
            const [source, target] = direction === "upstream" ? [client, server] : [server, client];
            proxy[direction] = { latency: 0, jitter: 80, bandwidth: 0, stalled: false };
            const start = performance.now();
            source.send(new Uint8Array([9]));
            expect(await receive(target, 1)).toEqual([9]);
            // First seeded draws impose 34.28 ms upstream and 20.41 ms downstream.
            expect(performance.now() - start).toBeGreaterThanOrEqual(
                direction === "upstream" ? 34 : 20,
            );
            for (let i = 0; i < 12; i++) {
                source.send(new Uint8Array([i]));
                await Bun.sleep(2);
            }
            expect(await receive(target, 12)).toEqual(Array.from({ length: 12 }, (_, i) => i));
        } finally {
            await session.dispose();
        }
    });
}

for (const direction of ["upstream", "downstream"] as const) {
    test(`${direction} stall holds every frame and release delivers its backlog as one burst`, async () => {
        const { session, client, server, proxy } = await traffic();
        try {
            const [source, target] = direction === "upstream" ? [client, server] : [server, client];
            proxy[direction].stalled = true;
            for (let i = 0; i < 4; i++) {
                source.send(new Uint8Array([i]));
                await Bun.sleep(5);
            }
            expect(target.receive()).toEqual([]);
            proxy[direction].stalled = false;
            // One drain must see all four frames, not only eventual delivery one at a time.
            await Bun.sleep(20);
            expect(target.receive().map((bytes) => [...bytes])).toEqual([[0], [1], [2], [3]]);
        } finally {
            await session.dispose();
        }
    });

    test(`${direction} bandwidth limits delivered bytes and preserves all WebSocket framing`, async () => {
        const { session, client, server, proxy } = await traffic();
        try {
            const [source, target] = direction === "upstream" ? [client, server] : [server, client];
            proxy[direction].bandwidth = 2000;
            const payload = Uint8Array.from({ length: 400 }, (_, i) => i % 251);
            const start = performance.now();
            source.send(payload);
            expect(await receive(target, payload.length)).toEqual([...payload]);
            expect(performance.now() - start).toBeGreaterThanOrEqual(200);
            proxy[direction].bandwidth = 0;
            source.send(new Uint8Array([42]));
            expect(await receive(target, 1)).toEqual([42]);
        } finally {
            await session.dispose();
        }
    });
}

test("reset_peer closes both peers, including a stalled direction", async () => {
    const { session, client, server, proxy } = await traffic();
    try {
        proxy.upstream.stalled = true;
        client.send(new Uint8Array([1]));
        proxy.resetPeer();
        await session.waitFor(
            session.clients[0]!,
            () => !!client.closed && !!server.closed,
            40,
            "reset closes both TCP legs",
        );
        expect(client.closed).toBe(true);
        expect(server.closed).toBe(true);
    } finally {
        await session.dispose();
    }
});

test("disposal releases stalled sockets with backpressured stream bytes still queued", async () => {
    const { session, client, server, proxy } = await traffic();
    proxy.upstream.stalled = proxy.downstream.stalled = true;
    const payload = new Uint8Array(256 * 1024);
    client.send(payload);
    server.send(payload);
    await Bun.sleep(10);
    await session.dispose();
    expect(client.closed).toBe(true);
    expect(server.closed).toBe(true);
});

test("disposal detects every leaked proxy listener and force-releases it", async () => {
    const { session } = await traffic(1234, 2);
    const close = spyOn(Server.prototype, "close").mockImplementation(function (this: Server) {
        return this;
    });
    let message = "";
    try {
        await session.dispose();
    } catch (error) {
        message = String(error);
    } finally {
        close.mockRestore();
    }
    expect(message).toContain("Disposal leaked loopback proxy 127.0.0.1:");
    expect(message).toContain("seed=1234");
    const ports = [...message.matchAll(/loopback proxy 127\.0\.0\.1:(\d+)/g)].map((match) =>
        Number(match[1]),
    );
    expect(ports).toHaveLength(2);
    expect(new Set(ports).size).toBe(2);
    const { listenerReleased } = await import("./proxy");
    for (const port of ports) expect(await listenerReleased(port)).toBe(true);
});
