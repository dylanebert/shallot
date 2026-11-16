import { expect, setDefaultTimeout, test } from "bun:test";
import { TIMEOUT } from "@dylanebert/shallot/testing";

setDefaultTimeout(TIMEOUT.node);

import { createApp } from "../../engine";
import {
    type Connection,
    type Game,
    MultiplayerPlugin,
    ServerSession,
    seededLink,
    server,
    WebSocketConnection,
} from ".";
import { serveGame } from "./bun";

const game: Game = {
    plugins: [],
    schema: [],
    maxClients: 1,
    rules: () => [],
    validateIntent: () => true,
    applyIntent() {},
};

for (const departure of ["disconnect", "destroy", "world", "shutdown"] as const) {
    test(`socket ${departure} closes the client and retains no late messages`, async () => {
        let transport!: Connection;
        const host = await serveGame(
            {
                ...game,
                onJoin(_world, _eid, connection) {
                    transport = connection;
                },
            },
            { hostname: "127.0.0.1", port: 0 },
        );
        const socket = new WebSocket(`ws://127.0.0.1:${host.port}`);
        const client = new WebSocketConnection(socket);
        const image = new Promise<void>((resolve) =>
            socket.addEventListener("message", () => resolve(), { once: true }),
        );
        try {
            await new Promise<void>((resolve, reject) => {
                socket.addEventListener("open", () => resolve(), { once: true });
                socket.addEventListener("error", reject, { once: true });
            });
            await image;
            const runtime = host.world.resource(ServerSession).server!;
            const eid = [...runtime.peers.keys()][0]!;
            const closed = new Promise<void>((resolve) =>
                socket.addEventListener("close", () => resolve(), { once: true }),
            );
            if (departure === "disconnect") runtime.disconnect(eid);
            else if (departure === "destroy") host.world.destroy(eid);
            else if (departure === "world") host.world.dispose();
            else host.stop();
            for (let i = 0; i < 5; i++)
                if (socket.readyState === 1) socket.send(new Uint8Array([i]));
            await Promise.race([closed, Bun.sleep(1000)]);
            if (!host.world.signal.aborted) host.world.tick();
            expect(runtime.peers.size).toBe(0);
            expect({ state: socket.readyState, retained: transport.receive().length }).toEqual({
                state: 3,
                retained: 0,
            });
            expect(client.receive()).toEqual([]);
        } finally {
            socket.close();
            host.stop();
        }
    });
}

for (const departure of ["disconnect", "destroy", "world", "app"] as const) {
    test(`seeded link ${departure} closes both ends and clears pending and future messages`, async () => {
        const app = await createApp({
            defaults: false,
            plugins: [MultiplayerPlugin, server(game)],
        });
        const runtime = app.world.resource(ServerSession).server!;
        const link = seededLink(1, { minDelay: 100, maxDelay: 100, loss: 0, duplication: 0 });
        try {
            const eid = runtime.connect(link.a)!;
            for (let i = 0; i < 5; i++) link.b.send(new Uint8Array([i]));
            if (departure === "disconnect") runtime.disconnect(eid);
            else if (departure === "destroy") app.world.destroy(eid);
            else if (departure === "world") app.world.dispose();
            else app.dispose();
            link.a.send(new Uint8Array([9]));
            link.b.send(new Uint8Array([9]));
            link.advance(1000);
            expect(runtime.peers.size).toBe(0);
            expect({
                a: link.a.closed,
                b: link.b.closed,
                receivedA: link.a.receive().length,
                receivedB: link.b.receive().length,
            }).toEqual({ a: true, b: true, receivedA: 0, receivedB: 0 });
        } finally {
            if (!app.world.signal.aborted) app.dispose();
        }
    });
}
