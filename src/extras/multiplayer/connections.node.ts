import { expect, test } from "bun:test";
import type { ServerWebSocket } from "bun";
import { component, createApp, u32 } from "../../engine";
import { ClientOf, type Game, MultiplayerPlugin, ServerSession, server } from ".";
import { ServerWebSocketConnection } from "./bun";
import { seededLink } from "./link";

const Seat = component("ConnectionTestSeat", { value: u32 });
async function session(maxClients = 2) {
    const leaves: number[] = [];
    const Log = { create: () => leaves };
    const game: Game = {
        plugins: [{ name: "ConnectionTest", components: [Seat] }],
        schema: [],
        maxClients,
        rules: () => [],
        onJoin(world, eid) {
            world.add(eid, Seat, { value: 17 });
        },
        onLeave(world, eid) {
            world.resource(Log).push(world.get(eid, Seat)?.value ?? -1);
        },
        validateIntent: () => true,
        applyIntent() {},
    };
    const app = await createApp({
        defaults: false,
        plugins: [...game.plugins, MultiplayerPlugin, server(game)],
    });
    return { app, runtime: app.world.resource(ServerSession).server!, leaves };
}
function connection() {
    return seededLink(1, { minDelay: 0, maxDelay: 0, loss: 0, duplication: 0 }).a;
}

test("join spawns one connection entity and keys its runtime handle by that entity", async () => {
    const { app, runtime } = await session();
    try {
        const transport = connection();
        const eid = runtime.connect(transport)!;
        expect([...app.world.query([ClientOf])]).toEqual([eid]);
        expect(app.world.entities()).toEqual([eid]);
        expect(app.world.get(eid, Seat)!.value).toBe(17);
        expect(runtime.peers.get(eid)!.connection).toBe(transport);
        expect(runtime.peers.size).toBe(1);
    } finally {
        app.dispose();
    }
});

test("leave despawns the entity and releases its runtime handle exactly once", async () => {
    const { app, runtime, leaves } = await session();
    try {
        const eid = runtime.connect(connection())!;
        runtime.disconnect(eid);
        runtime.disconnect(eid);
        expect(app.world.exists(eid)).toBe(false);
        expect([...app.world.query([ClientOf])]).toEqual([]);
        expect(runtime.peers.has(eid)).toBe(false);
        expect(leaves).toEqual([17]);
    } finally {
        app.dispose();
    }
});

test("a dead socket despawns its entity and releases its runtime handle without a close callback", async () => {
    const { app, runtime, leaves } = await session();
    const socket = { readyState: 1, sendBinary: () => 1, close() {} };
    try {
        const transport = new ServerWebSocketConnection(
            socket as unknown as ServerWebSocket<ServerWebSocketConnection | undefined>,
        );
        const eid = runtime.connect(transport)!;
        socket.readyState = 3;
        app.world.tick();
        expect(app.world.exists(eid)).toBe(false);
        expect(runtime.peers.has(eid)).toBe(false);
        expect(leaves).toEqual([17]);
    } finally {
        app.dispose();
    }
});

test("departure clears queued socket bytes and ignores callbacks arriving after release", async () => {
    const { app, runtime } = await session();
    const socket = { readyState: 1, sendBinary: () => 1, close() {} };
    const transport = new ServerWebSocketConnection(
        socket as unknown as ServerWebSocket<ServerWebSocketConnection | undefined>,
    );
    try {
        const eid = runtime.connect(transport)!;
        transport.push(new Uint8Array([1]));
        runtime.disconnect(eid);
        transport.push(new Uint8Array([2]));
        expect(transport.closed).toBe(true);
        expect(transport.receive()).toEqual([]);
    } finally {
        app.dispose();
    }
});

test("a join past the peer limit is refused without an entity or runtime handle", async () => {
    const { app, runtime } = await session(1);
    try {
        const eid = runtime.connect(connection())!;
        expect(runtime.connect(connection())).toBeUndefined();
        expect(app.world.entities()).toEqual([eid]);
        expect([...app.world.query([ClientOf])]).toEqual([eid]);
        expect([...app.world.query([Seat])]).toEqual([eid]);
        expect(runtime.peers.size).toBe(1);
    } finally {
        app.dispose();
    }
});

test("app disposal runs leave hooks while resources are live and releases every handle", async () => {
    const { app, runtime, leaves } = await session();
    runtime.connect(connection());
    runtime.connect(connection());
    app.dispose();
    app.dispose();
    expect(runtime.peers.size).toBe(0);
    expect(leaves).toEqual([17, 17]);
});

test("destroying a connection entity releases its handle before eid reuse", async () => {
    const { app, runtime, leaves } = await session();
    try {
        const eid = runtime.connect(connection())!;
        app.world.destroy(eid);
        expect(runtime.peers.size).toBe(0);
        expect(leaves).toHaveLength(1);
        const replacement = runtime.connect(connection())!;
        expect(runtime.peers.size).toBe(1);
        expect(app.world.has(replacement, ClientOf)).toBe(true);
    } finally {
        app.dispose();
    }
});
