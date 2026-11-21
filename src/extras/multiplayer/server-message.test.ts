import { expect, test } from "bun:test";
import { type App, createApp, type World } from "../../engine";
import { client, type Game, ReplicationReceiver, type ServerMessage, ServerSession } from ".";
import { serveRooms } from "./bun";
import { createProxy } from "./proxy";

async function until(ready: () => boolean, reason: string) {
    for (let i = 0; i < 100; i++) {
        if (ready()) return;
        await Bun.sleep(1);
    }
    throw new Error(reason);
}

test("client systems read private and room messages once in order, with no replay on resume", async () => {
    const seats: { world: World; eid: number }[] = [];
    const receivers = new Map<World, ReplicationReceiver>();
    const game: Game = {
        plugins: [],
        schema: [],
        maxClients: 2,
        reconnectMs: 100,
        rules: () => [],
        onJoin(world, eid) {
            seats.push({ world, eid });
        },
        onClient(world, connection) {
            receivers.set(world, new ReplicationReceiver(world, [], connection));
        },
        validateIntent: () => false,
        applyIntent() {},
    };
    const host = serveRooms(game, { hostname: "127.0.0.1", port: 0, reconnectNow: () => 0 });
    const apps: App[] = [];
    const received: ServerMessage[][] = [];
    let proxy: Awaited<ReturnType<typeof createProxy>> | undefined;
    let retry: (() => void) | undefined;
    try {
        proxy = await createProxy(host.port!, 4);
        for (const port of [proxy.port, host.port, host.port]) {
            const inbox: ServerMessage[] = [];
            received.push(inbox);
            const app = await createApp({
                defaults: false,
                plugins: [
                    client(game, {
                        url: `ws://127.0.0.1:${port}/?intent=joinOrCreate`,
                        retryAfter(_delay, run) {
                            retry = run;
                            return () => {
                                retry = undefined;
                            };
                        },
                    }),
                    {
                        name: "ReadServerMessages",
                        systems: [
                            {
                                name: "read-messages",
                                group: "fixed",
                                update(world) {
                                    const receiver = receivers.get(world);
                                    if (!receiver) return;
                                    receiver.flush();
                                    inbox.push(...receiver.readMessages());
                                },
                            },
                        ],
                    },
                ],
            });
            apps.push(app);
            app.world.step(0);
            await until(() => seats.length === apps.length, "client was not admitted");
        }
        const poll = () => {
            for (const app of apps) app.world.tick();
        };
        const readThrough = (expected: ServerMessage) =>
            until(() => {
                poll();
                return received.every((inbox) => inbox.at(-1)?.topic === expected.topic);
            }, `clients did not reach message ${expected.topic}`);
        const [first, second, elsewhere] = seats;
        expect(second!.world).toBe(first!.world);
        expect(elsewhere!.world).not.toBe(first!.world);
        const server = first!.world.resource(ServerSession).server!;
        const other = elsewhere!.world.resource(ServerSession).server!;
        const burst = Array.from({ length: 16 }, (_, topic) => ({
            topic,
            values: [first!.eid, topic + 0.5],
            text: `catch 🐟 ${topic}`,
        }));
        for (const message of burst) server.sendMessage(first!.eid, message);
        const broadcast = { topic: 16, values: [first!.eid] };
        server.sendMessage("all", broadcast);
        other.sendMessage("all", broadcast);
        await readThrough(broadcast);
        poll();
        expect(received).toEqual([[...burst, broadcast], [broadcast], [broadcast]]);

        proxy.controls.resetPeer();
        await until(() => !!retry && !!server.peers.get(first!.eid)?.held, "seat was not held");
        const held = { topic: 17, values: [first!.eid] };
        server.sendMessage(first!.eid, { topic: 99, values: [] });
        server.sendMessage("all", held);
        retry!();
        await until(() => server.peers.get(first!.eid)?.held === false, "seat did not resume");
        const resumed = { topic: 18, values: [first!.eid], text: "" };
        server.sendMessage("all", resumed);
        other.sendMessage("all", resumed);
        await readThrough(resumed);
        poll();
        expect(received).toEqual([
            [...burst, broadcast, resumed],
            [broadcast, held, resumed],
            [broadcast, resumed],
        ]);
        expect(seats).toHaveLength(3);
    } finally {
        for (const app of apps) app.dispose();
        proxy?.stop();
        host.stop();
    }
});
