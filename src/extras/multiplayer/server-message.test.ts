import { expect, test } from "bun:test";
import { type App, createApp, type World } from "../../engine";
import {
    type ClientId,
    client,
    type Game,
    local,
    ReplicationReceiver,
    type ServerMessage,
    ServerSession,
} from ".";
import { serveRooms } from "./bun";
import { createProxy } from "./proxy";
import { PredictionRecording } from "./recording-hooks";
import { ServerMessageKind } from "./server-message";

const emptyGame: Game = {
    plugins: [],
    schema: [],
    maxClients: 2,
    rules: () => [],
    validateIntent: () => false,
    applyIntent() {},
};
const faults = { minDelay: 0, maxDelay: 0, loss: 0, duplication: 0 };

function messageTopics(packets: Uint8Array[]): number[] {
    const topics: number[] = [];
    for (const bytes of packets) {
        const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
        if (view.getUint32(0, true) === ServerMessageKind) topics.push(view.getUint32(4, true));
    }
    return topics;
}

async function until(ready: () => boolean, reason: string) {
    for (let i = 0; i < 100; i++) {
        if (ready()) return;
        await Bun.sleep(1);
    }
    throw new Error(reason);
}

test("private messages stay in their room and retain their target across resume without replay", async () => {
    const seats: { world: World; eid: number; id: ClientId }[] = [];
    const receivers = new Map<World, ReplicationReceiver>();
    const game: Game = {
        ...emptyGame,
        reconnectMs: 100,
        onJoin(world, eid, _transport, id) {
            seats.push({ world, eid, id });
        },
        onClient(world, connection) {
            receivers.set(world, new ReplicationReceiver(world, [], connection));
        },
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
        const target = first!.id;
        for (const message of burst) server.sendMessage(target, message);
        expect(elsewhere!.eid).toBe(first!.eid);
        other.sendMessage(target, { topic: 97, values: [123] });
        const broadcast = { topic: 16, values: [first!.eid] };
        server.sendMessage("all", broadcast);
        other.sendMessage("all", broadcast);
        await readThrough(broadcast);
        poll();
        expect(received).toEqual([[...burst, broadcast], [broadcast], [broadcast]]);

        proxy.controls.resetPeer();
        await until(() => !!retry && !!server.peers.get(first!.eid)?.held, "seat was not held");
        const held = { topic: 17, values: [first!.eid] };
        server.sendMessage(target, { topic: 99, values: [] });
        server.sendMessage("all", held);
        retry!();
        await until(() => server.peers.get(first!.eid)?.held === false, "seat did not resume");
        const privateAfterResume = { topic: 98, values: [first!.eid] };
        server.sendMessage(target, privateAfterResume);
        const resumed = { topic: 18, values: [first!.eid], text: "" };
        server.sendMessage("all", resumed);
        other.sendMessage("all", resumed);
        await readThrough(resumed);
        poll();
        expect(received).toEqual([
            [...burst, broadcast, privateAfterResume, resumed],
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

test("a bot discards one-off messages after each flush, including private sends and broadcasts", async () => {
    let authority!: World;
    let botEid!: number;
    let receiver!: ReplicationReceiver;
    const topics: number[] = [];
    const host = await local(
        {
            ...emptyGame,
            onCreate(world) {
                authority = world;
            },
            plugins: [
                {
                    name: "ObserveBotMessageWire",
                    initialize(world) {
                        world.resource(PredictionRecording).receiver = (observed) => {
                            receiver = observed;
                            const receive = observed.connection.receive.bind(observed.connection);
                            observed.connection.receive = () => {
                                const packets = receive();
                                topics.push(...messageTopics(packets));
                                return packets;
                            };
                        };
                    },
                },
            ],
        },
        { rooms: true, seed: 1, faults },
    );
    try {
        const human = await host.join();
        await human.addBot({
            seed: 2,
            admit(_world, eid) {
                botEid = eid;
                return true;
            },
            input() {},
        });
        const server = authority.resource(ServerSession).server!;
        const bot = server.peers.get(botEid)!.clientId;
        for (let topic = 0; topic < 6; topic += 2) {
            server.sendMessage(bot, { topic, values: [] });
            server.sendMessage("all", { topic: topic + 1, values: [] });
            host.pump(0);
            expect(topics).toEqual(Array.from({ length: topic + 2 }, (_, i) => i));
            expect(receiver.readMessages()).toEqual([]);
        }
    } finally {
        host.dispose();
    }
});
