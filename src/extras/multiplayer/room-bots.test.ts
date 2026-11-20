import { expect, test } from "bun:test";
import { component, u32, World } from "../../engine";
import {
    type Game,
    IntentSender,
    Replicate,
    ReplicationReceiver,
    RoomBot,
    RoomBots,
    RoomLeader,
    WebSocketConnection,
} from ".";
import { serveRooms } from "./bun";
import { LEAVE_CLOSE_CODE } from "./websocket";

const Member = component("BotTestMember", { order: u32 });
const schema = [Member, RoomBot, RoomLeader];

async function room(
    run: (leader: Peer, friend: Peer, host: ReturnType<typeof fixture>) => Promise<void>,
) {
    const host = fixture();
    try {
        const leader = host.peer("joinOrCreate");
        const { code } = await leader.admitted;
        const friend = host.peer(`join&code=${code}`);
        await Promise.all([leader.sees(2, [], 1), friend.sees(2, [], 1)]);
        await run(leader, friend, host);
    } finally {
        host.stop();
    }
}

type Peer = ReturnType<ReturnType<typeof fixture>["peer"]>;

function fixture() {
    const worlds: World[] = [];
    const results: Promise<boolean>[] = [];
    let joined = 0;
    let botInputs = 0;
    const game: Game = {
        plugins: [{ name: "BotTestMember", components: schema }],
        schema,
        maxClients: 3,
        reconnectMs: 100,
        rules: (_world, recipient) => [new Map([[recipient, schema]])],
        onJoin(world, eid) {
            world.add(eid, Member, { order: ++joined });
            world.add(eid, Replicate, { rule: 0 });
        },
        validateIntent: () => true,
        applyIntent(world, eid, tick, records, input) {
            if (input && world.has(eid, RoomBot)) botInputs++;
            const bots = world.resource(RoomBots).controller!;
            for (const record of records) {
                if (record.tick !== tick) continue;
                if (record.values.length === 0) {
                    results.push(
                        bots.add(eid, {
                            seed: 41,
                            input(_world, sender, tick) {
                                sender.record(tick, "input", [1]);
                            },
                        }),
                    );
                } else {
                    const target = [...world.query([Member])].find(
                        (member) => world.get(member, Member)!.order === record.values[0],
                    );
                    results.push(Promise.resolve(target !== undefined && bots.remove(eid, target)));
                }
            }
        },
    };
    const host = serveRooms(game, { hostname: "127.0.0.1", port: 0, reconnectNow: () => 0 });
    const origin = `127.0.0.1:${host.port}`;

    function open(intent: string) {
        const socket = new WebSocket(`ws://${origin}/?announceRoom=1&intent=${intent}`);
        const closed = new Promise<CloseEvent>((resolve) =>
            socket.addEventListener("close", resolve, { once: true }),
        );
        return { socket, closed };
    }

    function peer(intent: string) {
        const { socket, closed } = open(intent);
        const world = new World();
        worlds.push(world);
        const admission = Promise.withResolvers<{ code: string; resume: string }>();
        let code = "";
        const connection = new WebSocketConnection(socket, (message) => {
            const [kind, first, second] = message.split(":");
            if (kind === "room") code = first!;
            else if (kind === "resume") {
                admission.resolve({ code, resume: `&seat=${first}&resume=${second}` });
            } else return false;
            return true;
        });
        const receiver = new ReplicationReceiver(world, schema, connection);
        const sender = new IntentSender(connection);
        function until(confirm: () => boolean) {
            return new Promise<void>((resolve, reject) => {
                function cleanup() {
                    socket.removeEventListener("message", check);
                    socket.removeEventListener("close", fail);
                }
                function fail() {
                    cleanup();
                    reject(new Error("Peer closed before receiving state"));
                }
                function check() {
                    sender.flush(receiver.flush());
                    if (!confirm()) return;
                    cleanup();
                    resolve();
                }
                socket.addEventListener("message", check);
                socket.addEventListener("close", fail, { once: true });
                if (connection.closed) fail();
                else check();
            });
        }
        return {
            admitted: admission.promise,
            sees(seats: number, bots: number[], leader: number) {
                return until(() => {
                    const orders = (marker: typeof RoomBot) =>
                        [...world.query([Member, marker])].map(
                            (eid) => world.get(eid, Member)!.order,
                        );
                    return (
                        [...world.query([Member])].length === seats &&
                        orders(RoomBot).join() === bots.join() &&
                        orders(RoomLeader).join() === String(leader)
                    );
                });
            },
            async request(...values: number[]) {
                const result = results.length;
                sender.record(receiver.confirmed, "operation", values);
                sender.flush();
                await until(() => results.length > result);
                return results[result];
            },
            close(code?: number) {
                socket.close(code);
                return closed;
            },
            until,
        };
    }
    return {
        peer,
        botInputs: () => botInputs,
        async refused(code: string) {
            return (await open(`join&code=${code}`).closed).code;
        },
        async occupancy() {
            const response = await fetch(`http://${origin}/rooms`);
            const [entry] = (await response.json()) as { clients: number }[];
            return entry?.clients;
        },
        stop() {
            host.stop();
            for (const world of worlds) world.dispose();
        },
    };
}

test("leader adds and removes a thin client every human sees", () =>
    room(async (leader, friend, host) => {
        expect(await leader.request()).toBe(true);
        await Promise.all([leader.sees(3, [3], 1), friend.sees(3, [3], 1)]);
        await leader.until(() => host.botInputs() > 0);
        expect(await leader.request(3)).toBe(true);
        await Promise.all([leader.sees(2, [], 1), friend.sees(2, [], 1)]);
    }));

test("a non-leader cannot add or remove a bot", () =>
    room(async (leader, friend) => {
        expect(await friend.request()).toBe(false);
        expect(await leader.request()).toBe(true);
        await friend.sees(3, [3], 1);
        expect(await friend.request(3)).toBe(false);
        await leader.sees(3, [3], 1);
    }));

test("a bot fills a seat and the full room refuses a second bot and a human", () =>
    room(async (leader, friend, host) => {
        expect(await leader.request()).toBe(true);
        await friend.sees(3, [3], 1);
        expect(await host.occupancy()).toBe(3);
        expect(await host.refused((await leader.admitted).code)).toBe(1013);
        expect(await leader.request()).toBe(false);
    }));

test("the bot survives leader reconnect and the successor may remove it", () =>
    room(async (leader, friend, host) => {
        expect(await leader.request()).toBe(true);
        await friend.sees(3, [3], 1);
        const { code, resume } = await leader.admitted;
        await leader.close();
        const resumed = host.peer(`join&code=${code}${resume}`);
        await resumed.sees(3, [3], 1);
        await friend.close(LEAVE_CLOSE_CODE);
        await resumed.sees(2, [3], 1);
        const successor = host.peer(`join&code=${code}`);
        await successor.sees(3, [3], 1);
        await resumed.close(LEAVE_CLOSE_CODE);
        await successor.sees(2, [3], 4);
        expect(await successor.request(3)).toBe(true);
        await successor.sees(1, [], 4);
    }));
