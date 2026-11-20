import { expect, test } from "bun:test";
import { component, u32, World } from "../../engine";
import { Replicate, ReplicationReceiver, RoomLeader, WebSocketConnection } from ".";
import { serveRooms } from "./bun";
import type { Game } from "./game";
import { LEAVE_CLOSE_CODE } from "./websocket";

const SeatOrder = component("LeaderTestSeatOrder", { order: u32 });

function fixture(startup?: Promise<void>) {
    const starting = Promise.withResolvers<void>();
    let now = 0;
    let joined = 0;
    const game: Game = {
        plugins: [
            {
                name: "SeatOrder",
                components: [SeatOrder],
                initialize() {
                    starting.resolve();
                    return startup;
                },
            },
        ],
        schema: [SeatOrder, RoomLeader],
        maxClients: 4,
        reconnectMs: 100,
        rules: (world, eid) => [
            new Map([[eid, world.get(eid, SeatOrder)!.order === 4 ? [SeatOrder] : game.schema]]),
        ],
        onJoin(world, eid) {
            world.add(eid, SeatOrder, { order: ++joined });
            world.add(eid, Replicate, { rule: 0 });
        },
        validateIntent: () => false,
        applyIntent() {},
    };
    const host = serveRooms(game, {
        hostname: "127.0.0.1",
        port: 0,
        reservationNow: () => now,
        reconnectNow: () => now,
    });
    const endpoint = `127.0.0.1:${host.port}`;
    const worlds: World[] = [];
    function open(intent: string) {
        const socket = new WebSocket(`ws://${endpoint}/?announceRoom=1&intent=${intent}`);
        const closed = new Promise<CloseEvent>((resolve) =>
            socket.addEventListener("close", resolve, { once: true }),
        );
        return { socket, closed };
    }
    function peer(intent: string) {
        const { socket, closed } = open(intent);
        const world = new World();
        worlds.push(world);
        const admitted = Promise.withResolvers<{ code: string; resume: string }>();
        let code = "";
        const connection = new WebSocketConnection(socket, (message) => {
            const [kind, first, second] = message.split(":");
            if (kind === "room") code = first!;
            else if (kind === "resume")
                admitted.resolve({ code, resume: `&seat=${first}&resume=${second}` });
            else return false;
            return true;
        });
        const receiver = new ReplicationReceiver(world, game.schema, connection);
        return {
            admitted() {
                return Promise.race([
                    admitted.promise,
                    closed.then(({ code }) => {
                        throw new Error(`Seat refused: ${code}`);
                    }),
                ]);
            },
            sees(order: number | undefined, seats: number) {
                return new Promise<void>((resolve, reject) => {
                    const cleanup = () => {
                        socket.removeEventListener("message", check);
                        socket.removeEventListener("close", fail);
                    };
                    const fail = () => {
                        cleanup();
                        reject(new Error("Seat closed before receiving state"));
                    };
                    const check = () => {
                        receiver.flush();
                        const leaders = [...world.query([SeatOrder, RoomLeader])].map(
                            (id) => world.get(id, SeatOrder)!.order,
                        );
                        if (
                            [...world.query([SeatOrder])].length !== seats ||
                            leaders.length !== (order === undefined ? 0 : 1) ||
                            leaders[0] !== order
                        )
                            return;
                        cleanup();
                        resolve();
                    };
                    socket.addEventListener("message", check);
                    socket.addEventListener("close", fail, { once: true });
                    if (connection.closed) fail();
                    else check();
                });
            },
            close(code?: number) {
                socket.close(code);
                return closed;
            },
        };
    }
    return {
        peer,
        starting: starting.promise,
        async reserve() {
            const response = await fetch(`http://${endpoint}/rooms`, { method: "POST" });
            expect(response.status).toBe(201);
            return (await response.json()) as { code: string; creator: string };
        },
        async refused(intent: string) {
            return (await open(intent).closed).code;
        },
        advance(ms: number) {
            now += ms;
        },
        stop() {
            host.stop();
            for (const world of worlds) world.dispose();
        },
    };
}

test("replication discloses one leader, preserves a held seat, and hands off in join order", async () => {
    const f = fixture();
    try {
        const first = f.peer("joinOrCreate");
        const { code, resume } = await first.admitted();
        const join = `join&code=${code}`;
        const second = f.peer(join);
        await second.admitted();
        const third = f.peer(join);
        await Promise.all([first.sees(1, 3), second.sees(1, 3), third.sees(1, 3)]);
        await first.close();
        const undisclosed = f.peer(join);
        await Promise.all([second.sees(1, 4), third.sees(1, 4), undisclosed.sees(undefined, 4)]);
        const resumed = f.peer(join + resume);
        await resumed.sees(1, 4);
        await resumed.close();
        expect(await f.refused(join)).toBe(1013);
        f.advance(100);
        const replacement = f.peer(join);
        await Promise.all([second.sees(2, 4), third.sees(2, 4), replacement.sees(2, 4)]);
        await second.close(LEAVE_CLOSE_CODE);
        await Promise.all([third.sees(3, 3), replacement.sees(3, 3)]);
    } finally {
        f.stop();
    }
});

test("the private code refuses early friends, admits its creator first, and unlocks on admission", async () => {
    const startup = Promise.withResolvers<void>();
    const f = fixture(startup.promise);
    try {
        const { code, creator } = await f.reserve();
        const join = `join&code=${code}`;
        expect(await f.refused(join)).toBe(4001);
        expect(await f.refused(`${join}&creator=wrong`)).toBe(4001);
        const owner = f.peer(`${join}&creator=${creator}`);
        await f.starting;
        expect(await f.refused(join)).toBe(4001);
        startup.resolve();
        await owner.sees(1, 1);
        const friend = f.peer(join);
        await Promise.all([owner.sees(1, 2), friend.sees(1, 2)]);
    } finally {
        startup.resolve();
        f.stop();
    }
});

test("the code admits anyone once the creator hold expires, before the code itself expires", async () => {
    const f = fixture();
    try {
        const { code, creator } = await f.reserve();
        const join = `join&code=${code}`;
        f.advance(29_999);
        expect(await f.refused(join)).toBe(4001);
        f.advance(1);
        const friend = f.peer(join);
        await friend.sees(1, 1);
        const formerCreator = f.peer(`${join}&creator=${creator}`);
        await Promise.all([friend.sees(1, 2), formerCreator.sees(1, 2)]);
    } finally {
        f.stop();
    }
});
