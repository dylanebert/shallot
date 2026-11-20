import { expect, test } from "bun:test";
import { expectError } from "../../../scripts/test-console";
import { component, u32, World } from "../../engine";
import { Replicate, ReplicationReceiver, WebSocketConnection } from ".";
import { serveRooms } from "./bun";
import type { Game } from "./game";
import type { Connection } from "./link";

const RoomValue = component("RoomValue", { value: u32 });

async function until(predicate: () => boolean) {
    while (!predicate()) await Bun.sleep(1);
}

function requestCode(port: number | undefined) {
    return fetch(`http://127.0.0.1:${port}/rooms`, { method: "POST" });
}

async function privateCode(port: number | undefined) {
    const response = await requestCode(port);
    expect(response.status).toBe(201);
    const { code } = (await response.json()) as { code: string };
    expect(code).toMatch(/^[0-9A-HJKMNP-TV-Z]{10}$/);
    return code;
}

function roomSocket(port: number | undefined, query: string) {
    const socket = new WebSocket(`ws://127.0.0.1:${port}/?intent=${query}`);
    const closed = new Promise<CloseEvent>((resolve) =>
        socket.addEventListener("close", resolve, { once: true }),
    );
    return { socket, closed };
}

test("room sockets isolate Worlds and discovery follows public seats only", async () => {
    const worlds: World[] = [];
    const disposed = new Set<World>();
    const transports = new Map<World, Connection[]>();
    const game: Game = {
        plugins: [{ name: "RoomValue", components: [RoomValue], recovery: "stateless" }],
        schema: [RoomValue],
        maxClients: 2,
        rules: (_world, eid) => [new Map([[eid, [RoomValue]]])],
        onCreate(world) {
            worlds.push(world);
            transports.set(world, []);
            world.onDispose(() => disposed.add(world));
            const entity = world.create();
            world.add(entity, RoomValue, { value: worlds.length });
            world.add(entity, Replicate, { rule: 0 });
        },
        onJoin(world, _eid, transport) {
            transports.get(world)!.push(transport);
        },
        validateIntent: () => false,
        applyIntent() {},
    };
    const roomsOrigin = "https://game.example";
    const host = serveRooms(game, { hostname: "127.0.0.1", port: 0, roomsOrigin });
    async function listing() {
        const response = await fetch(`http://127.0.0.1:${host.port}/rooms`);
        expect(response.status).toBe(200);
        expect(response.headers.get("Access-Control-Allow-Origin")).toBe(roomsOrigin);
        return (await response.json()) as {
            code: string;
            clients: number;
            maxClients: number;
        }[];
    }
    const clients: { socket: WebSocket; world: World }[] = [];
    function connect(query: string) {
        const { socket, closed } = roomSocket(host.port, query);
        const world = new World();
        clients.push({ socket, world });
        const connection = new WebSocketConnection(socket);
        const receiver = new ReplicationReceiver(world, game.schema, connection);
        return {
            closed,
            async sees(value: number) {
                await until(() => {
                    receiver.flush();
                    return [...world.query([RoomValue])].length > 0;
                });
                expect(
                    [...world.query([RoomValue])].map((eid) => world.get(eid, RoomValue)!.value),
                ).toEqual([value]);
            },
            async close() {
                connection.close();
                await closed;
            },
        };
    }
    try {
        const code = await privateCode(host.port);
        expect(worlds).toEqual([]);
        expect(await listing()).toEqual([]);
        expect((await connect("create&code=friend").closed).code).toBe(1008);
        const owner = connect(`join&code=${code.toLowerCase()}`);
        await owner.sees(1);
        expect(await listing()).toEqual([]);
        const a = connect("joinOrCreate");
        await a.sees(2);
        const [entry] = await listing();
        expect(entry?.code).toMatch(/^[0-9A-HJKMNP-TV-Z]{10}$/);
        const publicRoom = { code: entry!.code, clients: 1, maxClients: 2 };
        expect(publicRoom.code).not.toBe(code);
        expect(await listing()).toEqual([publicRoom]);
        const b = connect(`join&code=${publicRoom.code}`);
        await b.sees(2);
        expect(await listing()).toEqual([{ ...publicRoom, clients: 2 }]);
        expect(worlds).toHaveLength(2);
        const friend = connect(`join&code=${code}`);
        await friend.sees(1);
        expect((await connect(`join&code=${code}`).closed).code).toBe(1013);
        const overflow = connect("joinOrCreate");
        await overflow.sees(3);

        await owner.close();
        expect(disposed.has(worlds[0]!)).toBe(false);
        await friend.close();
        await until(() => disposed.has(worlds[0]!));
        expect([...disposed]).toEqual([worlds[0]!]);
        for (const transport of transports.get(worlds[0]!)!) {
            expect(transport.closed).toBe(true);
            expect(transport.receive()).toEqual([]);
        }
        expect((await connect(`join&code=${code}`).closed).code).toBe(1008);
        const replacement = connect(`join&code=${await privateCode(host.port)}`);
        await replacement.sees(4);
        await a.close();
        await until(() => transports.get(worlds[1]!)![0]!.closed === true);
        const [first, second] = await listing();
        expect(first).toEqual(publicRoom);
        const refill = connect("joinOrCreate");
        await refill.sees(2);
        expect(worlds).toHaveLength(4);
        await b.close();
        await refill.close();
        await until(() => disposed.has(worlds[1]!));
        expect(await listing()).toEqual([{ code: second!.code, clients: 1, maxClients: 2 }]);
        host.stop();
        expect(disposed).toEqual(new Set(worlds));
        await Promise.all([b.closed, overflow.closed, replacement.closed, refill.closed]);
    } finally {
        host.stop();
        for (const { socket, world } of clients) {
            socket.close();
            world.dispose();
        }
    }
});

test("private code reservations are bounded host-wide and claim or expiry frees a slot", async () => {
    let now = 0;
    const joined = Promise.withResolvers<void>();
    const host = serveRooms(
        {
            plugins: [],
            schema: [],
            maxClients: 1,
            rules: () => [],
            onJoin: () => joined.resolve(),
            validateIntent: () => false,
            applyIntent() {},
        },
        { hostname: "127.0.0.1", port: 0, reservationNow: () => now },
    );
    try {
        const expiring = await privateCode(host.port);
        now = 1000;
        const claimable = await privateCode(host.port);
        const codes = new Set([expiring, claimable]);
        for (let i = 2; i < 128; i++) codes.add(await privateCode(host.port));
        expect(codes.size).toBe(128);
        const refused = await requestCode(host.port);
        expect(refused.status).toBe(503);
        expect(refused.headers.get("Retry-After")).toBe("60");

        const claim = roomSocket(host.port, `join&code=${claimable}`);
        await joined.promise;
        expect(codes.has(await privateCode(host.port))).toBe(false);
        expect((await requestCode(host.port)).status).toBe(503);

        now = 60_001;
        const expired = roomSocket(host.port, `join&code=${expiring}`);
        const outcome = await Promise.race([
            expired.closed.then((event) => event.code),
            new Promise((resolve) =>
                expired.socket.addEventListener("message", () => resolve("admitted"), {
                    once: true,
                }),
            ),
        ]);
        expect(outcome).toBe(1008);
        await privateCode(host.port);
        expect((await requestCode(host.port)).status).toBe(503);

        now = 61_001;
        await privateCode(host.port);
        host.stop();
        await claim.closed;
    } finally {
        host.stop();
    }
});

test.each(["last leave", "stop", "failure"] as const)(
    "a pending initialization retires on %s without admitting its seat or affecting a new room",
    async (event) => {
        const gate = Promise.withResolvers<void>();
        const initializing = Promise.withResolvers<World>();
        const replacementJoined = Promise.withResolvers<World>();
        const disposed: World[] = [];
        const joined: World[] = [];
        let builds = 0;
        const game: Game = {
            plugins: [
                {
                    name: "Gated room",
                    recovery: "stateless",
                    async initialize(world) {
                        world.onDispose(() => disposed.push(world));
                        if (++builds !== 1) return;
                        initializing.resolve(world);
                        await gate.promise;
                        if (event === "failure") throw new Error("gated initializer failed");
                    },
                },
            ],
            schema: [],
            maxClients: 1,
            rules: () => [],
            onJoin(world) {
                joined.push(world);
                replacementJoined.resolve(world);
            },
            validateIntent: () => false,
            applyIntent() {},
        };
        const options = { hostname: "127.0.0.1", port: 0 };
        const host = serveRooms(game, options);
        let replacementHost = host;
        try {
            const code = await privateCode(host.port);
            const pending = roomSocket(host.port, `join&code=${code}`);
            const oldWorld = await initializing.promise;
            expect(joined).toEqual([]);
            expect(disposed).toEqual([]);
            expect((await roomSocket(host.port, `join&code=${code}`).closed).code).toBe(1013);
            if (event === "last leave") {
                pending.socket.close();
                await pending.closed;
            } else if (event === "stop") {
                host.stop();
                await pending.closed;
                replacementHost = serveRooms(game, options);
            } else {
                expectError(/gated initializer failed/);
                gate.resolve();
                expect((await pending.closed).code).toBe(1011);
            }
            if (event !== "stop") {
                expect((await roomSocket(host.port, `join&code=${code}`).closed).code).toBe(1008);
            }
            const nextCode = await privateCode(replacementHost.port);
            const replacement = roomSocket(replacementHost.port, `join&code=${nextCode}`);
            gate.resolve();
            const nextWorld = await replacementJoined.promise;
            expect(nextWorld).not.toBe(oldWorld);
            expect(joined).toEqual([nextWorld]);
            expect(disposed).toEqual([oldWorld]);
            replacementHost.stop();
            expect(disposed).toEqual([oldWorld, nextWorld]);
            await replacement.closed;
        } finally {
            host.stop();
            replacementHost.stop();
            gate.resolve();
        }
    },
);
