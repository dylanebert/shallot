import { expect, test } from "bun:test";
import { component, u32, World } from "../../engine";
import { Replicate, ReplicationReceiver, WebSocketConnection } from ".";
import { serveRooms } from "./bun";
import type { Game } from "./game";
import type { Connection } from "./link";

const RoomValue = component("RoomValue", { value: u32 });

async function until(predicate: () => boolean) {
    while (!predicate()) await Bun.sleep(1);
}

test("room sockets reserve seats, isolate images and release empty Worlds", async () => {
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
    const host = serveRooms(game, { hostname: "127.0.0.1", port: 0 });
    const clients: { socket: WebSocket; world: World }[] = [];
    function connect(query: string) {
        const socket = new WebSocket(`ws://127.0.0.1:${host.port}/?intent=${query}`);
        const closed = new Promise<CloseEvent>((resolve) =>
            socket.addEventListener("close", resolve, { once: true }),
        );
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
        expect(worlds).toEqual([]);
        const owner = connect("create&code=friend");
        await owner.sees(1);
        const a = connect("joinOrCreate");
        const b = connect("joinOrCreate");
        await a.sees(2);
        await b.sees(2);
        expect(worlds).toHaveLength(2);
        const friend = connect("join&code=friend");
        await friend.sees(1);
        expect((await connect("join&code=friend").closed).code).toBe(1013);
        expect((await connect("create&code=friend").closed).code).toBe(1008);
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
        expect((await connect("join&code=friend").closed).code).toBe(1008);
        const replacement = connect("create&code=friend");
        await replacement.sees(4);
        await a.close();
        await until(() => transports.get(worlds[1]!)![0]!.closed === true);
        const refill = connect("joinOrCreate");
        await refill.sees(2);
        expect(worlds).toHaveLength(4);
        host.stop();
        await Promise.all([b.closed, overflow.closed, replacement.closed, refill.closed]);
        expect(disposed).toEqual(new Set(worlds));
    } finally {
        host.stop();
        for (const { socket, world } of clients) {
            socket.close();
            world.dispose();
        }
    }
});
