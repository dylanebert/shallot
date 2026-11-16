import { expect, setDefaultTimeout, spyOn, test } from "bun:test";
import { TIMEOUT } from "@dylanebert/shallot/testing";
import { component, u32, type World } from "../../engine";
import { type Connection, type Game, Replicate, ReplicationReceiver, ServerSession } from ".";
import { createSession } from "./testing";

setDefaultTimeout(TIMEOUT.node);
const Value = component("SessionTestValue", { value: u32 });
const Confirmed = { create: (): { receiver?: ReplicationReceiver } => ({}) };
const game: Game = {
    plugins: [
        {
            name: "SessionTest",
            components: [Value],
            systems: [
                {
                    name: "receive-confirmed",
                    group: "postFixed",
                    update(world) {
                        world.resource(Confirmed).receiver?.flush();
                    },
                },
            ],
        },
    ],
    schema: [Value],
    maxClients: 8,
    onCreate(world) {
        const eid = world.create();
        world.add(eid, Value, { value: 42 });
        world.add(eid, Replicate);
    },
    rules: (_world, eid) => [new Map([[eid, [Value]]])],
    validateIntent: () => true,
    applyIntent() {},
    onClient(world, connection) {
        world.resource(Confirmed).receiver = new ReplicationReceiver(world, [Value], connection);
    },
};

for (const transport of ["link", "websocket"] as const) {
    test(`${transport} confirms the same state on two clients and a thin World and releases its resources`, async () => {
        const session = await createSession(game, {
            seed: 19,
            clients: 2,
            thinClients: 1,
            transport,
        });
        const peers = [...session.clients, ...session.thinClients];
        try {
            for (const peer of peers) {
                await session.waitFor(
                    peer,
                    () =>
                        [...peer.world.query([Value])].some(
                            (eid) => peer.world.get(eid, Value)!.value === 42,
                        ),
                    40,
                    "confirmed value 42",
                );
                expect(
                    [...peer.world.query([Value])].map((eid) => peer.world.get(eid, Value)!.value),
                ).toEqual([42]);
            }
            expect(session.thinClients[0]!.world.time.fixedTick).toBe(0);
            expect(session.server.world.resource(ServerSession).server!.peers.size).toBe(3);
        } finally {
            await session.dispose();
        }
        expect([session.server, ...peers].every((peer) => peer.world.signal.aborted)).toBe(true);
        expect(peers.every((peer) => peer.connection!.closed)).toBe(true);
        await session.dispose();
    });
}

test("an impossible wait names its peer, tick, state and replay seed", async () => {
    const session = await createSession(game, { seed: 27, clients: 1 });
    try {
        let message = "";
        try {
            await session.waitFor(
                session.clients[0]!,
                () => false,
                3,
                "planted impossible condition",
            );
        } catch (error) {
            message = String(error);
        }
        expect(message).toContain("Wait failed: planted impossible condition; budget=3");
        expect(message).toContain("client[0] tick=3 connection=open disposed=false seed=27");
        expect(session.server.world.time.fixedTick).toBe(3);
    } finally {
        await session.dispose();
    }
});

test("disposal fails on a client left undisposed, then force-cleans it", async () => {
    const session = await createSession(game, { seed: 28, clients: 1 });
    const peer = session.clients[0]!;
    const dispose = spyOn(peer, "dispose").mockImplementation(() => {});
    let message = "";
    try {
        await session.dispose();
    } catch (error) {
        message = String(error);
    } finally {
        dispose.mockRestore();
    }
    expect(message).toContain(
        "Disposal leaked peer: client[0] tick=0 connection=closed disposed=false seed=28",
    );
    expect(peer.world.signal.aborted).toBe(true);
});

test("targeted stepping does not tick siblings and a closed peer is named in wait failure", async () => {
    const session = await createSession(game, { seed: 29, clients: 2 });
    try {
        await session.step(session.clients[1]!, 4);
        expect(session.clients[1]!.world.time.fixedTick).toBe(4);
        expect(session.clients[0]!.world.time.fixedTick).toBe(0);
        expect(session.server.world.time.fixedTick).toBe(0);
        session.clients[0]!.connection!.close();
        let message = "";
        try {
            await session.waitFor(session.clients[0]!, () => false, 1, "closed control");
        } catch (error) {
            message = String(error);
        }
        expect(message).toContain("client[0] tick=1 connection=closed");
    } finally {
        await session.dispose();
    }
});

for (const transport of ["link", "websocket"] as const) {
    test(`${transport} releases partially built sessions after a client initializer fails`, async () => {
        const worlds: World[] = [];
        let connection: Connection | undefined;
        let message = "";
        try {
            await createSession(
                {
                    ...game,
                    plugins: [
                        ...game.plugins,
                        {
                            name: "PlantedBuildFailure",
                            initialize(world) {
                                worlds.push(world);
                                if (worlds.length === 2)
                                    throw new Error("planted client build failure");
                            },
                        },
                    ],
                    onJoin(_world, _eid, transport) {
                        connection = transport;
                    },
                },
                { seed: 30, clients: 1, transport },
            );
        } catch (error) {
            message = String(error);
        }
        expect(message).toContain("planted client build failure");
        expect(worlds).toHaveLength(2);
        expect(worlds.every((world) => world.signal.aborted)).toBe(true);
        expect(connection!.closed).toBe(true);
    });
}

test("waits test immediately and predicate exceptions identify the peer", async () => {
    const session = await createSession(game, { seed: 31, clients: 1 });
    try {
        await session.waitFor(session.clients[0]!, () => true, 0, "already true");
        expect(session.server.world.time.fixedTick).toBe(0);
        let message = "";
        try {
            await session.waitFor(
                session.clients[0]!,
                () => {
                    throw new Error("predicate control");
                },
                10,
                "broken condition",
            );
        } catch (error) {
            message = String(error);
        }
        expect(message).toContain(
            "Wait predicate failed: broken condition (Error: predicate control)",
        );
        expect(message).toContain("client[0] tick=0 connection=open disposed=false seed=31");
    } finally {
        await session.dispose();
    }
});

test("faulted sessions print their seed and reproduce transport delivery", async () => {
    const output = spyOn(console, "info").mockImplementation(() => {});
    const run = async () => {
        const session = await createSession(game, {
            seed: 1234,
            clients: 1,
            faults: { minDelay: 20, maxDelay: 80, loss: 0.2, duplication: 0.1 },
        });
        try {
            const values: number[] = [];
            for (let tick = 0; tick < 30; tick++) {
                await session.advance();
                values.push([...session.clients[0]!.world.query([Value])].length);
            }
            expect(values[0]).toBe(0);
            expect(values.at(-1)).toBe(1);
            return values;
        } finally {
            await session.dispose();
        }
    };
    try {
        expect(await run()).toEqual(await run());
        expect(output).toHaveBeenCalledWith("multiplayer session seed=1234 transport=link");
    } finally {
        output.mockRestore();
    }
});
