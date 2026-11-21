import { expect, setDefaultTimeout, test } from "bun:test";
import { TIMEOUT } from "@dylanebert/shallot/testing";
import { component, createApp, type Plugin, Time, u32, type World } from "../../engine";
import { ClientOf, type Game, local, Replicate, ReplicationReceiver, ServerSession } from ".";

// Thousands of server ticks against peers whose unconfirmed history grows to the limit.
setDefaultTimeout(TIMEOUT.node);

const Player = component("LocalRoomPlayer", { name: u32 });
const faults = { minDelay: 0, maxDelay: 0, loss: 0, duplication: 0 };

function fixture() {
    const joins: { world: World; eid: number }[] = [];
    const leaves: { world: World; eid: number }[] = [];
    const receivers = new Map<World, ReplicationReceiver>();
    const game: Game = {
        plugins: [{ name: "LocalRoomPlayer", components: [Player] }],
        schema: [Player],
        maxClients: 2,
        reconnectMs: 100,
        rules: (_world, recipient) => [new Map([[recipient, [Player]]])],
        onJoin(world, eid) {
            joins.push({ world, eid });
            world.add(eid, Player, { name: joins.length });
            world.add(eid, Replicate, { rule: 0 });
        },
        onLeave(world, eid) {
            leaves.push({ world, eid });
        },
        onClient(world, connection) {
            receivers.set(world, new ReplicationReceiver(world, game.schema, connection));
        },
        validateIntent: () => true,
        applyIntent() {},
    };
    async function view(plugin: Plugin) {
        const app = await createApp({ defaults: false, plugins: [plugin] });
        app.world.step(0);
        return {
            app,
            sees() {
                receivers.get(app.world)!.flush();
                return [...app.world.query([Player])]
                    .map((eid) => app.world.get(eid, Player)!.name)
                    .sort();
            },
        };
    }
    return { game, joins, leaves, view };
}

async function unconfirmedSeats(reconnectMs: number | undefined) {
    const limit = Math.round(30 / Time.FIXED_DT);
    const { game, joins, leaves, view } = fixture();
    const host = await local(
        { ...game, maxClients: 3, reconnectMs },
        { seed: 31, faults, rooms: true },
    );
    const silent = await host.join();
    const stopping = await host.join();
    const live = await host.join();
    const stoppingClient = await view(stopping.plugin);
    const liveClient = await view(live.plugin);
    const world = joins[0]!.world;
    const authority = world.resource(ServerSession).server!;
    const peer = (index: number) => authority.peers.get(joins[index]!.eid);
    const closed = (index: number) =>
        reconnectMs ? peer(index)?.held === true : !authority.peers.has(joins[index]!.eid);
    const advance = (acknowledging: boolean) => {
        host.pump(Time.FIXED_DT);
        liveClient.sees();
        if (acknowledging) stoppingClient.sees();
    };
    try {
        advance(true);
        const admitted = peer(0)!.since;
        while (world.time.fixedTick < admitted + limit) advance(true);
        expect(closed(0)).toBe(false);
        advance(true);
        expect(closed(0)).toBe(true);
        expect(closed(1)).toBe(false);
        // The stopping client's last acknowledgement reaches the server on this tick.
        advance(false);
        const confirmed = peer(1)!.sender.acknowledged;
        expect(confirmed).toBeGreaterThan(admitted);
        while (world.time.fixedTick < confirmed + limit) advance(false);
        expect(closed(1)).toBe(false);
        advance(false);
        expect(closed(1)).toBe(true);
        if (reconnectMs) {
            expect(leaves).toEqual([]);
            expect(host.list()).toEqual([{ code: silent.code, clients: 3, maxClients: 3 }]);
            const seat = joins[1]!;
            world.storage(Player).name.set(seat.eid, 9);
            expect(stopping.resume()).toBe(true);
            expect(stoppingClient.sees()).toEqual([1, 3, 9]);
            expect([...world.query([ClientOf])]).toContain(seat.eid);
            expect(peer(1)?.held).toBeFalsy();
            advance(true);
            expect(peer(1)!.sender.acknowledged).toBeGreaterThan(confirmed + limit);
            expect(joins).toHaveLength(3);
        } else {
            expect(leaves).toEqual([joins[0]!, joins[1]!]);
            expect(stopping.resume()).toBe(false);
        }
        for (let i = 0; i < 120; i++) advance(false);
        expect(closed(2)).toBe(false);
        expect(world.time.fixedTick - peer(2)!.sender.acknowledged).toBeLessThan(4);
    } finally {
        stoppingClient.app.dispose();
        liveClient.app.dispose();
        host.dispose();
    }
}

test("a seat whose oldest unconfirmed frame is 30 s old is held and resumes on a fresh image", () =>
    unconfirmedSeats(100_000));

test("a seat whose oldest unconfirmed frame is 30 s old leaves without a reconnect window", () =>
    unconfirmedSeats(undefined));
