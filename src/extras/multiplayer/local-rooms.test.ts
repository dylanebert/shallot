import { expect, test } from "bun:test";
import { component, createApp, type Plugin, Time, u32, type World } from "../../engine";
import { ClientOf, type Game, local, Replicate, ReplicationReceiver } from ".";

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

test("a running room replicates an added bot only there and immediately leaves on removal", async () => {
    const { game, joins, leaves, view } = fixture();
    const host = await local(game, { seed: 11, faults, rooms: true });
    const a = await host.join();
    const { code: privateCode, creator } = host.create()!;
    const b = await host.join(privateCode.toLowerCase(), creator);
    const av = await view(a.plugin);
    const bv = await view(b.plugin);
    try {
        host.pump(Time.FIXED_DT);
        expect(av.sees()).toEqual([1]);
        expect(bv.sees()).toEqual([2]);
        let botPlayers: number[] = [];
        const bot = await a.addBot({
            seed: 42,
            input(world) {
                botPlayers = [...world.query([Player])]
                    .map((eid) => world.get(eid, Player)!.name)
                    .sort();
            },
        });
        host.pump(Time.FIXED_DT);
        expect(av.sees()).toEqual([1, 3]);
        expect(bv.sees()).toEqual([2]);
        expect(botPlayers).toEqual([1, 3]);
        expect(joins[2]!.world).toBe(joins[0]!.world);
        expect(host.list()).toEqual([{ code: a.code, clients: 2, maxClients: 2 }]);
        bot.remove();
        bot.remove();
        expect(leaves).toEqual([joins[2]!]);
        host.pump(Time.FIXED_DT);
        expect(av.sees()).toEqual([1]);
        expect(bv.sees()).toEqual([2]);
        av.app.dispose();
        expect(leaves).toEqual([joins[2]!, joins[0]!]);
        expect(joins[0]!.world.signal.aborted).toBe(true);
        expect(joins[1]!.world.signal.aborted).toBe(false);
    } finally {
        av.app.dispose();
        bv.app.dispose();
        host.dispose();
    }
});

test("a cut local stream resumes its original seat at a fresh image, then expires on another cut", async () => {
    const { game, joins, leaves, view } = fixture();
    const host = await local({ ...game, maxClients: 1 }, { seed: 21, faults, rooms: true });
    const seat = await host.join();
    const client = await view(seat.plugin);
    try {
        host.pump(Time.FIXED_DT);
        expect(client.sees()).toEqual([1]);
        seat.cut();
        host.pump(0.001);
        expect(leaves).toEqual([]);
        expect([...joins[0]!.world.query([ClientOf])]).toEqual([joins[0]!.eid]);
        expect(host.list()).toEqual([{ code: seat.code, clients: 1, maxClients: 1 }]);
        await expect(host.join(seat.code)).rejects.toThrow("Local client refused");
        const original = joins[0]!;
        original.world.storage(Player).name.set(original.eid, 9);
        expect(seat.resume()).toBe(true);
        expect(client.sees()).toEqual([9]);
        expect(joins).toEqual([original]);
        expect(leaves).toEqual([]);

        seat.cut();
        host.pump(0.099);
        expect(leaves).toEqual([]);
        host.pump(0.002);
        expect(leaves).toEqual([original]);
        expect(host.list()).toEqual([]);
        expect(original.world.signal.aborted).toBe(true);
        expect(seat.resume()).toBe(false);
        const next = await host.join();
        expect(next.code).not.toBe("");
        expect(joins).toHaveLength(2);
        expect(joins[1]!.world).not.toBe(original.world);
    } finally {
        client.app.dispose();
        host.dispose();
    }
});
