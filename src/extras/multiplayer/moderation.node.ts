import { expect, setDefaultTimeout, test } from "bun:test";
import { TIMEOUT } from "@dylanebert/shallot/testing";

setDefaultTimeout(TIMEOUT.node);

import { component, u32, World } from "../../engine";
import {
    type ClientId,
    type Game,
    IntentSender,
    Replicate,
    ReplicationReceiver,
    RoomLeader,
    RoomModeration,
    ServerSession,
    WebSocketConnection,
} from ".";
import { serveRooms } from "./bun";

const Member = component("ModerationMember", { order: u32 });
const schema = [Member, RoomLeader];

function fixture(authenticated: boolean) {
    const clients: World[] = [];
    const ids: ClientId[] = [];
    const results: boolean[] = [];
    let authority: World | undefined;
    const game: Game = {
        plugins: [{ name: "ModerationMember", components: schema }],
        schema,
        maxClients: 3,
        reconnectMs: 500,
        rules: (_world, eid) => [new Map([[eid, schema]])],
        onAuth: authenticated
            ? async (credential) => ({ playerId: credential!.split("/")[0]! })
            : undefined,
        onCreate(world) {
            authority = world;
        },
        onJoin(world, eid, _transport, id) {
            ids.push(id);
            world.add(eid, Member, { order: ids.length });
            world.add(eid, Replicate, { rule: 0 });
        },
        validateIntent: (_world, _eid, record) => record.kind === "operation",
        applyIntent(world, eid, tick, records) {
            const moderation = world.resource(RoomModeration).controller!;
            const requester = world.resource(ServerSession).server!.peers.get(eid)!.clientId;
            for (const record of records) {
                if (record.tick !== tick) continue;
                const [kick, target] = record.values;
                results.push(
                    kick
                        ? moderation.kick(requester, target as ClientId)
                        : moderation.promote(requester, target as ClientId),
                );
            }
        },
    };
    const host = serveRooms(game, { hostname: "127.0.0.1", port: 0, reconnectNow: () => 0 });
    function peer(query: Record<string, string>) {
        const params = new URLSearchParams({ announceRoom: "1", ...query });
        const socket = new WebSocket(`ws://127.0.0.1:${host.port}/?${params}`);
        const closed = new Promise<CloseEvent>((resolve) =>
            socket.addEventListener("close", resolve, { once: true }),
        );
        const grant = Promise.withResolvers<{ seat: string; resume: string }>();
        let code = "";
        const connection = new WebSocketConnection(socket, (message) => {
            const [kind, value, token] = message.split(":");
            if (kind === "room") code = value!;
            else if (kind === "resume") grant.resolve({ seat: value!, resume: token! });
            else return false;
            return true;
        });
        const world = new World();
        clients.push(world);
        const receiver = new ReplicationReceiver(world, schema, connection);
        const sender = new IntentSender(connection);
        async function until(ready: () => boolean) {
            const deadline = performance.now() + 500;
            do {
                sender.flush(receiver.flush());
                if (ready()) return;
                if (connection.closed) throw new Error("Peer closed before confirmation");
                await Bun.sleep(1);
            } while (performance.now() < deadline);
            throw new Error("Peer did not confirm room state");
        }
        return {
            socket,
            closed,
            grant: grant.promise,
            get code() {
                return code;
            },
            sees(members: number, leader: number) {
                return until(() => {
                    const [eid] = world.query([Member, RoomLeader]);
                    return (
                        [...world.query([Member])].length === members &&
                        eid !== undefined &&
                        world.get(eid, Member)!.order === leader
                    );
                });
            },
            async request(kick: boolean, target: number) {
                const index = results.length;
                sender.record(receiver.confirmed, "operation", [Number(kick), ids[target - 1]!]);
                sender.flush();
                await until(() => results.length > index);
                return results[index];
            },
        };
    }
    return {
        peer,
        async held(order: number) {
            const server = authority!.resource(ServerSession).server!;
            const deadline = performance.now() + 500;
            while (!server.peers.get(server.entity(ids[order - 1]!)!)?.held) {
                if (performance.now() >= deadline) throw new Error("Seat was not held");
                await Bun.sleep(1);
            }
        },
        stop() {
            host.stop();
            for (const client of clients) client.dispose();
        },
    };
}

test.each(["live", "held"] as const)(
    "only the leader can promote or kick; a kicked %s seat loses resume and its PlayerId cannot rejoin",
    async (state) => {
        const host = fixture(true);
        try {
            const leader = host.peer({ intent: "joinOrCreate", credential: "leader/1" });
            await leader.grant;
            const join = { intent: "join", code: leader.code };
            const successor = host.peer({ ...join, credential: "successor/1" });
            await successor.grant;
            const target = host.peer({ ...join, credential: "target/1" });
            const grant = await target.grant;
            const peers = [leader, successor, target];
            await Promise.all(peers.map((peer) => peer.sees(3, 1)));
            expect(await successor.request(false, 3)).toBe(false);
            expect(await successor.request(true, 3)).toBe(false);
            expect(await leader.request(false, 2)).toBe(true);
            await Promise.all(peers.map((peer) => peer.sees(3, 2)));
            expect(await leader.request(true, 3)).toBe(false);
            expect(await successor.request(true, 2)).toBe(false);
            if (state === "held") {
                target.socket.close();
                await host.held(3);
                expect(await successor.request(false, 3)).toBe(false);
            }
            expect(await successor.request(true, 3)).toBe(true);
            if (state === "live") expect((await target.closed).reason).toBe("Kicked from room");
            const resumed = await host.peer({ ...join, ...grant }).closed;
            if (state === "held") {
                expect(resumed).toMatchObject({ code: 4003, reason: "Kicked from room" });
            } else {
                expect(resumed).toMatchObject({
                    code: 1008,
                    reason: "Invalid or expired reconnection token",
                });
            }
            expect((await host.peer({ ...join, credential: "target/2" }).closed).reason).toBe(
                "Kicked from room",
            );
            const fresh = host.peer({ ...join, credential: "fresh/1" });
            await Promise.all([leader, successor, fresh].map((peer) => peer.sees(3, 2)));
            expect(await successor.request(true, 3)).toBe(false);
        } finally {
            host.stop();
        }
    },
);

test("an anonymous kick revokes the seat grant, not a fresh unidentified join", async () => {
    const host = fixture(false);
    try {
        const leader = host.peer({ intent: "joinOrCreate" });
        await leader.grant;
        const join = { intent: "join", code: leader.code };
        const target = host.peer(join);
        const grant = await target.grant;
        await leader.sees(2, 1);
        expect(await leader.request(true, 2)).toBe(true);
        expect((await host.peer({ ...join, ...grant }).closed).reason).toMatch(/token/i);
        await host.peer(join).sees(2, 1);
    } finally {
        host.stop();
    }
});
