import { expect, setDefaultTimeout, test } from "bun:test";
import {
    Character,
    createApp,
    GlobalTransform,
    Time,
    Transform,
    type World,
} from "@dylanebert/shallot";
import { local } from "@dylanebert/shallot/multiplayer";
import { createSession } from "@dylanebert/shallot/multiplayer/testing";
import { TIMEOUT } from "@dylanebert/shallot/testing";
import { ClientSession, GAME, scene } from "./network";
import { LocalPlayer, Player } from "./player";

setDefaultTimeout(TIMEOUT.node);

test("a room's roster and avatar survive reconnection and reset a vacated seat for a fresh occupant", async () => {
    let joinedWorld: World | undefined;
    const host = await local(
        {
            ...GAME,
            onJoin(...args) {
                GAME.onJoin!(...args);
                joinedWorld = args[0];
            },
        },
        {
            rooms: true,
            seed: 7,
            faults: { minDelay: 0, maxDelay: 0, loss: 0, duplication: 0 },
        },
    );
    try {
        const seat = await host.join();
        const { code, creator } = host.create()!;
        const privateSeat = await host.join(code, creator);
        const a = await createApp({ defaults: false, plugins: [seat.plugin] });
        const b = await createApp({ defaults: false, plugins: [privateSeat.plugin] });
        try {
            const step = (count: number) => {
                for (let i = 0; i < count; i++) {
                    host.pump(Time.FIXED_DT);
                    a.world.step(Time.FIXED_DT);
                    b.world.step(Time.FIXED_DT);
                }
            };
            step(12);
            const peer = a.world.resource(ClientSession).client!;
            const other = b.world.resource(ClientSession).client!;
            expect(peer.roster).toBe(0b0001);
            expect(other.roster).toBe(0b0001);
            const botAvatar = scene(a.world).players[1]!.player;
            const botZ = () => a.world.storage(GlobalTransform).translation.z.get(botAvatar);
            const bot = await seat.addBot({
                seed: 11,
                input(_world, sender, tick) {
                    sender.record(tick + 1, "input", [0, 1, 0.2, 0.1, 0, 0]);
                },
            });
            step(12);
            expect(peer.roster).toBe(0b0011);
            expect(other.roster).toBe(0b0001);
            const avatar = scene(a.world).players[peer.slot]!.player;
            const z = () => a.world.storage(GlobalTransform).translation.z.get(avatar);
            peer.record([0, 1, 0, 0, 0, 0]);
            step(30);
            expect(z()).toBeLessThan(11);
            seat.cut();
            peer.record([0, 0, 0, 0, 0, 0]);
            step(3);
            expect(seat.resume()).toBe(true);
            step(12);
            expect(peer.connection.generation).toBe(1);
            expect(peer.slot).toBe(0);
            expect(z()).toBeLessThan(11);
            expect(botZ()).toBeLessThan(5);
            bot.remove();
            step(12);
            expect(peer.roster).toBe(0b0001);
            const replacement = await host.join();
            const authority = joinedWorld!;
            const { player, eye } = scene(authority).players[1]!;
            const look = authority.get(player, Player)!;
            expect(look.camera).toBe(eye);
            expect(look.yaw).toBe(0);
            expect(look.pitch).toBe(0);
            expect(authority.storage(Character).velocity.z.get(player)).toBe(0);
            const c = await createApp({ defaults: false, plugins: [replacement.plugin] });
            try {
                step(12);
                c.world.step(Time.FIXED_DT);
                const fresh = c.world.resource(ClientSession).client!;
                expect(fresh.slot).toBe(1);
                expect(peer.roster).toBe(0b0011);
                expect(botZ()).toBeCloseTo(5.3);
            } finally {
                c.dispose();
            }
        } finally {
            a.dispose();
            b.dispose();
        }
        expect(host.list()).toEqual([]);
    } finally {
        host.dispose();
    }
});

test("the local camera follows its avatar's interpolated placement between fixed ticks", async () => {
    const session = await createSession(GAME, { seed: 10, clients: 1 });
    try {
        const peer = session.clients[0]!;
        peer.world.step(0);
        const client = peer.world.resource(ClientSession).client!;
        const { player, eye } = scene(peer.world).players[client.slot]!;
        peer.world.add(player, LocalPlayer);
        peer.world.storage(LocalPlayer).camera.set(player, eye);
        for (let i = 0; i < 60; i++) {
            client.record([0, 1, 0, 0, 0, 0]);
            await session.advance();
        }
        const z: number[] = [];
        for (let i = 0; i < 12; i++) {
            client.record([0, 1, 0, 0, 0, 0]);
            peer.world.step(Time.FIXED_DT / 2);
            z.push(peer.world.storage(Transform).translation.z.get(eye));
        }
        for (let i = 1; i < z.length; i++) expect(z[i]).toBeLessThan(z[i - 1]!);
    } finally {
        await session.dispose();
    }
});
